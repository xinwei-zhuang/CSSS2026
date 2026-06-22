from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from einops import reduce

from config import Config
from model import CASunGroup


@contextmanager
def device_autocast(device, dtype=torch.bfloat16):
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


class Feature:
    def __init__(self, config):
        pass

    def on_init(self, world):
        pass

    def before_step(self, world):
        pass

    def after_step(self, world, grid):
        pass


class SimpleBurnInFeature(Feature):
    """
    Increments the number of steps you want to take
    """

    def __init__(self, config: Config):
        # TODO: You can also make a new feature to customize how burn-in works
        self.initial_steps_before = 0
        self.initial_steps_per = 1
        self.target_steps_before = config.steps_before_update
        self.target_steps_per = config.steps_per_update
        self.increment = config.burn_in_increment
        self.increment_epochs = config.burn_in_increment_epochs

    def on_init(self, world):
        world.state = getattr(world, "state", {})
        world.state["steps_before_update"] = self.initial_steps_before
        world.state["steps_per_update"] = self.initial_steps_per
        world.state["last_burn_in_epoch"] = 0

    def before_step(self, world):
        # Check if we should increment
        if world.epoch - world.state["last_burn_in_epoch"] >= self.increment_epochs:
            world.state["steps_before_update"] = min(
                world.state["steps_before_update"] + self.increment,
                self.target_steps_before,
            )
            world.state["steps_per_update"] = min(
                world.state["steps_per_update"] + self.increment, self.target_steps_per
            )
            world.state["last_burn_in_epoch"] = world.epoch

            # Log progress
            # progress = world.state['steps_per_update'] / self.target_steps_per
            # stats['burn_in_progress'] = progress


class SunUpdateFeature(Feature):
    def __init__(self, config: Config):
        self.update_interval = config.sun_update_epoch_wait

    def on_init(self, world):
        world.state = getattr(world, "state", {})
        world.state["update_sun"] = False

    def before_step(self, world):
        # Determine if sun should update this epoch
        world.state["update_sun"] = (
            world.epoch > 0
            and self.update_interval > 0
            and world.epoch % self.update_interval == 0
        )


class UpdatePoolWithNondeadFeature(Feature):
    """
    Goes through and updates pool with any runs which doesn't have a NCA which died out
    """

    def __init__(self, config: Config):
        self.n_ncas = config.n_ncas

    def after_step(self, world, grid):
        # If there are any in the batch where a NCA has died out, replace it with something in pool
        nca_growth = reduce(grid[:, 1 : self.n_ncas + 1], "b c h w -> b c", "sum")
        alive_mask = torch.all(nca_growth > 0, dim=1)  # [B, ]

        # Update pool with anything still valid
        # NOTE: If this happens after the new_seeds, then pool_idxs is updated which is bad
        world.pool[world.state["pool_idxs"][alive_mask]] = grid[alive_mask]


class World:
    """World class for managing the training environment.

    Foundation for the entire simulation
    """

    def __init__(self, config: Config) -> None:
        """
        Takes in a config and initializes the world.
        """
        self.config = config
        self.device = config.device

        self.epoch = 0
        self.steps_taken = 0

        # Nice to saves
        self.batch_size = config.batch_size
        self.pool_size = config.pool_size
        self.seed_dim = config.cell_wo_alive_dim
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float

        # State dict that features can modify
        self.state = {
            "steps_before_update": config.steps_before_update,
            "steps_per_update": config.steps_per_update,
            "update_sun": False,
            "pool_idxs": [],
            "mode": config.mode,
        }

        self._init_pool(config)
        self.features = self._build_features(config)

        for feature in self.features:
            feature.on_init(self)

    def get_seed(self):
        if self.batch_size < self.pool_size:
            self.state["pool_idxs"] = torch.randint(
                self.pool_size, size=(self.batch_size,), device=self.device
            )
        else:
            self.state["pool_idxs"] = torch.arange(self.batch_size, device=self.device)

        return self.pool[self.state["pool_idxs"]]

    def step(self, group, grid):
        # Run all the before steps of the features
        for feature in self.features:
            feature.before_step(self)

        if self.config.city_mode and hasattr(group, "set_city_environment_phase"):
            group.set_city_environment_phase(self.steps_taken)

        # Calculations
        steps_before = self.state.get("steps_before_update", 0)
        steps_per = self.state.get("steps_per_update", 1)
        epoch_steps = steps_before + steps_per

        # Run steps
        grid_storage = torch.zeros((epoch_steps, *grid.shape))
        # Use conditional autocast
        with device_autocast(self.device):
            with torch.no_grad():
                _, grids, _ = group(grid, steps_before)
                if steps_before:
                    grid_storage[:steps_before] = grids
                    grid = grids[-1]

        with device_autocast(self.device):
            grid_batch, grids, forward_stats = group(grid, steps_per)
            grid_storage[steps_before:] = grids
            grid = grids[-1]

        # Run all the after steps of the features
        for feature in self.features:
            feature.after_step(self, grid)

        # Update internal stats
        self.epoch += 1
        self.steps_taken += epoch_steps

        return self._get_stats_and_new_grid(
            group, grid_batch, grid, grid_storage.detach(), forward_stats
        )

    def save(self, config: Config, run_name: str) -> None:
        np.save(f"{run_name}/seed.npy", self.seed_update.float().detach().cpu().numpy())

    def load(self, loc: str) -> bool:
        """Load previous parameters for the World.

        Args:
            loc: Directory path containing saved world files.

        Returns:
            True if loading was successful, False otherwise.
        """
        self.seed_update = torch.from_numpy(np.load(f"{loc}/seed.npy")).to(self.device)
        self._init_pool()
        return True

    def _init_pool(self, config: Config):
        """Setup the pool of possible training configurations"""
        self.pool = torch.zeros(
            config.pool_size,
            config.cell_dim,
            *config.grid_size,
            device=config.device,
            dtype=self.dtype,
        )

        if config.seed_dist == "scatter":
            # Generate seeds
            all_coords = torch.cartesian_prod(
                torch.arange(config.grid_size[0], device=config.device),
                torch.arange(config.grid_size[1], device=config.device),
            )

            # Generate all seeds for all NCAs at once
            seed_idxs = torch.stack(
                [
                    torch.randperm(config.total_grid_size, device=config.device)[
                        : config.n_ncas * config.n_seeds
                    ].view(config.n_ncas, config.n_seeds)
                    for _ in range(config.pool_size)
                ]
            )
            seed_pts = all_coords[seed_idxs]
            init_xs = seed_pts[:, :, :, 0]
            init_ys = seed_pts[:, :, :, 1]
        elif config.seed_dist == "city_anchors":
            init_xs, init_ys = self._city_anchor_seed_points(config)
        else:
            raise ValueError(f"Unknown seed_dist: {config.seed_dist}")

        batch_indices = (
            torch.arange(config.pool_size)
            .view(-1, 1, 1)
            .expand(-1, config.n_ncas, config.n_seeds)
        )
        nca_indices = (
            torch.arange(config.n_ncas)
            .view(1, -1, 1)
            .expand(config.pool_size, -1, config.n_seeds)
        )

        batch_flat = batch_indices.flatten()
        nca_flat = nca_indices.flatten()
        xs_flat = init_xs.flatten()
        ys_flat = init_ys.flatten()

        if config.seed_mode == "solid":
            seed_vals = 1.0
        elif config.seed_mode == "random":
            if config.city_mode:
                seed_vals = self._city_seed_vectors(config)
            else:
                seed_vals = torch.randn(
                    (config.n_ncas, self.seed_dim),
                    device=config.device,
                    dtype=self.dtype,
                )  # [N, S]

            # Handle loading
            # TODO: This is a tad suspicious isn't it...
            if hasattr(self, "seed_update"):
                seed_vals = config.seed_update
            else:
                self.seed_update = seed_vals

            seed_vals = seed_vals / seed_vals.norm(dim=-1, keepdim=True).to(self.dtype)
            seed_vals = seed_vals.repeat_interleave(config.n_seeds, dim=0)
            seed_vals = (
                seed_vals.unsqueeze(0)  # [1, NS*N, SS]
                .expand(config.pool_size, -1, -1, -1)
                .reshape(-1, self.seed_dim)
            )

        # Set so the sun is immediately alive at all places
        self.pool[:, 0] = 1.0

        self.pool[batch_flat, config.alive_dim :, xs_flat, ys_flat] = seed_vals
        self.pool[batch_flat, 0, xs_flat, ys_flat] = 0.0
        self.pool[batch_flat, nca_flat + 1, xs_flat, ys_flat] = 1.0

        # Normalize the aliveness
        self.pool[:, : config.alive_dim] /= self.pool[:, : config.alive_dim].sum(
            dim=1, keepdim=True
        )

    def _city_seed_vectors(self, config: Config) -> torch.Tensor:
        """Structured initial cell states for city tribes.

        The first three NCAs are interpreted as:
        0. solar-rich / low-rise support tissue
        1. dense high-load / commercial tissue
        2. storage / critical-support tissue
        Extra NCAs get small random variants.
        """
        archetypes = torch.randn(
            (config.n_ncas, self.seed_dim),
            device=config.device,
            dtype=self.dtype,
        ) * 0.08

        base = torch.tensor(
            [
                [0.90, -0.25, 0.35, -0.10],
                [-0.15, 0.90, -0.20, 0.35],
                [0.20, 0.10, 0.95, -0.35],
            ],
            device=config.device,
            dtype=self.dtype,
        )
        state_dim = min(config.cell_state_dim, base.shape[1])
        for i in range(min(config.n_ncas, base.shape[0])):
            archetypes[i, :state_dim] = base[i, :state_dim]
            if config.cell_hidden_dim > 0:
                start = config.cell_state_dim
                stop = min(self.seed_dim, start + config.cell_hidden_dim)
                archetypes[i, start:stop] = (i + 1) / max(1, config.n_ncas)

        return archetypes

    def _city_anchor_seed_points(self, config: Config) -> tuple[torch.Tensor, torch.Tensor]:
        """Seed each NCA around a persistent city niche."""
        height, width = config.grid_size
        anchors = torch.tensor(
            [
                [int(height * 0.24), int(width * 0.72)],  # solar support tissue
                [int(height * 0.72), int(width * 0.24)],  # dense load tissue
                [int(height * 0.50), int(width * 0.50)],  # storage / support tissue
            ],
            device=config.device,
        )
        if config.n_ncas > anchors.shape[0]:
            angles = torch.linspace(
                0,
                2 * torch.pi,
                config.n_ncas - anchors.shape[0] + 1,
                device=config.device,
            )[:-1]
            radius = max(1, min(height, width) // 4)
            center = torch.tensor([height // 2, width // 2], device=config.device)
            ring = torch.stack(
                [
                    center[0] + torch.round(torch.sin(angles) * radius).long(),
                    center[1] + torch.round(torch.cos(angles) * radius).long(),
                ],
                dim=1,
            )
            anchors = torch.cat([anchors, ring], dim=0)
        anchors = anchors[: config.n_ncas]
        anchors[:, 0] = anchors[:, 0].clamp(0, height - 1)
        anchors[:, 1] = anchors[:, 1].clamp(0, width - 1)

        jitter = torch.randint(
            -1,
            2,
            (config.pool_size, config.n_ncas, config.n_seeds, 2),
            device=config.device,
        )
        pts = anchors.view(1, config.n_ncas, 1, 2) + jitter
        pts[..., 0] = pts[..., 0].clamp(0, height - 1)
        pts[..., 1] = pts[..., 1].clamp(0, width - 1)

        return pts[..., 0], pts[..., 1]

    def _build_features(self, config):
        features = []

        if config.burn_in:
            features.append(SimpleBurnInFeature(config))

        if config.sun_update_epoch_wait > 0:
            features.append(SunUpdateFeature(config))

        if config.mode == "train":
            features.append(UpdatePoolWithNondeadFeature(config))
        return features

    def _get_stats_and_new_grid(
        self,
        group: CASunGroup,
        grid_batch: torch.Tensor,
        grid: torch.Tensor,
        grids: torch.Tensor,
        forward_stats: dict[str, Any],
    ):
        stats = {}
        if self.state["mode"] != "frozen_eval" and self.state["steps_per_update"] > 0:
            stats = group.update_models(grid_batch, self.state["update_sun"])
        stats["forward"] = forward_stats

        grid = grid.detach().requires_grad_(True)

        return stats, grid, grids
