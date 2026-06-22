from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


TRIBES = [
    {
        "key": "A",
        "name": "solar",
        "color": (37, 99, 235),
        "morph": 0.25,
        "load": 0.82,
        "solar": 1.45,
        "storage": 0.76,
    },
    {
        "key": "B",
        "name": "load",
        "color": (217, 119, 6),
        "morph": 0.72,
        "load": 1.28,
        "solar": 0.62,
        "storage": 0.92,
    },
    {
        "key": "C",
        "name": "storage",
        "color": (21, 128, 61),
        "morph": 0.48,
        "load": 0.95,
        "solar": 0.95,
        "storage": 1.52,
    },
]

# A beats B, B beats C, C beats A.
BEATS = {0: 1, 1: 2, 2: 0}


@dataclass
class Cell:
    alive: bool = False
    tribe: int = -1
    morph: float = 0.5
    health: float = 0.0
    storage: float = 0.0
    storage_cap: float = 0.0
    load: float = 0.0
    solar_cap: float = 0.0
    unmet: float = 0.0
    critical: bool = False


class CityPreview:
    def __init__(self, size: int, seed: int):
        self.size = size
        self.rng = random.Random(seed)
        self.t = 0
        self.cells = [Cell() for _ in range(size * size)]
        self.history = []
        self.reset()

    def reset(self) -> None:
        self.cells = [Cell() for _ in range(self.size * self.size)]
        anchors = [
            (int(self.size * 0.25), int(self.size * 0.72), 0),
            (int(self.size * 0.70), int(self.size * 0.28), 1),
            (int(self.size * 0.50), int(self.size * 0.50), 2),
        ]
        for row, col, tribe in anchors:
            for _ in range(5):
                rr = min(self.size - 1, max(0, row + self.rng.randint(-1, 1)))
                cc = min(self.size - 1, max(0, col + self.rng.randint(-1, 1)))
                self.activate(rr * self.size + cc, tribe, critical=(tribe == 2 and self.rng.random() < 0.45))

    def activate(self, idx: int, tribe: int, critical: bool = False, morph: float | None = None) -> None:
        spec = TRIBES[tribe]
        m = spec["morph"] + self.rng.gauss(0, 0.055) if morph is None else morph
        m = min(0.97, max(0.03, m))
        cell = self.cells[idx]
        cell.alive = True
        cell.tribe = tribe
        cell.morph = m
        cell.health = 0.86 + self.rng.random() * 0.14
        cell.critical = critical
        roof = max(0.18, 1.18 - m)
        density = 0.62 + 1.18 * m
        cell.load = density * spec["load"] * (1.34 if critical else 1.0)
        cell.solar_cap = roof * spec["solar"] * 1.25
        cell.storage_cap = (1.2 + 1.7 * (1.0 - abs(m - 0.5))) * spec["storage"] * (1.7 if critical else 1.0)
        cell.storage = cell.storage_cap * (0.38 + self.rng.random() * 0.18)
        cell.unmet = 0.0

    def neighbors(self, idx: int) -> list[int]:
        row, col = divmod(idx, self.size)
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    out.append(rr * self.size + cc)
        return out

    def sunlight(self) -> float:
        phase = (self.t % 24) / 24.0
        return max(0.0, math.sin(math.pi * phase))

    def demand_wave(self) -> float:
        phase = (self.t % 24) / 24.0
        evening = math.exp(-((phase - 0.78) / 0.16) ** 2)
        morning = 0.55 * math.exp(-((phase - 0.34) / 0.13) ** 2)
        return 0.76 + 0.36 * evening + 0.20 * morning

    def resource_step(self) -> tuple[float, float]:
        sun = self.sunlight()
        demand_mult = self.demand_wave()
        total_served = 0.0
        total_load = 0.0

        for cell in self.cells:
            if not cell.alive:
                continue
            generation = cell.solar_cap * sun
            demand = cell.load * demand_mult
            total_load += demand
            available = cell.storage + generation
            served = min(demand, available)
            total_served += served
            cell.storage = min(cell.storage_cap, max(0.0, available - demand))
            cell.unmet = 0.0 if demand <= 0 else max(0.0, demand - available) / demand
            night_stress = 0.025 if sun < 0.08 and cell.tribe != 2 else 0.0
            cell.health -= 0.105 * cell.unmet + night_stress
            cell.health = min(1.0, cell.health + 0.018 * (1.0 - cell.unmet) + 0.01 * (cell.storage / max(cell.storage_cap, 1e-6)))

        donors = [i for i, c in enumerate(self.cells) if c.alive and c.storage > c.storage_cap * 0.58]
        receivers = [i for i, c in enumerate(self.cells) if c.alive and c.unmet > 0.03]
        for receiver_idx in receivers:
            receiver = self.cells[receiver_idx]
            neigh_donors = [i for i in self.neighbors(receiver_idx) if i in donors]
            neigh_donors.sort(
                key=lambda i: (
                    -self.cells[i].storage,
                    abs(self.cells[i].morph - receiver.morph),
                )
            )
            for donor_idx in neigh_donors[:3]:
                donor = self.cells[donor_idx]
                coherence = math.exp(-2.1 * abs(donor.morph - receiver.morph))
                gift = min(donor.storage - donor.storage_cap * 0.35, receiver.unmet * receiver.load * 0.38 * coherence)
                if gift <= 0:
                    continue
                donor.storage -= gift
                receiver.storage = min(receiver.storage_cap, receiver.storage + gift)
                receiver.unmet = max(0.0, receiver.unmet - gift / max(receiver.load, 1e-6))
                receiver.health = min(1.0, receiver.health + gift / max(receiver.load, 1e-6) * 0.075)

        return total_served, total_load

    def choose_birth_tribe(self, live_neighbors: list[Cell]) -> int:
        counts = [0, 0, 0]
        scores = [0.0, 0.0, 0.0]
        for cell in live_neighbors:
            counts[cell.tribe] += 1
        alive_total = sum(counts) or 1
        for cell in live_neighbors:
            minority = 1.0 + 0.9 * (1.0 - counts[cell.tribe] / alive_total)
            energy = cell.storage / max(cell.storage_cap, 1e-6)
            scores[cell.tribe] += (0.75 + cell.health + 0.45 * energy) * minority
        total = sum(scores)
        roll = self.rng.random() * total
        for tribe, score in enumerate(scores):
            roll -= score
            if roll <= 0:
                return tribe
        return max(range(3), key=lambda t: scores[t])

    def dynamics_step(self) -> dict[str, float]:
        deaths = []
        births = []
        takeovers = []
        alive_count = sum(1 for c in self.cells if c.alive)
        tribe_counts = [sum(1 for c in self.cells if c.alive and c.tribe == t) for t in range(3)]

        for idx, cell in enumerate(self.cells):
            neigh = [self.cells[i] for i in self.neighbors(idx)]
            live = [c for c in neigh if c.alive]
            if cell.alive:
                same = sum(1 for c in live if c.tribe == cell.tribe)
                enemy = [c for c in live if c.tribe != cell.tribe]
                crowd = max(0, len(live) - 5)
                isolation = 1 if len(live) <= 1 else 0
                majority_pressure = tribe_counts[cell.tribe] / max(1, alive_count)
                death_p = (
                    0.018
                    + 0.085 * cell.unmet
                    + 0.045 * isolation
                    + 0.018 * crowd
                    + 0.045 * max(0.0, majority_pressure - 0.48)
                    - 0.050 * (cell.storage / max(cell.storage_cap, 1e-6))
                    - (0.035 if cell.critical else 0.0)
                )
                if cell.health <= 0.04 or self.rng.random() < max(0.0, death_p):
                    deaths.append(idx)
                    continue

                if enemy:
                    attack = [0.0, 0.0, 0.0]
                    for e in enemy:
                        advantage = 0.34 if BEATS[e.tribe] == cell.tribe else -0.18
                        energy = e.storage / max(e.storage_cap, 1e-6)
                        minority = 1.0 + 0.65 * (1.0 - tribe_counts[e.tribe] / max(1, alive_count))
                        attack[e.tribe] += (0.22 + e.health + energy + advantage) * minority
                    attacker = max(range(3), key=lambda t: attack[t])
                    defense = (
                        0.72
                        + 0.72 * cell.health
                        + 0.55 * (cell.storage / max(cell.storage_cap, 1e-6))
                        + 0.16 * same
                        + (0.42 if cell.critical else 0.0)
                    )
                    if attack[attacker] > defense * (1.04 + self.rng.random() * 0.55):
                        takeovers.append((idx, attacker))
            elif len(live) >= 2:
                tribe = self.choose_birth_tribe(live)
                same = sum(1 for c in live if c.tribe == tribe)
                energy = sum(c.storage / max(c.storage_cap, 1e-6) for c in live) / len(live)
                density_penalty = max(0.0, alive_count / len(self.cells) - 0.72)
                birth_p = 0.07 + 0.06 * same + 0.065 * energy - 0.13 * density_penalty
                if self.rng.random() < max(0.0, birth_p):
                    avg_morph = sum(c.morph for c in live if c.tribe == tribe) / max(1, same)
                    target = TRIBES[tribe]["morph"]
                    morph = min(0.97, max(0.03, 0.56 * avg_morph + 0.44 * target + self.rng.gauss(0, 0.035)))
                    births.append((idx, tribe, morph))

        for idx in deaths:
            self.cells[idx] = Cell()
        for idx, tribe, morph in births:
            self.activate(idx, tribe, morph=morph)
        for idx, tribe in takeovers:
            cell = self.cells[idx]
            old_critical = cell.critical and self.rng.random() < 0.65
            morph = 0.54 * cell.morph + 0.46 * TRIBES[tribe]["morph"] + self.rng.gauss(0, 0.025)
            self.activate(idx, tribe, critical=old_critical, morph=min(0.97, max(0.03, morph)))
            self.cells[idx].health *= 0.68

        if alive_count < len(self.cells) * 0.16:
            empty = [i for i, c in enumerate(self.cells) if not c.alive]
            for _ in range(min(4, len(empty))):
                idx = empty.pop(self.rng.randrange(len(empty)))
                tribe = min(range(3), key=lambda t: tribe_counts[t])
                self.activate(idx, tribe)

        mix = 0.0
        alive_after = sum(1 for c in self.cells if c.alive)
        if alive_after:
            counts_after = [sum(1 for c in self.cells if c.alive and c.tribe == t) for t in range(3)]
            mix = 1.0 - sum((c / alive_after) ** 2 for c in counts_after)
            mix = mix / (1.0 - 1.0 / 3.0)

        return {
            "alive": alive_after,
            "births": len(births),
            "deaths": len(deaths),
            "takeovers": len(takeovers),
            "mix": mix,
        }

    def step(self) -> dict[str, float]:
        served, load = self.resource_step()
        stats = self.dynamics_step()
        stats["served_fraction"] = served / load if load else 1.0
        stats["sun"] = self.sunlight()
        self.t += 1
        self.history.append(stats)
        return stats

    def render(self, scale: int = 12, margin: int = 18) -> Image.Image:
        width = self.size * scale + 2 * margin
        height = self.size * scale + 2 * margin + 34
        img = Image.new("RGB", (width, height), (248, 250, 252))
        draw = ImageDraw.Draw(img)
        draw.rectangle([margin - 1, margin - 1, margin + self.size * scale, margin + self.size * scale], outline=(15, 23, 42))

        for idx, cell in enumerate(self.cells):
            row, col = divmod(idx, self.size)
            x = margin + col * scale
            y = margin + row * scale
            if cell.alive:
                base = TRIBES[cell.tribe]["color"]
                energy = cell.storage / max(cell.storage_cap, 1e-6)
                health = max(0.0, min(1.0, cell.health))
                bright = 0.48 + 0.34 * health + 0.18 * energy
                color = tuple(max(0, min(255, int(v * bright + 255 * 0.05 * energy))) for v in base)
            else:
                color = (226, 232, 240)
            draw.rectangle([x, y, x + scale - 1, y + scale - 1], fill=color)
            if cell.critical and cell.alive:
                draw.rectangle([x + scale // 3, y + scale // 3, x + scale * 2 // 3, y + scale * 2 // 3], fill=(245, 245, 245))

        stats = self.history[-1] if self.history else {"alive": 0, "mix": 0, "served_fraction": 0, "takeovers": 0, "sun": 0}
        label = (
            f"t={self.t:03d}  alive={stats['alive']:03.0f}  mix={stats['mix']:.2f}  "
            f"served={stats['served_fraction']:.2f}  takeovers={stats['takeovers']:02.0f}  sun={stats['sun']:.2f}"
        )
        draw.text((margin, margin + self.size * scale + 11), label, fill=(15, 23, 42))
        return img


def write_html(path: Path, seed: int, size: int, steps: int) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>City Petri Concept Preview</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #cbd5e1;
      --panel: #f8fafc;
      --paper: #ffffff;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: #eef2f7;
    }}
    main {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 1fr) 300px;
      gap: 0;
    }}
    .stage {{
      display: grid;
      place-items: center;
      padding: 24px;
      background: #e8eef5;
    }}
    canvas {{
      width: min(78vmin, 820px);
      aspect-ratio: 1;
      background: var(--paper);
      border: 1px solid var(--line);
      box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
    }}
    aside {{
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 760;
      letter-spacing: 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: var(--paper);
      border-radius: 6px;
      padding: 10px;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }}
    .metric strong {{
      display: block;
      margin-top: 5px;
      font-size: 19px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    button {{
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--paper);
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{ border-color: #64748b; }}
    .legend {{
      display: grid;
      gap: 8px;
      font-size: 13px;
    }}
    .row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 16px;
      height: 16px;
      border: 1px solid rgba(15, 23, 42, 0.18);
    }}
    .a {{ background: #2563eb; }}
    .b {{ background: #d97706; }}
    .c {{ background: #15803d; }}
    .dead {{ background: #e2e8f0; }}
    .note {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    @media (max-width: 760px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
      canvas {{ width: min(92vw, 620px); }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="stage">
      <canvas id="grid" width="840" height="840"></canvas>
    </section>
    <aside>
      <h1>City Petri Preview</h1>
      <div class="metrics">
        <div class="metric"><span>step</span><strong id="mStep">0</strong></div>
        <div class="metric"><span>alive</span><strong id="mAlive">0</strong></div>
        <div class="metric"><span>mix</span><strong id="mMix">0.00</strong></div>
        <div class="metric"><span>served</span><strong id="mServed">0.00</strong></div>
        <div class="metric"><span>takeovers</span><strong id="mTake">0</strong></div>
        <div class="metric"><span>sun</span><strong id="mSun">0.00</strong></div>
      </div>
      <div class="controls">
        <button id="toggle">Pause</button>
        <button id="reset">Reset</button>
        <button id="step">Step</button>
        <button id="speed">1x</button>
      </div>
      <div class="legend">
        <div class="row"><span class="swatch a"></span><span>A solar tissue</span></div>
        <div class="row"><span class="swatch b"></span><span>B dense load tissue</span></div>
        <div class="row"><span class="swatch c"></span><span>C storage tissue</span></div>
        <div class="row"><span class="swatch dead"></span><span>empty or failed cell</span></div>
      </div>
      <div class="note">A beats B, B beats C, C beats A. Cells can grow, die, or be absorbed when local energy, storage, crowding, and neighbor pressure shift.</div>
    </aside>
  </main>
  <script>
    const SIZE = {size};
    const TRIBES = [
      {{ key: "A", color: [37, 99, 235], morph: 0.25, load: 0.82, solar: 1.45, storage: 0.76 }},
      {{ key: "B", color: [217, 119, 6], morph: 0.72, load: 1.28, solar: 0.62, storage: 0.92 }},
      {{ key: "C", color: [21, 128, 61], morph: 0.48, load: 0.95, solar: 0.95, storage: 1.52 }}
    ];
    const BEATS = [1, 2, 0];
    let seed = {seed};
    function rnd() {{
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    }}
    function gauss() {{
      const u = Math.max(1e-9, rnd());
      const v = Math.max(1e-9, rnd());
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }}
    function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
    let cells = [];
    let t = 0;
    let running = true;
    let speed = 1;
    const canvas = document.getElementById("grid");
    const ctx = canvas.getContext("2d");
    function blank() {{
      return {{ alive: false, tribe: -1, morph: 0.5, health: 0, storage: 0, storageCap: 0, load: 0, solarCap: 0, unmet: 0, critical: false }};
    }}
    function activate(i, tribe, critical = false, morph = null) {{
      const spec = TRIBES[tribe];
      const m = clamp(morph ?? spec.morph + gauss() * 0.055, 0.03, 0.97);
      const roof = Math.max(0.18, 1.18 - m);
      const density = 0.62 + 1.18 * m;
      cells[i] = {{
        alive: true,
        tribe,
        morph: m,
        health: 0.86 + rnd() * 0.14,
        critical,
        load: density * spec.load * (critical ? 1.34 : 1),
        solarCap: roof * spec.solar * 1.25,
        storageCap: (1.2 + 1.7 * (1 - Math.abs(m - 0.5))) * spec.storage * (critical ? 1.7 : 1),
        storage: 0,
        unmet: 0
      }};
      cells[i].storage = cells[i].storageCap * (0.38 + rnd() * 0.18);
    }}
    function reset() {{
      cells = Array.from({{ length: SIZE * SIZE }}, blank);
      t = 0;
      seed = {seed};
      [[0.25, 0.72, 0], [0.70, 0.28, 1], [0.50, 0.50, 2]].forEach(([r, c, tribe]) => {{
        for (let k = 0; k < 5; k++) {{
          const rr = clamp(Math.floor(SIZE * r) + Math.floor(rnd() * 3) - 1, 0, SIZE - 1);
          const cc = clamp(Math.floor(SIZE * c) + Math.floor(rnd() * 3) - 1, 0, SIZE - 1);
          activate(rr * SIZE + cc, tribe, tribe === 2 && rnd() < 0.45);
        }}
      }});
      draw({{alive: cells.filter(c => c.alive).length, mix: 0, served: 1, takeovers: 0, sun: 0}});
    }}
    function neigh(i) {{
      const r = Math.floor(i / SIZE);
      const c = i % SIZE;
      const out = [];
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {{
        if (!dr && !dc) continue;
        const rr = r + dr, cc = c + dc;
        if (rr >= 0 && rr < SIZE && cc >= 0 && cc < SIZE) out.push(rr * SIZE + cc);
      }}
      return out;
    }}
    function sunlight() {{ return Math.max(0, Math.sin(Math.PI * ((t % 24) / 24))); }}
    function demandWave() {{
      const p = (t % 24) / 24;
      return 0.76 + 0.36 * Math.exp(-Math.pow((p - 0.78) / 0.16, 2)) + 0.20 * Math.exp(-Math.pow((p - 0.34) / 0.13, 2));
    }}
    function chooseTribe(live) {{
      const counts = [0,0,0], scores = [0,0,0];
      live.forEach(c => counts[c.tribe]++);
      const total = Math.max(1, counts.reduce((a,b) => a + b, 0));
      live.forEach(c => {{
        const energy = c.storage / Math.max(c.storageCap, 1e-6);
        scores[c.tribe] += (0.75 + c.health + 0.45 * energy) * (1 + 0.9 * (1 - counts[c.tribe] / total));
      }});
      let roll = rnd() * scores.reduce((a,b) => a + b, 0);
      for (let i = 0; i < 3; i++) {{ roll -= scores[i]; if (roll <= 0) return i; }}
      return 0;
    }}
    function simStep() {{
      const sun = sunlight();
      const demand = demandWave();
      let served = 0, load = 0;
      cells.forEach(c => {{
        if (!c.alive) return;
        const d = c.load * demand;
        const generation = c.solarCap * sun;
        const available = c.storage + generation;
        served += Math.min(d, available);
        load += d;
        c.storage = clamp(available - d, 0, c.storageCap);
        c.unmet = d > 0 ? Math.max(0, d - available) / d : 0;
        c.health = clamp(c.health - 0.105 * c.unmet - (sun < 0.08 && c.tribe !== 2 ? 0.025 : 0) + 0.018 * (1 - c.unmet) + 0.01 * c.storage / Math.max(c.storageCap, 1e-6), 0, 1);
      }});
      const deaths = [], births = [], takeovers = [];
      const alive = cells.filter(c => c.alive).length;
      const counts = [0,1,2].map(tr => cells.filter(c => c.alive && c.tribe === tr).length);
      cells.forEach((c, i) => {{
        const live = neigh(i).map(j => cells[j]).filter(x => x.alive);
        if (c.alive) {{
          const same = live.filter(x => x.tribe === c.tribe).length;
          const enemy = live.filter(x => x.tribe !== c.tribe);
          const deathP = 0.018 + 0.085 * c.unmet + (live.length <= 1 ? 0.045 : 0) + 0.018 * Math.max(0, live.length - 5) + 0.045 * Math.max(0, counts[c.tribe] / Math.max(1, alive) - 0.48) - 0.05 * c.storage / Math.max(c.storageCap, 1e-6) - (c.critical ? 0.035 : 0);
          if (c.health <= 0.04 || rnd() < Math.max(0, deathP)) {{ deaths.push(i); return; }}
          if (enemy.length) {{
            const attack = [0,0,0];
            enemy.forEach(e => {{
              const adv = BEATS[e.tribe] === c.tribe ? 0.34 : -0.18;
              attack[e.tribe] += (0.22 + e.health + e.storage / Math.max(e.storageCap, 1e-6) + adv) * (1 + 0.65 * (1 - counts[e.tribe] / Math.max(1, alive)));
            }});
            const attacker = attack.indexOf(Math.max(...attack));
            const defense = 0.72 + 0.72 * c.health + 0.55 * c.storage / Math.max(c.storageCap, 1e-6) + 0.16 * same + (c.critical ? 0.42 : 0);
            if (attack[attacker] > defense * (1.04 + rnd() * 0.55)) takeovers.push([i, attacker]);
          }}
        }} else if (live.length >= 2) {{
          const tribe = chooseTribe(live);
          const same = live.filter(x => x.tribe === tribe);
          const energy = live.reduce((s, x) => s + x.storage / Math.max(x.storageCap, 1e-6), 0) / live.length;
          const densityPenalty = Math.max(0, alive / cells.length - 0.72);
          if (rnd() < Math.max(0, 0.07 + 0.06 * same.length + 0.065 * energy - 0.13 * densityPenalty)) {{
            const avg = same.length ? same.reduce((s, x) => s + x.morph, 0) / same.length : TRIBES[tribe].morph;
            births.push([i, tribe, clamp(0.56 * avg + 0.44 * TRIBES[tribe].morph + gauss() * 0.035, 0.03, 0.97)]);
          }}
        }}
      }});
      deaths.forEach(i => cells[i] = blank());
      births.forEach(([i, tribe, morph]) => activate(i, tribe, false, morph));
      takeovers.forEach(([i, tribe]) => activate(i, tribe, cells[i].critical && rnd() < 0.65, clamp(0.54 * cells[i].morph + 0.46 * TRIBES[tribe].morph + gauss() * 0.025, 0.03, 0.97)));
      const aliveAfter = cells.filter(c => c.alive).length;
      if (aliveAfter < cells.length * 0.16) {{
        for (let k = 0; k < 4; k++) {{
          const empty = cells.map((c, i) => c.alive ? -1 : i).filter(i => i >= 0);
          if (!empty.length) break;
          const tribe = counts.indexOf(Math.min(...counts));
          activate(empty[Math.floor(rnd() * empty.length)], tribe);
        }}
      }}
      const afterCounts = [0,1,2].map(tr => cells.filter(c => c.alive && c.tribe === tr).length);
      const mix = aliveAfter ? (1 - afterCounts.reduce((s, c) => s + Math.pow(c / aliveAfter, 2), 0)) / (1 - 1 / 3) : 0;
      t++;
      const stats = {{alive: aliveAfter, mix, served: load ? served / load : 1, takeovers: takeovers.length, sun}};
      draw(stats);
    }}
    function draw(stats) {{
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const pad = 34;
      const cell = (canvas.width - pad * 2) / SIZE;
      cells.forEach((c, i) => {{
        const r = Math.floor(i / SIZE), col = i % SIZE;
        const x = pad + col * cell, y = pad + r * cell;
        if (c.alive) {{
          const base = TRIBES[c.tribe].color;
          const energy = c.storage / Math.max(c.storageCap, 1e-6);
          const bright = 0.48 + 0.34 * c.health + 0.18 * energy;
          ctx.fillStyle = `rgb(${{base.map(v => Math.round(clamp(v * bright + 255 * 0.05 * energy, 0, 255))).join(",")}})`;
        }} else {{
          ctx.fillStyle = "#e2e8f0";
        }}
        ctx.fillRect(x, y, cell - 1, cell - 1);
        if (c.critical && c.alive) {{
          ctx.fillStyle = "#f8fafc";
          ctx.fillRect(x + cell * 0.36, y + cell * 0.36, cell * 0.28, cell * 0.28);
        }}
      }});
      document.getElementById("mStep").textContent = t;
      document.getElementById("mAlive").textContent = stats.alive;
      document.getElementById("mMix").textContent = stats.mix.toFixed(2);
      document.getElementById("mServed").textContent = stats.served.toFixed(2);
      document.getElementById("mTake").textContent = stats.takeovers;
      document.getElementById("mSun").textContent = stats.sun.toFixed(2);
    }}
    function loop() {{
      if (running) for (let i = 0; i < speed; i++) simStep();
      requestAnimationFrame(loop);
    }}
    document.getElementById("toggle").onclick = () => {{
      running = !running;
      document.getElementById("toggle").textContent = running ? "Pause" : "Run";
    }};
    document.getElementById("reset").onclick = reset;
    document.getElementById("step").onclick = simStep;
    document.getElementById("speed").onclick = () => {{
      speed = speed === 1 ? 3 : speed === 3 ? 8 : 1;
      document.getElementById("speed").textContent = speed + "x";
    }};
    reset();
    loop();
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def make_contact_sheet(frames: list[Image.Image], path: Path, columns: int = 5) -> None:
    thumbs = [frame.resize((220, 244), Image.Resampling.NEAREST) for frame in frames]
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 220, rows * 244), (248, 250, 252))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % columns) * 220, (i // columns) * 244))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="outputs/city_concept_preview")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=144)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = CityPreview(size=args.size, seed=args.seed)
    gif_frames = []
    sheet_frames = []
    for step in range(args.steps):
        sim.step()
        if step % 2 == 0:
            gif_frames.append(sim.render(scale=10, margin=16))
        if step in {0, 12, 24, 36, 48, 72, 96, 120, args.steps - 1}:
            sheet_frames.append(sim.render(scale=10, margin=16))

    gif_path = out_dir / "city_concept_preview.gif"
    png_path = out_dir / "city_concept_contact_sheet.png"
    html_path = out_dir / "city_concept_preview.html"
    json_path = out_dir / "city_concept_metrics.json"

    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=95,
        loop=0,
        optimize=False,
    )
    make_contact_sheet(sheet_frames, png_path)
    write_html(html_path, seed=args.seed, size=args.size, steps=args.steps)
    json_path.write_text(json.dumps(sim.history, indent=2), encoding="utf-8")

    print(f"gif={gif_path.resolve()}")
    print(f"png={png_path.resolve()}")
    print(f"html={html_path.resolve()}")
    print(f"metrics={json_path.resolve()}")


if __name__ == "__main__":
    main()
