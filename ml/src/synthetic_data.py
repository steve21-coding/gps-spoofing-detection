"""Reproducible synthetic drone-navigation data with labelled spoofing-like attacks.

IMPORTANT: this is SIMULATED data. Models trained on it learn to recognise the simulated
patterns below. It is not real GNSS measurement data and does not demonstrate real-world
spoofing detection. See README "Limitations".

Each trajectory is a 1 Hz track with one scenario:

  normal            plain flight with correlated receiver noise
  glitch            (benign, label 0) a few isolated multipath-like position outliers
  dropout           (benign, label 0) a gap with missing measurements
  low_sat_noise     (benign, label 0) a stretch of poor satellite visibility, noisier fixes
  jump              (attack) position offset by 150-3000 m from onset to end of trajectory
  drift             (attack) slowly growing offset; reported speed is NOT altered
  consistent_drift  (attack) like drift but the reported speed is also made consistent
  freeze            (attack) position frozen while the reported speed continues

`label` is 1 for every row from the attack onset onward, 0 otherwise.
Receiver effects (stronger C/N0 during attacks) are an ASSUMPTION controlled by
`attack_cn0_shift_prob`; they are not measured from real spoofers.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALL_SCENARIOS, ATTACK_SCENARIOS, DEFAULT_SEED, SAMPLE_DATA_DIR
from .geo import enu_to_latlon

DEFAULT_WEIGHTS = {
    "normal": 0.30, "glitch": 0.10, "dropout": 0.07, "low_sat_noise": 0.08,
    "jump": 0.15, "drift": 0.12, "consistent_drift": 0.10, "freeze": 0.08,
}


@dataclass
class GeneratorConfig:
    n_trajectories: int = 400
    duration_s: int = 300
    seed: int = DEFAULT_SEED
    scenario_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    attack_cn0_shift_prob: float = 0.5   # share of attacks that also raise C/N0 (assumption)
    horiz_noise_m: float = 1.5           # receiver position noise (1 sigma, AR(1))
    vert_noise_m: float = 2.5
    noise_corr: float = 0.8


def _ar1(rng: np.random.Generator, n: int, sigma: float, rho: float) -> np.ndarray:
    """Zero-mean AR(1) noise with stationary standard deviation sigma."""
    x = np.zeros(n)
    eps = rng.normal(0.0, sigma * np.sqrt(1 - rho**2), n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return x


def _true_track(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Smooth drone kinematics in a local east/north/up frame (metres, 1 s steps)."""
    v_target = rng.uniform(4, 14)
    v = v_target * rng.uniform(0.7, 1.0)
    heading = rng.uniform(0, 2 * np.pi)
    hr, turn_left, turn_rate = 0.0, 0, 0.0
    alt_target = rng.uniform(40, 120)
    alt, vz = alt_target, 0.0
    e = nn = 0.0
    out = {k: np.zeros(n) for k in ("e", "n", "alt", "v", "vE", "vN")}
    for i in range(n):
        v = float(np.clip(v + 0.1 * (v_target - v) + rng.normal(0, 0.3), 1.0, 18.0))
        if turn_left <= 0 and rng.random() < 0.02:
            turn_left = int(rng.integers(3, 9))
            turn_rate = rng.choice([-1, 1]) * rng.uniform(5, 25)  # deg/s
        hr_target = turn_rate if turn_left > 0 else 0.0
        turn_left -= 1
        hr = 0.6 * hr + 0.4 * hr_target + rng.normal(0, 0.5)
        heading += np.radians(hr)
        vE, vN = v * np.sin(heading), v * np.cos(heading)
        e, nn = e + vE, nn + vN
        if rng.random() < 0.01:
            alt_target = rng.uniform(40, 120)
        vz = float(np.clip(0.7 * vz + 0.015 * (alt_target - alt) + rng.normal(0, 0.3), -3, 3))
        alt += vz
        out["e"][i], out["n"][i], out["alt"][i] = e, nn, alt
        out["v"][i], out["vE"][i], out["vN"][i] = v, vE, vN
    return out


def _satellites(rng: np.random.Generator, n: int) -> np.ndarray:
    sat = int(rng.integers(9, 14))
    out = np.zeros(n)
    for i in range(n):
        if rng.random() < 0.05:
            sat = int(np.clip(sat + rng.choice([-1, 1]), 7, 14))
        out[i] = sat
    return out


def _simulate_trajectory(rng: np.random.Generator, scenario: str, cfg: GeneratorConfig, traj_id: str) -> pd.DataFrame:
    n = cfg.duration_s
    tr = _true_track(rng, n)
    e_m = tr["e"] + _ar1(rng, n, cfg.horiz_noise_m, cfg.noise_corr)
    n_m = tr["n"] + _ar1(rng, n, cfg.horiz_noise_m, cfg.noise_corr)
    alt_m = tr["alt"] + _ar1(rng, n, cfg.vert_noise_m, cfg.noise_corr)
    speed = np.clip(tr["v"] + rng.normal(0, 0.2, n), 0, None)
    sat = _satellites(rng, n)
    cn0 = np.clip(42 + rng.normal(0, 1.5) + rng.normal(0, 2.0, n), 25, 55)
    label = np.zeros(n, dtype=int)
    missing = np.zeros(n, dtype=bool)
    onset = np.nan

    if scenario == "glitch":
        for _ in range(int(rng.integers(3, 9))):
            i = int(rng.integers(5, n - 5))
            width = int(rng.integers(1, 4))
            ang, mag = rng.uniform(0, 2 * np.pi), rng.uniform(20, 400)
            sl = slice(i, i + width)
            e_m[sl] += mag * np.sin(ang)
            n_m[sl] += mag * np.cos(ang)
            cn0[sl] -= rng.uniform(5, 15)
            sat[sl] = np.maximum(3, sat[sl] - rng.integers(2, 6))
    elif scenario == "dropout":
        a = int(rng.integers(30, n - 30))
        missing[a:a + int(rng.integers(3, 16))] = True
    elif scenario == "low_sat_noise":
        a = int(rng.integers(30, n - 60))
        sl = slice(a, a + int(rng.integers(15, 41)))
        width = len(range(*sl.indices(n)))
        e_m[sl] += rng.normal(0, cfg.horiz_noise_m * 5, width)
        n_m[sl] += rng.normal(0, cfg.horiz_noise_m * 5, width)
        sat[sl] = rng.integers(4, 6, width)
        cn0[sl] -= 6
    elif scenario in ATTACK_SCENARIOS:
        t0 = int(rng.integers(60, n - 60))
        onset = t0
        label[t0:] = 1
        if rng.random() < cfg.attack_cn0_shift_prob:
            cn0[t0:] += rng.uniform(2, 6)
        ang = rng.uniform(0, 2 * np.pi)
        ux, uy = np.sin(ang), np.cos(ang)
        k = np.arange(n - t0)
        if scenario == "jump":
            mag = rng.uniform(150, 3000)
            e_m[t0:] += mag * ux
            n_m[t0:] += mag * uy
        elif scenario in ("drift", "consistent_drift"):
            rate = rng.uniform(0.4, 3.0)  # m/s
            e_m[t0:] += rate * (k + 1) * ux
            n_m[t0:] += rate * (k + 1) * uy
            if scenario == "consistent_drift":
                # attacker also shifts the reported velocity to match the spoofed track
                speed[t0:] = np.hypot(tr["vE"][t0:] + rate * ux, tr["vN"][t0:] + rate * uy) + rng.normal(0, 0.2, n - t0)
        elif scenario == "freeze":
            e_m[t0:] = e_m[t0]
            n_m[t0:] = n_m[t0]
    elif scenario != "normal":
        raise ValueError(f"unknown scenario {scenario!r}")

    lat0, lon0 = rng.uniform(-50, 55), rng.uniform(-170, 170)
    lat, lon = enu_to_latlon(lat0, lon0, e_m, n_m)
    ts = float(rng.integers(1_700_000_000, 1_790_000_000)) + np.arange(n)
    frame = pd.DataFrame({
        "trajectory_id": traj_id, "timestamp": ts, "lat": lat, "lon": lon, "alt": alt_m,
        "speed_mps": speed, "sat_count": sat, "cn0_dbhz": cn0,
        "label": label, "scenario": scenario,
        "attack_onset_ts": ts[int(onset)] if not np.isnan(onset) else np.nan,
    })
    frame.loc[missing, ["lat", "lon", "alt", "speed_mps", "sat_count", "cn0_dbhz"]] = np.nan
    return frame


def _scenario_plan(cfg: GeneratorConfig, rng: np.random.Generator) -> list[str]:
    names = list(cfg.scenario_weights)
    unknown = set(names) - set(ALL_SCENARIOS)
    if unknown:
        raise ValueError(f"unknown scenarios in weights: {sorted(unknown)}")
    w = np.array([cfg.scenario_weights[k] for k in names], float)
    w /= w.sum()
    counts = np.floor(w * cfg.n_trajectories).astype(int)
    counts[np.argsort(-w)[: cfg.n_trajectories - counts.sum()]] += 1  # distribute remainder
    plan = [name for name, c in zip(names, counts) for _ in range(c)]
    rng.shuffle(plan)
    return plan


def generate_dataset(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    """Generate the full labelled dataset. Same config + seed -> identical output."""
    cfg = cfg or GeneratorConfig()
    if cfg.duration_s < 150:
        raise ValueError("duration_s must be at least 150 so attacks have room before and after onset")
    root = np.random.SeedSequence(cfg.seed)
    plan_rng = np.random.default_rng(root.spawn(1)[0])
    plan = _scenario_plan(cfg, plan_rng)
    children = root.spawn(len(plan) + 1)[1:]  # one independent stream per trajectory
    frames = [
        _simulate_trajectory(np.random.default_rng(children[i]), sc, cfg, f"traj_{i:04d}")
        for i, sc in enumerate(plan)
    ]
    return pd.concat(frames, ignore_index=True)


DEMO_SEED = 2024  # differs from the training seed: demo flights are freshly simulated


def demo_dataset(seed: int = DEMO_SEED) -> pd.DataFrame:
    """Two simulated 200 s flights per scenario for demonstrations (labels included).

    Freshly simulated with a seed other than the training seed, so none of these flights were
    seen in training. Still SIMULATED data."""
    cfg = GeneratorConfig(n_trajectories=2 * len(ALL_SCENARIOS), duration_s=200, seed=seed,
                          scenario_weights={k: 1.0 for k in ALL_SCENARIOS})
    return generate_dataset(cfg)


def write_sample_csv(path: Path | None = None, seed: int = DEFAULT_SEED) -> Path:
    """Small downloadable example (one trajectory per scenario) with documented columns."""
    cfg = GeneratorConfig(n_trajectories=len(ALL_SCENARIOS), duration_s=200, seed=seed,
                          scenario_weights={k: 1.0 for k in ALL_SCENARIOS})
    df = generate_dataset(cfg)
    path = Path(path or SAMPLE_DATA_DIR / "example_navigation.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.round({"lat": 7, "lon": 7, "alt": 2, "speed_mps": 2, "cn0_dbhz": 2}).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic navigation data")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-trajectories", type=int, default=400)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args()
    d = generate_dataset(GeneratorConfig(n_trajectories=a.n_trajectories, seed=a.seed))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    print(f"wrote {len(d):,} rows to {a.out}")
