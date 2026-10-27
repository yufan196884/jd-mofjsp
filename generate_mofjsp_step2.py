#!/usr/bin/env python3
"""Step 2: build self-contained MOFJSP instances using the pinned upstream generator.

Copy this file to the ROOT of preference-conditioned-rl-mofjsp and run it there.
By default the upstream job/machine sampler is retained but external energy inputs
are replaced with deterministic, hourly piecewise-constant synthetic profiles.

The optional 'original' mode saves and reuses a cache of PVGIS/SMARD raw days,
then aggregates each sampled upstream profile into 100-timestep intervals. Both
modes serialize precisely the profile later used by the reward evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import random
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data_generator.sp_factory import SPFactory
from src.data_generator.GreenFJSP_Utils.functions.energy_manager import EnergyManager

TICKS_PER_HOUR = 100
HOURS_PER_DAY = 24
# Deliberately synthetic ct/kWh and kW; not measurements from SMARD/PVGIS.
PRICE_CT_PER_KWH = (
    12, 11, 10, 10, 10, 12, 15, 19, 25, 31, 35, 34,
    30, 29, 32, 38, 48, 55, 49, 37, 27, 22, 18, 14,
)
SOLAR_KW = (
    0, 0, 0, 0, 0, 0, 4, 12, 35, 72, 116, 149,
    172, 165, 142, 105, 60, 22, 5, 0, 0, 0, 0, 0,
)
FORMAT_VERSION = "jd-mofjsp-step2-v2"
ENERGY_DECIMALS = 2
REPO_ROOT = Path(__file__).resolve().parent


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class SyntheticEnergyShim:
    """Provide upstream's EnergyManager interface without CSV, API, or RNG use.

    Upstream builds and slices temporary arrays. We replace them with the final,
    completely specified synthetic profile BEFORE serializing each instance.
    """

    def get_solar_energy(self) -> np.ndarray:
        return np.zeros(1440, dtype=np.float64)

    def get_energy_prices(self) -> np.ndarray:
        return np.zeros(1440, dtype=np.float64)

    def convert_minutes_to_industrial_minutes(self, values: np.ndarray) -> np.ndarray:
        if len(values) != 1440:
            raise ValueError("Expected 1440 source minutes")
        return np.zeros(2400, dtype=np.float64)


class CachedOriginalEnergy:
    """Sample locally cached daily arrays with upstream-compatible semantics."""

    def __init__(self, solar_days: np.ndarray, price_days: np.ndarray):
        self.solar_days = solar_days
        self.price_days = price_days
        self.old_idx = np.linspace(0, 1439, 1440)
        self.new_idx = np.linspace(0, 1439, 2400)

    def get_solar_energy(self) -> np.ndarray:
        return self.solar_days[np.random.randint(len(self.solar_days))].copy()

    def get_energy_prices(self) -> np.ndarray:
        return self.price_days[np.random.randint(len(self.price_days))].copy()

    def convert_minutes_to_industrial_minutes(self, values: np.ndarray) -> np.ndarray:
        if len(values) != 1440:
            raise ValueError("Expected 1440 source minutes")
        return np.interp(self.new_idx, self.old_idx, values)


def install_profile_manager(mode: str, month: int, cache_dir: Path) -> dict:
    """Prime SPFactory's month cache so its original sampler needs no online calls."""
    SPFactory._energy_manager_cache.clear()
    if mode == "simplified":
        SPFactory._energy_manager_cache[month] = SyntheticEnergyShim()
        return {
            "mode": "simplified",
            "source": "Fixed synthetic daily template (not SMARD/PVGIS data)",
            "price_ct_per_kwh_by_hour": list(PRICE_CT_PER_KWH),
            "solar_kw_by_hour": list(SOLAR_KW),
            "days_repeated": None,  # Set by main() from --days.
            "external_inputs_used": False,
        }

    price_csv = (
        REPO_ROOT / "src/data_generator/GreenFJSP_Utils/data/Spotmarktpreis.csv"
    )
    if not price_csv.exists():
        raise FileNotFoundError(
            f"Original mode requires {price_csv}; use --profile-mode simplified "
            "to run without the CSV."
        )
    csv_digest = digest_file(price_csv)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / f"original_energy_month{month:02d}.npz"
    meta_file = cache_dir / f"original_energy_month{month:02d}.json"

    if npz.exists() != meta_file.exists():
        raise RuntimeError("Incomplete profile cache: NPZ and JSON must both exist")
    if npz.exists():
        metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        if metadata.get("electricity_price_csv_sha256") != csv_digest:
            raise RuntimeError(
                "Price CSV differs from the cached version. Inspect the inputs and "
                "remove both original_energy_monthXX files to explicitly rebuild."
            )
        if metadata.get("cache_npz_sha256") != digest_file(npz):
            raise RuntimeError("Cached energy NPZ checksum mismatch")
        with np.load(npz, allow_pickle=False) as saved:
            solar = saved["solar_days"]
            prices = saved["price_days"]
    else:
        # Fetch/cache *before* resetting generator RNGs. Fail closed if PVGIS falls
        # back to synthetic sine data; such fallback must never be mislabelled original.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager = EnergyManager(month=month)
        fallbacks = [str(w.message) for w in caught if "fallback" in str(w.message).lower()
                     or "pvgis api error" in str(w.message).lower()]
        if fallbacks:
            raise RuntimeError(
                "PVGIS real-data retrieval failed; refusing a synthetic fallback "
                "in original mode: " + "; ".join(fallbacks)
            )
        solar = np.stack(manager._solar_profiles).astype(np.float64)
        prices = np.stack(manager._price_profiles).astype(np.float64)
        np.savez_compressed(npz, solar_days=solar, price_days=prices)
        metadata = {
            "mode": "original",
            "solar_source": "PVGIS-SARAH3 via pvlib, upstream EnergyManager",
            "solar_year": manager._YEAR_SOLAR,
            "electricity_price_source": "Local SMARD Spotmarktpreis.csv",
            "electricity_price_year": manager._YEAR_PRICES,
            "electricity_price_csv_sha256": csv_digest,
            "target_month": month,
            "num_solar_days": int(len(solar)),
            "num_price_days": int(len(prices)),
            "cache_npz_sha256": digest_file(npz),
            "notes": "Cached raw daily profiles; final exported profiles are hourly means.",
        }
        meta_file.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if solar.ndim != 2 or prices.ndim != 2 or solar.shape[1] != 1440 or prices.shape[1] != 1440:
        raise ValueError("Cached energy profiles must have shape (days, 1440)")
    if not np.isfinite(solar).all() or not np.isfinite(prices).all():
        raise ValueError("Non-finite values in original energy profile cache")
    SPFactory._energy_manager_cache[month] = CachedOriginalEnergy(solar, prices)
    return {**metadata, "cache_file": str(npz.relative_to(REPO_ROOT))
            if npz.is_relative_to(REPO_ROOT) else str(npz)}


def energy_pair(price: float, solar: float) -> list[float]:
    """Round values BEFORE persisting: exactly these are used by reward.py."""
    p, s = round(float(price), ENERGY_DECIMALS), round(float(solar), ENERGY_DECIMALS)
    if not np.isfinite(p) or not np.isfinite(s) or s < 0:
        raise ValueError("Invalid price or solar power")
    # Avoid serializing negative zeros.
    return [p if p != 0 else 0.0, s if s != 0 else 0.0]


def simplified_intervals(days: int) -> tuple[dict, int]:
    if days < 1:
        raise ValueError("--days must be >= 1")
    pairs = [energy_pair(PRICE_CT_PER_KWH[h % HOURS_PER_DAY],
                         SOLAR_KW[h % HOURS_PER_DAY])
             for h in range(days * HOURS_PER_DAY)]
    return {"interval_ticks": TICKS_PER_HOUR, "price_solar": pairs}, (
        days * HOURS_PER_DAY * TICKS_PER_HOUR
    )


def intervals_from_original(instance: dict) -> tuple[dict, int]:
    """100-tick means, rounded BEFORE writing; no hidden higher-res evaluator."""
    price = np.asarray(instance["energy_prices"], dtype=np.float64)
    solar = np.asarray(instance["solar_profile"], dtype=np.float64)
    if price.ndim != 1 or solar.ndim != 1 or not len(price) or len(price) != len(solar):
        raise ValueError("Original profiles must be nonempty, equal-length 1D arrays")
    if not np.isfinite(price).all() or not np.isfinite(solar).all():
        raise ValueError("Non-finite original energy profile")
    horizon = len(price)
    pairs = [
        energy_pair(np.mean(price[begin:begin + TICKS_PER_HOUR]),
                    np.mean(solar[begin:begin + TICKS_PER_HOUR]))
        for begin in range(0, horizon, TICKS_PER_HOUR)
    ]
    return {"interval_ticks": TICKS_PER_HOUR, "price_solar": pairs}, horizon


def serialize_instance(raw: dict, profile_mode: str, days: int) -> dict:
    machines = raw["machines"]
    n_machines = len(machines)
    if n_machines < 1:
        raise ValueError("Instance has no machines")
    machine_info = []
    for m, machine in enumerate(machines):
        kw = float(machine.standby_energy)
        if not np.isfinite(kw) or kw < 0:
            raise ValueError(f"Invalid standby power on machine {m}: {kw}")
        machine_info.append({"machine": m, "standby_kw": kw})

    jobs = defaultdict(list)
    for task in raw["tasks"]:
        j, op = int(task.job_index), int(task.task_index)
        if j < 0 or op < 0:
            raise ValueError("Negative job/operation ID")
        if len(task.runtimes) != n_machines or len(task.machine_running_energy) != n_machines:
            raise ValueError("Operation machine-duration/power arrays do not match machine count")
        allowed = sorted(set(map(int, task.machines)))
        options = []
        for m in allowed:
            if not 0 <= m < n_machines:
                raise ValueError(f"Machine ID {m} out of bounds")
            duration = int(task.runtimes[m])
            active = float(task.machine_running_energy[m])
            if duration <= 0 or not np.isfinite(active) or active < 0:
                raise ValueError(f"Invalid duration/active power for job {j}, op {op}, machine {m}")
            options.append({"machine": m, "duration": duration, "active_kw": active})
        if not options:
            raise ValueError(f"Job {j} op {op} has no compatible machines")
        jobs[j].append({"operation": op, "options": options})

    ids = sorted(jobs)
    if ids != list(range(len(ids))):
        raise ValueError(f"Job identifiers are not consecutive from zero: {ids}")
    job_data = []
    for j in ids:
        ops = sorted(jobs[j], key=lambda op: op["operation"])
        if [op["operation"] for op in ops] != list(range(len(ops))):
            raise ValueError(f"Operation identifiers for job {j} are not consecutive")
        job_data.append({"job": j, "operations": ops})

    if profile_mode == "simplified":
        energy_intervals, horizon = simplified_intervals(days)
    else:
        energy_intervals, horizon = intervals_from_original(raw)

    content = {
        "format_version": FORMAT_VERSION,
        "num_jobs": len(job_data),
        "num_machines": n_machines,
        "num_operations": sum(len(j["operations"]) for j in job_data),
        "horizon": horizon,
        "ticks_per_hour": TICKS_PER_HOUR,
        "jobs": job_data,
        "machines": machine_info,
        "energy_profile": energy_intervals,
    }
    # Deterministic, content-addressed ID (unlike upstream Task.instance_hash).
    instance_id = hashlib.sha256(canonical_bytes(content)).hexdigest()
    return {
        "instance_id": instance_id,
        **content,
        "metadata": {
            "profile_mode": profile_mode,
            "upstream_start_offset": int(raw.get("start_offset", 0)),
            "profile_processing": (
                "fixed hourly synthetic template repeated daily"
                if profile_mode == "simplified" else
                "100-timestep means of upstream price and solar arrays"
            ),
            "energy_round_decimals": ENERGY_DECIMALS,
        },
    }


def validate_instance(instance: dict) -> None:
    if instance["num_jobs"] != len(instance["jobs"]):
        raise ValueError("Incorrect number of jobs")
    if instance["num_machines"] != len(instance["machines"]):
        raise ValueError("Incorrect number of machines")
    if instance["num_operations"] != sum(len(j["operations"]) for j in instance["jobs"]):
        raise ValueError("Incorrect number of operations")
    if instance["num_operations"] < 1:
        raise ValueError("Empty schedule")
    for j, job in enumerate(instance["jobs"]):
        if job["job"] != j or not job["operations"]:
            raise ValueError("Missing or improperly ordered job")
        for k, op in enumerate(job["operations"]):
            if op["operation"] != k or not op["options"]:
                raise ValueError("Invalid operation order/options")
            for opt in op["options"]:
                if not 0 <= opt["machine"] < instance["num_machines"] or opt["duration"] < 1:
                    raise ValueError("Invalid compatible machine or duration")
    profile = instance["energy_profile"]
    if not isinstance(profile, dict) or profile.get("interval_ticks") != TICKS_PER_HOUR:
        raise ValueError("Invalid compact energy-profile schema")
    pairs = profile.get("price_solar")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Missing compact energy profile")
    if len(pairs) != (instance["horizon"] + TICKS_PER_HOUR - 1) // TICKS_PER_HOUR:
        raise ValueError("Energy intervals do not cover exactly the scheduling horizon")
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Each energy interval must contain [price_ct_per_kwh, solar_kw]")
        price, solar = pair
        if not np.isfinite(price) or not np.isfinite(solar) or solar < 0:
            raise ValueError("Non-finite or negative solar / non-finite electricity price")
        if round(price, ENERGY_DECIMALS) != price or round(solar, ENERGY_DECIMALS) != solar:
            raise ValueError("Energy values exceed the specified decimal precision")
    if instance["instance_id"] != hashlib.sha256(canonical_bytes({
        k: v for k, v in instance.items() if k not in ("instance_id", "metadata")
    })).hexdigest():
        raise ValueError("Instance hash mismatch")


def get_upstream_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    if not config_file.exists():
        raise FileNotFoundError(config_file)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("sp_type") != "green_fjsp":
        raise ValueError("Expected sp_type: green_fjsp")
    if args.num_instances < 1:
        raise ValueError("--num-instances must be >= 1")
    month = int(config.get("Solar_Power_Profile", {}).get("target_month", 6))
    loading_days = int(config.get("Energy_info", {}).get("loading_days", 5))
    if args.profile_mode == "original" and args.days != loading_days:
        raise ValueError(
            "In original mode, --days must match the upstream Energy_info.loading_days "
            f"({loading_days})."
        )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "instances.jsonl"
    manifest_file = output_dir / "dataset_manifest_step2.json"
    if args.no_overwrite and (out_file.exists() or manifest_file.exists()):
        raise FileExistsError(
            f"Output exists in {output_dir}; omit --no-overwrite to regenerate"
        )

    provenance = install_profile_manager(
        args.profile_mode, month, REPO_ROOT / args.cache_dir
    )
    if args.profile_mode == "simplified":
        provenance["days_repeated"] = args.days

    random.seed(args.seed)
    np.random.seed(args.seed)
    # Do not let the upstream pickle writer run. Generate 1 instance at a time to
    # avoid both its quadratic accumulated-pickle rewrite and excessive RAM use.
    parameters = dict(config)
    parameters.update(num_instances=1, write_to_file=False)
    seen_ids: set[str] = set()
    operation_counts = []
    # Build a temporary JSONL first: failed generation must not destroy old output.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output_dir,
        prefix="._instances_", suffix=".jsonl", delete=False,
    ) as out:
        temp_out = Path(out.name)
        for i in range(args.num_instances):
            raw = SPFactory.generate_instances(**parameters)[0]
            instance = serialize_instance(raw, args.profile_mode, args.days)
            validate_instance(instance)
            ident = instance["instance_id"]
            if ident in seen_ids:
                raise RuntimeError(f"Duplicate instance generated at index {i}: {ident}")
            seen_ids.add(ident)
            operation_counts.append(instance["num_operations"])
            out.write(canonical_bytes(instance).decode("utf-8") + "\n")
            if (i + 1) % 100 == 0 or i + 1 == args.num_instances:
                print(f"Generated {i + 1}/{args.num_instances} instances", flush=True)

    manifest = {
        "format_version": FORMAT_VERSION,
        "upstream_repository": "https://github.com/tmdt-buw/preference-conditioned-rl-mofjsp",
        "upstream_commit": get_upstream_commit(),
        "seed": args.seed,
        "num_instances": args.num_instances,
        "profile_mode": args.profile_mode,
        "profile_provenance": provenance,
        "config_path": str(config_file.relative_to(REPO_ROOT))
            if config_file.is_relative_to(REPO_ROOT) else str(config_file),
        "config_sha256": digest_file(config_file),
        "generator_script_sha256": digest_file(Path(__file__)),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "total_operations_min": min(operation_counts),
        "total_operations_max": max(operation_counts),
        "instance_file": out_file.name,
        "instance_file_sha256": digest_file(temp_out),
        "energy_profile_encoding": "100-tick implicit intervals; [price_ct_per_kwh, solar_kw]",
        "energy_round_decimals": ENERGY_DECIMALS,
        "unit_definitions": {
            "time": "industrial timestep; 100 ticks per hour",
            "active_and_standby_power": "kW",
            "solar": "kW",
            "electricity_price": "ct/kWh",
            "electricity_cost_eur_per_tick": "grid_kw * price_ct_per_kwh / 10000",
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output_dir,
        prefix="._manifest_", suffix=".json", delete=False,
    ) as out:
        temp_manifest = Path(out.name)
        out.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Replace existing files instead of requiring an empty output directory.
    os.replace(temp_out, out_file)
    os.replace(temp_manifest, manifest_file)
    return out_file, manifest_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=(
        "config/data_generation/green_fjsp/config_GreenFJSP_5x5.yaml"))
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--days", type=int, default=5,
                        help="Full days of 24 hourly profiles (simplified mode)")
    parser.add_argument("--profile-mode", choices=["simplified", "original"],
                        default="simplified")
    parser.add_argument("--output-dir", default="data/jd_mofjsp/step2_smoke")
    parser.add_argument("--cache-dir", default="data/jd_mofjsp/profile_cache")
    parser.add_argument("--overwrite", action="store_true", help="Deprecated: files are overwritten by default")
    parser.add_argument("--no-overwrite", action="store_true", help="Refuse to replace existing outputs")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        instances, manifest = run(parse_args())
    except Exception as exc:
        print(f"Step 2 generation FAILED: {exc}", file=sys.stderr)
        raise
    print(f"SUCCESS: {instances}")
    print(f"Manifest: {manifest}")
