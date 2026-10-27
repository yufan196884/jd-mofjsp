#!/usr/bin/env python3
"""Convert an existing Step 2 v1 dataset to compact v2 IN PLACE.

No upstream imports, PVGIS, SMARD or re-sampling. Writes to temporary files,
validates the original file checksum and instance IDs, and replaces the existing
instances.jsonl and dataset_manifest_step2.json when the conversion succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

OLD = "jd-mofjsp-step2-v1"
NEW = "jd-mofjsp-step2-v2"
TICKS = 100
ROUND = 2


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def rounded(x):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("Non-finite energy value")
    x = round(x, ROUND)
    return 0.0 if x == 0 else x


def convert_one(obj):
    if obj.get("format_version") != OLD:
        raise ValueError("Expected a v1 instance")
    old_id = obj["instance_id"]
    old_core = {k: v for k, v in obj.items() if k not in ("instance_id", "metadata")}
    if hashlib.sha256(canon(old_core)).hexdigest() != old_id:
        raise ValueError(f"Old instance hash mismatch: {old_id}")
    intervals = obj["energy_profile"]
    horizon = obj["horizon"]
    if not isinstance(horizon, int) or horizon <= 0 or not isinstance(intervals, list):
        raise ValueError("Invalid v1 horizon or energy intervals")
    if len(intervals) != (horizon + TICKS - 1) // TICKS:
        raise ValueError("Unexpected number of original 100-tick intervals")
    pairs = []
    for i, iv in enumerate(intervals):
        begin = i * TICKS
        end = min((i + 1) * TICKS, horizon)
        if iv["start"] != begin or iv["end"] != end:
            raise ValueError("Original intervals cannot be represented by fixed 100-tick stride")
        price = rounded(iv["price_ct_per_kwh"])
        solar = rounded(iv["solar_kw"])
        if solar < 0:
            raise ValueError("Negative solar generation")
        pairs.append([price, solar])
    obj["format_version"] = NEW
    obj["energy_profile"] = {"interval_ticks": TICKS, "price_solar": pairs}
    metadata = obj.setdefault("metadata", {})
    metadata["energy_round_decimals"] = ROUND
    metadata["profile_processing"] = (
        metadata.get("profile_processing", "100-timestep profile")
        + "; compressed into ordered [price,solar] pairs and rounded to 2 decimals"
    )
    obj["instance_id"] = hashlib.sha256(canon({
        k: v for k, v in obj.items() if k not in ("instance_id", "metadata")
    })).hexdigest()
    return obj


def convert_directory(directory: Path):
    mf_path = directory / "dataset_manifest_step2.json"
    manifest = json.loads(mf_path.read_text(encoding="utf-8"))
    if manifest["format_version"] == NEW:
        print("Already compact v2: nothing changed")
        return
    if manifest["format_version"] != OLD:
        raise ValueError("Unsupported manifest format version")
    instances_path = directory / manifest["instance_file"]
    old_sha = digest_file(instances_path)
    if old_sha != manifest["instance_file_sha256"]:
        raise ValueError("Original file SHA-256 mismatch; refusing to overwrite")

    temp_data = None
    temp_manifest = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=directory,
            prefix="._compact_instances_", suffix=".jsonl", delete=False,
        ) as f:
            temp_data = Path(f.name)
            count = 0
            seen = set()
            op_counts = []
            for line in instances_path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                obj = convert_one(json.loads(line))
                ident = obj["instance_id"]
                if ident in seen:
                    raise ValueError("Duplicate instance ID after compression")
                seen.add(ident)
                f.write(canon(obj).decode("utf-8") + "\n")
                count += 1
                op_counts.append(obj["num_operations"])
        if count != manifest["num_instances"]:
            raise ValueError("Instance count differs from original manifest")
        if min(op_counts) != manifest["total_operations_min"] or max(op_counts) != manifest["total_operations_max"]:
            raise ValueError("Operation-count summary differs from original manifest")
        new_sha = digest_file(temp_data)
        manifest["format_version"] = NEW
        manifest["instance_file_sha256"] = new_sha
        manifest["energy_profile_encoding"] = "100-tick implicit intervals; [price_ct_per_kwh, solar_kw]"
        manifest["energy_round_decimals"] = ROUND
        manifest["conversion"] = {
            "source_format_version": OLD,
            "source_instance_file_sha256": old_sha,
            "converter_script_sha256": digest_file(Path(__file__)),
            "operation": "v1 intervals -> v2 [price,solar] pairs, rounded to 2 decimals",
            "original_sampling_preserved": True,
            "upstream_daily_profile_cache_unchanged": True,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=directory,
            prefix="._compact_manifest_", suffix=".json", delete=False,
        ) as f:
            temp_manifest = Path(f.name)
            f.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        # Do not modify the original output until both replacements are complete.
        os.replace(temp_data, instances_path)
        temp_data = None
        os.replace(temp_manifest, mf_path)
        temp_manifest = None
        print(f"Converted {count} instances IN PLACE:")
        print(instances_path)
        print(mf_path)
        print("Old JSONL SHA-256:", old_sha)
        print("New JSONL SHA-256:", new_sha)
    finally:
        if temp_data and temp_data.exists():
            temp_data.unlink()
        if temp_manifest and temp_manifest.exists():
            temp_manifest.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()
    convert_directory(args.output_dir)
