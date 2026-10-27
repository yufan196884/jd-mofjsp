#!/usr/bin/env python3
"""Verify Step 2 JSONL instances and their manifest (compact v2; legacy v1 too)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


FORMAT_V2 = "jd-mofjsp-step2-v2"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def verify_energy(instance: dict) -> int:
    profile = instance["energy_profile"]
    horizon = instance["horizon"]
    ticks = instance["ticks_per_hour"]
    assert isinstance(horizon, int) and horizon > 0
    assert ticks == 100
    if instance["format_version"] == FORMAT_V2:
        assert isinstance(profile, dict), "v2 energy_profile must be an object"
        assert profile.get("interval_ticks") == 100
        pairs = profile.get("price_solar")
        assert isinstance(pairs, list)
        assert len(pairs) == (horizon + 99) // 100, "Energy interval coverage mismatch"
        for pair in pairs:
            assert isinstance(pair, list) and len(pair) == 2
            price, solar = pair
            assert isinstance(price, (int, float)) and not isinstance(price, bool)
            assert isinstance(solar, (int, float)) and not isinstance(solar, bool)
            assert math.isfinite(price) and math.isfinite(solar) and solar >= 0
            assert round(price, 2) == price and round(solar, 2) == solar
        return len(pairs)
    else:
        assert instance["format_version"] == "jd-mofjsp-step2-v1"
        t = 0
        for iv in profile:
            assert iv["start"] == t and iv["end"] > iv["start"]
            assert iv["end"] - iv["start"] <= 100
            assert math.isfinite(iv["price_ct_per_kwh"]) and math.isfinite(iv["solar_kw"])
            assert iv["solar_kw"] >= 0
            t = iv["end"]
        assert t == horizon
        return len(profile)


def verify(directory: Path) -> None:
    manifest = json.loads((directory / "dataset_manifest_step2.json").read_text())
    path = directory / manifest["instance_file"]
    digest = hash_file(path)
    assert digest == manifest["instance_file_sha256"], "File hash mismatch"
    ids = set()
    count = 0
    operation_counts = []
    example = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            instance = json.loads(line)
            if example is None:
                example = dict(instance)
            ident = instance.pop("instance_id")
            instance.pop("metadata", None)
            assert hashlib.sha256(canon(instance)).hexdigest() == ident, "Instance ID mismatch"
            assert ident not in ids, "Duplicate instance ID"
            ids.add(ident)
            count += 1
            assert instance["format_version"] == manifest["format_version"]
            assert instance["num_jobs"] == len(instance["jobs"])
            assert instance["num_machines"] == len(instance["machines"])
            num_ops = sum(len(job["operations"]) for job in instance["jobs"])
            assert num_ops == instance["num_operations"] and num_ops > 0
            operation_counts.append(num_ops)
            for m, machine in enumerate(instance["machines"]):
                assert machine["machine"] == m and machine["standby_kw"] >= 0
            for j, job in enumerate(instance["jobs"]):
                assert job["job"] == j and job["operations"]
                for op_i, op in enumerate(job["operations"]):
                    assert op["operation"] == op_i and op["options"]
                    machines_seen = set()
                    for opt in op["options"]:
                        assert 0 <= opt["machine"] < instance["num_machines"]
                        assert opt["machine"] not in machines_seen
                        machines_seen.add(opt["machine"])
                        assert opt["duration"] >= 1 and opt["active_kw"] >= 0
            verify_energy(instance)
    assert count == manifest["num_instances"] and count > 0, "Count mismatch"
    assert min(operation_counts) == manifest["total_operations_min"]
    assert max(operation_counts) == manifest["total_operations_max"]
    print("VERIFIED", count, "instances")
    print("SHA256:", digest)
    print("Format:", manifest["format_version"])
    print("Upstream commit:", manifest["upstream_commit"])
    print("Seed:", manifest["seed"])
    print("Profile mode:", manifest["profile_mode"])
    print("Operations per instance:", min(operation_counts), "to", max(operation_counts))
    print("Example instance ID:", example["instance_id"])
    print("Example horizon:", example["horizon"])
    print("Example energy intervals:", len(example["energy_profile"]["price_solar"])
          if example["format_version"] == FORMAT_V2 else len(example["energy_profile"]))
    print("Example first operation:", example["jobs"][0]["operations"][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    verify(parser.parse_args().output_dir)
