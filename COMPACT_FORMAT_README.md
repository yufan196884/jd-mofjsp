# MOFJSP Step 2: Compact v2 representation

## What changed

The old `jd-mofjsp-step2-v1` energy profile contained one dictionary per interval:

```json
{"start":0,"end":100,"price_ct_per_kwh":-1.2191399999999988,"solar_kw":130.05640200597747}
```

The compact `jd-mofjsp-step2-v2` profile is a single object whose interval starts are implicit:

```json
{"interval_ticks":100,"price_solar":[[-1.22,130.06],[-2.67,123.89]]}
```

For interval `i`, `start = i * interval_ticks` and `end = min((i + 1) * interval_ticks, horizon)`.

Pairs are `[price_ct_per_kwh, solar_kw]` and **both are rounded to two decimal places before serialization and later reward evaluation**. Other job/machine information is unchanged. The reward evaluator must consume the serialized, rounded profile and must **not** silently revert to more precise raw PVGIS/SMARD values.

Changing the canonical instance content changes its SHA-256 `instance_id`; format version changes to `jd-mofjsp-step2-v2`. Regenerate downstream prompts, reference data and any instance-ID indexes if they have already been created.

## Recommended: overwrite the already-generated files WITHOUT PVGIS calls

From the upstream cloned repository root (where `src/`, `config/` and both Python scripts exist):

```bash
python compact_existing_mofjsp_step2.py data/jd_mofjsp/original_smoke
python verify_mofjsp_step2.py data/jd_mofjsp/original_smoke
```

This validates the old JSONL's manifest hash and every instance hash, converts each row, updates the manifest with the original SHA-256 checksum and conversion record, and replaces both original output files in place. It does **not** run the upstream generator again or touch cached external energy data. Repeating the command on v2 is a no-op.

To convert simplified mode too:

```bash
python compact_existing_mofjsp_step2.py data/jd_mofjsp/step2_smoke
python verify_mofjsp_step2.py data/jd_mofjsp/step2_smoke
```

## New generation overwrites old files by default

```bash
python generate_mofjsp_step2.py \
  --profile-mode original --num-instances 10 --seed 2026 \
  --output-dir data/jd_mofjsp/original_smoke
python verify_mofjsp_step2.py data/jd_mofjsp/original_smoke
```

The new generator automatically replaces `instances.jsonl` and `dataset_manifest_step2.json` after successful generation. Use `--no-overwrite` to refuse replacement. `--overwrite` is accepted for compatibility but is no longer necessary.

The generator builds the replacement JSONL and manifest in temporary files first. A failure while generating instances leaves both existing files untouched, though temporary files may remain in the output directory.

## Important: negative prices and scientific precision

Keep signed energy prices, including negative values. Round each source **interval average**, not the raw one-minute data before averaging. Since reward values will change slightly at two-decimal precision, treat v1 and v2 as separate evaluator/instance-schema versions. The final partial interval still has a duration of `horizon % 100`, rather than an entire hour.
