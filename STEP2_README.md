# MOFJSP — Step 2 CPU-only instance generator

## Prerequisites

Run from **WSL2 Ubuntu**. You must already have cloned `tmdt-buw/preference-conditioned-rl-mofjsp`, preferably pinned at upstream commit:

`fa15955e14d50b70a4dea1853433cfd47c7b5d3f`

Copy `generate_mofjsp_step2.py` and `verify_mofjsp_step2.py` into the **root of that upstream clone** (same directory as `src/` and `config/`). Do not run them from outside that clone.

Install lightweight CPU generator dependencies within a dedicated environment:

```bash
python -m pip install numpy pandas matplotlib pvlib tqdm pyyaml jsonschema
```

## Smoke test — simplified profiles

```bash
python generate_mofjsp_step2.py \
  --config config/data_generation/green_fjsp/config_GreenFJSP_5x5.yaml \
  --num-instances 10 --seed 2026 --days 5 \
  --output-dir data/jd_mofjsp/step2_smoke

python verify_mofjsp_step2.py data/jd_mofjsp/step2_smoke
```

The default mode **does not read `Spotmarktpreis.csv` or contact PVGIS**. It retains the upstream sampling for operations, eligible machines, processing times, and machine active/standby consumption. A compatibility shim supplies temporary zero arrays to upstream; these are **discarded** and replaced with specified synthetic profiles before serialization.

All outputs are self-contained JSONL with a corresponding manifest:

```text
data/jd_mofjsp/step2_smoke/
├── instances.jsonl
└── dataset_manifest_step2.json
```

For each instance, JSON includes the ordered operations of each job, all eligible machine options with duration and active kW, each machine's standby kW, and the entire price and solar forecast in 100-industrial-timestep intervals. There are 100 timesteps/hour, 24 hours/day, and (by default) 5 days, or 120 intervals per instance. Prices are in ct/kWh; power is kW.

**Caution:** These profiles are deliberately synthetic rather than factual SMARD/PVGIS measurements. A preference-sensitive Pareto trade-off is **not guaranteed** by any individual instance and must be screened later (Step 9).

## Reproducibility check

Generate the same instances twice into different directories. File-content hashes should match:

```bash
python generate_mofjsp_step2.py --num-instances 10 --seed 2026 --output-dir data/jd_mofjsp/check_a
python generate_mofjsp_step2.py --num-instances 10 --seed 2026 --output-dir data/jd_mofjsp/check_b
sha256sum data/jd_mofjsp/check_{a,b}/instances.jsonl
cmp data/jd_mofjsp/check_a/instances.jsonl data/jd_mofjsp/check_b/instances.jsonl
```

For a distinct 10x10 config, use:

```bash
python generate_mofjsp_step2.py \
 --config config/data_generation/green_fjsp/config_GreenFJSP_10x10x10.yaml \
 --num-instances 100 --seed 2026 \
 --output-dir data/jd_mofjsp/step2_10x10
```

This writes one instance at a time, avoiding the original `DataHandler`'s repeated loading and rewrite of accumulated pickle data.

## Optional original external profile mode

Once simplified mode passes, supply the official SMARD CSV at:

`src/data_generator/GreenFJSP_Utils/data/Spotmarktpreis.csv`

The CSV must contain `Datum`, `von`, `Spotmarktpreis in ct/kWh`, formatted using semicolon field separators and comma decimal points, with records for the upstream `target_month` in 2025.

Then:

```bash
python generate_mofjsp_step2.py \
 --profile-mode original \
 --num-instances 10 --seed 2026 \
 --output-dir data/jd_mofjsp/original_smoke
```

The first original-mode invocation requires network access to PVGIS through pvlib. Original daily solar and electricity-price arrays are saved under `data/jd_mofjsp/profile_cache/` as `.npz` and a provenance `.json`. Subsequent runs reuse the cached arrays. If PVGIS fails, this mode refuses the upstream's synthetic fallback rather than incorrectly labelling it original.

The upstream price and solar series are sampled and then **hourly-averaged into nonoverlapping 100-timestep intervals**. The simplified mode instead uses an explicitly fixed 24-hour template. Both modes export only the intervals used by the future evaluator; do not later score against hidden, higher-resolution originals.

The original-mode horizon is the array length **after the upstream's random `start_offset`** (usually less than 5 days). A final partial hourly interval is permitted.

## Important scientific distinctions

- No reference solutions or objective values are supplied to the LLM here; this stage only creates instances.
- No scheduling solver or reward evaluator is implemented in Step 2. Those are separate steps.
- The upstream Python `Task.instance_hash` is not a stable identity. We use content-based SHA-256 over each instance's static problem specification.
- The original generator can produce different numbers of operations per job; do not assume exactly `5x5` operations or `10x10` operations in a sample without inspecting it.
- The extra `delay` field in the planned LLM response extends the upstream action space. This feature belongs to Step 4/5, not Step 2.
- The eventual energy reward should use the published formula of active power replacing standby power, solar offset, and grid cost in euros `sum(grid_kw * price_ct_per_kwh / 10000)`.
- `instances.jsonl` is a preparatory format. Step 3 can refine/rename the schema, while Step 12 handles Parquet integration into veRL/JD.
