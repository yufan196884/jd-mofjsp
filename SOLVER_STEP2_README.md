# Dual-objective Step-2 MOFJSP scheduling solver

`solve_mofjsp_step2.py` is a deterministic (seeded), CPU-only heuristic to produce **two complete, valid schedules per input row** of the instance generator's `instances.jsonl`:

- `makespan`: best-found minimum completion time (maximizes the unscaled makespan reward).
- `cost`: best-found minimum purchased electricity cost (maximizes the unscaled cost reward).

These are approximate best-found solutions, **not global optimality certificates**. Both objectives are evaluated independently for both schedules.

## Run in the JD parent repository

The parent repo tracks this repository as the Git submodule `third_party/jd-mofjsp`. From the parent root, after checking out this solver branch in the submodule (or after its PR is merged and the parent submodule pointer updated):

```bash
python third_party/jd-mofjsp/solve_mofjsp_step2.py \
  --instances "$PWD/data/mofjsp/instances.jsonl" \
  --output "$PWD/data/mofjsp/solutions.jsonl" \
  --seed 2026 --iterations 400 --restarts 8
```

Smoke test on 10 rows:

```bash
python third_party/jd-mofjsp/solve_mofjsp_step2.py \
  --instances "$PWD/data/mofjsp/instances.jsonl" \
  --output "$PWD/data/mofjsp/solutions_smoke.jsonl" --limit 10
```

Run unit tests from the submodule root:

```bash
cd third_party/jd-mofjsp
python -m unittest -v test_solve_mofjsp_step2.py
```

The Python solver only uses the standard library. No solver license, GPU, PVGIS, or network connection is needed.

## Output schema

Each JSONL line matches one nonblank input instance line (original index stored as `input_line`):

```json
{
  "instance_id": "original-instance-sha256",
  "input_line": 1,
  "makespan": {
    "actions": [{"job": 0, "machine": 1, "delay": 0}],
    "metrics": {"makespan_ticks": 100, "energy_cost_eur": 12.3,
                "makespan_reward_unscaled": -100, "cost_reward_unscaled": -12.3}
  },
  "cost": {
    "actions": [{"job": 0, "machine": 1, "delay": 25}],
    "metrics": {"makespan_ticks": 125, "energy_cost_eur": 9.5,
                "makespan_reward_unscaled": -125, "cost_reward_unscaled": -9.5}
  }
}
```

Illustrative numbers above, not a real output. Each `actions` array has **exactly one decision per operation**, in the single-turn LLM format: zero-based job, eligible machine, and additional delay in ticks. Each use of a job selects its next unexecuted operation. The decoder starts it at `max(job_ready, machine_ready) + delay`, enforcing precedence and resource capacity. The default delay quantum is 25 ticks, default per-action cap 2400 ticks; every schedule must end before `horizon`.

The energy evaluator sums **all machines' standby draw** until the overall makespan, replaces standby by active draw during processing, subtracts available solar, floors grid purchases at zero (no export), and integrates the piecewise price profile. A negative electricity price is supported. Currency is euros: `grid_kw * price_ct_per_kwh * duration_ticks / (100 * ticks_per_hour)`.

The `*_reward_unscaled` values are diagnostic negative raw objectives, **not** the final normalized reward functions from the JD dataset plan. When reference scales are defined, compute `r_time=-min(makespan/C_ref,5)` and `r_energy=-clip(cost/E_ref,-5,5)` in the training evaluator; these are not assigned here because the generator has not provided reference values.

## Search options

`--iterations` controls the proposal count **per objective and instance** (default 400). `--restarts` controls randomized restarts (default 8). `--seed` makes results reproducible by instance ID independently of input ordering. The search explores topological job dispatch orders, eligible machines, and permitted delays (especially for cost). `--max-delay`, `--delay-step`, and `--limit` are configurable. More iterations generally improve the search budget but do not guarantee global optimality.

If any input is invalid or no feasible schedule is found, execution fails without replacing the existing output file. Input must be compact v2 or expanded v1 Step-2 JSONL with complete energy profile. Repeated instances remain separate output rows.

**Important:** The solver code lives in the **submodule repository**. Its pull request does not automatically change the parent repository's pinned submodule commit. After merging, update and commit the submodule pointer in `experiment_JD_LLM_v2` to expose the solver by default.
