from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np
import time
from scipy.spatial import cKDTree

Point = Tuple[float, float]

REF_POINT_NORM: Point = (1.1, 1.1)
# Maximum attainable HV in the normalised space: a single point at the
# utopia point (0, 0) dominates the entire reference box [0, ref_x] x [0, ref_y].
# Dividing by this bounds the reported HV to [0, 1] as a constant, instance-
# independent factor (preserves ranking and relative gaps).
HV_MAX: float = REF_POINT_NORM[0] * REF_POINT_NORM[1]   # = 1.21


# --------------------------------------------------------------------------
# Non-dominated sorting (2D, minimization)
# --------------------------------------------------------------------------
def non_dominated_front(points: Sequence[Point]) -> List[Point]:
    if not points:
        return []
    pts = sorted(set(points), key=lambda p: (p[0], p[1]))
    front = []
    best_ec = np.inf
    for ms, ec in pts:
        if ec < best_ec:
            front.append((ms, ec))
            best_ec = ec
    return front


# --------------------------------------------------------------------------
# Hypervolume
# --------------------------------------------------------------------------
def hypervolume_2d(points: Sequence[Point], ref: Point) -> float:
    if not points:
        return 0.0
    objs = np.asarray(points, dtype=np.float64)
    mask = (objs[:, 0] < ref[0]) & (objs[:, 1] < ref[1])
    objs = objs[mask]
    if len(objs) == 0:
        return 0.0
    objs = objs[objs[:, 0].argsort()]
    hv = 0.0
    prev_ec = ref[1]
    for ms, ec in objs:
        if ec < prev_ec:
            hv += (ref[0] - ms) * (prev_ec - ec)
            prev_ec = ec
    return float(hv)


# --------------------------------------------------------------------------
# Spacing (KDTree, O(n log n))
# --------------------------------------------------------------------------
def spacing(points: Sequence[Point]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    arr = np.unique(arr, axis=0)
    if len(arr) < 2:
        return 0.0
    tree = cKDTree(arr)
    dists, _ = tree.query(arr, k=2)
    nn_dist = dists[:, 1]
    return float(np.std(nn_dist))


# --------------------------------------------------------------------------
# IGD (fully vectorized)
# --------------------------------------------------------------------------
def igd(reference: np.ndarray, solution: np.ndarray) -> float:
    if len(reference) == 0 or len(solution) == 0:
        return float("nan")
    diff = reference[:, None, :] - solution[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return float(dist.min(axis=1).mean())


# --------------------------------------------------------------------------
# Reference front + normalisation range
# --------------------------------------------------------------------------
def instance_reference(
    solutions_by_algo: Dict[str, Sequence[Point]],
):
    pool = []
    for pts in solutions_by_algo.values():
        pool.extend(pts)
    ref_front = non_dominated_front(pool)

    # Ideal and nadir are taken over the FULL pool of all algorithms.
    # ideal (componentwise min) is identical whether taken from the pool or
    # from the reference front. nadir (componentwise max) MUST come from the
    # pool: taking it from the reference front lets a dominating algorithm
    # collapse the normalisation range, pushing dominated solutions far
    # outside [0,1]^2 and producing degenerate HV / IGD values.
    arr_pool = np.asarray(pool, dtype=np.float64)
    ideal = (arr_pool[:, 0].min(), arr_pool[:, 1].min())
    nadir = (arr_pool[:, 0].max(), arr_pool[:, 1].max())
    return ref_front, ideal, nadir


def normalise(points, ideal, nadir):
    eps = 1e-12
    rx = max(nadir[0] - ideal[0], eps)
    ry = max(nadir[1] - ideal[1], eps)
    return np.array(
        [((p[0] - ideal[0]) / rx, (p[1] - ideal[1]) / ry) for p in points],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# MAIN EVALUATION (no inference timing inside)
# --------------------------------------------------------------------------
def evaluate_fronts(
    solutions: Dict[str, List[Sequence[Point]]],
    times: Optional[Dict[str, Dict[int, float]]] = None,
):
    algos = list(solutions.keys())
    n_instances = len(next(iter(solutions.values())))
    for a in algos:
        assert len(solutions[a]) == n_instances

    per_instance = {}
    acc = {a: {"norm_hv": [], "spacing": [], "igd": [], "time": []} for a in algos}

    for i in range(n_instances):
        per_algo = {a: solutions[a][i] for a in algos}

        ref_front, ideal, nadir = instance_reference(per_algo)
        ref_front_np = np.asarray(ref_front, dtype=np.float64)
        ref_norm = normalise(ref_front_np, ideal, nadir)

        rec = {"reference_front_size": len(ref_norm)}

        for a in algos:
            pts_norm = normalise(per_algo[a], ideal, nadir)
            pts_nd = np.asarray(
                non_dominated_front([tuple(p) for p in pts_norm]),
                dtype=np.float64,
            )

            # Raw HV in normalised objective space, then rescaled to [0,1]
            # by the maximum attainable HV (constant factor HV_MAX).
            hv_raw = hypervolume_2d(pts_norm.tolist(), REF_POINT_NORM)
            hv = hv_raw / HV_MAX

            sp = spacing(pts_nd)
            ig = igd(ref_norm, pts_nd)

            t = None
            if times is not None and a in times and i in times[a]:
                t = float(times[a][i])

            rec[a] = {
                "norm_hv": round(hv, 6),
                "spacing": round(sp, 6),
                "igd": None if np.isnan(ig) else round(ig, 6),
                "time": None if t is None else round(t, 6),
            }

            acc[a]["norm_hv"].append(hv)
            acc[a]["spacing"].append(sp)
            if not np.isnan(ig):
                acc[a]["igd"].append(ig)
            if t is not None:
                acc[a]["time"].append(t)

        per_instance[f"instance_{i}"] = rec

    def mean_std(x):
        x = np.asarray(x, dtype=np.float64)
        return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0

    aggregate = {}
    for a in algos:
        hv_m, hv_s = mean_std(acc[a]["norm_hv"])
        sp_m, sp_s = mean_std(acc[a]["spacing"])
        ig_m, ig_s = mean_std(acc[a]["igd"])
        t_m, t_s = mean_std(acc[a]["time"])
        aggregate[a] = {
            "norm_hv_mean": hv_m,
            "norm_hv_std": hv_s,
            "spacing_mean": sp_m,
            "spacing_std": sp_s,
            "igd_mean": ig_m,
            "igd_std": ig_s,
            "time_mean": t_m,
            "time_std": t_s,
            "n_instances": len(acc[a]["norm_hv"]),
        }

    return {"per_instance": per_instance, "aggregate": aggregate}