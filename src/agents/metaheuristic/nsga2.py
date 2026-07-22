"""
NSGA-II Solver for Multi-Objective Flexible Job Shop Problem (FJSP)
Objectives: Makespan + Energy Cost (with Solar Profile)

"""

import sys
import os
import time
import random
import collections
import csv
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

# ==========================================================
# 1. PROJECT PATH SETUP
# ==========================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================================
# 2. PROJECT IMPORTS
# ==========================================================
try:
    from src.utils.file_handler.data_handler import DataHandler
    from src.data_generator.task import Task
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Global seed for reproducibility
BASE_SEED = 42


# ==========================================================
# 3. INDIVIDUAL  —  dual chromosome (sequence + machine assignment)
# ==========================================================
class Individual:
    """
    Represents an individual in the evolutionary algorithm.

      seq_chrom  : Permutation of job indices (order of operations).
      mach_chrom : Machine assignment per operation.
    """

    def __init__(
        self,
        seq_chrom: List[int],
        mach_chrom: Optional[List[int]] = None,
        alpha: Optional[float] = None,
    ):
        self.seq_chrom: List[int]  = seq_chrom
        self.mach_chrom: Optional[List[int]] = mach_chrom
        self.chromosome = self.seq_chrom

        self.objectives: List[float] = []
        self.rank: Optional[int] = None
        self.crowding_distance: float = 0.0
        self.dominates: set = set()
        self.dominated_by: int = 0
        self.schedule_details: Dict = {}
        self.alpha: float = alpha if alpha is not None else random.random()

    def dominates_other(self, other: "Individual") -> bool:
        """Pareto dominance"""
        better = False
        for s, o in zip(self.objectives, other.objectives):
            if s > o:
                return False
            if s < o:
                better = True
        return better

    def __repr__(self) -> str:
        return (f"Individual(rank={self.rank}, "
                f"obj={[round(o, 2) for o in self.objectives]}, "
                f"alpha={self.alpha:.3f})")


# ==========================================================
# 4. NSGA-II SCHEDULER
# ==========================================================
class NSGA2Scheduler:
    """
    NSGA-II-based scheduler for the Green FJSP.
    """

    def __init__(
        self,
        instance_data: Dict[str, Any],
        population_size: int = 100,         
        generations: int = 100,
        mutation_rate: float = 0.15,
        mach_mutation_rate: float = 0.15,   
        alpha_mutation_rate: float = 0.2,
        alpha_mutation_strength: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.tasks = instance_data["tasks"]
        self.machines_objs = instance_data["machines"]
        self.solar_profile = np.array(instance_data["solar_profile"])
        self.energy_prices = np.array(instance_data["energy_prices"])
        self.num_machines = len(self.machines_objs)

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mach_mutation_rate = mach_mutation_rate
        self.alpha_mutation_rate = alpha_mutation_rate
        self.alpha_mutation_strength = alpha_mutation_strength

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Group and sort jobs
        self.jobs_dict: Dict[int, List[Task]] = collections.defaultdict(list)
        for t in self.tasks:
            self.jobs_dict[t.job_index].append(t)
        for j in self.jobs_dict:
            self.jobs_dict[j].sort(key=lambda x: x.task_index)

        # Base sequence chromosome
        self._base_chromosome: List[int] = []
        for j, ops in self.jobs_dict.items():
            self._base_chromosome.extend([j] * len(ops))

        # Ordered list of all (job, op_idx) pairs
        self._op_order: List[Tuple[int, int]] = []
        for j in sorted(self.jobs_dict.keys()):
            for op_idx in range(len(self.jobs_dict[j])):
                self._op_order.append((j, op_idx))

        # valid_machines[k] = list of valid machines for operation k in _op_order
        self._valid_machines: List[List[int]] = []
        for (j, op_idx) in self._op_order:
            task = self.jobs_dict[j][op_idx]
            self._valid_machines.append(
                [m for m, rt in enumerate(task.runtimes) if rt > 0]
            )

        # Lookup & cache
        self._running_power_lookup: Dict[Tuple[int, int, int], float] = self._build_power_lookup()
        self._total_standby_power: float = sum(getattr(m, "standby_energy", 0.0) for m in self.machines_objs)

        # Precomputed absolute machine bounds per operation (for the greedy fallback)
        self._op_min_runtime: np.ndarray = np.array([
            min(self.jobs_dict[j][op_idx].runtimes[m] for m in vms)
            for (j, op_idx), vms in zip(self._op_order, self._valid_machines)
        ], dtype=np.float64)
        self._op_max_runtime: np.ndarray = np.array([
            max(self.jobs_dict[j][op_idx].runtimes[m] for m in vms)
            for (j, op_idx), vms in zip(self._op_order, self._valid_machines)
        ], dtype=np.float64)

    def _initialize_population(self) -> List[Individual]:
        population = []
        for _ in range(self.population_size):
            seq = self._base_chromosome[:]
            random.shuffle(seq)
            mach = [random.randrange(len(vms)) if vms else 0 for vms in self._valid_machines]
            population.append(Individual(seq, mach_chrom=mach, alpha=random.random()))
        return population

    def _fresh_mach_chrom(self) -> List[int]:
        return [random.randrange(len(vms)) if vms else 0 for vms in self._valid_machines]

    def _build_power_lookup(self) -> Dict[Tuple[int, int, int], float]:
        lookup: Dict[Tuple[int, int, int], float] = {}
        for ops in self.jobs_dict.values():
            for task in ops:
                for m, rt in enumerate(task.runtimes):
                    if rt > 0:
                        lookup[(task.job_index, task.task_index, m)] = float(task.machine_running_energy[m])
        return lookup

    def _compute_total_cost(self, scheduled: Dict, makespan: float) -> float:

        horizon = max(int(np.ceil(makespan)), 1)

        # 1. Build array of total power per timestep
        total_power = np.full(horizon, self._total_standby_power, dtype=np.float64)

        # 2. Accumulate net running energy (active - standby) for scheduled jobs
        for (job_idx, task_idx), info in scheduled.items():
            m_idx = info["machine"]
            t_s = int(info["start"])
            t_e = min(int(np.ceil(info["end"])), horizon)

            run_pwr = self._running_power_lookup.get((job_idx, task_idx, m_idx), 0.0)
            standby_pwr = getattr(self.machines_objs[m_idx], "standby_energy", 0.0)

            # Subtract standby power
            net_run_pwr = max(0.0, run_pwr - standby_pwr)

            if t_s < t_e:
                total_power[t_s:t_e] += net_run_pwr

        # 3. Vectorized subtraction of solar and multiplication by prices
        solar_slice = np.zeros(horizon, dtype=np.float64)
        n_solar = min(horizon, len(self.solar_profile))
        if n_solar > 0:
            solar_slice[:n_solar] = self.solar_profile[:n_solar]

        price_slice = np.full(horizon, float(self.energy_prices[-1]) if len(self.energy_prices) > 0 else 0.1, dtype=np.float64)
        n_price = min(horizon, len(self.energy_prices))
        if n_price > 0:
            price_slice[:n_price] = self.energy_prices[:n_price]

        # Net grid draw: (consumption - solar), negative draw clamped to 0
        net_power = np.maximum(0.0, total_power - solar_slice)
        total_cost = np.sum(net_power * price_slice)*(1/10000) # convert IM to h and ct to €
        
        return float(total_cost)

    def _surrogate_cost(self, power: float, t_start: int, t_end: int) -> float:
        """Fast cost approximation for the score-greedy fallback in _decode"""
        duration = max(0, t_end - t_start)
        if duration <= 0: return 0.0
        
        t_start = max(0, t_start)
        H_price = len(self.energy_prices)
        t_end_price = min(t_end, H_price)
        
        if t_start < t_end_price:
            avg_price = float(np.mean(self.energy_prices[t_start:t_end_price]))
        else:
            avg_price = float(self.energy_prices[-1]) if H_price > 0 else 1.0
            
        return duration * power * avg_price

    def _decode(
        self, seq_chrom: List[int], alpha: float,
        mach_chrom: Optional[List[int]] = None,
    ) -> Tuple[Dict, float, float]:
        
        op_to_k: Dict[Tuple[int, int], int] = {
            (j, op_idx): k for k, (j, op_idx) in enumerate(self._op_order)
        }

        machine_free  = np.zeros(self.num_machines, dtype=np.float64)
        job_free: Dict[int, float] = {j: 0.0 for j in self.jobs_dict}
        job_next: Dict[int, int]   = {j: 0   for j in self.jobs_dict}
        scheduled: Dict[Tuple[int, int], Dict] = {}

        fallback_price = float(self.energy_prices[-1]) if len(self.energy_prices) > 0 else 1.0
        max_price      = float(self.energy_prices.max()) if len(self.energy_prices) > 0 else 1.0

        for job_id in seq_chrom:
            op_idx = job_next.get(job_id, 0)
            ops    = self.jobs_dict.get(job_id, [])
            if op_idx >= len(ops):
                continue

            task           = ops[op_idx]
            k              = op_to_k.get((job_id, op_idx))
            valid_machines = self._valid_machines[k] if k is not None else [m for m, rt in enumerate(task.runtimes) if rt > 0]

            if not valid_machines:
                job_next[job_id] = op_idx + 1
                continue

            # Step 1: evaluate the mach_chrom gene
            chosen_m: Optional[int] = None
            if mach_chrom is not None and k is not None:
                gene   = mach_chrom[k] % len(valid_machines)
                chosen_m = valid_machines[gene]

            # Step 2: if no chromosome gene, fall back to score-greedy selection
            if chosen_m is None:
                est_makespan = float(machine_free.max())
                cand_ends, cand_costs = [], []

                min_rt = float(self._op_min_runtime[k]) if k is not None else 1.0
                max_rt = float(self._op_max_runtime[k]) if k is not None else 1.0
                rt_range = max(max_rt - min_rt, 1e-6)  

                for m in valid_machines:
                    start    = max(float(machine_free[m]), job_free[job_id])
                    end      = start + task.runtimes[m]
                    duration = end - start
                    run_pwr  = float(task.machine_running_energy[m])

                    run_cost = self._surrogate_cost(run_pwr, int(start), int(np.ceil(end)))

                    tail_s = int(np.ceil(end))
                    tail_e = max(tail_s + 1, int(np.ceil(est_makespan)))
                    H      = len(self.energy_prices)
                    tail_sl = self.energy_prices[tail_s : min(tail_e, H)]
                    tail_avg = float(tail_sl.mean()) if len(tail_sl) > 0 else fallback_price
                    standby_marginal = duration * self._total_standby_power * tail_avg

                    cand_ends.append(end)
                    cand_costs.append(run_cost + standby_marginal)

                ends  = np.array(cand_ends,  dtype=np.float64)
                costs = np.array(cand_costs, dtype=np.float64)

                base_start = max(float(machine_free.min()), job_free[job_id])
                end_norm   = (ends - (base_start + min_rt)) / (rt_range + 1e-9)
                end_norm   = np.clip(end_norm, 0.0, 1.0)

                max_pwr        = max(float(task.machine_running_energy[m]) for m in valid_machines)
                cost_ref       = max(max_pwr * max_price * max_rt * 2.0, 1e-9)
                cost_norm      = costs / cost_ref
                cost_norm      = np.clip(cost_norm, 0.0, 1.0)

                scores   = alpha * end_norm + (1.0 - alpha) * cost_norm
                best_i   = int(np.argmin(scores))
                chosen_m = valid_machines[best_i]

            # Schedule the operation
            start = max(float(machine_free[chosen_m]), job_free[job_id])
            end   = start + task.runtimes[chosen_m]

            machine_free[chosen_m] = end
            job_free[job_id]       = end
            job_next[job_id]       = op_idx + 1
            scheduled[(task.job_index, task.task_index)] = {
                "start": start, "end": end, "machine": chosen_m,
            }

        makespan     = float(machine_free.max()) if len(machine_free) else 0.0
        # Compute exact vectorized cost at the end
        total_energy = self._compute_total_cost(scheduled, makespan)
        
        return scheduled, makespan, total_energy

    def _evaluate(self, population: List[Individual]) -> None:
        for ind in population:
            sched, ms, ec = self._decode(
                ind.seq_chrom, ind.alpha, mach_chrom=ind.mach_chrom
            )
            ind.objectives = [ms, ec]
            ind.schedule_details = sched

    def _fast_non_dominated_sort(self, pop: List[Individual]) -> List[List[Individual]]:
        fronts: List[List[Individual]] = [[]]
        for p in pop:
            p.dominates = set()
            p.dominated_by = 0
            for q in pop:
                if p is q:
                    continue
                if p.dominates_other(q):
                    p.dominates.add(q)
                elif q.dominates_other(p):
                    p.dominated_by += 1
            if p.dominated_by == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front: List[Individual] = []
            for p in fronts[i]:
                for q in p.dominates:
                    q.dominated_by -= 1
                    if q.dominated_by == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def _crowding(self, front: List[Individual]) -> None:
        if len(front) <= 1:
            for p in front: p.crowding_distance = float("inf")
            return

        for p in front: p.crowding_distance = 0.0

        n_obj = len(front[0].objectives)
        for m in range(n_obj):
            front.sort(key=lambda x: x.objectives[m])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")
            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0:
                continue
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives[m] - front[i - 1].objectives[m]
                ) / obj_range

    def _tournament(self, pop: List[Individual]) -> Individual:
        a, b = random.sample(pop, 2)
        if a.rank < b.rank: return a
        if b.rank < a.rank: return b
        return a if a.crowding_distance >= b.crowding_distance else b

    def _order_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        size = len(p1.seq_chrom)
        s, e = sorted(random.sample(range(size), 2))

        def ox_correct(par_a: List[int], par_b: List[int]) -> List[int]:
            segment = par_a[s:e]
            child   = [None] * size
            child[s:e] = segment
            fill_order = par_b[e:] + par_b[:e]
            count_in_seg: Dict[int, int] = {}
            for g in segment:
                count_in_seg[g] = count_in_seg.get(g, 0) + 1
            available: List[int] = []
            for g in fill_order:
                if count_in_seg.get(g, 0) > 0:
                    count_in_seg[g] -= 1
                else:
                    available.append(g)
            pos = list(range(e, size)) + list(range(s))
            for p_idx, val in zip(pos, available):
                child[p_idx] = val
            return child

        c1_seq = ox_correct(p1.seq_chrom, p2.seq_chrom)
        c2_seq = ox_correct(p2.seq_chrom, p1.seq_chrom)

        m1  = p1.mach_chrom or self._fresh_mach_chrom()
        m2  = p2.mach_chrom or self._fresh_mach_chrom()
        n_m = len(m1)
        mask   = np.random.randint(0, 2, size=n_m).astype(bool)
        c1_mac = [m1[k] if mask[k] else m2[k] for k in range(n_m)]
        c2_mac = [m2[k] if mask[k] else m1[k] for k in range(n_m)]

        alpha_avg = (p1.alpha + p2.alpha) / 2.0
        return (
            Individual(c1_seq, mach_chrom=c1_mac, alpha=alpha_avg),
            Individual(c2_seq, mach_chrom=c2_mac, alpha=alpha_avg),
        )

    def _mutate(self, ind: Individual) -> Individual:
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(ind.seq_chrom)), 2)
            ind.seq_chrom[i], ind.seq_chrom[j] = ind.seq_chrom[j], ind.seq_chrom[i]
            ind.chromosome = ind.seq_chrom 

        if ind.mach_chrom and random.random() < self.mach_mutation_rate:
            k = random.randrange(len(ind.mach_chrom))
            n_valid = len(self._valid_machines[k])
            if n_valid > 1:
                current = ind.mach_chrom[k] % n_valid
                choices = [x for x in range(n_valid) if x != current]
                ind.mach_chrom[k] = random.choice(choices)

        if random.random() < self.alpha_mutation_rate:
            ind.alpha = float(np.clip(
                ind.alpha + np.random.normal(0.0, self.alpha_mutation_strength),
                0.0, 1.0,
            ))
        return ind

    def compute_nadir_reference(self, buffer=1.1):
        # Worst-case makespan
        wc_machine_free = np.zeros(self.num_machines, dtype=np.float64)
        wc_job_free = {j: 0.0 for j in self.jobs_dict}
        wc_scheduled = {}
        
        for j in sorted(self.jobs_dict.keys()):
            for task in self.jobs_dict[j]:
                valid = [m for m, rt in enumerate(task.runtimes) if rt > 0]
                if not valid: continue
                slowest_m = max(valid, key=lambda m: task.runtimes[m])
                start = max(float(wc_machine_free[slowest_m]), wc_job_free[j])
                end = start + task.runtimes[slowest_m]
                wc_machine_free[slowest_m] = end
                wc_job_free[j] = end
                wc_scheduled[(task.job_index, task.task_index)] = {
                    "start": start, "end": end, "machine": slowest_m,
                }
        
        wc_makespan = float(wc_machine_free.max())

        # Worst-case energy - no solar, maximum price
        max_price = float(self.energy_prices.max()) if len(self.energy_prices) > 0 else 1.0

        # Maximum power across all tasks
        total_max_power = self._total_standby_power
        for (j, op_idx) in self._op_order:
            task = self.jobs_dict[j][op_idx]
            valid = self._valid_machines[self._op_order.index((j, op_idx))]
            if valid:
                max_run_pwr = max(task.machine_running_energy[m] for m in valid)
                standby = getattr(self.machines_objs[valid[0]], "standby_energy", 0.0)
                total_max_power += max(0.0, max_run_pwr - standby)
        
        # Upper bound: maximum power * maximum price * total time,
        # no solar, all tasks on the most expensive machine
        wc_energy_upper = (
            total_max_power      # kW
            * wc_makespan        # IM
            / 100.0              # -> kWh
            * max_price          # ct/kWh
            / 100.0              # -> euro
        )
        
        return wc_makespan * buffer, wc_energy_upper * buffer

    @staticmethod
    def _hypervolume_2d(front, ref=None):
        if not front: return 0.0
        objs = np.array([ind.objectives for ind in front])
        
        if ref is None:
            raise ValueError("Nadir reference point must be provided — use compute_nadir_reference()")
        else:
            ref_arr = np.array([ref[0], ref[1]], dtype=np.float64)
        
        
        mask = (objs[:, 0] < ref_arr[0]) & (objs[:, 1] < ref_arr[1])
        objs = objs[mask]
        if len(objs) == 0: return 0.0
        
        sorted_objs = objs[objs[:, 0].argsort()]
        hv = 0.0
        prev_ec = ref_arr[1]
        
        for ms, ec in sorted_objs:
            if ec < prev_ec:
                hv += (ref_arr[0] - ms) * (prev_ec - ec)
                prev_ec = ec
        
        return hv

    @staticmethod
    def spacing(front: List[Individual]) -> float:
        if len(front) < 2:
            return 0.0
        objs = np.array([ind.objectives for ind in front], dtype=np.float64)

        # Min-max normalization per objective
        mins = objs.min(axis=0)
        maxs = objs.max(axis=0)
        ranges = np.where(maxs - mins > 1e-9, maxs - mins, 1.0)
        objs_norm = (objs - mins) / ranges

        # Euclidean distance to the nearest neighbor
        n = len(objs_norm)
        d = np.full(n, np.inf)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist = np.sqrt(np.sum((objs_norm[i] - objs_norm[j]) ** 2))
                d[i] = min(d[i], dist)

        d_mean = d.mean()
        return float(np.sqrt(np.mean((d - d_mean) ** 2)))
    
    
    def plot_pareto_front(
        self,
        pareto: List[Individual],
        instance_idx: int = 0,
        weighted_best: Optional[Individual] = None,
    ) -> None:
        """
        Pareto-Front als Scatter-Plot.
        """

        if not pareto:
            print("Keine Pareto-Punkte vorhanden.")
            return

        makespans = [ind.objectives[0] for ind in pareto]
        energies  = [ind.objectives[1] for ind in pareto]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            makespans, energies,
            c="steelblue", s=60, alpha=0.8, zorder=2,
            label=f"Pareto front ({len(pareto)} solutions)",
        )

        sorted_pairs = sorted(zip(makespans, energies), key=lambda x: x[0])
        ax.step(
            [p[0] for p in sorted_pairs],
            [p[1] for p in sorted_pairs],
            where="post", color="steelblue", linewidth=1.0, alpha=0.4, zorder=1,
        )

        if weighted_best is not None:
            ax.scatter(
                weighted_best.objectives[0],
                weighted_best.objectives[1],
                c="crimson", s=120, zorder=3, marker="*",
                label=(f"Gewichteter Favorit\n"
                    f"(MS={weighted_best.objectives[0]:.1f}, "
                    f"EC={weighted_best.objectives[1]:.2f}€)"),
            )

        ax.set_xlabel("Makespan [Zeiteinheiten]", fontsize=11)
        ax.set_ylabel("Energiekosten [€]", fontsize=11)
        ax.set_title(f"Pareto-Front – Instanz {instance_idx}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    def run(self) -> Tuple[List[Individual], List[float], Tuple[float, float], float]:
        nadir_ref = self.compute_nadir_reference(buffer=1.1)
        pop = self._initialize_population()
        self._evaluate(pop)
        hv_progress: List[float] = []

        for gen in range(self.generations):
            fronts = self._fast_non_dominated_sort(pop)
            for f in fronts: self._crowding(f)

            rank0 = [ind for ind in pop if ind.rank == 0]
            hv_progress.append(NSGA2Scheduler._hypervolume_2d(rank0, ref=nadir_ref))

            offspring: List[Individual] = []
            while len(offspring) < self.population_size:
                p1 = self._tournament(pop)
                p2 = self._tournament(pop)
                c1, c2 = self._order_crossover(p1, p2)
                offspring.append(self._mutate(c1))
                if len(offspring) < self.population_size:
                    offspring.append(self._mutate(c2))

            self._evaluate(offspring)

            combined = pop + offspring
            fronts   = self._fast_non_dominated_sort(combined)
            new_pop: List[Individual] = []
            for f in fronts:
                self._crowding(f)
                if len(new_pop) + len(f) <= self.population_size:
                    new_pop.extend(f)
                else:
                    remaining = self.population_size - len(new_pop)
                    f.sort(key=lambda x: x.crowding_distance, reverse=True)
                    new_pop.extend(f[:remaining])
                    break
            pop = new_pop

        final_fronts = self._fast_non_dominated_sort(pop)
        if final_fronts:
            for f in final_fronts: self._crowding(f)
            
        pareto  = [ind for ind in pop if ind.rank == 0]
        sp      = self.spacing(pareto)
        return pareto, hv_progress, nadir_ref, sp


# ==========================================================
# 5. HELPER: SELECT WEIGHTED SOLUTION
# ==========================================================
def select_best_weighted_solution(pareto_front: List[Individual], weights: List[float] = [0.5, 0.5]) -> Individual:
    """
    Selects the best solution from the Pareto front based on user-defined weights,
    using min-max normalization.
    """
    if not pareto_front:
        return None
    if len(pareto_front) == 1:
        return pareto_front[0]

    makespans = np.array([ind.objectives[0] for ind in pareto_front])
    energies = np.array([ind.objectives[1] for ind in pareto_front])
    
    min_ms, max_ms = makespans.min(), makespans.max()
    min_ec, max_ec = energies.min(), energies.max()
    
    range_ms = max(max_ms - min_ms, 1e-9)
    range_ec = max(max_ec - min_ec, 1e-9)
    
    best_ind = None
    best_score = float('inf')
    
    for ind in pareto_front:
        ms, ec = ind.objectives[0], ind.objectives[1]
        norm_ms = (ms - min_ms) / range_ms
        norm_ec = (ec - min_ec) / range_ec
        
        score = weights[0] * norm_ms + weights[1] * norm_ec
        if score < best_score:
            best_score = score
            best_ind = ind
            
    return best_ind


# ==========================================================
# 6. SAVE RESULTS
# ==========================================================
def save_pareto_metrics(
    instance_idx: int,
    pareto: List[Individual],
    nadir_ref: Tuple[float, float],
    spacing_val: float,
    output_dir: str = "results",
) -> None:
    """
    Saves the Pareto front's numerical results as a CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"pareto_instance_{instance_idx}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"# nadir_ref_ms={nadir_ref[0]:.4f}",
                         f"nadir_ref_ec={nadir_ref[1]:.4f}",
                         f"spacing={spacing_val:.4f}"])
        writer.writerow(["makespan", "energy_cost", "alpha"])
        for ind in pareto:
            writer.writerow([ind.objectives[0], ind.objectives[1], ind.alpha])


# ==========================================================
# 7. MAIN
# ==========================================================
if __name__ == "__main__":
    import json

    config = {
        "instances_file": "green_fjsp/GreenFJSP_10_instances.pkl",
        "sp_type": "green_fjsp",
    }

    all_instances = DataHandler.load_instances_data_file(config)

    # Define your weights for the comparison here
    OBJECTIVE_WEIGHTS = [0.5, 0.5]

    summary_rows  = []
    nadir_registry: Dict[str, Dict] = {}

    for idx, instance in enumerate(all_instances):
        print(f"\n{'='*50}")
        print(f"  Solving instance {idx} ...")
        print(f"{'='*50}")

        scheduler = NSGA2Scheduler(
            instance,
            population_size=100,
            generations=100,
            mutation_rate=0.15,
            mach_mutation_rate=0.15,
            alpha_mutation_rate=0.2,
            alpha_mutation_strength=0.1,
            seed=BASE_SEED + idx,
        )

        t0 = time.time()
        pareto, hv_prog, nadir_ref, spacing_val = scheduler.run()
        duration = time.time() - t0

        best_ms  = min(ind.objectives[0] for ind in pareto) if pareto else 0.0
        best_ec  = min(ind.objectives[1] for ind in pareto) if pareto else 0.0
        final_hv = hv_prog[-1] if hv_prog else 0.0

        # Select the best weighted solution
        weighted_best_ind = select_best_weighted_solution(pareto, weights=OBJECTIVE_WEIGHTS)
        weighted_ms = weighted_best_ind.objectives[0] if weighted_best_ind else best_ms
        weighted_ec = weighted_best_ind.objectives[1] if weighted_best_ind else best_ec

        print(f"  Done in {duration:.2f}s | Pareto size: {len(pareto)}")
        print(f"  Best makespan (absolute): {best_ms:.1f} | Best energy (absolute): {best_ec:.2f}")
        print(f"  Weighted favorite ({OBJECTIVE_WEIGHTS}): MS={weighted_ms:.1f} | EC={weighted_ec:.2f} euro")

        nadir_registry[f"instance_{idx}"] = {
            "nadir_ref_ms":  round(nadir_ref[0], 4),
            "nadir_ref_ec":  round(nadir_ref[1], 4),
            "nsga2_hv":      round(final_hv,     4),
            "nsga2_spacing": round(spacing_val,  4),
            "nsga2_best_ms": round(best_ms,      4),
            "nsga2_best_ec": round(best_ec,      4),
            "nsga2_weighted_ms": round(weighted_ms, 4),
            "nsga2_weighted_ec": round(weighted_ec, 4),
        }

        save_pareto_metrics(
            instance_idx=idx, 
            pareto=pareto, 
            nadir_ref=nadir_ref,
            spacing_val=spacing_val
        )

        summary_rows.append({
            "instance":        idx,
            "pareto_size":     len(pareto),
            "best_makespan":   best_ms,
            "best_energy":     best_ec,
            "weighted_best_ms": weighted_ms,  
            "weighted_best_ec": weighted_ec,  
            "nadir_ms":        nadir_ref[0],
            "nadir_ec":        nadir_ref[1],
            "final_hv":        final_hv,
            "spacing":         spacing_val,
            "duration_s":      round(duration, 2),
        })

    os.makedirs("results", exist_ok=True)

    summary_path = os.path.join("results", "summary.csv")
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary saved: {summary_path}")

    nadir_path = os.path.join("results", "nadir_reference.json")
    with open(nadir_path, "w") as f:
        json.dump(nadir_registry, f, indent=2)
    print(f"Nadir reference points saved: {nadir_path}")