import random
import numpy as np
import copy
from typing import Callable, Tuple, Dict, List, Optional
from zenml.logger import get_logger

logger = get_logger(__name__)

Particle = Dict[str, float]

def dpso_ga(
    fitness_fn: Callable[[Particle], float],
    space: Dict[str, Tuple[float, float]],
    pop_size: int,
    ga_generations: int,
    crossover_rate: float,
    mutation_rate: float,
    pso_iterations: int,
    w_max: float,
    w_min: float,
    c1: float,
    c2: float,
    vmax_fraction: float,
    early_stop_iters: int,
    on_iteration_end: Optional[Callable[[int, Particle, float, List[float]], None]] = None,
) -> Tuple[Particle, List[float]]:

    lo = {k: lo for k, (lo, hi) in space.items()}
    rng = {k: hi - lo for k, (lo, hi) in space.items()}
    def encode(cfg): return {k: (cfg[k] - lo[k]) / rng[k] for k in space}
    def decode(u):   return {k: u[k] * rng[k] + lo[k] for k in space}

    def _ga_step(pop_u, fitness_fn, space, crossover_rate, mutation_rate, elite_count=1):
        # 1) score all individuals
        decoded = [decode(u) for u in pop_u]
        scores = [fitness_fn(cfg) for cfg in decoded]

        # 2) sort by fitness (lower = better)
        sorted_pop = [u for _, u in sorted(zip(scores, pop_u), key=lambda x: x[0])]
        elites = sorted_pop[:elite_count]  # elite individuals to keep

        # 3) tournament selection
        def tourney():
            i, j = random.sample(range(len(pop_u)), 2)
            return pop_u[i] if scores[i] < scores[j] else pop_u[j]

        # 4) crossover & mutation
        keys = list(space)
        children = []
        while len(children) < len(pop_u) - elite_count:
            p1, p2 = tourney(), tourney()
            if random.random() < crossover_rate:
                pt1, pt2 = sorted(random.sample(range(len(keys)), 2))
                c1 = {
                    k: (
                        p2[k] if pt1 <= i < pt2 else p1[k]
                    ) for i, k in enumerate(keys)
                }
                c2 = {
                    k: (
                        p1[k] if pt1 <= i < pt2 else p2[k]
                    ) for i, k in enumerate(keys)
                }
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                for k in keys:
                    if random.random() < mutation_rate:
                        child[k] = random.random()
                children.append(child)

        # 5) return elites + children
        return elites + children[:len(pop_u) - elite_count]

    # ——— 1) initialize GA population in unit‐space ———
    particles_u = [encode({k: random.uniform(*space[k]) for k in space})
                   for _ in range(pop_size)]

    logger.info(f"Initial particles: {particles_u}")

    # ——— 2) GA warm‐up ———
    for _ in range(ga_generations):
        particles_u = _ga_step(
            pop_u=particles_u,
            fitness_fn=fitness_fn,
            space=space,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_count=1 # keep the best individual
        )

        logger.info(f"Particles after GA step: {particles_u}")

    # ——— 3) initialize PSO from GA‐warmed population ———
    velocities_u = [{k: 0.0 for k in space} for _ in range(pop_size)]
    personal_best_u    = copy.deepcopy(particles_u)
    personal_best_score = [fitness_fn(decode(u)) for u in personal_best_u]
    g_idx = int(np.argmin(personal_best_score))
    global_best_u   = copy.deepcopy(personal_best_u[g_idx])
    global_best_score = personal_best_score[g_idx]

    trajectory = [global_best_score]
    no_improve = 0

    # ——— 4) pure PSO loop ———
    for it in range(pso_iterations):
        w = w_max - (w_max - w_min) * it / pso_iterations
        for i,u in enumerate(particles_u):
            for k in space:
                r1, r2 = random.random(), random.random()
                # velocity update
                velocities_u[i][k] = (
                    w*velocities_u[i][k]
                    + c1*r1*(personal_best_u[i][k] - u[k])
                    + c2*r2*(global_best_u[k]        - u[k])
                )
                # position update
                u[k] = max(0.0, min(1.0, u[k] + velocities_u[i][k]))

            v_vec = np.array([velocities_u[i][k] for k in space])
            norm = np.linalg.norm(v_vec)

            if norm > vmax_fraction:
                for k in space:
                    velocities_u[i][k] *= vmax_fraction / norm

            score = fitness_fn(decode(u))
            if score < personal_best_score[i]:
                personal_best_u[i], personal_best_score[i] = copy.deepcopy(u), score
                if score < global_best_score:
                    global_best_u, global_best_score = copy.deepcopy(u), score

        trajectory.append(global_best_score)
        if on_iteration_end:
            on_iteration_end(it, decode(global_best_u), global_best_score, trajectory)

        # early stopping
        if trajectory[-1] < trajectory[-2] - 1e-8:
            no_improve = 0
        else:
            no_improve += 1
        if early_stop_iters is not None and no_improve >= early_stop_iters:
            print(f"[early stopping] stopped at iter {it}")
            break

    return decode(global_best_u), trajectory