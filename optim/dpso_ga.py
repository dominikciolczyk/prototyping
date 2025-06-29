import random
import numpy as np
import copy
from typing import Callable, Tuple, Dict, List, Optional

Particle = Dict[str, float]

def _clip(cfg: Particle, space: Dict[str, Tuple[float, float]]) -> None:
    """Keep every dim inside its (min,max) box."""
    for k, (lo, hi) in space.items():
        cfg[k] = max(lo, min(cfg[k], hi))

def _initial_particle(space: Dict[str, Tuple[float, float]]) -> Particle:
    return {k: random.uniform(lo, hi) for k, (lo, hi) in space.items()}

def dpso_ga(
    fitness_fn: Callable[[Particle], float],
    space: Dict[str, Tuple[float, float]],
    pop_size: int,
    max_iter: int,
    w: float,
    c1: float,
    c2: float,
    mutation_rate: float,
    vmax_fraction:  float,
    early_stop_iters: int,
    on_iteration_end: Optional[Callable[[int, Particle, float, List[float]], None]] = None,
) -> Tuple[Particle, List[float]]:
    """
    Returns best hyper-param *dict* and the fitness trajectory.
    """

    # --- helpers: encode(real) <--> unit ∈ [0,1] --------------------------
    lo = {k: lo for k, (lo, hi) in space.items()}
    rng = {k: hi - lo for k, (lo, hi) in space.items()}

    def encode(cfg: Particle) -> Particle:
        return {k: (cfg[k] - lo[k]) / rng[k] for k in space}

    def decode(unit: Particle) -> Particle:
        return {k: unit[k] * rng[k] + lo[k] for k in space}

    V_MAX = vmax_fraction  # same for every dim

    # --- initial swarm ----------------------------------------------------
    particles_u = [encode({k: random.uniform(*space[k]) for k in space})
                   for _ in range(pop_size)]
    vel_u = [{k: 0.0 for k in space} for _ in range(pop_size)]

    p_best_u = copy.deepcopy(particles_u)
    p_best_s = [fitness_fn(decode(u)) for u in particles_u]

    g_idx = int(np.argmin(p_best_s))
    g_best_u = copy.deepcopy(p_best_u[g_idx])
    g_best_s = p_best_s[g_idx]

    trajectory = [g_best_s]

    no_improve = 0
    best_so_far = g_best_s

    # --- main loop --------------------------------------------------------
    for it in range(max_iter):
        for i, u_i in enumerate(particles_u):
            # 1️⃣  PSO update in unit-space
            for k in space:
                r1, r2 = random.random(), random.random()
                vel_u[i][k] = (
                        w * vel_u[i][k]
                        + c1 * r1 * (p_best_u[i][k] - u_i[k])
                        + c2 * r2 * (g_best_u[k] - u_i[k])
                )
                # velocity clamp
                vel_u[i][k] = max(-V_MAX, min(vel_u[i][k], V_MAX))
                # position update
                u_i[k] = max(0.0, min(u_i[k] + vel_u[i][k], 1.0))

                # 2️⃣  GA mutation (also in unit-space)
                if random.random() < mutation_rate:
                    u_i[k] = random.random()

            # 3️⃣  Evaluate decoded particle
            score = fitness_fn(decode(u_i))
            if score < p_best_s[i]:
                p_best_u[i], p_best_s[i] = copy.deepcopy(u_i), score
                if score < g_best_s:
                    g_best_u, g_best_s = copy.deepcopy(u_i), score

        trajectory.append(g_best_s)
        if on_iteration_end is not None:
            best_cfg = decode(g_best_u)
            on_iteration_end(it, best_cfg, g_best_s, trajectory)

        # EARLY STOPPING LOGIC
        if g_best_s < best_so_far - 1e-8:
            best_so_far = g_best_s
            no_improve = 0
        else:
            no_improve += 1

        if early_stop_iters is not None and no_improve >= early_stop_iters:
            print(
                f"[early stopping] No improvement in {early_stop_iters} iterations. Stopping early at iteration {it}.")
            break

    return decode(g_best_u), trajectory