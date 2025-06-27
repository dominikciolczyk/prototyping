"""
DPSO-GA :

    • Particle position  -> x_i
    • Velocity           -> v_i
    • Best local position-> p_best_i
    • Best global        -> g_best

Particles encode hyper-parameters of CNN-LSTM,
e.g. [n_cnn_layers, ch_1, k_1, ..., lstm_hidden, lr, dropout]
"""
import random
import numpy as np
import copy
from typing import Callable, Tuple, Dict, List

Particle = Dict[str, float]  # hyperparameter dictionary


def _clip(cfg: Particle, space: Dict[str, Tuple[float, float]]) -> None:
    """Keep every dim inside its (min,max) box."""
    for k, (lo, hi) in space.items():
        cfg[k] = max(lo, min(cfg[k], hi))


def _initial_particle(space: Dict[str, Tuple[float, float]]) -> Particle:
    return {k: random.uniform(lo, hi) for k, (lo, hi) in space.items()}


def dpso_ga(
    fitness_fn: Callable[[Particle], float],
    space: Dict[str, Tuple[float, float]],
    pop_size: int = 20,
    max_iter: int = 30,
    w: float = 0.5,
    c1: float = 1.5,
    c2: float = 1.5,
    mutation_rate: float = 0.1,
    vmax_fraction:  float = 0.20,
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

    # --- main loop --------------------------------------------------------
    for _ in range(max_iter):
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

    return decode(g_best_u), trajectory