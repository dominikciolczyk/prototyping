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
from copy import deepcopy
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
) -> Tuple[Particle, List[float]]:
    """
    Returns best hyper-param *dict* and the fitness trajectory.
    """
    # === Initialization ====================================================
    particles = [_initial_particle(space) for _ in range(pop_size)]
    vel = [{k: 0.0 for k in space} for _ in range(pop_size)]
    p_best = deepcopy(particles)
    p_best_score = [fitness_fn(p) for p in particles]
    g_best_idx = int(np.argmin(p_best_score))
    g_best = deepcopy(p_best[g_best_idx])
    g_best_score = p_best_score[g_best_idx]
    trajectory = [g_best_score]

    # === Main loop =========================================================
    for t in range(max_iter):
        for i, x_i in enumerate(particles):
            # 1️⃣  PSO velocity / position update
            for k in space:
                r1, r2 = random.random(), random.random()
                vel[i][k] = (
                    w * vel[i][k]
                    + c1 * r1 * (p_best[i][k] - x_i[k])
                    + c2 * r2 * (g_best[k] - x_i[k])
                )
                x_i[k] += vel[i][k]
            _clip(x_i, space)

            # 2️⃣  GA mutation
            for k in space:
                if random.random() < mutation_rate:
                    lo, hi = space[k]
                    x_i[k] = random.uniform(lo, hi)

            # 3️⃣  Evaluate
            score = fitness_fn(x_i)
            if score < p_best_score[i]:
                p_best[i], p_best_score[i] = deepcopy(x_i), score
                if score < g_best_score:
                    g_best, g_best_score = deepcopy(x_i), score

        trajectory.append(g_best_score)

    return g_best, trajectory
