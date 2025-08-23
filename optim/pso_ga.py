import random
import numpy as np
import copy
from typing import Callable, Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from zenml.logger import get_logger

logger = get_logger(__name__)

Particle = Dict[str, float]

def save_pso_ga_plots(score_history, prefix="pso_ga"):
    scores = np.array(score_history)  # shape: (iterations, particles)

    # === 1) All particles' scores over time ===
    plt.figure(figsize=(10, 6))
    for i in range(scores.shape[1]):
        plt.plot(scores[:, i], alpha=0.4)
    plt.xlabel("Iteration")
    plt.ylabel("Score (Loss)")
    plt.title("Scores of All Particles Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_all_particles.png", dpi=150)
    plt.close()

    # === 2) Best score convergence ===
    best_scores = scores.min(axis=1)
    plt.figure(figsize=(8, 5))
    plt.plot(best_scores, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Best Score (Loss)")
    plt.title("Best Score Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_best_score.png", dpi=150)
    plt.close()


def visualize_training(param_history, score_history):
    for i, (params, scores) in enumerate(zip(param_history, score_history)):
        logger.info(f"Iteration {i}:")
        for j, (param, score) in enumerate(zip(params, scores)):
            logger.info(f"  Particle {j}: {param} => Score: {score}")

    if score_history:
        save_pso_ga_plots(score_history, prefix="pso_ga")

def pso_ga(
    fitness_fn: Callable[[Particle], float],
    space: Dict[str, Tuple[float, float]],
    pop_size: int,
    ga_generations: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_std: float,
    pso_iterations: int,
    w_max: float,
    w_min: float,
    c1: float,
    c2: float,
    vmax_fraction: float,
    early_stop_iters: int,
    on_iteration_end: Optional[Callable[[int, Particle, float, List[float]], None]] = None,
    seed_cfg: Optional[Particle] = None
) -> Tuple[Particle, List[float]]:
    seed = 42
    py_rng = random.Random(seed) if seed is not None else random
    np_rng = np.random.default_rng(seed) if seed is not None else np.random

    vmax = vmax_fraction * np.sqrt(len(space))
    lo = {k: low for k, (low, high) in space.items()}
    rng = {k: high - low for k, (low, high) in space.items()}

    logger.info(f"optim/pso_ga with arguments:\n"
                f"space: {space}\n"
                f"pop_size: {pop_size}\n"
                f"ga_generations: {ga_generations}\n"
                f"crossover_rate: {crossover_rate}\n"
                f"mutation_rate: {mutation_rate}\n"
                f"mutation_std: {mutation_std}\n"
                f"pso_iterations: {pso_iterations}\n"
                f"w_max: {w_max}\n"
                f"w_min: {w_min}\n"
                f"c1: {c1}\n"
                f"c2: {c2}\n"
                f"vmax_fraction: {vmax_fraction}\n"
                f"early_stop_iters: {early_stop_iters}\n"
                f"seed_cfg: {seed_cfg}\n"
                f"seed: {seed}\n"
                f"vmax: {vmax}\n"
                f"lo: {lo}\n"
                f"rng: {rng}\n\n")

    def encode(cfg): return {k: (cfg[k] - lo[k]) / rng[k] for k in space}
    def decode(u):   return {k: u[k] * rng[k] + lo[k] for k in space}

    assert all(abs(seed_cfg[k] - decode(encode(seed_cfg))[k]) < 1e-8 for k in seed_cfg), "Encode/decode round-trip failed!"

    def _ga_step(pop_u, fitness_fn, space, crossover_rate, mutation_rate, elite_count):
        # 1) score all individuals
        decoded = [decode(u) for u in pop_u]
        scores = [fitness_fn(cfg) for cfg in decoded]

        param_history.append(decoded)
        score_history.append(scores)
        visualize_training(param_history, score_history)

        # 2) sort by fitness (lower = better)
        sorted_pairs = sorted(zip(scores, pop_u), key=lambda x: x[0])
        sorted_scores = [s for s, _ in sorted_pairs]
        sorted_pop = [u for _, u in sorted_pairs]
        elites = sorted_pop[:elite_count]  # elite individuals to keep
        logger.info(f"Elites: {elites} with scores: {sorted_scores[:elite_count]}")

        # 3) tournament selection
        def tourney(k=3):
            idx = [py_rng.randrange(len(pop_u)) for _ in range(k)]
            best = min(idx, key=lambda t: scores[t])
            logger.info(f"Tournament participants: {[pop_u[i] for i in idx]}")
            logger.info(f"Scores: {[scores[i] for i in idx]}")
            logger.info(f"Tournament selected index: {best} with score: {scores[best]}")
            return pop_u[best]

        # 4) crossover & mutation
        keys = sorted(space.keys())
        children = []
        while len(children) < len(pop_u) - elite_count:
            p1, p2 = tourney(), tourney()
            logger.info(f"Selected parents: {p1} and {p2}")
            if py_rng.random() < crossover_rate:
                logger.info("Performing crossover")
                pt1, pt2 = sorted(py_rng.sample(range(len(keys)), 2))
                child1 = {
                    k: (
                        p2[k] if pt1 <= i < pt2 else p1[k]
                    ) for i, k in enumerate(keys)
                }
                child2 = {
                    k: (
                        p1[k] if pt1 <= i < pt2 else p2[k]
                    ) for i, k in enumerate(keys)
                }
            else:
                logger.info("No crossover, copying parents")
                child1, child2 = p1.copy(), p2.copy()

            logger.info(f"Children before mutation: {child1}, {child2}")

            for child in (child1, child2):
                for k in keys:
                    if py_rng.random() < mutation_rate:
                        logger.info(f"Mutating {k} in child {child}")
                        child[k] = max(0.0, min(1.0, child[k] + float(np_rng.normal(0, mutation_std))))
                children.append(child)

            logger.info(f"Children after mutation: {child1}, {child2}")

        # 5) return elites + children
        return elites + children[:len(pop_u) - elite_count]

    # ——— 1) initialize GA population in unit‐space ———
    particles_u = [encode({k: py_rng.uniform(*space[k]) for k in space})
                   for _ in range(pop_size)]

    if seed_cfg is not None:
        seed_u = encode({k: seed_cfg[k] for k in space})
        particles_u[0] = seed_u
        logger.info("Seeded 1 particle at init.")

    logger.info(f"Initial particles: {particles_u}\n")

    param_history = []
    score_history = []

    # ——— 2) GA warm‐up ———
    for i in range(ga_generations):
        particles_u = _ga_step(
            pop_u=particles_u,
            fitness_fn=fitness_fn,
            space=space,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_count=1 # keep the best individual
        )

        logger.info(f"Particles after GA generation {i+1}: {particles_u}")

    # ——— 3) initialize PSO from GA‐warmed population ———
    velocities_u = [{k: 0.0 for k in space} for _ in range(pop_size)]
    personal_best_u    = copy.deepcopy(particles_u)
    personal_best_score = [fitness_fn(decode(u)) for u in personal_best_u]
    g_idx = int(np.argmin(personal_best_score))
    global_best_u   = copy.deepcopy(personal_best_u[g_idx])
    global_best_score = personal_best_score[g_idx]

    logger.info(f"PSO arguments:\n"
                f"w_max: {w_max}\n"
                f"w_min: {w_min}\n"
                f"c1: {c1}\n"
                f"c2: {c2}\n"
                f"vmax: {vmax}\n"
                f"personal_best_u: {personal_best_u}\n"
                f"personal_best_score: {personal_best_score}\n"
                f"global_best_u: {global_best_u}\n"
                f"global_best_score: {global_best_score}\n"
                f"velocities_u: {velocities_u}\n\n")

    trajectory = [global_best_score]
    no_improve = 0

    # ——— 4) pure PSO loop ———
    for it in range(pso_iterations):
        w = w_max - (w_max - w_min) * it / pso_iterations

        iter_params: List[Particle] = []
        iter_scores: List[float] = []


        for i,u in enumerate(particles_u):
            for k in space:
                r1, r2 = py_rng.random(), py_rng.random()
                # velocity update
                velocities_u[i][k] = (
                    w*velocities_u[i][k]
                    + c1*r1*(personal_best_u[i][k] - u[k])
                    + c2*r2*(global_best_u[k]        - u[k])
                )

            norm = np.linalg.norm(list(velocities_u[i].values()))
            if norm > vmax:
                for k in space:
                    velocities_u[i][k] *= vmax / norm

            for k in space:
                u[k] = max(0.0, min(1.0, u[k] + velocities_u[i][k]))

            decoded = decode(u)
            score = fitness_fn(decoded)
            if score < personal_best_score[i]:
                personal_best_u[i], personal_best_score[i] = copy.deepcopy(u), score
                if score < global_best_score:
                    global_best_u, global_best_score = copy.deepcopy(u), score

            iter_params.append(decoded)
            iter_scores.append(score)

        param_history.append(iter_params)
        score_history.append(iter_scores)
        trajectory.append(global_best_score)
        visualize_training(param_history, score_history)

        if on_iteration_end:
            on_iteration_end(it, decode(global_best_u), global_best_score, trajectory)

        tol = 1e-5
        if trajectory[-2] - trajectory[-1] > tol:
            no_improve = 0
        else:
            no_improve += 1
        if early_stop_iters is not None and no_improve >= early_stop_iters:
            logger.info(f"[early stopping] stopped at iter {it}")
            break

    return decode(global_best_u), trajectory