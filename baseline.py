"""
Baseline hand‑crafted metrics for scoring Rubik's cube algorithms.
Compute baseline objectives on the validation set.
"""
import os
from typing import Dict, List

from algSpeed import alg_speed
from utils import load_algorithm_data, load_ranking_reference, split_states


# ---- Per‑move difficulty weights (Weighted Move Count) ----
MOVE_WEIGHTS = {
    "R": 1.0, "U": 1.0, "D": 1.2, "F": 1.5, "B": 1.8, "L": 2.0,
    "r": 1.3, "l": 2.3, "u": 1.3, "d": 1.5, "f": 1.8, "b": 2.1,
    "M": 2.2, "S": 2.0, "E": 1.8,
    "x": 2.5, "y": 2.5, "z": 2.5,
}
DEFAULT_WEIGHT = 2.0


def weighted_move_count(algorithm: str) -> float:
    """Sum per‑move difficulty weights (Weighted Move Count)."""
    total = 0.0
    for token in algorithm.split():
        if token.endswith(("'", "2")):
            base = token[:-1]
            multiplier = 2 if token.endswith("2") else 1
        else:
            base = token
            multiplier = 1
        total += MOVE_WEIGHTS.get(base, DEFAULT_WEIGHT) * multiplier
    return total


def count_moves(algorithm: str, metric: str) -> float:
    """
    Count moves according to one of the supported metrics:
        - stm  : slice turn metric
        - qtm  : quarter turn metric
        - wmc  : weighted move count
        - mcc  : movecount coefficient
    """
    if metric == "stm":
        return len(algorithm.split())
    if metric == "qtm":
        return sum(2 if t.endswith("2") else 1 for t in algorithm.split())
    if metric == "mcc":
        return alg_speed(algorithm)
    if metric == "wmc":
        return weighted_move_count(algorithm)
    raise ValueError(f"Unknown metric: {metric}")


def baseline_objective(
    states: List[str],
    alg_data: Dict,
    rank_ranks: Dict,
    num_ref_algorithms: int,
    metric: str,
) -> float:
    """
    Compute objective using a hand‑crafted metric as the scoring function.

    The objective is the average over states of the penalised preference score,
    exactly as defined in the training objective but using the metric scores
    instead of a learned model.
    """
    total = 0.0
    count = 0
    for state in states:
        if state not in alg_data or state not in rank_ranks:
            continue
        rank_scores = [count_moves(alg, metric) for alg in rank_ranks[state]]
        for entry in alg_data[state]:
            s = count_moves(entry["alg"], metric)
            # Number of reference algorithms that are better or equal
            better = sum(1 for rs in rank_scores if rs < s) + 0.5 * sum(1 for rs in rank_scores if rs == s)
            total += entry["rate"] * (better / num_ref_algorithms)
        count += 1
    return total / count if count > 0 else 0.0


def main():
    """Compute baseline objectives on the validation split."""
    DATA_DIR = "./corner/"
    MANMADE_FILE = "cornerManmade.json"
    RANKING_FILE = "cornerranking.json"
    NUM_REF_ALGORITHMS = 200
    SEED = 42
    SPLIT_RATIO = 0.8

    manmade_path = os.path.join(DATA_DIR, MANMADE_FILE)
    ranking_path = os.path.join(DATA_DIR, RANKING_FILE)

    alg_data = load_algorithm_data(manmade_path)
    rank_ranks = load_ranking_reference(ranking_path)

    common_states = list(set(alg_data.keys()) & set(rank_ranks.keys()))
    alg_data = {s: alg_data[s] for s in common_states}
    rank_ranks = {s: rank_ranks[s] for s in common_states}

    _, val_states = split_states(common_states, seed=SEED, split_ratio=SPLIT_RATIO)

    metrics = ["stm", "qtm", "wmc", "mcc"]
    results = {m: baseline_objective(val_states, alg_data, rank_ranks, NUM_REF_ALGORITHMS, m) for m in metrics}

    print("📊 Baseline objectives (validation set):")
    print(", ".join(f"{m} = {results[m]:.6f}" for m in metrics))


if __name__ == "__main__":
    main()