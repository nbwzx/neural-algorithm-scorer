"""Common utilities."""

import json
import math
import os
import random
from typing import Dict, List, Tuple, Any


def load_algorithm_data(manmade_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load human preference data.

    Returns a dict mapping state (string) to a list of entries, each with
    'alg' (algorithm string) and 'rate' (normalised preference score).
    """
    if not os.path.exists(manmade_path):
        return {}

    with open(manmade_path, "r", encoding="utf-8") as f:
        manmade = json.load(f)

    combined = {}
    for state, entries in manmade.items():
        total_users = sum(len(entry[1]) for entry in entries)
        sum_sqrt_users = sum(math.sqrt(len(entry[1])) for entry in entries)

        arr = []
        for entry in entries:
            votes = len(entry[1])
            # Normalise by sqrt(total_users) to dampen the effect of many voters
            rate = votes / math.sqrt(total_users) / sum_sqrt_users if total_users and sum_sqrt_users else 0.0
            for alg in entry[0]:
                arr.append({"alg": alg, "rate": rate / len(entry[0])})

        if arr:
            arr.sort(key=lambda x: -x["rate"])
            combined[state] = arr

    return combined


def load_ranking_reference(ranking_path: str) -> Dict[str, List[str]]:
    """Load the algorithm ranking reference (JSON)."""
    if not os.path.exists(ranking_path):
        raise FileNotFoundError(f"Ranking file not found: {ranking_path}")
    with open(ranking_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_states(states: List[str], seed: int = 42, split_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """
    Shuffle and split states into train/validation lists.

    Args:
        states: list of state identifiers.
        seed: random seed for reproducibility.
        split_ratio: fraction of states used for training.

    Returns:
        (train_states, val_states)
    """
    random.seed(seed)
    shuffled = sorted(states)          # deterministic ordering before shuffle
    random.shuffle(shuffled)
    split_idx = int(split_ratio * len(shuffled))
    return shuffled[:split_idx], shuffled[split_idx:]


class ActionTokenizer:
    def __init__(self, algorithms):
        actions = set()
        for f in algorithms:
            for act in f.split():
                actions.add(act)
        self.stoi = {'<PAD>': 0}
        for a in sorted(actions):
            self.stoi[a] = len(self.stoi)
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, algorithm, max_len=None):
        ids = [self.stoi[act] for act in algorithm.split() if act in self.stoi]
        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [0] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        return ' '.join([self.itos[i] for i in ids if i != 0])