"""
Neural Algorithm Scorer: A deep‑learning model to score Rubik's cube algorithms.
"""
import datetime
import json
import os
import pickle
import random
import sys
from typing import Dict, List, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from loguru import logger

from utils import load_algorithm_data, load_ranking_reference, split_states
from model import AlgorithmScorer


# =============================================================================
#  CONFIGURATION
# =============================================================================
CONFIG = {
    # ---- Data ----
    "DATA_DIR": "./corner/",
    "MANMADE_FILE": "cornerManmade.json",
    "RANKING_FILE": "cornerRanking.json",

    # ---- Model ----
    "DIM": 64,
    "FF_HIDDEN_DIM": 256,
    "NUM_LAYERS": 3,
    "NUM_HEADS": 4,
    "DROPOUT": 0.1,
    "MAX_LEN": 50,

    # ---- Training ----
    "BATCH_SIZE": 16,
    "EPOCHS": 100,
    "LR": 1e-3,
    "WEIGHT_DECAY": 0.01,
    "PATIENCE": 10,
    "TEMPERATURE": 0.1,
    "NUM_REF_ALGORITHMS": 200,

    # ---- Reproducibility ----
    "SEED": 42,

    # ---- Output ----
    "RUN_DIR": None,          # auto‑timestamp if None

    # ---- Progress ----
    "SHOW_PROGRESS": True,
}


# ---------- Tokenizer ---------------------------------------------------------
class ActionTokenizer:
    """Simple tokenizer for Rubik's cube algorithms."""

    def __init__(self, sequences: List[str]):
        """
        Build vocabulary from all tokens appearing in the given sequences.

        Special tokens: <PAD>, <UNK> are added first.
        """
        tokens = set()
        for seq in sequences:
            tokens.update(seq.split())

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.special_tokens = [self.pad_token, self.unk_token]

        self.token_to_id = {tok: i for i, tok in enumerate(self.special_tokens)}
        for tok in sorted(tokens):
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)

        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

    def encode(self, sequence: str, max_len: int) -> List[int]:
        """Tokenise and pad/truncate to max_len."""
        tokens = sequence.split()
        ids = [self.token_to_id.get(tok, self.token_to_id[self.unk_token]) for tok in tokens]
        if len(ids) < max_len:
            ids += [self.token_to_id[self.pad_token]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids


# ---------- Dataset -----------------------------------------------------------
class StateDataset(Dataset):
    """Dataset for Rubik's cube states, with on‑the‑fly token caching."""

    def __init__(
        self,
        state_keys: List[str],
        alg_data: Dict,
        rank_ranks: Dict,
        tokenizer: ActionTokenizer,
        max_len: int,
        cache: Optional[Dict] = None,
    ):
        self.state_keys = state_keys
        self.alg_data = alg_data
        self.rank_ranks = rank_ranks
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache = cache if cache is not None else {}

        # Pre‑tokenise all algorithms to avoid repeated work
        all_algorithms = set()
        for state in state_keys:
            for entry in alg_data[state]:
                all_algorithms.add(entry["alg"])
            for alg in rank_ranks[state]:
                all_algorithms.add(alg)

        for seq in all_algorithms:
            if seq not in self.cache:
                ids = tokenizer.encode(seq, max_len=max_len)
                self.cache[seq] = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.state_keys)

    def __getitem__(self, idx: int) -> Dict:
        state = self.state_keys[idx]
        entries = self.alg_data[state]

        entry_rates = [e["rate"] for e in entries]
        entry_ids = (
            torch.stack([self.cache[e["alg"]] for e in entries])
            if entries
            else torch.empty(0, self.max_len, dtype=torch.long)
        )

        rank_ids = (
            torch.stack([self.cache[alg] for alg in self.rank_ranks[state]])
            if self.rank_ranks[state]
            else torch.empty(0, self.max_len, dtype=torch.long)
        )

        return {
            "state": state,
            "entry_rates": torch.tensor(entry_rates, dtype=torch.float),
            "entry_ids": entry_ids,
            "rank_ids": rank_ids,
        }


# ---------- Exact objective (validation) --------------------------------------
def compute_objective_exact(
    model: torch.nn.Module,
    alg_data: Dict,
    rank_ranks: Dict,
    num_ref_algorithms: int,
    device: torch.device,
    tokenizer: ActionTokenizer,
    max_len: int,
    states: Optional[List[str]] = None,
    cache: Optional[Dict] = None,
    score_batch_size: int = 64,
) -> float:
    """
    Compute the exact objective over the given states.
    Lower is better.
    """
    model.eval()
    if states is None:
        states = list(set(alg_data.keys()) & set(rank_ranks.keys()))
    if not states:
        return 0.0

    # Collect all algorithms and metadata
    all_algorithms = []
    entry_rates_per_state = []
    entry_counts = []
    rank_counts = []

    for state in states:
        entries = alg_data.get(state, [])
        ranks = rank_ranks.get(state, [])
        entry_rates_per_state.append([e["rate"] for e in entries])
        entry_counts.append(len(entries))
        rank_counts.append(len(ranks))
        all_algorithms.extend(e["alg"] for e in entries)
        all_algorithms.extend(ranks)

    if not all_algorithms:
        return 0.0

    if cache is None:
        cache = {}

    # Score all algorithms in batches
    all_scores = []
    total_algorithms = len(all_algorithms)
    for start_idx in range(0, total_algorithms, score_batch_size):
        end_idx = min(start_idx + score_batch_size, total_algorithms)
        batch_algorithms = all_algorithms[start_idx:end_idx]

        ids_list = []
        for seq in batch_algorithms:
            if seq not in cache:
                ids = tokenizer.encode(seq, max_len=max_len)
                cache[seq] = torch.tensor(ids, dtype=torch.long, device=device)
            ids_list.append(cache[seq])

        batch_ids = torch.stack(ids_list).to(device)
        with torch.no_grad():
            batch_scores = model(batch_ids).squeeze(-1)
        all_scores.append(batch_scores)

    scores = torch.cat(all_scores, dim=0)

    # Compute objective per state
    total_obj = 0.0
    count = 0
    idx = 0
    for si, state in enumerate(states):
        E = entry_counts[si]
        R = rank_counts[si]
        if E == 0 or R == 0:
            idx += E + R
            continue

        entry_scores = scores[idx : idx + E]
        idx += E
        rank_scores = scores[idx : idx + R]
        idx += R

        rates = torch.tensor(entry_rates_per_state[si], dtype=torch.float, device=device)
        diff = entry_scores.unsqueeze(1) - rank_scores.unsqueeze(0)  # [E, R]
        better = (diff > 0).float().sum(dim=1) + 0.5 * (diff == 0).float().sum(dim=1)
        better = better / num_ref_algorithms
        penalty = rates * better
        total_obj += penalty.sum().item()
        count += 1

    return total_obj / count if count > 0 else 0.0


# ---------- Differentiable training loss --------------------------------------
def compute_batch_loss(
    model: torch.nn.Module,
    batch: List[Dict],
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the differentiable training loss for a batch of states.
    Loss = average penalised objective
    """
    all_entry_ids, all_rank_ids = [], []
    entry_counts, rank_counts = [], []
    entry_rates_list = []

    for state_dict in batch:
        entry_ids = state_dict["entry_ids"]
        rank_ids = state_dict["rank_ids"]
        entry_rates = state_dict["entry_rates"]

        all_entry_ids.append(entry_ids)
        all_rank_ids.append(rank_ids)
        entry_counts.append(entry_ids.size(0))
        rank_counts.append(rank_ids.size(0))
        entry_rates_list.append(entry_rates.to(device))

    # Concatenate all sequences
    all_entry_ids = torch.cat(all_entry_ids, dim=0).to(device) if all_entry_ids else None
    all_rank_ids = torch.cat(all_rank_ids, dim=0).to(device) if all_rank_ids else None

    # Forward pass
    entry_scores_all = (
        model(all_entry_ids).squeeze(-1) if all_entry_ids is not None else torch.tensor([], device=device)
    )
    rank_scores_all = (
        model(all_rank_ids).squeeze(-1) if all_rank_ids is not None else torch.tensor([], device=device)
    )

    # Per‑state loss
    batch_loss = torch.tensor(0.0, device=device)
    e_idx, r_idx = 0, 0
    for i in range(len(batch)):
        E = entry_counts[i]
        R = rank_counts[i]
        e_scores = entry_scores_all[e_idx : e_idx + E] if E > 0 else torch.tensor([], device=device)
        r_scores = rank_scores_all[r_idx : r_idx + R] if R > 0 else torch.tensor([], device=device)
        e_idx += E
        r_idx += R

        if E == 0 or R == 0:
            continue

        rates = entry_rates_list[i]
        diff = e_scores.unsqueeze(1) - r_scores.unsqueeze(0)          # [E, R]
        probs = torch.sigmoid(diff / temperature)                     # differentiable ranking
        better = probs.mean(dim=1)                                    # approx better / num_ref_algorithms
        penalty = rates * better
        state_obj = penalty.sum()
        batch_loss = batch_loss + state_obj

    return batch_loss / len(batch)


# ---------- Training loop -----------------------------------------------------
def train_direct(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_states: List[str],
    alg_data: Dict,
    rank_ranks: Dict,
    tokenizer: ActionTokenizer,
    config: Dict,
    device: torch.device,
    resume_from: Optional[str] = None,
    history_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    best_model_file: Optional[str] = None,
) -> torch.nn.Module:
    """
    Main training loop with early stopping and checkpointing.

    Args:
        model: the AlgorithmScorer instance.
        train_loader, val_loader: DataLoaders for training and validation.
        val_states: list of state keys used for validation.
        alg_data, rank_ranks: data dictionaries.
        tokenizer: ActionTokenizer.
        config: configuration dictionary.
        device: torch device.
        resume_from: path to checkpoint to resume from (if any).
        history_file: JSONL file to write epoch metrics.
        checkpoint_file: path to save periodic checkpoints.
        best_model_file: path to save the best model weights.

    Returns:
        The trained model (with best weights loaded).
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    start_epoch = 1
    best_obj = float("inf")
    best_epoch = 0
    patience_counter = 0
    cache = {}  # shared token cache for validation

    # Resume if checkpoint exists
    if resume_from and os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_obj = compute_objective_exact(
            model, alg_data, rank_ranks, config["NUM_REF_ALGORITHMS"], device, tokenizer, config["MAX_LEN"],
            states=val_states
        )
        best_epoch = checkpoint.get("best_epoch", 0)
        patience_counter = checkpoint.get("patience_counter", 0)
        cache = checkpoint.get("cache", {})
        logger.info(
            f"Resumed at epoch {start_epoch}, best val obj = {best_obj:.6f}, "
            f"patience = {patience_counter}/{config['PATIENCE']}"
        )

    for epoch in range(start_epoch, config["EPOCHS"] + 1):
        # ---- Training ----
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['EPOCHS']}", disable=not config["SHOW_PROGRESS"])
        for batch in loop:
            loss = compute_batch_loss(model, batch, config["TEMPERATURE"], device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_train_loss = total_loss / len(train_loader)

        # ---- Validation loss ----
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_loss_total += compute_batch_loss(
                    model, val_batch, config["TEMPERATURE"], device
                ).item()
        avg_val_loss = val_loss_total / len(val_loader) if len(val_loader) > 0 else float("nan")

        # ---- Validation objective ----
        val_obj = compute_objective_exact(
            model,
            alg_data,
            rank_ranks,
            config["NUM_REF_ALGORITHMS"],
            device,
            tokenizer,
            config["MAX_LEN"],
            states=val_states,
            cache=cache,
        )

        logger.info(
            f"Epoch {epoch}/{config['EPOCHS']} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val Objective: {val_obj:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # ---- Best model & early stopping ----
        if val_obj < best_obj:
            best_obj = val_obj
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_file)
            logger.success(f"Best model saved (val objective: {val_obj:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{config['PATIENCE']}")
            if patience_counter >= config["PATIENCE"]:
                logger.warning("Early stopping triggered.")
                break

        # ---- Learning rate scheduling ----
        scheduler.step(val_obj)

        # ---- Save checkpoint ----
        torch.save(
            {
                "epoch": epoch,
                "best_obj": best_obj,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cache": cache,
            },
            checkpoint_file,
        )

        # ---- Write history ----
        if history_file:
            with open(history_file, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "val_objective": val_obj,
                        "lr": optimizer.param_groups[0]["lr"],
                        "best_val_objective": best_obj,
                        "best_epoch": best_epoch,
                        "patience_counter": patience_counter,
                        "timestamp": datetime.datetime.now().isoformat(),
                    },
                    f,
                )
                f.write("\n")

    # Load best model
    model.load_state_dict(torch.load(best_model_file))
    logger.success(f"Training complete. Best val objective: {best_obj:.6f}")
    return model


# ---------- Main -------------------------------------------------------------
def main():
    """Set up everything and run training."""
    config = CONFIG

    # ---- Reproducibility ----
    random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])

    # ---- Output directory ----
    if config["RUN_DIR"] is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(os.getcwd(), timestamp)
    else:
        run_dir = config["RUN_DIR"]
    os.makedirs(run_dir, exist_ok=True)

    # ---- File paths ----
    config_path = os.path.join(run_dir, "config.json")
    checkpoint_file = os.path.join(run_dir, "checkpoint.pt")
    best_model_file = os.path.join(run_dir, "best_model.pth")
    tokenizer_file = os.path.join(run_dir, "tokenizer.pkl")
    history_file = os.path.join(run_dir, "epoch_history.jsonl")
    log_file = os.path.join(run_dir, "training.log")

    # ---- Save config ----
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # ---- Logging ----
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> <level>{level}</level> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        encoding="utf-8",
    )

    logger.info("=" * 70)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config saved to: {config_path}")
    logger.info(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 70)

    # ---- Load data ----
    data_dir = config["DATA_DIR"]
    manmade_path = os.path.join(data_dir, config["MANMADE_FILE"])
    ranking_path = os.path.join(data_dir, config["RANKING_FILE"])

    alg_data = load_algorithm_data(manmade_path)
    rank_ranks = load_ranking_reference(ranking_path)

    common_states = list(set(alg_data.keys()) & set(rank_ranks.keys()))
    alg_data = {s: alg_data[s] for s in common_states}
    rank_ranks = {s: rank_ranks[s] for s in common_states}
    logger.info(f"Loaded {len(alg_data)} common states.")

    # ---- Train/validation split ----
    train_states, val_states = split_states(
        common_states, seed=config["SEED"], split_ratio=0.8
    )
    logger.info(f"Train states: {len(train_states)}, Val states: {len(val_states)}")

    # ---- Tokenizer ----
    if os.path.exists(tokenizer_file):
        with open(tokenizer_file, "rb") as f:
            tokenizer = pickle.load(f)
        logger.info(f"Loaded tokenizer from {tokenizer_file}")
    else:
        all_algorithms = []
        for items in alg_data.values():
            all_algorithms.extend(item["alg"] for item in items)
        for algs in rank_ranks.values():
            all_algorithms.extend(algs)
        tokenizer = ActionTokenizer(all_algorithms)
        with open(tokenizer_file, "wb") as f:
            pickle.dump(tokenizer, f)
        logger.info(f"Built tokenizer (vocab size: {tokenizer.vocab_size})")

    # ---- Datasets & loaders ----
    cache = {}
    train_dataset = StateDataset(
        train_states, alg_data, rank_ranks, tokenizer, config["MAX_LEN"], cache=cache
    )
    val_dataset = StateDataset(
        val_states, alg_data, rank_ranks, tokenizer, config["MAX_LEN"], cache=cache
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=lambda x: x
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=lambda x: x
    )

    # ---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = AlgorithmScorer(
        vocab_size=tokenizer.vocab_size,
        dim=config["DIM"],
        ff_hidden_dim=config["FF_HIDDEN_DIM"],
        num_layers=config["NUM_LAYERS"],
        num_heads=config["NUM_HEADS"],
        max_seq_len=config["MAX_LEN"],
        dropout=config["DROPOUT"],
    ).to(device)

    # ---- Resume ----
    resume_path = checkpoint_file if os.path.exists(checkpoint_file) else None
    if resume_path:
        logger.info(f"Found checkpoint: {resume_path}")
    else:
        logger.info("No checkpoint found – starting fresh.")

    # ---- Train ----
    model = train_direct(
        model,
        train_loader,
        val_loader,
        val_states,
        alg_data,
        rank_ranks,
        tokenizer,
        config,
        device,
        resume_from=resume_path,
        history_file=history_file,
        checkpoint_file=checkpoint_file,
        best_model_file=best_model_file,
    )

    # ---- Final save ----
    with open(tokenizer_file, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.success("Training complete. Best model and tokenizer saved.")


if __name__ == "__main__":
    main()