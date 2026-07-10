"""
Score and sort Rubik's cube algorithms using a trained neural model.

Usage:
  # Sort a simple ranking JSON (state → list of algs)
  python export.py ranking.json output.json [--top-k 18]

  # Sort cornerManmade.json (nested entries) by model score
  python export.py cornerManmade.json cornerManmadeSorted.json

  # Sort a plain text file (one alg per line)
  python export.py algs.txt algs_sorted.txt --txt [--with-scores] [--top-k 100]

  # Sort all .txt files in a directory
  python export.py --dir data/ [--output-dir sorted/] [--top-k 100] [--with-scores]
"""

import os
import sys
import json
import pickle
import argparse
import math
from typing import List, Dict, Optional, Tuple

import torch
from tqdm import tqdm

# Local modules
from model import AlgorithmScorer
from utils import ActionTokenizer


# =============================================================================
#  CONFIGURATION
# =============================================================================
RUN_DIR = "best"   # folder containing best_model.pth, tokenizer.pkl, config.json
BATCH_SIZE = 512           # number of algorithms to score in one forward pass
# =============================================================================


# ---------- Model loading (reads config.json) -------------------------------
def load_model_and_tokenizer(run_dir: str) -> Tuple[AlgorithmScorer, ActionTokenizer, dict]:
    """Load the trained model, tokenizer, and training config from a run directory."""
    model_path = os.path.join(run_dir, "best_model.pth")
    tokenizer_path = os.path.join(run_dir, "tokenizer.pkl")
    config_path = os.path.join(run_dir, "config.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        print(f"⚠️ config.json not found in {run_dir} – using hardcoded defaults!")
        config = {"DIM": 128, "NUM_LAYERS": 4, "NUM_HEADS": 4, "MAX_LEN": 50}

    dim = config.get("DIM", 128)
    num_layers = config.get("NUM_LAYERS", 4)
    num_heads = config.get("NUM_HEADS", 4)
    max_len = config.get("MAX_LEN", 50)
    ff_hidden_dim = config.get("FF_HIDDEN_DIM", 4 * dim)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    model = AlgorithmScorer(
        vocab_size=tokenizer.vocab_size,
        dim=dim,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_len,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"✅ Loaded model: DIM={dim}, LAYERS={num_layers}, HEADS={num_heads}, MAX_LEN={max_len}")
    return model, tokenizer, config


# ---------- Batched scoring with proper progress bar ------------------------
def score_algorithms_batch(
    algs: List[str],
    model: AlgorithmScorer,
    tokenizer: ActionTokenizer,
    max_len: int,
    batch_size: int = BATCH_SIZE,
    desc: str = "Scoring",
    show_progress: bool = True,
) -> List[float]:
    """
    Score a list of algorithms in batches.
    Shows a progress bar over the number of batches.
    """
    total = len(algs)
    if total == 0:
        return []

    num_batches = math.ceil(total / batch_size)
    scores = []

    with torch.no_grad():
        # Progress bar over batches
        with tqdm(total=num_batches, desc=desc, unit="batch", disable=not show_progress) as pbar:
            for start in range(0, total, batch_size):
                batch = algs[start:start + batch_size]
                # Encode all algorithms in the batch
                ids_list = []
                for alg in batch:
                    ids = tokenizer.encode(alg, max_len=max_len)
                    ids_list.append(ids)
                input_tensor = torch.tensor(ids_list, dtype=torch.long)
                batch_scores = model(input_tensor).squeeze(-1).flatten().tolist()
                scores.extend(batch_scores)
                pbar.update(1)

    return scores


# ---------- Sorting helpers --------------------------------------------------
def sort_simple_json(
    data: Dict[str, List[str]],
    model,
    tokenizer,
    max_len: int,
    top_k: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Sort a simple JSON (state → list of algorithms) by model score (ascending)."""
    sorted_data = {}
    for state, algs in tqdm(data.items(), desc="Sorting states"):
        if not algs:
            sorted_data[state] = []
            continue
        # Inner scoring progress is suppressed – only the outer state bar is shown
        scores = score_algorithms_batch(algs, model, tokenizer, max_len, desc=f"Scoring {state}", show_progress=False)
        scored = sorted(zip(scores, algs), key=lambda x: x[0])
        sorted_algs = [alg for _, alg in scored]
        if top_k is not None:
            sorted_algs = sorted_algs[:top_k]
        sorted_data[state] = sorted_algs
    return sorted_data


def sort_nested_json(
    data: Dict[str, List[List]],
    model,
    tokenizer,
    max_len: int,
    top_k: Optional[int] = None,
) -> Dict[str, List[List]]:
    """Sort nested JSON (cornerManmade format) by the score of the first algorithm in each entry."""
    sorted_data = {}
    for state, entries in tqdm(data.items(), desc="Sorting states"):
        if not entries:
            sorted_data[state] = []
            continue

        # Extract first algorithm from each entry (if any)
        first_algs = []
        valid_indices = []
        for idx, entry in enumerate(entries):
            if entry and isinstance(entry[0], list) and len(entry[0]) > 0:
                first_algs.append(entry[0][0])
                valid_indices.append(idx)

        # Score all first algorithms in batch – inner progress suppressed
        scores = score_algorithms_batch(first_algs, model, tokenizer, max_len, desc=f"Scoring {state}", show_progress=False)

        # Reconstruct scored entries
        scored_entries = []
        score_idx = 0
        for idx, entry in enumerate(entries):
            if idx in valid_indices:
                score = scores[score_idx]
                score_idx += 1
            else:
                score = float("inf")
            scored_entries.append((score, entry))

        scored_entries.sort(key=lambda x: x[0])
        sorted_entries = [entry for _, entry in scored_entries]
        if top_k is not None:
            sorted_entries = sorted_entries[:top_k]
        sorted_data[state] = sorted_entries
    return sorted_data


def sort_text_algs(
    algs: List[str],
    model,
    tokenizer,
    max_len: int,
    with_scores: bool = False,
    desc: str = "Scoring",
    show_progress: bool = True,
    top_k: Optional[int] = None,
) -> List:
    """
    Score and sort a list of algorithms (plain text).
    If top_k is given, keep only the top‑K algorithms (lowest scores).
    """
    scores = score_algorithms_batch(algs, model, tokenizer, max_len, desc=desc, show_progress=show_progress)
    scored = sorted(zip(scores, algs), key=lambda x: x[0])
    if top_k is not None:
        scored = scored[:top_k]
    if with_scores:
        return [(score, alg) for score, alg in scored]
    else:
        return [alg for _, alg in scored]


# ---------- File processing --------------------------------------------------
def sort_json_file(
    input_path: str,
    output_path: str,
    top_k: Optional[int] = None,
    run_dir: str = RUN_DIR,
):
    """Sort a JSON file (either simple or nested)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model, tokenizer, config = load_model_and_tokenizer(run_dir)
    max_len = config.get("MAX_LEN", 50)

    sample_key = next(iter(data.keys()))
    sample_value = data[sample_key]
    is_nested = (
        isinstance(sample_value, list)
        and len(sample_value) > 0
        and isinstance(sample_value[0], list)
    )

    if is_nested:
        sorted_data = sort_nested_json(data, model, tokenizer, max_len, top_k)
    else:
        sorted_data = sort_simple_json(data, model, tokenizer, max_len, top_k)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=4, separators=(",", ":  "))
    print(f"✅ Sorted JSON saved to {output_path}")


def sort_text_file(
    input_path: str,
    output_path: str,
    run_dir: str = RUN_DIR,
    with_scores: bool = False,
    top_k: Optional[int] = None,
):
    """Sort a plain text file (one algorithm per line)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        algs = [line.strip() for line in f if line.strip()]

    if not algs:
        print(f"⚠️ No algorithms found in {input_path}")
        return

    model, tokenizer, config = load_model_and_tokenizer(run_dir)
    max_len = config.get("MAX_LEN", 50)

    result = sort_text_algs(
        algs,
        model,
        tokenizer,
        max_len,
        with_scores,
        desc=f"Scoring {os.path.basename(input_path)}",
        show_progress=True,   # single file → show progress bar
        top_k=top_k,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        if with_scores:
            for score, alg in result:
                f.write(f"{score}\t{alg}\n")
        else:
            for alg in result:
                f.write(alg + "\n")
    print(f"✅ {'Scored and ' if with_scores else ''}Sorted algorithms saved to {output_path}")


def sort_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    top_k: Optional[int] = None,
    run_dir: str = RUN_DIR,
    with_scores: bool = False,
):
    """Sort all .txt files in a directory."""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    if not txt_files:
        print(f"⚠️ No .txt files found in {input_dir}")
        return

    model, tokenizer, config = load_model_and_tokenizer(run_dir)
    max_len = config.get("MAX_LEN", 50)
    print(f"Model loaded. Processing {len(txt_files)} files...")

    for filename in tqdm(txt_files, desc="Processing files"):
        in_path = os.path.join(input_dir, filename)
        if output_dir == input_dir:
            base, ext = os.path.splitext(filename)
            suffix = "_scores" if with_scores else "_sorted"
            out_path = os.path.join(output_dir, base + suffix + ext)
        else:
            out_path = os.path.join(output_dir, filename)

        with open(in_path, "r", encoding="utf-8") as f:
            algs = [line.strip() for line in f if line.strip()]
        if not algs:
            continue

        # For directory mode, we suppress the per‑file progress bar (show_progress=False)
        # so only the outer "Processing files" bar is shown.
        result = sort_text_algs(
            algs,
            model,
            tokenizer,
            max_len,
            with_scores,
            desc=f"Scoring {filename}",
            show_progress=False,
            top_k=top_k,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            if with_scores:
                for score, alg in result:
                    f.write(f"{score}\t{alg}\n")
            else:
                for alg in result:
                    f.write(alg + "\n")

    print(f"\n✅ All files processed. Sorted files are in {output_dir}")


# ---------- Main -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Score and sort Rubik's cube algorithms using a trained model."
    )
    parser.add_argument("input", nargs="?", help="Input file path (JSON or plain text).")
    parser.add_argument("output", nargs="?", help="Output file path (optional).")
    parser.add_argument("--txt", action="store_true", help="Treat input as plain text.")
    parser.add_argument("--dir", type=str, help="Process all .txt files in directory.")
    parser.add_argument("--output-dir", type=str, help="Output directory (with --dir).")
    parser.add_argument("--top-k", type=int, default=None, help="Keep only top‑K algorithms/entries.")
    parser.add_argument("--with-scores", "-s", action="store_true", help="Include scores in text output.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=RUN_DIR,
        help="Folder containing best_model.pth, tokenizer.pkl, and config.json."
    )

    args = parser.parse_args()

    if args.dir:
        try:
            sort_directory(
                args.dir,
                args.output_dir,
                args.top_k,
                args.run_dir,
                args.with_scores,
            )
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        if args.txt:
            suffix = "_scores" if args.with_scores else "_sorted"
            out_path = base + suffix + ext
        else:
            out_path = base + "_sorted" + ext

    try:
        if args.txt:
            sort_text_file(args.input, out_path, args.run_dir, args.with_scores, args.top_k)
        else:
            sort_json_file(args.input, out_path, args.top_k, args.run_dir)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()