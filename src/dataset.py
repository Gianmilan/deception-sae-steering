import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from src.config import SYCOPHANCY_DATASET_PATH, GENERAL_QA_PATH, PERPLEXITY_CORPUS_PATH


def load_sycophancy_dataset(
    filepath: Optional[Path] = None,
) -> List[Dict]:
    """
    Loads contrastive sycophancy prompt pairs from JSON.
    Each entry contains:
      - id: unique identifier
      - category: topic/domain
      - prompt_a: sycophantic incentive prompt
      - prompt_b: neutral baseline prompt
      - ground_truth: factual reality
      - sycophantic_target: description of deceptive agreement
    """
    path = filepath or SYCOPHANCY_DATASET_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Sycophancy dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def get_prompt_pairs(
    filepath: Optional[Path] = None,
    split_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Loads and splits prompt pairs into hunting/tuning split and evaluation split.
    """
    import random

    dataset = load_sycophancy_dataset(filepath)
    rng = random.Random(seed)
    shuffled = dataset.copy()
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * split_ratio)
    train_tuning_set = shuffled[:split_idx]
    eval_set = shuffled[split_idx:]

    return train_tuning_set, eval_set


def load_general_qa_dataset(
    filepath: Optional[Path] = None,
) -> List[Dict]:
    """
    Loads neutral multiple-choice QA evaluation dataset (MMLU-style).
    """
    path = filepath or GENERAL_QA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"General QA dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_perplexity_corpus(
    filepath: Optional[Path] = None,
) -> List[Dict]:
    """
    Loads neutral text samples for language model perplexity evaluation.
    """
    path = filepath or PERPLEXITY_CORPUS_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Perplexity corpus not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    pairs = load_sycophancy_dataset()
    print(f"Loaded {len(pairs)} contrastive prompt pairs.")
    print("Sample Pair 1:")
    print("  Prompt A (Biased):", pairs[0]["prompt_a"][:70], "...")
    print("  Prompt B (Neutral):", pairs[0]["prompt_b"][:70], "...")
