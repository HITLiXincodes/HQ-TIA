import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from train.config import add_config_arguments, load_config


def get_all_embeddings(embedding_dir: str) -> List[str]:
    if not os.path.isdir(embedding_dir):
        raise FileNotFoundError("Embedding dir not found: {}".format(embedding_dir))
    files: List[str] = []
    for embedding_file in sorted(os.listdir(embedding_dir)):
        if embedding_file.endswith(".npy"):
            file_path = os.path.join(embedding_dir, embedding_file)
            if os.path.isfile(file_path):
                files.append(file_path)
    if not files:
        raise ValueError("No .npy files found in: {}".format(embedding_dir))
    return files


def cosine_metric(x1: np.ndarray, x2: np.ndarray) -> float:
    x1 = x1.reshape(-1).astype(np.float32)
    x2 = x2.reshape(-1).astype(np.float32)
    denom = np.linalg.norm(x1) * np.linalg.norm(x2)
    if denom == 0:
        return 0.0
    return float(np.dot(x1, x2) / denom)


def build_pairs(real_embeddings: List[str], fake_embeddings: List[str]) -> List[Tuple[str, str]]:
    real_map = {os.path.basename(p): p for p in real_embeddings}
    fake_map = {os.path.basename(p): p for p in fake_embeddings}
    same_names = sorted(set(real_map.keys()) & set(fake_map.keys()))

    if same_names:
        return [(real_map[name], fake_map[name]) for name in same_names]

    pair_num = min(len(real_embeddings), len(fake_embeddings))
    if pair_num == 0:
        raise ValueError("No valid pairs to compare.")
    print(
        "[Warning] No same filenames found. Fallback to sorted index pairing, pair_num={}".format(
            pair_num
        )
    )
    return [(real_embeddings[i], fake_embeddings[i]) for i in range(pair_num)]


def calculate_similarities(real_embeddings: List[str], fake_embeddings: List[str]) -> List[float]:
    similarities: List[float] = []
    for real_path, fake_path in build_pairs(real_embeddings, fake_embeddings):
        real = np.load(real_path)
        fake = np.load(fake_path)
        sim = cosine_metric(real, fake)
        similarities.append(sim)
    return similarities


def summarize_threshold_ratios(similarities: List[float]) -> List[Tuple[float, float]]:
    if not similarities:
        raise ValueError("similarities is empty")
    sims = np.array(similarities, dtype=np.float32)
    thresholds = [round(x, 2) for x in np.arange(0.0, 1.0 + 1e-8, 0.01)]

    results: List[Tuple[float, float]] = []
    for th in thresholds:
        ratio = float(np.mean(sims > th))
        results.append((th, round(ratio, 4)))
    return results


def save_results(output_csv: str, threshold_ratios: List[Tuple[float, float]]) -> None:
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("threshold,ratio_over_threshold\n")
        for th, ratio in threshold_ratios:
            f.write("{:.4f},{:.4f}\n".format(th, ratio))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate cosine similarity and ratio over threshold."
    )
    add_config_arguments(parser)
    parser.add_argument("--real-embedding-root", type=str, default=None)
    parser.add_argument("--fake-embedding-root", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    attack_config = config["attack_metrics"]

    real_embedding_root = args.real_embedding_root or attack_config["real_embedding_root"]
    fake_embedding_root = args.fake_embedding_root or attack_config["fake_embedding_root"]
    output_csv = args.output_csv or attack_config["output_csv"]

    real_embeddings = get_all_embeddings(real_embedding_root)
    fake_embeddings = get_all_embeddings(fake_embedding_root)
    similarities = calculate_similarities(real_embeddings, fake_embeddings)
    threshold_ratios = summarize_threshold_ratios(similarities)
    save_results(output_csv, threshold_ratios)

    print("pairs: {}".format(len(similarities)))
    print("similarity mean: {:.4f}".format(float(np.mean(similarities))))
    print("saved threshold ratios to: {}".format(output_csv))
    for th, ratio in threshold_ratios:
        print("threshold {:.4f} -> ratio {:.4f}".format(th, ratio))


if __name__ == "__main__":
    main()
