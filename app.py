# app.py
import argparse
import yaml

from src.utils.seed import set_seed
from src.data.load_telecomts import load_telecomts
from src.training.train import train_from_config
from src.inference.score import score_sample_from_config
from src.inference.evaluate import evaluate_from_config


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cmd_train(args):
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    dataset = load_telecomts(
        cfg["data"]["dataset_name"],
        cfg["data"]["data_files_pattern"]
    )
    train_from_config(cfg, dataset)


def cmd_score(args):
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    dataset = load_telecomts(
        cfg["data"]["dataset_name"],
        cfg["data"]["data_files_pattern"]
    )
    score_sample_from_config(cfg, dataset)


def cmd_evaluate(args):
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    dataset = load_telecomts(
        cfg["data"]["dataset_name"],
        cfg["data"]["data_files_pattern"]
    )
    evaluate_from_config(cfg, dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Telecom Network Anomaly Detection CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- TRAIN ---
    p_train = sub.add_parser(
        "train", help="Train LSTM autoencoder on NORMAL samples"
    )
    p_train.add_argument(
        "--config", required=True, help="Path to configs/train.yaml"
    )
    p_train.set_defaults(func=cmd_train)

    # --- SCORE ---
    p_score = sub.add_parser(
        "score", help="Score one sample and export anomaly scores"
    )
    p_score.add_argument(
        "--config", required=True, help="Path to configs/score.yaml"
    )
    p_score.set_defaults(func=cmd_score)

    # --- EVALUATE ---
    p_eval = sub.add_parser(
        "evaluate", help="Evaluate model on multiple samples"
    )
    p_eval.add_argument(
        "--config", required=True, help="Path to configs/eval.yaml"
    )
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
