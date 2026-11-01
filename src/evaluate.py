"""Evaluates a saved checkpoint and saves one prediction per example.

The test set is only used once, after all choices are made: it needs --final, and
it refuses to run if a test result for this checkpoint already exists.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import ArtDataset, load_processed, make_loader
from src.lit_module import ArtModule
from src.models import build_model
from src.paths import ARTIFACTS_DIR
from src.train import make_collate


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--final", action="store_true", help="Confirms that this is the final test evaluation.")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.final:
        sys.exit("Refusing to use the test set without --final.")
    return args


def main(argv=None):
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    output = ARTIFACTS_DIR / "predictions" / f"{checkpoint.parent.name}_{args.split}.json"
    if args.split == "test" and output.exists():
        sys.exit(f"Refusing to use the test set twice: {output} exists.")

    config = torch.load(checkpoint, map_location="cpu", weights_only=False)["hyper_parameters"]["config"]
    rows, vocabulary, matrix = load_processed()
    module = ArtModule.load_from_checkpoint(
        checkpoint, model=build_model(config, matrix), config=config, weights_only=False, map_location="cuda"
    ).eval()

    dataset = ArtDataset(rows[args.split], vocabulary, config["keep_punctuation"])
    loader = make_loader(dataset, config["batch_size"], make_collate(config, vocabulary))
    total_loss, scores, labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.cuda() for key, value in batch.items()}
            logits = module.model(batch)
            total_loss += F.cross_entropy(logits, batch["label"], reduction="sum").item()
            scores += logits.tolist()
            labels += batch["label"].tolist()

    predictions = [int(a < b) for a, b in scores]
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    result = {"split": args.split, "loss": total_loss / len(labels), "accuracy": accuracy, "n": len(labels)}
    print(checkpoint.parent.name, result)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump({**result, "scores": scores, "labels": labels, "predictions": predictions}, file)


if __name__ == "__main__":
    main()
