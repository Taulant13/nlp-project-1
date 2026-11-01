"""Trains one model. W&B logging only happens with --wandb.

Example: python -m src.train --arch a1 --lr 1e-3
"""

import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from src.data import ArtDataset, StoryCollator, collate_fields, load_processed, make_loader
from src.embeddings import SEP
from src.lit_module import ArtModule, BestValidation
from src.models import build_model
from src.paths import ARTIFACTS_DIR, CHECKPOINTS_DIR, WANDB_DIR

WANDB_PROJECT = "nlp-project1"
DEFAULT_SEED = 42  # Left out of run names; other seeds are shown.
DEFAULT_WEIGHT_DECAY = 0.01  # Same for run names.
ARCH_1_SETTINGS = {"input_format", "pooling"}
ARCH_2_SETTINGS = {"cell", "bidirectional", "rnn_hidden", "story_format", "no_sep"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["a1", "a2"], required=True)
    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # Classifier, both architectures
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--layer-norm", action="store_true")
    # Preprocessing variant
    parser.add_argument("--keep-punctuation", action="store_true")
    # Arch 1
    parser.add_argument("--input-format", choices=["shared", "joint", "interaction", "h_only"], default="shared")
    parser.add_argument("--pooling", choices=["mean", "sum"], default="mean")
    # Arch 2
    parser.add_argument("--cell", choices=["lstm", "gru"], default="gru")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--rnn-hidden", type=int, default=256)
    parser.add_argument("--story-format", choices=["o1_h_o2", "o1_o2_h", "separate", "h_only"], default="o1_h_o2")
    parser.add_argument("--no-sep", action="store_true")
    # Runs
    parser.add_argument("--wandb", action="store_true", help="Log this run to W&B. Off by default.")
    parser.add_argument("--tags", nargs="*", default=[], help="W&B tags, e.g. the experiment group.")
    parser.add_argument("--overfit-batches", type=int, default=0, help="Sanity check: train on this many batches.")
    return parser.parse_args(argv)


def run_name(config: dict) -> str:
    """Readable name that shows what differs between runs."""
    common = f"h{config['hidden']}_d{config['dropout']:g}_lr{config['lr']:g}"
    if config["seed"] != DEFAULT_SEED:
        common += f"_s{config['seed']}"
    if config["arch"] == "a1":
        parts = ["a1", config["input_format"], config["pooling"]]
    else:
        direction = "bi" if config["bidirectional"] else "uni"
        story = config["story_format"] + ("_nosep" if config["no_sep"] else "")
        parts = ["a2", config["cell"], direction, story, f"r{config['rnn_hidden']}"]
    if config["keep_punctuation"]:
        parts.append("punct")
    if config["layer_norm"]:
        parts.append("ln")
    if config["weight_decay"] != DEFAULT_WEIGHT_DECAY:
        parts.append(f"wd{config['weight_decay']:g}")
    if config["overfit_batches"]:
        parts.append("overfit")
    return "_".join(parts + [common])


def make_collate(config: dict, vocabulary: dict):
    if config["arch"] == "a1" or config["story_format"] == "separate":
        return collate_fields
    return StoryCollator(vocabulary[SEP], order=config["story_format"], use_sep=not config["no_sep"])


def logged_settings(config: dict) -> dict:
    """Only the settings that affect this architecture, without the run-control flags."""
    hidden = {"wandb", "tags", "overfit_batches"} | (ARCH_2_SETTINGS if config["arch"] == "a1" else ARCH_1_SETTINGS)
    return {key: value for key, value in config.items() if key not in hidden}


def make_logger(config: dict, name: str):
    """W&B is only touched with --wandb. Otherwise the curves go to a local CSV file."""
    if not config["wandb"]:
        return CSVLogger(save_dir=ARTIFACTS_DIR / "csv", name=name)

    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(
        project=WANDB_PROJECT,
        name=name,
        group=config["arch"],
        tags=config["tags"],
        config=logged_settings(config),
        save_dir=str(WANDB_DIR),
    )
    run = logger.experiment
    # A sweep starts the run before this point, so these are set on the run directly.
    run.name = name
    run.tags = tuple(config["tags"])
    run.config.update(logged_settings(config), allow_val_change=True)
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    # One commit at the end can't identify the code of each run, so each run keeps a copy.
    run.log_code("src")
    return logger


def main(argv=None):
    config = vars(parse_args(argv))
    L.seed_everything(config["seed"], workers=True)

    rows, vocabulary, matrix = load_processed()
    collate = make_collate(config, vocabulary)
    keep = config["keep_punctuation"]
    train_loader = make_loader(ArtDataset(rows["train"], vocabulary, keep), config["batch_size"], collate, shuffle=True)
    val_loader = make_loader(ArtDataset(rows["validation"], vocabulary, keep), config["batch_size"], collate)

    model = build_model(config, matrix)
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    module = ArtModule(model, config)

    name = run_name(config)
    logger = make_logger(config, name)
    best = BestValidation()
    # enable_version_counter=False: a repeated run replaces best.ckpt instead of adding best-v1.ckpt.
    checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR / name,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        enable_version_counter=False,
    )
    stopping = EarlyStopping(monitor="val/loss", mode="min", patience=config["patience"])

    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        logger=logger,
        callbacks=[checkpoint, stopping, best],
        gradient_clip_val=config["grad_clip"],
        overfit_batches=config["overfit_batches"],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(module, train_loader, val_loader)
    print(f"{name}: best val loss {best.loss:.4f}, val acc {best.accuracy:.4f}, epoch {best.epoch}")


if __name__ == "__main__":
    main()
