"""Locations of large files that are kept out of the repository."""

import os
from pathlib import Path

# Set ART_ARTIFACTS_DIR to keep large files elsewhere, e.g. outside a synced folder.
ARTIFACTS_DIR = Path(os.environ.get("ART_ARTIFACTS_DIR", "artifacts"))
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
WANDB_DIR = ARTIFACTS_DIR / "wandb"

GLOVE_ZIP = EMBEDDINGS_DIR / "glove.840B.300d.zip"
TOKENS_FILE = PROCESSED_DIR / "tokens.json"
EMBEDDING_FILE = PROCESSED_DIR / "embedding.pt"
