"""Builds the frozen embedding table from the pretrained GloVe vectors."""

import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

PAD = "<pad>"
UNK = "<unk>"
SEP = "<sep>"
PAD_ID = 0
DIM = 300


def build_embedding_table(words, glove_zip: Path, seed: int = 42):
    """Returns the vocabulary and the embedding matrix for the given words.

    Rows: <pad> = zeros, <unk> = mean of all GloVe vectors, <sep> = fixed random
    vector with the same spread as GloVe, then every given word that has a vector.
    Words without a vector are left out and later map to <unk>.
    """
    wanted = set(words)
    found = {}
    total = np.zeros(DIM)
    total_squared = np.zeros(DIM)
    count = 0

    with zipfile.ZipFile(glove_zip) as archive:
        with archive.open(archive.namelist()[0]) as file:
            for line in io.TextIOWrapper(file, encoding="utf-8", errors="replace", newline="\n"):
                # Some GloVe tokens contain spaces, so split from the right.
                word, *numbers = line.rstrip("\n").rsplit(" ", DIM)
                vector = np.array(numbers, dtype=np.float64)
                total += vector
                total_squared += vector**2
                count += 1
                if word in wanted and word not in found:
                    found[word] = vector.astype(np.float32)

    mean = total / count
    std = np.sqrt((total_squared / count - mean**2).mean())
    sep = np.random.default_rng(seed).normal(0.0, std, DIM)

    vocabulary = {PAD: PAD_ID, UNK: 1, SEP: 2}
    rows = [np.zeros(DIM, dtype=np.float32), mean.astype(np.float32), sep.astype(np.float32)]
    for word in sorted(found):
        vocabulary[word] = len(rows)
        rows.append(found[word])
    return vocabulary, torch.from_numpy(np.stack(rows))


def frozen_embedding(matrix: torch.Tensor) -> nn.Embedding:
    """Embedding layer whose weights are never trained."""
    return nn.Embedding.from_pretrained(matrix, freeze=True, padding_idx=PAD_ID)
