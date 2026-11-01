"""Tokenization, datasets and batching for the ART task."""

import json
import string

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.embeddings import PAD_ID, UNK
from src.paths import EMBEDDING_FILE, TOKENS_FILE

FIELDS = ("observation_1", "observation_2", "hypothesis_1", "hypothesis_2")
PUNCTUATION = set(string.punctuation + "’‘“”—–…")


def is_punctuation(token: str) -> bool:
    """True only for tokens made entirely of punctuation, so `n't` and `'s` stay."""
    return all(character in PUNCTUATION for character in token)


def encode(tokens: list[str], vocabulary: dict, keep_punctuation: bool = False) -> list[int]:
    unk = vocabulary[UNK]
    return [vocabulary.get(t, unk) for t in tokens if keep_punctuation or not is_punctuation(t)]


def has_words(row: dict) -> bool:
    """False for rows with a text of only punctuation, such as a hypothesis `{ } .`."""
    return all(any(not is_punctuation(token) for token in row[field]) for field in FIELDS)


class ArtDataset(Dataset):
    """Word ids per text field, with the label mapped from 1/2 to 0/1.

    Rows with a text of no words are dropped: they can't be answered and a length
    of 0 can't be averaged or passed to an RNN. This affects 6 training rows.
    """

    def __init__(self, rows: list[dict], vocabulary: dict, keep_punctuation: bool = False):
        self.examples = [
            {
                **{field: encode(row[field], vocabulary, keep_punctuation) for field in FIELDS},
                "label": row["label"] - 1,
            }
            for row in rows
            if has_words(row)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _pad(sequences: list[list[int]]):
    tensors = [torch.tensor(s, dtype=torch.long) for s in sequences]
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    return pad_sequence(tensors, batch_first=True, padding_value=PAD_ID), lengths


def collate_fields(batch: list[dict]) -> dict:
    """Keeps the four texts separate: each field is (B, L) with lengths (B)."""
    output = {"label": torch.tensor([example["label"] for example in batch], dtype=torch.long)}
    for field in FIELDS:
        output[field], output[f"{field}_length"] = _pad([example[field] for example in batch])
    return output


class StoryCollator:
    """Builds one sequence per hypothesis, so the batch holds 2B sequences.

    Rows alternate H1, H2, H1, H2, ... and the model reshapes its scores to (B, 2).
    """

    def __init__(self, sep_id: int, order: str = "o1_h_o2", use_sep: bool = True):
        if order not in ("o1_h_o2", "o1_o2_h", "h_only"):
            raise ValueError(f"Unknown order: {order}")
        self.separator = [sep_id] if use_sep else []
        self.order = order

    def _story(self, example: dict, hypothesis: str) -> list[int]:
        if self.order == "h_only":
            return example[hypothesis]
        o1, o2, h = example["observation_1"], example["observation_2"], example[hypothesis]
        parts = (o1, h, o2) if self.order == "o1_h_o2" else (o1, o2, h)
        return parts[0] + self.separator + parts[1] + self.separator + parts[2]

    def __call__(self, batch: list[dict]) -> dict:
        stories = [self._story(e, h) for e in batch for h in ("hypothesis_1", "hypothesis_2")]
        story, length = _pad(stories)
        label = torch.tensor([example["label"] for example in batch], dtype=torch.long)
        return {"story": story, "story_length": length, "label": label}


def load_processed():
    """Loads the output of `src.prepare_data`: rows per split, vocabulary, matrix."""
    with open(TOKENS_FILE, encoding="utf-8") as file:
        rows = json.load(file)
    table = torch.load(EMBEDDING_FILE, weights_only=True)
    return rows, table["vocabulary"], table["matrix"]


def make_loader(dataset: ArtDataset, batch_size: int, collate_fn, shuffle: bool = False) -> DataLoader:
    # The data is already in memory, so worker processes would only add start-up time.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0)
