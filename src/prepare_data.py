"""Tokenizes the dataset and builds the embedding table. Run once: python -m src.prepare_data"""

import json

import datasets
import nltk
import torch
from nltk.tokenize import word_tokenize

from src.data import FIELDS, is_punctuation
from src.embeddings import build_embedding_table
from src.paths import EMBEDDING_FILE, GLOVE_ZIP, PROCESSED_DIR, TOKENS_FILE


def main():
    nltk.download("punkt_tab", quiet=True)
    dataset = datasets.load_dataset("amarfurt/art-nlp-hs26")

    rows = {
        split: [{**{field: word_tokenize(row[field]) for field in FIELDS}, "label": row["label"]} for row in dataset[split]]
        for split in dataset
    }
    # The table only has to hold words that occur in the data. Each word keeps its
    # pretrained vector, so this gives the same result as the full GloVe table.
    words = {token for split in rows.values() for row in split for field in FIELDS for token in row[field]}
    vocabulary, matrix = build_embedding_table(words, GLOVE_ZIP)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(TOKENS_FILE, "w", encoding="utf-8") as file:
        json.dump(rows, file)
    torch.save({"vocabulary": vocabulary, "matrix": matrix}, EMBEDDING_FILE)

    shortest = min(
        sum(not is_punctuation(token) for token in row[field])
        for split in rows.values() for row in split for field in FIELDS
    )
    print(f"Unique tokens: {len(words)} | table: {matrix.shape} | {matrix.numel() * 4 / 1e6:.0f} MB")
    print(f"Shortest text without punctuation: {shortest} words")


if __name__ == "__main__":
    main()
