# Abductive Reasoning with Word Embeddings and RNNs

Project 1 for the NLP course HS26 HSLU.

**Task:** given two observations O1 and O2, choose the more plausible of two hypotheses H1 and H2 (ART dataset, Bhagavatula et al., 2020).

**Experiment tracking:** [W&B saved view](https://forge.coreweave.com/wandb/tqntaulant-hochschule-luzern/nlp-project1/workspace?nw=rn5arnml0vb)


## Results

Final models, one seed (42), each evaluated once on the test set.

| Model | Validation accuracy | Validation loss | Test accuracy | Test loss |
|---|---|---|---|---|
| Architecture 1: GloVe + classifier | 73.3% | 0.5396 | **52.7%** | 0.9812 |
| Architecture 2: GloVe + GRU + classifier | 71.6% | 0.5430 | **54.8%** | 0.9998 |

- The test accuracy margin is about ±2.5 points (95%, 1,532 examples).
- Hypotheses only, without O1 and O2, on validation: 69.0% (Architecture 1), 68.3% (Architecture 2).


## Data

- Dataset: [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26)
- Splits: train 16,801, validation 1,000, test 1,532.
- 6 training rows are dropped because a hypothesis contains no words (`{ } .` or `.`).
## Preprocessing

- Tokenizer: NLTK `word_tokenize`.
- Punctuation: tokens made up only of punctuation are removed after tokenizing.
- No lowercasing, stemming, lemmatizing, stopword removal, rare word removal, text cleaning or truncation.
- Embeddings: GloVe 840B, 300 dimensions, frozen. They cover 99.93% of training tokens.
- Special tokens: `<pad>` (zero vector), `<unk>` (mean of all GloVe vectors), `<sep>` (fixed random vector).
- Padding: to the longest sequence in each batch.
- Labels: 1/2 mapped to 0/1.

## Models

Both models score each hypothesis with shared weights and output logits of shape `(batch, 2)`.

| | Architecture 1 | Architecture 2 |
|---|---|---|
| **Input per hypothesis** | Mean vectors of O1, H and O2, concatenated (900 dimensions) | Sequence `O1 <sep> H <sep> O2` |
| **RNN** | None | 2-layer bidirectional GRU, hidden size 256, dropout 0.2 between layers. Output: final hidden states of both directions, concatenated (512 dimensions) |
| **Classifier** | `Linear(900, 1024)`, ReLU, Dropout 0.5, `Linear(1024, 1)` | `Linear(512, 256)`, ReLU, Dropout 0.2, `Linear(256, 1)` |
| **Trainable parameters** | 923,649 | 2,171,393 |

## Training

- Framework: PyTorch models with PyTorch Lightning.
- Loss: cross-entropy over the two scores.
- Optimizer: AdamW, learning rate 1e-3, weight decay 0.01.
- Batch size 1,024, at most 200 epochs.
- Early stopping on validation loss with patience 5, keeping the best checkpoint.
- Seed 42.

## Experiments

44 runs, tracked in W&B, tuned on the validation set.

| Tag | Experiment | Values |
|---|---|---|
| A1-2 | Learning rate | 1e-2, 1e-3, 1e-4 |
| A1-3 | Input format and pooling | Shared, interaction features, joint, hypotheses only, sum |
| A1-4 | Capacity and regularization | Hidden 64, 256, 1024, 2048; dropout 0, 0.2, 0.5, 0.7 |
| A1-4b | Weight decay, normalization | Weight decay 0, LayerNorm |
| A1-5 | Punctuation | Kept |
| A2-1 | Learning rate | 1e-1, 1e-2, 10^-2.5, 1e-3, 1e-4 |
| A2-2 | Cell and direction | GRU and LSTM, unidirectional and bidirectional |
| A2-3 | Story format | No `<sep>`, `O1 O2 H`, separate encodings |
| A2-4 | Capacity and regularization | RNN hidden 128, 256, 512; dropout 0, 0.2, 0.5 |
| A2-H | Hypotheses only | No observations |

## Repository

```
configs/                     W&B sweep configurations
notebooks/                   data exploration
src/prepare_data.py          tokenizes the data, builds the embedding table
src/embeddings.py            GloVe table and frozen embedding layer
src/data.py                  dataset and batching
src/models.py                the two architectures
src/lit_module.py            loss, metrics, optimizer
src/train.py                 trains one model
src/sweep.py                 runs a W&B sweep
src/evaluate.py              evaluates a checkpoint
```

## Reproduce

Python 3.11 and a CUDA GPU.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Download [`glove.840B.300d.zip`](https://nlp.stanford.edu/data/glove.840B.300d.zip) to `artifacts/embeddings/`. Large files go to `artifacts/`; set `ART_ARTIFACTS_DIR` to use another folder.

```bash
# Tokenize the data and build the embedding table
python -m src.prepare_data

# Train the final models (add --wandb to log to W&B)
python -m src.train --arch a1 --hidden 1024 --dropout 0.5
python -m src.train --arch a2 --bidirectional

# Evaluate on validation
python -m src.evaluate --checkpoint artifacts/checkpoints/a1_shared_mean_h1024_d0.5_lr0.001/best.ckpt --split validation
python -m src.evaluate --checkpoint artifacts/checkpoints/a2_gru_bi_o1_h_o2_r256_h256_d0.2_lr0.001/best.ckpt --split validation

# Evaluate on test: needs --final and refuses to run twice for a checkpoint
python -m src.evaluate --checkpoint artifacts/checkpoints/a1_shared_mean_h1024_d0.5_lr0.001/best.ckpt --final
python -m src.evaluate --checkpoint artifacts/checkpoints/a2_gru_bi_o1_h_o2_r256_h256_d0.2_lr0.001/best.ckpt --final

# Run a sweep
python -m src.sweep configs/a1_lr.yaml
```

Runs outside the sweeps use `src.train` flags, for example `--pooling sum`, `--weight-decay 0`, `--layer-norm`, `--keep-punctuation`, `--no-sep`, `--story-format h_only` and `--lr`.

## References

- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *Proceedings of EMNLP 2014*, 1532–1543.
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media. (NLTK)
