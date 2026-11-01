"""The two architectures. Each maps a batch to logits of shape (B, 2)."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.embeddings import DIM, PAD_ID, frozen_embedding


def pool(vectors: torch.Tensor, ids: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Combines word vectors (B, L, E) into one vector (B, E), ignoring padding."""
    mask = (ids != PAD_ID).unsqueeze(-1)
    total = (vectors * mask).sum(dim=1)
    if mode == "sum":
        return total
    return total / mask.sum(dim=1)


def classifier(input_size: int, hidden: int, dropout: float, layer_norm: bool, output_size: int = 1) -> nn.Sequential:
    """Two linear layers with ReLU. No dropout after the output."""
    layers = [nn.Linear(input_size, hidden)]
    if layer_norm:
        layers.append(nn.LayerNorm(hidden))
    layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, output_size)]
    return nn.Sequential(*layers)


class EmbeddingClassifier(nn.Module):
    """Architecture 1: frozen embeddings, pooling per text, two-layer classifier.

    Input formats:
    - shared: score [O1; H; O2] once per hypothesis with the same classifier.
    - interaction: as shared, plus |H - O| and H * O for both observations.
    - h_only: score H alone, without the observations.
    - joint: one input [O1; O2; H1; H2] that gives both logits at once.
    """

    INPUT_SIZES = {"shared": 3 * DIM, "interaction": 7 * DIM, "h_only": DIM, "joint": 4 * DIM}

    def __init__(self, matrix, hidden=256, dropout=0.2, layer_norm=False, input_format="shared", pooling="mean"):
        super().__init__()
        self.embedding = frozen_embedding(matrix)
        self.input_format = input_format
        self.pooling = pooling
        output_size = 2 if input_format == "joint" else 1
        self.classifier = classifier(self.INPUT_SIZES[input_format], hidden, dropout, layer_norm, output_size)

    def _text(self, batch: dict, field: str) -> torch.Tensor:
        return pool(self.embedding(batch[field]), batch[field], self.pooling)

    def _features(self, o1: torch.Tensor, h: torch.Tensor, o2: torch.Tensor) -> torch.Tensor:
        if self.input_format == "h_only":
            return h
        if self.input_format == "interaction":
            return torch.cat([o1, h, o2, (h - o1).abs(), (h - o2).abs(), h * o1, h * o2], dim=-1)
        return torch.cat([o1, h, o2], dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        o1 = self._text(batch, "observation_1")
        o2 = self._text(batch, "observation_2")
        h1 = self._text(batch, "hypothesis_1")
        h2 = self._text(batch, "hypothesis_2")
        if self.input_format == "joint":
            return self.classifier(torch.cat([o1, o2, h1, h2], dim=-1))
        # (B, 2, features) -> (B, 2, 1) -> (B, 2): the same classifier scores both.
        features = torch.stack([self._features(o1, h1, o2), self._features(o1, h2, o2)], dim=1)
        return self.classifier(features).squeeze(-1)


class RNNClassifier(nn.Module):
    """Architecture 2: frozen embeddings, two-layer LSTM or GRU, two-layer classifier.

    Story formats:
    - o1_h_o2 / o1_o2_h: one sequence per hypothesis, so the batch holds 2B sequences.
    - h_only: the hypothesis alone, as a control without the observations.
    - separate: O1, H and O2 are encoded on their own and the final states concatenated.
    The same dropout applies between the RNN layers and in the classifier.
    """

    def __init__(
        self, matrix, rnn_hidden=256, hidden=256, dropout=0.2, layer_norm=False, cell="gru", bidirectional=False, story_format="o1_h_o2"
    ):
        super().__init__()
        self.embedding = frozen_embedding(matrix)
        self.bidirectional = bidirectional
        self.story_format = story_format
        rnn_class = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_class(DIM, rnn_hidden, num_layers=2, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        state_size = rnn_hidden * (2 if bidirectional else 1)
        input_size = state_size * (3 if story_format == "separate" else 1)
        self.classifier = classifier(input_size, hidden, dropout, layer_norm)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Final hidden state of the top layer, taken at each sequence's true last word."""
        # Packing stops the RNN at the true length, so it never reads padding.
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.rnn(packed)
        final = state[0] if isinstance(self.rnn, nn.LSTM) else state  # An LSTM also returns its cell state.
        if self.bidirectional:
            return torch.cat([final[-2], final[-1]], dim=1)
        return final[-1]

    def forward(self, batch: dict) -> torch.Tensor:
        if self.story_format == "separate":
            encoded = {
                field: self.encode(batch[field], batch[f"{field}_length"])
                for field in ("observation_1", "observation_2", "hypothesis_1", "hypothesis_2")
            }
            scores = [
                self.classifier(torch.cat([encoded["observation_1"], encoded[h], encoded["observation_2"]], dim=1))
                for h in ("hypothesis_1", "hypothesis_2")
            ]
            return torch.cat(scores, dim=1)
        # Rows alternate H1, H2, so 2B scores reshape to one (H1, H2) pair per example.
        return self.classifier(self.encode(batch["story"], batch["story_length"])).view(-1, 2)


def build_model(config: dict, matrix: torch.Tensor) -> nn.Module:
    if config["arch"] == "a1":
        return EmbeddingClassifier(
            matrix,
            hidden=config["hidden"],
            dropout=config["dropout"],
            layer_norm=config["layer_norm"],
            input_format=config["input_format"],
            pooling=config["pooling"],
        )
    return RNNClassifier(
        matrix,
        rnn_hidden=config["rnn_hidden"],
        hidden=config["hidden"],
        dropout=config["dropout"],
        layer_norm=config["layer_norm"],
        cell=config["cell"],
        bidirectional=config["bidirectional"],
        story_format=config["story_format"],
    )
