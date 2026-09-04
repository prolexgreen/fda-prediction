"""Tests for v15 chunked long-doc text encoding (options: 2048-token context
via 4x512 chunks + title prepend)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fda_predictor.data.tokenizers import encode_text_chunks  # noqa: E402
from fda_predictor.models.encoders import ProtocolEncoder  # noqa: E402


class _StubTok:
    """Word-level tokenizer: each word -> distinct id >= 4; 1=PAD, 2=CLS, 3=SEP."""

    pad_token_id = 1
    cls_token_id = 2
    sep_token_id = 3

    def __init__(self):
        self._counts: dict[str, int] = {}

    def __call__(self, text, add_special_tokens=True, **_kwargs):
        ids = []
        for w in text.split():
            if w not in self._counts:
                self._counts[w] = 4 + len(self._counts)
            ids.append(self._counts[w])
        if add_special_tokens:
            ids = [2] + ids + [3]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _StubBackbone(nn.Module):
    """Returns last_hidden_state = input_ids as float embeddings, dim 8."""

    def forward(self, input_ids, attention_mask=None):
        h = torch.nn.functional.one_hot(input_ids.clamp(max=511), num_classes=8).float()
        return type("Out", (), {"last_hidden_state": h})


def test_chunks_cover_full_document():
    tok = _StubTok()
    words = [f"w{i}" for i in range(1200)]
    chunks = encode_text_chunks(tok, " ".join(words), max_length=512, max_chunks=4)
    assert len(chunks) == 4
    # No content token may be dropped: concatenated chunk ids (minus specials)
    # must equal the full word-id sequence.
    flat = [i for c in chunks for i in c["input_ids"] if i not in (2, 3)]
    assert flat == [4 + i for i in range(1200)]


def test_short_text_zeroes_later_chunks():
    tok = _StubTok()
    chunks = encode_text_chunks(tok, "alpha beta", max_length=512, max_chunks=4)
    assert sum(chunks[0]["attention_mask"]) > 0
    for c in chunks[1:]:
        assert sum(c["attention_mask"]) == 0


def test_empty_text_produces_valid_stub():
    tok = _StubTok()
    chunks = encode_text_chunks(tok, "", max_length=512, max_chunks=4)
    assert len(chunks) == 4
    assert len(chunks[0]["input_ids"]) == 2


def test_protocol_encoder_chunked_forward_shape_and_weights():
    enc = ProtocolEncoder(_StubBackbone(), hidden_size=8)
    b, k, l = 2, 4, 10
    ids = torch.randint(4, 512, (b, k, l))
    am = torch.ones(b, k, l, dtype=torch.long)
    am[:, 3, :] = 0  # last chunk empty
    out = enc({"input_ids": ids, "attention_mask": am})
    assert out.shape == (b, 8)
    # 2D legacy path still works
    out2 = enc({"input_ids": ids[:, 0, :], "attention_mask": am[:, 0, :]})
    assert out2.shape == (b, 8)
    # Chunk validity weighting: empty chunk contributes nothing. Compare against
    # manual mean of the first 3 chunks only.
    vecs = []
    for ci in range(3):
        r = enc({"input_ids": ids[:, ci, :], "attention_mask": am[:, ci, :]})
        vecs.append(r)
    ref = torch.stack(vecs).mean(0)
    assert torch.allclose(out, ref, atol=1e-5)
