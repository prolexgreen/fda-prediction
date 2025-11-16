"""Per-checkpoint tokenizer registry for the three-stream model.

ChemBERTa and MoLFormer ship incompatible tokenizers; token IDs are never
shared across molecule streams. Every entry point imports this module after
`fda_predictor.utils.paths`, so downloads resolve to D:\\Models\\hub.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer

from fda_predictor.utils.paths import HUB_CACHE


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    name: str
    revision: str
    trust_remote_code: bool
    max_length: int


def specs_from_config(config: dict) -> dict[str, EncoderSpec]:
    enc = config["encoders"]

    def _make(key: str) -> EncoderSpec:
        entry = enc[key]
        # Local DAPT checkpoint wins over HF hub name/revision when present.
        dapt_path = entry.get("dapt_path")
        if dapt_path:
            p = Path(str(dapt_path))
            if (p / "config.json").exists():
                return EncoderSpec(
                    key=key,
                    name=str(p),
                    revision="",
                    trust_remote_code=bool(entry["trust_remote_code"]),
                    max_length=int(entry["max_length"]),
                )
        return EncoderSpec(
            key=key,
            name=entry["name"],
            revision=entry["revision"],
            trust_remote_code=bool(entry["trust_remote_code"]),
            max_length=int(entry["max_length"]),
        )

    return {
        "chemberta": _make("chemberta"),
        "molformer": _make("molformer"),
        "clinicalbert": _make("clinicalbert"),
    }


@lru_cache(maxsize=None)
def get_tokenizer(spec: EncoderSpec) -> AutoTokenizer:
    kwargs = dict(
        trust_remote_code=spec.trust_remote_code,
        cache_dir=str(HUB_CACHE),
    )
    if spec.revision:
        kwargs["revision"] = spec.revision
    return AutoTokenizer.from_pretrained(spec.name, **kwargs)


def encode_smiles_list(tokenizer, smiles_list: list[str], max_length: int) -> list[dict]:
    """Tokenize each molecule separately; returns one token dict per SMILES."""
    encoded = tokenizer(
        list(smiles_list),
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    return [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
    ]


def encode_text(tokenizer, text: str, max_length: int) -> dict:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
