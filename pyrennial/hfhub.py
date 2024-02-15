"""Pull metadata for models and datasets from huggingface hub."""
from __future__ import annotations

import dataclasses as dc

import huggingface_hub as hh
import pandas as pd
from huggingface_hub.hf_api import DatasetInfo, ModelInfo
from pandas import DataFrame

MODELCARD_FEATURES = {
    "tags": "card_tags",
    "datasets": "datasets",
    "license": "license",
    "metrics": "metrics",
}

DATACARD_FEATURES = {
    "language": "language",
    "license": "license",
    "annotations_creators": "annotations_creators",
    "language_creators": "language_creators",
    "multilinguality": "multilinguality",
    "size_categories": "size_categories",
    "source_datasets": "datasets",
    "task_categories": "task_categories",
    "task_ids": "task_ids",
    "paperswithcode_id": "paperswithcode_id",
    "pretty_name": "pretty_name",
}


def to_dict(obj: DatasetInfo | ModelInfo) -> dict:
    """Convert a HF Model or Dataset to dict."""
    if obj.card_data and not isinstance(obj.card_data, dict):
        obj.card_data = obj.card_data.to_dict()

    obj = dc.asdict(obj)
    obj.pop("siblings", None)
    return obj


def get_multivalued(card: dict | None, key: str) -> list:
    """Huggingface mixes scalars and lists in the same model card field."""
    if not card:
        return None

    val = card.get(key)
    if val and not isinstance(val, list):
        val = [val]

    return val


def models() -> DataFrame:
    """Create a DataFrame with all models."""
    models = (to_dict(model) for model in hh.list_models(full=True, cardData=True))
    df = pd.DataFrame(models)

    for key, name in MODELCARD_FEATURES.items():
        df[name] = df.card_data.apply(lambda x: x.get(key) if isinstance(x, dict) else None)

    df["language"] = df.card_data.apply(lambda card: get_multivalued(card, "language"))

    df = df.drop(columns="card_data")
    df = df.dropna(axis=1, how="all")
    return df


def datasets() -> DataFrame:
    """Create a DataFrame with all datasets."""
    dsets = (to_dict(ds) for ds in hh.list_datasets(full=True))
    df = pd.DataFrame(dsets)

    for key, name in DATACARD_FEATURES.items():
        df[name] = df.card_data.apply(lambda card: get_multivalued(card, key))

    df = df.drop(columns="card_data")
    df = df.dropna(axis=1, how="all")
    return df


def models_datasets() -> DataFrame:
    """One long dataframe with models and datasets."""
    ms = models()
    ds = datasets()
    return pd.concat([ms, ds], axis=0)
