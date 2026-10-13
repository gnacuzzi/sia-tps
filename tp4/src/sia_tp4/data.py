"""Carga validada del catálogo de creadores y selección base de variables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


BASE_FEATURES: Tuple[str, ...] = (
    "average_reach",
    "monthly_posts",
    "argentina_audience_pct",
    "non_follower_reach_pct",
    "total_engagement_per_1000",
    "profile_visits_per_1000",
    "price_usd",
    "sponsored_posts_pct",
    "topic_football_pct",
    "topic_makeup_pct",
    "topic_dogs_pct",
    "content_tutorial_pct",
    "content_review_pct",
    "content_entertainment_pct",
    "average_watch_time_seconds",
)

ENGAGEMENT_COMPONENTS: Tuple[str, ...] = (
    "likes_per_1000",
    "comments_per_1000",
    "saves_per_1000",
    "shares_per_1000",
)

ENGAGEMENT_COMPONENT_FEATURES: Tuple[str, ...] = tuple(
    feature
    for base_feature in BASE_FEATURES
    for feature in (
        ENGAGEMENT_COMPONENTS
        if base_feature == "total_engagement_per_1000"
        else (base_feature,)
    )
)


@dataclass(frozen=True)
class CreatorDataset:
    """Matriz de entradas junto con metadatos que no participan de PCA."""

    creator_ids: Tuple[str, ...]
    names: Tuple[str, ...]
    feature_names: Tuple[str, ...]
    values: np.ndarray
    followers: np.ndarray


def _required_columns(feature_names: Sequence[str]) -> set[str]:
    return {"creator_id", "name", "followers", *feature_names}


def load_creator_dataset(
    path: Path | str, feature_names: Sequence[str] = BASE_FEATURES
) -> CreatorDataset:
    """Leer las entradas sin imputar, descartar ni transformar observaciones."""

    path = Path(path)
    features = tuple(feature_names)
    if not features or len(set(features)) != len(features):
        raise ValueError("las variables deben ser únicas y no vacías")

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("el CSV no tiene encabezado")
        missing = sorted(_required_columns(features) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"faltan columnas requeridas: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError("el CSV no contiene creadores")

    creator_ids = tuple(row["creator_id"].strip() for row in rows)
    names = tuple(row["name"].strip() for row in rows)
    if any(not value for value in creator_ids) or any(not value for value in names):
        raise ValueError("creator_id y name no pueden estar vacíos")
    if len(set(creator_ids)) != len(creator_ids):
        raise ValueError("creator_id debe ser único")

    try:
        values = np.asarray(
            [[float(row[name]) for name in features] for row in rows],
            dtype=np.float64,
        )
        followers = np.asarray(
            [float(row["followers"]) for row in rows], dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise ValueError("las variables seleccionadas deben contener números") from error

    if not np.isfinite(values).all() or not np.isfinite(followers).all():
        raise ValueError("el dataset contiene valores no finitos")
    if np.any(followers < 0):
        raise ValueError("followers no puede contener valores negativos")
    if values.shape[0] < 2:
        raise ValueError("PCA necesita al menos dos creadores")
    deviations = np.std(values, axis=0, ddof=1)
    if np.any(deviations == 0):
        constant = [
            name
            for name, deviation in zip(features, deviations)
            if deviation == 0
        ]
        raise ValueError(f"hay variables constantes: {constant}")

    return CreatorDataset(
        creator_ids=creator_ids,
        names=names,
        feature_names=features,
        values=values,
        followers=followers,
    )
