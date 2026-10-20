"""Selección transparente de perfiles complementarios por campaña."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ROLES = ("alcance", "equilibrio", "especializacion_eficiencia")
ROLE_LABELS = {
    "alcance": "Alcance",
    "equilibrio": "Equilibrio",
    "especializacion_eficiencia": "Especialización y eficiencia",
}
CONTENT_FEATURES = (
    "content_tutorial_pct",
    "content_review_pct",
    "content_entertainment_pct",
    "content_showcase_pct",
)
ALLOWED_RESPONSE_FEATURES = {
    "total_engagement_per_1000",
    "profile_visits_per_1000",
}


@dataclass(frozen=True)
class CandidateProtocol:
    """Roles y variable de respuesta descriptiva de cada campaña."""

    candidate_count: int
    roles: tuple[str, ...]
    response_feature_by_campaign: Mapping[str, str]


def load_candidate_protocol(
    raw: object, campaign_names: Sequence[str]
) -> CandidateProtocol:
    """Validar que el protocolo versionado coincide con las reglas implementadas."""

    if not isinstance(raw, dict) or set(raw) != {
        "candidate_count",
        "roles",
        "response_feature_by_campaign",
    }:
        raise ValueError("candidate_protocol tiene campos inválidos")
    roles = raw["roles"]
    if raw["candidate_count"] != len(ROLES) or roles != list(ROLES):
        raise ValueError("candidate_protocol debe declarar los tres roles acordados")
    responses = raw["response_feature_by_campaign"]
    if (
        not isinstance(responses, dict)
        or set(responses) != set(campaign_names)
        or not set(responses.values()).issubset(ALLOWED_RESPONSE_FEATURES)
    ):
        raise ValueError("las variables de respuesta por campaña son inválidas")
    return CandidateProtocol(
        candidate_count=raw["candidate_count"],
        roles=tuple(roles),
        response_feature_by_campaign=dict(responses),
    )


def _first_by(rows: Sequence[dict], *keys) -> dict:
    if not rows:
        raise ValueError("el protocolo no produjo candidatos elegibles")
    return sorted(rows, key=lambda row: tuple(key(row) for key in keys))[0]


def _candidate_record(
    campaign,
    row: dict,
    role: str,
    selection_rule: str,
    topic_median: float,
    response_feature: str,
    response_median: float,
    price_median: float,
) -> dict:
    dominant_content = max(CONTENT_FEATURES, key=lambda feature: row[feature])
    return {
        "campaign": campaign.name,
        "campaign_label": campaign.label,
        "role": role,
        "role_label": ROLE_LABELS[role],
        "creator_id": row["creator_id"],
        "name": row["name"],
        "selection_rule": selection_rule,
        "region_stability_class": row["region_stability_class"],
        "region_stability_count": row["region_stability_count"],
        "region_stability_pct": row["region_stability_pct"],
        "frontier_warning": row["region_stability_class"] == "fronterizo",
        "topic_feature": campaign.topic_feature,
        "topic_affinity_pct": row[campaign.topic_feature],
        "campaign_median_topic_affinity_pct": topic_median,
        "response_feature": response_feature,
        "response_value": row[response_feature],
        "campaign_median_response_value": response_median,
        "campaign_median_price_usd": price_median,
        "dominant_content_feature": dominant_content,
        "dominant_content_pct": row[dominant_content],
        "followers": row["followers"],
        "estimated_argentina_reach": row["estimated_argentina_reach"],
        "average_reach": row["average_reach"],
        "argentina_audience_pct": row["argentina_audience_pct"],
        "non_follower_reach_pct": row["non_follower_reach_pct"],
        "total_engagement_per_1000": row["total_engagement_per_1000"],
        "profile_visits_per_1000": row["profile_visits_per_1000"],
        "price_usd": row["price_usd"],
        "sponsored_posts_pct": row["sponsored_posts_pct"],
        **{feature: row[feature] for feature in CONTENT_FEATURES},
        "average_watch_time_seconds": row["average_watch_time_seconds"],
        "pc1_score": row["pc1_score"],
        "pc2_score": row["pc2_score"],
        "oja_index": row["oja_index"],
    }


def select_final_candidates(
    creator_rows: Sequence[dict], campaigns: Sequence, protocol: CandidateProtocol
) -> list[dict]:
    """Elegir tres perfiles distintos sin combinar variables en un score."""

    selected_rows = []
    for campaign in campaigns:
        pool = [
            row
            for row in creator_rows
            if row["campaign"] == campaign.name and row["primary_candidate"]
        ]
        if len(pool) < protocol.candidate_count:
            raise ValueError(f"la campaña {campaign.name} no tiene suficientes candidatos")
        response_feature = protocol.response_feature_by_campaign[campaign.name]
        topic_median = float(np.median([row[campaign.topic_feature] for row in pool]))
        response_median = float(np.median([row[response_feature] for row in pool]))
        price_median = float(np.median([row["price_usd"] for row in pool]))

        reach_pool = [
            row for row in pool if row[campaign.topic_feature] >= topic_median
        ]
        reach = _first_by(
            reach_pool,
            lambda row: -row["estimated_argentina_reach"],
            lambda row: row["creator_id"],
        )
        balance_pool = [
            row
            for row in pool
            if row["creator_id"] != reach["creator_id"]
            and row[campaign.topic_feature] >= topic_median
            and row[response_feature] >= response_median
            and row["price_usd"] <= price_median
        ]
        balance = _first_by(
            balance_pool,
            lambda row: -row["estimated_argentina_reach"],
            lambda row: row["creator_id"],
        )
        used = {reach["creator_id"], balance["creator_id"]}
        efficiency_pool = [
            row
            for row in pool
            if row["creator_id"] not in used
            and row[campaign.topic_feature] >= topic_median
        ]
        efficiency = _first_by(
            efficiency_pool,
            lambda row: row["price_usd"],
            lambda row: -row[campaign.topic_feature],
            lambda row: row["creator_id"],
        )
        selections = (
            (
                "alcance",
                reach,
                "afinidad al menos mediana; máximo alcance argentino estimado",
            ),
            (
                "equilibrio",
                balance,
                "afinidad y respuesta al menos medianas, costo no mayor a la mediana; máximo alcance argentino restante",
            ),
            (
                "especializacion_eficiencia",
                efficiency,
                "afinidad al menos mediana; menor costo entre los candidatos restantes",
            ),
        )
        selected_rows.extend(
            _candidate_record(
                campaign,
                row,
                role,
                rule,
                topic_median,
                response_feature,
                response_median,
                price_median,
            )
            for role, row, rule in selections
        )
    return selected_rows
