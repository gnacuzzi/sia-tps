import csv
from pathlib import Path

import numpy as np

from sia_tp4.campaign_synthesis import (
    RAW_FEATURES,
    CampaignDefinition,
    build_creator_rows,
    build_profile_rows,
    build_region_summaries,
    build_shortlist_comparison,
    build_strategy_summaries,
    load_design,
    load_original_rows,
    write_tables,
)
from sia_tp4.data import BASE_FEATURES, CreatorDataset, load_creator_dataset


ROOT = Path(__file__).resolve().parents[1]


def _campaigns():
    return (
        CampaignDefinition("futbol", "Fútbol", "football", "topic_football_pct"),
        CampaignDefinition("belleza", "Belleza", "makeup", "topic_makeup_pct"),
        CampaignDefinition("perros", "Perros", "dogs", "topic_dogs_pct"),
        CampaignDefinition("arte", "Arte", "art", "topic_art_pct"),
    )


def _dataset():
    creator_ids = ("A", "B", "C", "D")
    values = np.arange(4 * len(BASE_FEATURES), dtype=float).reshape(
        4, len(BASE_FEATURES)
    )
    return CreatorDataset(
        creator_ids=creator_ids,
        names=("Alfa", "Beta", "Gamma", "Delta"),
        feature_names=BASE_FEATURES,
        values=values,
        followers=np.asarray([100.0, 200.0, 300.0, 400.0]),
    )


def _original_rows(dataset):
    rows = {}
    for index, creator_id in enumerate(dataset.creator_ids):
        row = {feature: float(index + 1) for feature in RAW_FEATURES}
        row["followers"] = float((index + 1) * 100)
        row["average_reach"] = float((index + 1) * 10)
        for topic_index, topic in enumerate(
            ("football", "makeup", "dogs", "art")
        ):
            row[f"topic_{topic}_pct"] = (
                60.0 if index == topic_index else 10.0
            )
        rows[creator_id] = row
    return rows


def _stability_row(creator_id, region, category, count):
    regions = region.split(";")
    return {
        "creator_id": creator_id,
        "modal_region": region,
        "modal_region_tie": len(regions) > 1,
        **{
            f"region_{topic}_count": count if topic in regions else 0
            for topic in ("football", "makeup", "dogs", "art")
        },
        "region_stability_class": category,
        "region_stability_count": count,
        "region_stability_pct": count * 10.0,
        "transition_count": 10 - count,
        "transition_pct": (10 - count) * 10.0,
    }


def test_load_design_references_the_existing_reproducible_protocols():
    design = load_design(ROOT / "configs/campaigns/synthesis.json")

    assert [campaign.region for campaign in design.campaigns] == [
        "football",
        "makeup",
        "dogs",
        "art",
    ]
    assert design.primary_stability_classes == ("estable", "fronterizo")
    assert design.baseline.feature == "followers"
    assert design.baseline.scope == "global"
    assert design.baseline.order == "descending"
    assert design.kohonen_stability_config.is_file()
    assert design.oja_config.is_file()


def test_load_original_rows_preserves_source_values():
    full_dataset = load_creator_dataset(ROOT / "datos/creadores.csv")
    rows = load_original_rows(ROOT / "datos/creadores.csv", full_dataset)

    assert rows["INF001"]["followers"] == 775589.0
    assert rows["INF001"]["estimated_argentina_reach"] == 204416.0
    assert rows["INF002"]["topic_makeup_pct"] == 70.43


def test_creator_rows_use_full_regions_and_keep_instability_visible():
    dataset = _dataset()
    stability_rows = [
        _stability_row(creator_id, region, category, count)
        for creator_id, region, category, count in (
            ("A", "football", "estable", 10),
            ("B", "makeup", "fronterizo", 7),
            ("C", "dogs", "inestable", 5),
            ("D", "art;dogs", "inestable", 5),
        )
    ]

    rows = build_creator_rows(
        dataset,
        _original_rows(dataset),
        stability_rows,
        _campaigns(),
        ("estable", "fronterizo"),
        np.asarray([[1.0, 2.0], [2.0, 1.0], [-1.0, 1.0], [0.0, -2.0]]),
        np.asarray([1.1, 2.1, -0.9, 0.1]),
    )

    assert len(rows) == 5
    assert {row["campaign"] for row in rows if row["creator_id"] == "D"} == {
        "perros",
        "arte",
    }
    assert next(row for row in rows if row["creator_id"] == "A")[
        "primary_candidate"
    ] is True
    assert next(row for row in rows if row["creator_id"] == "C")[
        "primary_candidate"
    ] is False
    assert rows[0]["followers"] == 100.0
    assert all("rank" not in key and "score_compuesto" not in key for key in rows[0])


def test_profiles_summaries_and_tables_are_aggregated(tmp_path):
    dataset = _dataset()
    stability_rows = [
        _stability_row(creator_id, region, category, count)
        for creator_id, region, category, count in zip(
            dataset.creator_ids,
            ("football", "makeup", "dogs", "art"),
            ("estable", "fronterizo", "fronterizo", "estable"),
            (10, 7, 6, 9),
        )
    ]
    creator_rows = build_creator_rows(
        dataset,
        _original_rows(dataset),
        stability_rows,
        _campaigns(),
        ("estable", "fronterizo"),
        np.asarray([[1.0, 2.0], [2.0, 1.0], [-1.0, 1.0], [0.0, -2.0]]),
        np.asarray([1.1, 2.1, -0.9, 0.1]),
    )
    standardized = np.arange(4 * len(BASE_FEATURES), dtype=float).reshape(
        4, len(BASE_FEATURES)
    )
    profiles = build_profile_rows(dataset, creator_rows, _campaigns(), standardized)
    summaries = build_region_summaries(creator_rows, profiles, _campaigns())
    comparison = build_shortlist_comparison(
        dataset,
        _original_rows(dataset),
        stability_rows,
        creator_rows,
        _campaigns(),
        np.asarray([[1.0, 2.0], [2.0, 1.0], [-1.0, 1.0], [0.0, -2.0]]),
        np.asarray([1.1, 2.1, -0.9, 0.1]),
    )
    strategy_summaries = build_strategy_summaries(comparison, _campaigns())

    write_tables(
        tmp_path,
        creator_rows,
        profiles,
        summaries,
        comparison,
        strategy_summaries,
    )

    generated = tmp_path / "generated"
    assert {path.name for path in generated.iterdir()} == {
        "creadores_por_campana.csv",
        "perfiles_estandarizados.csv",
        "resumen_regiones.csv",
        "shortlists_comparadas.csv",
        "comparacion_followers.csv",
    }
    assert len(profiles) == 4 * len(BASE_FEATURES)
    assert summaries[2]["fronterizo_count"] == 1
    with (generated / "creadores_por_campana.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 4
    assert exported[0]["followers"] == "100.0"


def test_shortlist_comparison_uses_the_same_k_and_global_followers_baseline():
    dataset = _dataset()
    original = _original_rows(dataset)
    stability_rows = [
        _stability_row(creator_id, region, category, count)
        for creator_id, region, category, count in zip(
            dataset.creator_ids,
            ("football", "makeup", "dogs", "art"),
            ("estable", "fronterizo", "estable", "estable"),
            (10, 7, 9, 10),
        )
    ]
    scores = np.asarray([[1.0, 2.0], [2.0, 1.0], [-1.0, 1.0], [0.0, -2.0]])
    indices = np.asarray([1.1, 2.1, -0.9, 0.1])
    creator_rows = build_creator_rows(
        dataset,
        original,
        stability_rows,
        _campaigns(),
        ("estable", "fronterizo"),
        scores,
        indices,
    )

    rows = build_shortlist_comparison(
        dataset,
        original,
        stability_rows,
        creator_rows,
        _campaigns(),
        scores,
        indices,
    )
    summaries = build_strategy_summaries(rows, _campaigns())

    for campaign in _campaigns():
        campaign_rows = [row for row in rows if row["campaign"] == campaign.name]
        regional = [
            row for row in campaign_rows if row["strategy"] == "regional_kohonen"
        ]
        baseline = [
            row for row in campaign_rows if row["strategy"] == "baseline_followers"
        ]
        assert len(regional) == len(baseline) == 1
        assert baseline[0]["creator_id"] == "D"
        assert baseline[0]["global_followers_position"] == 1
        assert baseline[0]["topic_feature"] == campaign.topic_feature
    art_regional = next(
        row
        for row in summaries
        if row["campaign"] == "arte" and row["strategy"] == "regional_kohonen"
    )
    football_regional = next(
        row
        for row in summaries
        if row["campaign"] == "futbol" and row["strategy"] == "regional_kohonen"
    )
    assert art_regional["overlap_count"] == 1
    assert football_regional["overlap_count"] == 0
    assert art_regional["total_price_usd"] == original["D"]["price_usd"]
    assert all("sales" not in key and "adoptions" not in key for key in art_regional)
