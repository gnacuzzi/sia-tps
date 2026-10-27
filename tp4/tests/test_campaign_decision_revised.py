from pathlib import Path

from sia_tp4.campaign_decision_revised import (
    Campaign,
    DecisionDesign,
    build_threshold_sensitivity,
    campaign_rows,
    select_shortlists,
)


def _raw(creator_id, affinity, reach, nonfollowers, followers, visits, price):
    return {
        "creator_id": creator_id,
        "name": f"@{creator_id.lower()}",
        "topic_football_pct": float(affinity),
        "estimated_argentina_reach": float(reach),
        "non_follower_reach_pct": float(nonfollowers),
        "profile_visits_per_1000": float(visits),
        "price_usd": float(price),
        "followers": float(followers),
    }


def _stability(creator_id, region):
    return {
        "creator_id": creator_id,
        "modal_region": region,
        "region_stability_class": "estable",
        "region_stability_pct": 100.0,
    }


def _campaign():
    return Campaign(
        "futbol",
        "Fútbol",
        "Ropa de entrenamiento",
        "Aumentar ventas",
        "topic_football_pct",
    )


def test_affinity_is_the_gate_and_kohonen_region_does_not_exclude():
    raw = [
        _raw("A", 80, 1000, 50, 100, 10, 100),
        _raw("B", 70, 900, 50, 200, 10, 100),
        _raw("C", 60, 800, 50, 300, 10, 100),
        _raw("D", 49, 5000, 90, 1000, 10, 100),
    ]
    stability = [
        _stability("A", "art"),
        _stability("B", "football"),
        _stability("C", "football"),
        _stability("D", "football"),
    ]

    rows = campaign_rows(raw, stability, _campaign(), 50.0)

    assert [row["creator_id"] for row in rows] == ["A", "B", "C"]
    assert rows[0]["kohonen_region_matches_topic"] is False
    assert rows[0]["estimated_new_relevant_reach"] == 400.0


def test_comparison_has_equal_k_and_a_topic_controlled_followers_baseline():
    raw = [
        _raw("A", 80, 1000, 50, 100, 10, 100),
        _raw("B", 70, 900, 50, 200, 10, 100),
        _raw("C", 60, 800, 50, 300, 10, 100),
        _raw("D", 10, 5000, 90, 1000, 10, 100),
    ]
    stability = [_stability(row["creator_id"], "football") for row in raw]
    campaign = _campaign()
    design = DecisionDesign(
        "test",
        50.0,
        3,
        Path("stability.json"),
        (campaign,),
    )
    eligible = campaign_rows(raw, stability, campaign, 50.0)

    rows = select_shortlists(raw, eligible, design)

    by_strategy = {
        strategy: [row for row in rows if row["strategy"] == strategy]
        for strategy in ("propuesta", "followers_elegibles", "followers_global")
    }
    assert all(len(members) == 3 for members in by_strategy.values())
    assert [row["creator_id"] for row in by_strategy["followers_elegibles"]] == [
        "C",
        "B",
        "A",
    ]
    assert by_strategy["followers_global"][0]["creator_id"] == "D"
    assert by_strategy["followers_global"][0]["eligible_by_affinity"] is False


def test_threshold_sensitivity_records_each_cut_and_rank():
    raw = [
        _raw("A", 80, 1000, 50, 100, 10, 100),
        _raw("B", 70, 900, 50, 200, 10, 100),
        _raw("C", 60, 800, 50, 300, 10, 100),
        _raw("D", 45, 700, 50, 400, 10, 100),
    ]
    stability = [_stability(row["creator_id"], "football") for row in raw]
    design = DecisionDesign(
        "test", 50.0, 3, Path("stability.json"), (_campaign(),)
    )

    rows = build_threshold_sensitivity(
        raw, stability, design, thresholds=(40.0, 50.0, 60.0)
    )

    assert len(rows) == 9
    assert {row["affinity_threshold_pct"] for row in rows} == {
        40.0,
        50.0,
        60.0,
    }
    assert [
        row["creator_id"]
        for row in rows
        if row["affinity_threshold_pct"] == 60.0
    ] == ["A", "B", "C"]
