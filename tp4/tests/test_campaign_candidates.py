from types import SimpleNamespace

from sia_tp4.campaign_candidates import (
    CandidateProtocol,
    ROLES,
    select_final_candidates,
)


def _row(
    creator_id,
    reach,
    affinity,
    response,
    price,
    stability="estable",
):
    return {
        "campaign": "futbol",
        "primary_candidate": True,
        "creator_id": creator_id,
        "name": f"@{creator_id.lower()}",
        "region_stability_class": stability,
        "region_stability_count": 10 if stability == "estable" else 7,
        "region_stability_pct": 100.0 if stability == "estable" else 70.0,
        "topic_football_pct": affinity,
        "total_engagement_per_1000": response,
        "profile_visits_per_1000": 10.0,
        "price_usd": price,
        "estimated_argentina_reach": reach,
        "average_reach": reach * 2,
        "argentina_audience_pct": 50.0,
        "non_follower_reach_pct": 40.0,
        "sponsored_posts_pct": 20.0,
        "content_tutorial_pct": 10.0,
        "content_review_pct": 20.0,
        "content_entertainment_pct": 60.0,
        "content_showcase_pct": 10.0,
        "average_watch_time_seconds": 15.0,
        "followers": reach * 4,
        "pc1_score": 1.0,
        "pc2_score": 0.5,
        "oja_index": 1.0,
    }


def test_selects_three_distinct_roles_without_a_composite_score():
    campaign = SimpleNamespace(
        name="futbol",
        label="Fútbol",
        topic_feature="topic_football_pct",
    )
    protocol = CandidateProtocol(
        candidate_count=3,
        roles=ROLES,
        response_feature_by_campaign={
            "futbol": "total_engagement_per_1000"
        },
    )
    rows = [
        _row("A", reach=1000, affinity=80, response=90, price=400),
        _row("B", reach=500, affinity=80, response=80, price=100),
        _row("C", reach=300, affinity=70, response=60, price=50),
        _row("D", reach=200, affinity=60, response=70, price=300),
        _row("E", reach=100, affinity=50, response=50, price=200),
    ]

    selected = select_final_candidates(rows, (campaign,), protocol)

    assert [(row["role"], row["creator_id"]) for row in selected] == [
        ("alcance", "A"),
        ("equilibrio", "B"),
        ("especializacion_eficiencia", "C"),
    ]
    assert selected[1]["campaign_median_topic_affinity_pct"] == 70.0
    assert selected[1]["campaign_median_response_value"] == 70.0
    assert selected[2]["price_usd"] == 50
    assert all(
        "composite_score" not in row and "selection_score" not in row
        for row in selected
    )


def test_frontier_candidate_remains_explicitly_flagged():
    campaign = SimpleNamespace(
        name="futbol",
        label="Fútbol",
        topic_feature="topic_football_pct",
    )
    protocol = CandidateProtocol(
        candidate_count=3,
        roles=ROLES,
        response_feature_by_campaign={
            "futbol": "total_engagement_per_1000"
        },
    )
    rows = [
        _row("A", reach=1000, affinity=80, response=90, price=400),
        _row("B", reach=500, affinity=80, response=80, price=100),
        _row("C", reach=300, affinity=70, response=60, price=50, stability="fronterizo"),
        _row("D", reach=200, affinity=60, response=70, price=300),
        _row("E", reach=100, affinity=50, response=50, price=200),
    ]

    selected = select_final_candidates(rows, (campaign,), protocol)

    efficiency = selected[2]
    assert efficiency["creator_id"] == "C"
    assert efficiency["frontier_warning"] is True
    assert efficiency["region_stability_class"] == "fronterizo"
