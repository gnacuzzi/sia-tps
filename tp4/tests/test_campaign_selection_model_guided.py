from scripts.analyze_campaigns_model_guided import select_campaign


def row(creator_id, affinity, pc1, reach, nonfollowers, visits, primary=True):
    return {
        "campaign": "futbol",
        "campaign_label": "Fútbol",
        "creator_id": creator_id,
        "name": f"@{creator_id.lower()}",
        "primary_candidate": str(primary),
        "topic_football_pct": str(affinity),
        "pc1_score": str(pc1),
        "oja_index": str(pc1),
        "estimated_argentina_reach": str(reach),
        "non_follower_reach_pct": str(nonfollowers),
        "profile_visits_per_1000": str(visits),
        "price_usd": "100",
        "region_stability_class": "estable",
        "region_stability_pct": "100",
    }


def test_models_filter_before_objective_metric():
    rows = [
        row("A", 80, 2, 1000, 50, 10),
        row("B", 70, 1, 900, 50, 10),
        row("C", 60, 3, 5000, 90, 10),
        row("D", 90, 4, 10000, 90, 10, primary=False),
    ]
    campaign = {
        "name": "futbol",
        "label": "Fútbol",
        "topic": "topic_football_pct",
        "pc1_side": "high",
        "objective": "ventas y descubrimiento",
        "decision_metric": "estimated_new_relevant_reach",
    }

    finalists, selected = select_campaign(rows, campaign)

    assert [candidate["creator_id"] for candidate in finalists] == ["A"]
    assert selected["creator_id"] == "A"
    assert selected["estimated_new_relevant_reach"] == 400.0
