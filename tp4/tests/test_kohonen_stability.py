"""Pruebas del diagnóstico agregado de estabilidad de Kohonen."""

import json
from pathlib import Path

import numpy as np
import pytest

from sia_tp4.data import CreatorDataset
from sia_tp4.kohonen_analysis import load_run
from sia_tp4.kohonen_stability import (
    CAMPAIGNS,
    CampaignShortlist,
    SeedStabilityResult,
    StabilityDesign,
    load_design,
    stability_class,
    summarize_creator_stability,
    summarize_shortlist_candidates,
    write_report,
    write_tables,
)


def _document():
    return {
        "name": "final-stability",
        "seeds": list(range(10)),
        "stable_min_count": 8,
        "frontier_min_count": 6,
        "transition_threshold_pct": 35.0,
        "kohonen": {
            "grid_size": 5,
            "epochs": 150,
            "learning_rate_start": 0.2,
            "learning_rate_end": 0.2,
            "radius_start": 2**0.5 * 4,
            "radius_end": 1.0,
            "initialization": "sample",
            "shuffle": True,
        },
    }


def _design():
    document = _document()
    return StabilityDesign(
        name=document["name"],
        seeds=tuple(document["seeds"]),
        stable_min_count=document["stable_min_count"],
        frontier_min_count=document["frontier_min_count"],
        transition_threshold_pct=document["transition_threshold_pct"],
        kohonen_parameters=document["kohonen"],
    )


def _dataset():
    return CreatorDataset(
        creator_ids=("c1", "c2", "c3"),
        names=("Uno", "Dos", "Tres"),
        feature_names=("x",),
        values=np.asarray([[1.0], [2.0], [3.0]]),
        followers=np.asarray([10.0, 20.0, 30.0]),
    )


def _seed_result(seed, regions, shortlist_indexes, transitions=(False, False, False)):
    shortlists = tuple(
        CampaignShortlist(
            campaign=campaign,
            topic=topic,
            row=0,
            column=0,
            topic_pct=70.0,
            dominant_region=topic,
            dominant_region_pct=70.0,
            creator_indexes=tuple(shortlist_indexes.get(campaign, ())),
        )
        for campaign, topic in CAMPAIGNS
    )
    return SeedStabilityResult(
        seed=seed,
        run=f"seed-{seed}",
        quantization_error=1.0,
        occupied_units=3,
        weak_topic_cells=0,
        creator_regions=tuple(regions),
        creator_transition_flags=tuple(transitions),
        shortlists=shortlists,
    )


def test_load_design_requires_exactly_ten_unique_seeds(tmp_path):
    path = tmp_path / "stability.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")

    design = load_design(path)

    assert design.seeds == tuple(range(10))
    assert design.run_for_seed(9).config.seed == 9
    invalid = _document()
    invalid["seeds"] = [0] * 10
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="10 enteros únicos"):
        load_design(path)


def test_repository_design_reuses_the_selected_final_configuration():
    root = Path(__file__).parents[1]
    design = load_design(root / "configs" / "kohonen" / "stability.json")
    selected = load_run(
        root
        / "configs"
        / "kohonen"
        / "06-initialization"
        / "k5-constant-global-sample-seed-00.json"
    )

    expected = vars(selected.config).copy()
    expected.pop("seed")
    assert design.kohonen_parameters == expected


def test_stability_thresholds_match_the_declared_protocol():
    design = _design()

    assert stability_class(10, design) == "estable"
    assert stability_class(8, design) == "estable"
    assert stability_class(7, design) == "fronterizo"
    assert stability_class(6, design) == "fronterizo"
    assert stability_class(5, design) == "inestable"


def test_creator_and_shortlist_frequencies_are_aggregated_across_seeds():
    results = []
    for seed in range(10):
        regions = (
            "football" if seed < 8 else "makeup",
            "dogs" if seed < 6 else "art",
            "art" if seed < 5 else "football",
        )
        results.append(
            _seed_result(
                seed,
                regions,
                {"futbol": (0,) if seed < 8 else (2,)},
                transitions=(False, seed < 2, False),
            )
        )

    creators = summarize_creator_stability(_dataset(), results, _design())
    candidates = summarize_shortlist_candidates(_dataset(), results, _design())

    assert creators[0]["modal_region"] == "football"
    assert creators[0]["region_stability_count"] == 8
    assert creators[0]["region_stability_class"] == "estable"
    assert creators[1]["region_stability_class"] == "fronterizo"
    assert creators[1]["transition_count"] == 2
    assert creators[2]["modal_region_tie"] is True
    assert creators[2]["region_stability_class"] == "inestable"
    football = [row for row in candidates if row["campaign"] == "futbol"]
    assert [(row["creator_id"], row["inclusion_count"]) for row in football] == [
        ("c1", 8),
        ("c3", 2),
    ]
    assert [row["candidate_class"] for row in football] == [
        "estable",
        "inestable",
    ]


def test_tables_use_one_aggregate_directory(tmp_path):
    results = [
        _seed_result(seed, ("football", "dogs", "art"), {"futbol": (0,)})
        for seed in range(10)
    ]

    creator_rows, _, campaign_rows = write_tables(
        tmp_path, _dataset(), results, _design()
    )
    write_report(tmp_path, _design(), results, creator_rows, campaign_rows)

    generated = tmp_path / "generated"
    assert sorted(path.name for path in generated.iterdir()) == [
        "candidatos_shortlist.csv",
        "corridas.csv",
        "estabilidad_creadores.csv",
        "resumen_campanas.csv",
        "shortlists_semillas.csv",
    ]
    assert not any(path.is_dir() for path in generated.iterdir())
    assert (tmp_path / "README.md").is_file()
