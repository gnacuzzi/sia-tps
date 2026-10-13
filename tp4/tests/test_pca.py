"""Pruebas de carga, estandarización y PCA."""

import csv

import numpy as np
import pytest

from sia_tp4.data import load_creator_dataset
from sia_tp4.pca import Standardizer, fit_pca


def test_standardizer_uses_sample_statistics_and_can_invert():
    values = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    standardizer = Standardizer.fit(values, ("a", "b"))
    transformed = standardizer.transform(values)

    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(transformed.std(axis=0, ddof=1), 1.0)
    np.testing.assert_allclose(standardizer.inverse_transform(transformed), values)


def test_pca_orders_components_and_diagonalizes_score_covariance():
    values = np.asarray(
        [[1.0, 2.0, 8.0], [2.0, 4.1, 5.0], [3.0, 5.9, 3.0], [4.0, 8.2, 1.0]]
    )
    result = fit_pca(values, ("a", "b", "c"))

    assert np.all(np.diff(result.eigenvalues) <= 0)
    np.testing.assert_allclose(result.explained_variance_ratio.sum(), 1.0)
    np.testing.assert_allclose(
        np.cov(result.scores, rowvar=False, ddof=1),
        np.diag(result.eigenvalues),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.components.T @ result.components, np.eye(3), atol=1e-12
    )
    for column in range(result.components.shape[1]):
        pivot = np.argmax(np.abs(result.components[:, column]))
        assert result.components[pivot, column] >= 0


def test_component_count_uses_first_component_reaching_threshold():
    values = np.asarray([[1.0, 1.0], [2.0, 2.1], [3.0, 2.9], [4.0, 4.0]])
    result = fit_pca(values, ("a", "b"))

    assert result.component_count(0.50) == 1
    assert result.component_count(1.00) == 2
    with pytest.raises(ValueError):
        result.component_count(0.0)


def test_loader_preserves_requested_order_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "creators.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["creator_id", "name", "followers", "x", "y"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"creator_id": "c1", "name": "Uno", "followers": 10, "x": 1, "y": 4},
                {"creator_id": "c2", "name": "Dos", "followers": 20, "x": 2, "y": 3},
            ]
        )

    dataset = load_creator_dataset(path, ("y", "x"))
    assert dataset.feature_names == ("y", "x")
    np.testing.assert_array_equal(dataset.values, [[4.0, 1.0], [3.0, 2.0]])

    text = path.read_text(encoding="utf-8").replace("c2,Dos", "c1,Dos")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="creator_id debe ser único"):
        load_creator_dataset(path, ("y", "x"))
