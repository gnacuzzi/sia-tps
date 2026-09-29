import numpy as np
import pytest

from sia_tp3 import classification_metrics, confusion_matrix


def test_binary_metrics_follow_real_rows_and_predicted_columns():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])

    report = classification_metrics(y_true, y_pred, labels=[0, 1])

    np.testing.assert_array_equal(report.confusion_matrix, [[1, 1], [1, 1]])
    assert report.accuracy == pytest.approx(0.5)
    for label in (0, 1):
        metrics = report.for_label(label)
        assert metrics.support == metrics.predicted == 2
        assert metrics.true_positive == metrics.true_negative == 1
        assert metrics.false_positive == metrics.false_negative == 1
        assert metrics.precision == pytest.approx(0.5)
        assert metrics.recall == metrics.tpr == pytest.approx(0.5)
        assert metrics.f1 == pytest.approx(0.5)
        assert metrics.fpr == pytest.approx(0.5)


def test_multiclass_metrics_are_one_vs_rest_and_keep_absent_classes():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 2, 2])

    report = classification_metrics(y_true, y_pred, labels=[0, 1, 2, 8])

    np.testing.assert_array_equal(report.confusion_matrix,
                                  [[1, 1, 0, 0],
                                   [0, 1, 1, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 0]])
    assert report.accuracy == pytest.approx(3 / 5)
    assert report.for_label(0).precision == pytest.approx(1)
    assert report.for_label(0).recall == pytest.approx(1 / 2)
    assert report.for_label(1).fpr == pytest.approx(1 / 3)
    assert report.for_label(2).recall == pytest.approx(1)

    absent = report.for_label(8)
    assert absent.support == absent.predicted == 0
    assert np.isnan(absent.precision)
    assert np.isnan(absent.recall)
    assert np.isnan(absent.f1)
    assert absent.fpr == pytest.approx(0)
    assert report.macro_precision == pytest.approx(2 / 3)
    assert report.macro_recall == pytest.approx(2 / 3)
    assert report.macro_f1 == pytest.approx(11 / 18)
    assert report.macro_tpr == report.macro_recall
    assert report.macro_fpr == pytest.approx(7 / 48)


def test_confusion_matrix_requires_declared_integer_classes():
    with pytest.raises(ValueError, match="no figuran"):
        confusion_matrix([0, 1], [0, 2], labels=[0, 1])
    with pytest.raises(ValueError, match="enteras"):
        confusion_matrix([0.0, 1.0], [0.0, 1.0], labels=[0, 1])
    with pytest.raises(ValueError, match="igual longitud"):
        confusion_matrix([0], [0, 1], labels=[0, 1])
