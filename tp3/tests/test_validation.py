import json
from pathlib import Path

import numpy as np
import pytest

from sia_tp3 import fit
from sia_tp3.validation import build_model, dataset, load_config


CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'validation.json'


def test_datasets_match_statement():
    X, y = dataset('and', sample_count=50, input_interval=[-1, 1])
    np.testing.assert_array_equal(X, [[-1, 1], [1, -1], [-1, -1], [1, 1]])
    np.testing.assert_array_equal(y[:, 0], [-1, -1, -1, 1])
    _, y = dataset('xor', sample_count=50, input_interval=[-1, 1])
    np.testing.assert_array_equal(y[:, 0], [1, 1, -1, -1])


@pytest.mark.parametrize('case_index', range(5))
def test_validation_converges_with_reference_seed(case_index):
    config = load_config(CONFIG)
    case = config['cases'][case_index]
    X, y = dataset(case['dataset'], sample_count=config['sample_count'],
                   input_interval=config['input_interval'])
    model = build_model(case, seed=0)
    history = fit(model, X, y, learning_rate=case['learning_rate'],
                  max_epochs=case['max_epochs'], target_mse=case['target_mse'],
                  shuffle=case['shuffle'], seed=0,
                  require_bipolar_accuracy=case['dataset'] in {'and', 'xor'})
    assert history[-1]['converged'], history[-1]


def test_unknown_config_key_is_rejected(tmp_path):
    config = json.loads(CONFIG.read_text())
    config['cases'][0]['learnig_rate'] = 0.1
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='campos de caso'):
        load_config(path)


def test_cli_reports_failure_and_refuses_to_overwrite(tmp_path):
    import subprocess
    import sys

    config = json.loads(CONFIG.read_text())
    config['seeds'] = [0]
    case = config['cases'][3]
    case['max_epochs'] = 1
    config['cases'] = [case]
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    output = tmp_path / 'run'
    command = [sys.executable, '-m', 'sia_tp3', '--config', str(path), '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert json.loads((output / 'summary.json').read_text())[0]['passed'] is False
    assert (output / 'xor-2-2-1-seed-0' / 'model.npz').exists()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'carpeta nueva o vacía' in result.stderr
