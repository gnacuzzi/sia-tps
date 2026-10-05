"""Construir las configuraciones del ejercicio 3 partiendo de RMSProp-64.

Cada etapa replica la del ejercicio 3 original y sólo reemplaza el punto de
partida: `[784,64,10]` con RMSProp en lugar de `[784,128,10]` con Adam. Las
decisiones de cada etapa se toman con los resultados de la anterior, por eso
este script recibe la receta vigente como argumentos en lugar de fijarla.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(name, stage, *, hidden=(64,), learning_rate=0.01, loss="mse",
        l2=None, augmentation=None, schedule=None, batch=128, epochs=200):
    softmax = loss == "categorical_cross_entropy"
    config = {
        "name": name,
        "stage": stage,
        "architecture": [784, *hidden, 10],
        "activations": ["tanh"] * len(hidden) + ["softmax" if softmax else "logistic"],
    }
    if softmax:
        config["loss"] = loss
    if l2 is not None:
        config["l2_lambda"] = l2
    if augmentation is not None:
        config["augmentation"] = augmentation
    config.update({
        "beta": 1.0,
        "init_scale": 0.5,
        "optimizer": {"name": "rmsprop", "learning_rate": learning_rate,
                      "gamma": 0.9, "epsilon": 1e-8},
    })
    if schedule is not None:
        config["learning_rate_schedule"] = [
            {"start_epoch": start, "learning_rate": rate} for start, rate in schedule]
    config.update({"batch_size": batch, "max_epochs": epochs,
                   "model_seed": 0, "shuffle_seed": 0, "shuffle": True})
    return config


def translation(shift=1, probability=0.5):
    return {"name": "translation", "max_shift": shift,
            "probability": probability, "seed": 0}


def rotation(angle, probability=0.5):
    return {"name": "translation_rotation", "max_shift": 1,
            "translation_probability": 0.5, "max_angle_degrees": angle,
            "rotation_probability": probability, "seed": 0}


def plan(runs, *, checkpoints, seeds=None):
    config = {"protocol": "search", "validation_fraction": 0.2,
              "validation_seed": 0, "checkpoints": checkpoints}
    if seeds is not None:
        config["repeat_seeds"] = seeds
    config["runs"] = runs
    return config


def write(name, config):
    path = HERE / f"{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module_code", help="código Python que llama a write(...)")
    exec(parser.parse_args().module_code)
