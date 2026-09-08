# TP2 - Algoritmos Genéticos

Motor de algoritmos geneticos que aproxima una imagen con triangulos
translucidos sobre un canvas blanco.

El ejercicio 1 (representacion ASCII, teorico) esta en `docs/ej1.md`.

## Como correrlo

Necesita Python 3.9+. Para las imagenes usa Pillow y numpy; los graficos usan
matplotlib.

Desde la carpeta `tp2`, armo un entorno virtual e instalo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Uso el venv porque en mac el `pip` del sistema esta bloqueado y tira error.
Con el venv activado ya anda el comando:

```bash
sia-tp2 --config configs/default.json evolve
```

Si no querés instalar nada, se puede correr directo:

```bash
PYTHONPATH=src python3 -m sia_tp2 --config configs/default.json evolve
```

Los tres subcomandos:

- `evolve`: corre la evolucion completa.
- `render-random`: solo genera y evalua la poblacion inicial (baseline sin evolucion).
- `inspect-config`: valida el config y muestra la configuracion efectiva.

## El config.json

Es donde se elige todo: la imagen, la cantidad de triangulos y los
hiperparametros del algoritmo. Ejemplo recortado (el completo esta en
`configs/default.json` y el contrato campo por campo en
`docs/config-contract.md`):

```json
{
  "input": { "image": "assets/targets/01_simple.png", "working_max_side": 64 },
  "representation": { "triangle_count": 20, "canvas_rgb": [255, 255, 255] },
  "genetic": {
    "population_size": 100,
    "offspring_count": 100,
    "parent_selection": { "method": "tournament_deterministic", "params": { "tournament_size": 3 } },
    "crossover": { "method": "uniform", "probability": 0.9, "params": { "swap_probability": 0.5 } },
    "mutation": { "method": "multigene_uniform", "probability": 0.05 },
    "survival": { "strategy": "additive", "selection": { "method": "elite" } }
  },
  "termination": { "max_generations": 1000 },
  "run": { "seed": 0 }
}
```

- `parent_selection.method`: elite, roulette, universal, ranking, boltzmann,
  tournament_deterministic o tournament_probabilistic.
- `crossover.method`: one_point o uniform.
- `mutation.method`: single_gene o multigene_uniform. El cambio de alelo puede
  ser local (`local_delta`) o un resampleo completo (`global_resample`).
- `survival.strategy`: additive o exclusive, cada una con su propio metodo de
  seleccion.
- `termination`: corta por generaciones, error objetivo (`target_nmse`),
  estancamiento o tiempo — el primero que se cumpla.
- Todos los campos son obligatorios y las claves desconocidas dan error: la
  configuracion efectiva queda siempre a la vista.

## Que imprime y que genera

Al terminar muestra el directorio de la corrida, el motivo de corte, la
generacion final, la mejor generacion y el NMSE/fitness del mejor individuo.

Cada corrida queda en `output/runs/<modo>-seed-<n>/` con:

```
best.png                la mejor imagen encontrada
triangles.json          la enumeracion de triangulos (posicion y color RGBA)
metrics.csv             metricas por generacion (error, fitness, diversidad, tiempo)
metadata.json           resumen de la corrida
config.effective.json   la configuracion exacta usada (para reproducir)
checkpoints/            el mejor individuo cada N generaciones
```

## Experimentos

El estudio comparativo corre por fases encadenadas (cada fase usa las
decisiones numericas de la anterior) a partir del manifiesto
`experiments/study-manifest.json`:

```bash
python3 scripts/run_study.py selection
python3 scripts/analyze_study.py selection --figures
```

Las fases: profile, resolution, capacity, selection, crossover, mutation,
survival, validation y showcase. Para correr todo el estudio de una:

```bash
python3 scripts/run_full_study.py
```

## Graficos y resultados

Los resultados del estudio final, con sus graficos y la discusion, estan en
`docs/experimental-results.md` y `docs/ej2-comparative-study.md`. Las figuras
sueltas quedan en `experiments/extra-figures/`.

## Tests

```bash
pytest
```
