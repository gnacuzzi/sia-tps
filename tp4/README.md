# TP4 - Aprendizaje no supervisado

## Preparación

Desde este directorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

## Análisis disponibles

El EDA inicial no requiere dependencias externas:

```bash
python3 scripts/analyze_creators.py
```

El análisis PCA usa las 15 variables documentadas en `docs/decisiones.md`:

```bash
python3 scripts/analyze_pca.py
```

Genera el informe en `docs/pca-creadores/README.md`, junto con tablas de
varianza, cargas, autovectores, scores y los gráficos correspondientes. El
umbral acordado es 90 %, que en estos datos conserva 8 componentes. Puede
indicarse otro umbral de forma explícita:

```bash
python3 scripts/analyze_pca.py --variance-threshold 0.90
```

La comparación entre engagement total y sus cuatro componentes se reproduce
con:

```bash
python3 scripts/compare_engagement_pca.py
```

El informe comparativo queda en `docs/pca-engagement/README.md`.

La red de Kohonen está implementada en `src/sia_tp4/kohonen.py`. Estandariza
las 15 entradas, permite configurar la grilla y el entrenamiento, y expone el
mapa de hits, la matriz U y los mapas por variable. Cada JSON representa una
sola corrida; la configuración base está en `configs/kohonen/base.json`. Cuando
corresponda ejecutarla, su informe se genera con:

```bash
python3 scripts/analyze_kohonen.py
```

Antes de usar los creadores, la implementación se valida con cuatro grupos
sintéticos conocidos:

```bash
python3 scripts/validate_kohonen.py
```

El resultado reproducible está en `docs/kohonen-sintetico/README.md`.

Para los experimentos se crearán otros JSON y se indicarán mediante `--config`.
Cada nombre de corrida produce por defecto una carpeta diferente dentro de
`docs/kohonen-creadores/`; `--output` permite sobrescribir ese destino. También
puede cambiarse el dataset con `--data`. Su diseño y las decisiones pendientes
están documentados en `docs/kohonen.md`.

El primer experimento compara `k = 2, ..., 8` con cinco semillas por tamaño:

```bash
python3 scripts/experiment_kohonen_grid.py
```

Cada una de las 35 corridas conserva su propio JSON dentro de
`configs/kohonen/01-grid/`. El informe reproducible queda en
`docs/kohonen-experimentos/01-grid/README.md`. Esta primera etapa dejó `k = 4`
y `k = 5` como candidatos, pero todavía no fijó los hiperparámetros finales.

Para comprobar la interacción entre el tamaño y la tasa se ejecuta:

```bash
python3 scripts/generate_kohonen_rate_configs.py
python3 scripts/experiment_kohonen_rate.py
```

El generador crea un JSON por corrida y el segundo comando compara los 245
casos de `k = 2, ..., 8`, siete tasas constantes y cinco semillas. El informe
queda en `docs/kohonen-experimentos/02-rate/README.md`.

La comparación completa entre tasa constante y disminución lineal al 10 % se
reproduce con:

```bash
python3 scripts/generate_kohonen_decay_configs.py
python3 scripts/experiment_kohonen_rate_schedule.py
```

El informe de las 490 redes queda en
`docs/kohonen-experimentos/03-rate-schedule/README.md`.

El siguiente experimento compara el radio inicial en las regiones útiles de
`k = 4`, `k = 5` y `k = 6`:

```bash
python3 scripts/generate_kohonen_radius_configs.py
python3 scripts/experiment_kohonen_radius.py
```

Son 300 corridas: cuatro perfiles de tasa por tamaño, cinco radios y cinco
semillas. Se registran métricas de calidad y tiempo de entrenamiento, pero se
generan únicamente curvas agregadas para no almacenar cientos de mapas
redundantes. El informe queda en
`docs/kohonen-experimentos/04-radius/README.md`.

El barrido posterior compara seis cantidades de épocas sobre las cinco
configuraciones finalistas:

```bash
python3 scripts/generate_kohonen_epoch_configs.py
python3 scripts/experiment_kohonen_epochs.py
```

Son 150 corridas independientes: `25`, `50`, `100`, `150`, `250` y `400`
épocas, con cinco semillas. El informe queda en
`docs/kohonen-experimentos/05-epochs/README.md`.

La comparación controlada de inicialización se reproduce con:

```bash
python3 scripts/generate_kohonen_initialization_configs.py
python3 scripts/experiment_kohonen_initialization.py
```

Compara `sample` y `uniform` sobre las cinco finalistas, con 150 épocas y cinco
semillas: 50 corridas. El informe queda en
`docs/kohonen-experimentos/06-initialization/README.md`.

El estudio interpretativo de las cuatro candidatas `k = 4` y `k = 5` se
genera con:

```bash
python3 scripts/analyze_kohonen_finalists.py
```

Produce mapas comparables para las 15 variables, variables reconstruidas y
auxiliares, además de asignaciones y perfiles neuronales. El informe queda en
`docs/kohonen-experimentos/07-interpretation/README.md`.

Para ejecutar las pruebas:

```bash
python3 -m pytest
```
