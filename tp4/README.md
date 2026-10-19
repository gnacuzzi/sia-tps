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
están documentados en `docs/kohonen.md`. Todavía no se ejecutó el entrenamiento
ni se eligieron hiperparámetros finales.

Para ejecutar las pruebas:

```bash
python3 -m pytest
```
