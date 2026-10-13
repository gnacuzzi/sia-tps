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
umbral del 80 % es sólo una referencia exploratoria y puede modificarse:

```bash
python3 scripts/analyze_pca.py --variance-threshold 0.85
```

Para ejecutar las pruebas:

```bash
python3 -m pytest
```
