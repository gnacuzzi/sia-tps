# Development del ejercicio 3

Este directorio registra la construcción reproducible de development sin
copiar nuevamente las imágenes originales.

Se combinaron `data/digits.csv` y `data/more_digits.csv`, se conservó una sola
copia de cada vector de 784 píxeles y se generó un holdout estratificado 80/20
con semilla 0.

## Resultado

| Partición | Imágenes únicas |
|---|---:|
| Training | 19.601 |
| Validation | 4.900 |
| Total | **24.501** |

Antes de deduplicar había 28.190 filas. Se eliminaron 3.689 copias exactas, sin
conflictos de etiqueta. Training y validation no comparten ninguna imagen.

| Dígito | Total | Training | Validation |
|---:|---:|---:|---:|
| 0 | 2.795 | 2.236 | 559 |
| 1 | 3.212 | 2.570 | 642 |
| 2 | 2.817 | 2.254 | 563 |
| 3 | 2.895 | 2.316 | 579 |
| 4 | 2.788 | 2.230 | 558 |
| 5 | 785 | 628 | 157 |
| 6 | 2.825 | 2.260 | 565 |
| 7 | 2.969 | 2.375 | 594 |
| 8 | 585 | 468 | 117 |
| 9 | 2.830 | 2.264 | 566 |

## Reproducción

La carpeta de salida debe ser nueva o estar vacía:

```bash
PYTHONPATH=src python3 docs/ejercicio3/prepare_development.py \
  --primary data/digits.csv \
  --additional data/more_digits.csv \
  --output /tmp/tp3-development \
  --validation-fraction 0.2 \
  --validation-seed 0
```

El runner de búsqueda puede reconstruir el mismo development directamente:

```bash
PYTHONPATH=src python3 docs/ejercicio2/run_experiment.py \
  --config CONFIGURACION.json \
  --data data/digits.csv \
  --additional-data data/more_digits.csv \
  --deduplicate-inputs \
  --output SALIDA
```

## Artefactos

- `manifest.csv`: una fila por imagen única, partición, etiqueta, filas de
  origen y SHA-256 de sus píxeles.
- `split-indices.npz`: índices canónicos exactos de training y validation.
- `class-balance.csv`: comprobación de la estratificación.
- `summary.json`: fuentes, hashes, parámetros y controles de integridad.

`digits_test.csv` no se abrió durante esta construcción.
