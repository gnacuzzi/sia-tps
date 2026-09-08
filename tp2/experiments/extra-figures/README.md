# Gráficos adicionales para la defensa

Esta carpeta reúne los gráficos usados durante el análisis y la preparación de
la presentación. No introduce resultados nuevos.

## Campañas formales

Los gráficos `01` a `07` corresponden a las campañas controladas de Colombia:
1000 generaciones, 5 semillas por condición y un único factor modificado por
etapa. Los gráficos de barras se generan desde los CSV versionados en
`../results/summaries/` mediante `generate_summary_charts.py`.

- `01-resolution-colombia`: resolución de trabajo.
- `02-triangle-count-colombia`: cantidad de triángulos.
- `03-selection-colombia`: métodos de selección.
- `04-crossover-summary-colombia`: NMSE final de cruza y selección.
- `05-crossover-evolution-colombia`: evolución desde la generación 200.
- `06-mutation-colombia`: métodos de mutación.
- `07-survival-colombia`: estrategias de supervivencia.

## Exploración extendida

Los gráficos `08` y `09` comparan 50 y 100 triángulos durante 5000 generaciones
para señal e ícono. Corresponden a una única corrida por condición con semilla
303. Sirven como exploración visual, no como evidencia estadística equivalente
a las campañas de cinco semillas.

Los gráficos `10` a `15` conservan la secuencia exploratoria completa que llevó
a extender el presupuesto de generaciones:

- `10` y `11`: 25, 50, 75, 100 y 125 triángulos durante 1000 generaciones,
  con 5 semillas, para señal e ícono. Son las figuras asociadas al estudio
  guardado en `.context/tp2-triangle-count-study`.
- `12` y `13`: 25, 50 y 100 triángulos durante 4000 generaciones, con semilla
  303. Permitieron observar que el error todavía continuaba descendiendo.
- `14` y `15`: 50 y 100 triángulos durante 5000 generaciones, con semilla 303.
  Son la continuación focalizada del estudio anterior.

Los resultados de 4000 y 5000 generaciones son exploratorios porque tienen una
sola semilla. Documentan la decisión de extender las corridas finales, pero no
reemplazan las comparaciones formales de cinco semillas.

Para regenerar los gráficos derivados de los resúmenes publicados:

```bash
MPLCONFIGDIR=/tmp/tp2-mplconfig \
  .venv/bin/python experiments/extra-figures/generate_summary_charts.py
```
