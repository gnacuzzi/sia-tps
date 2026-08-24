# Resultados de experimentos

Esta carpeta centraliza las ejecuciones usadas para comparar algoritmos de
búsqueda en Sokoban. Cada comparación tiene una ficha única con todos sus
casos y también una fila por ejecución en `registro.csv`, que se puede abrir
directamente con una planilla de cálculo.

## Campos registrados

- **Nivel:** archivo de tablero que se resolvió.
- **Algoritmo:** BFS, DFS, Greedy o A*.
- **Heurística:** identificador configurado; `null` cuando no corresponde.
- **Resultado:** `success`, `failure` o `cutoff` (y la causa de corte).
- **Costo de la solución:** cantidad de acciones, pues el modelo de costo es
  unitario.
- **Nodos expandidos:** estados que el algoritmo extrajo de la frontera y
  desarrolló.
- **Nodos frontera al finalizar:** nodos pendientes al terminar la búsqueda.
- **Máxima frontera:** máximo de entradas almacenadas simultáneamente. En A*
  corresponde al tamaño físico del heap y puede incluir entradas obsoletas;
  la frontera final de A* sí cuenta solamente estados activos.
- **Solución:** camino de movimientos desde el estado inicial al final.
- **Tiempo de procesamiento:** segundos que tardó la búsqueda. Depende de la
  máquina, por lo que se compara con cautela.

También se guardan los movimientos y empujes, límites de la ejecución y la
ruta a la animación cuando existe.

## Cómo repetir una prueba

Desde la carpeta `tp1`, usá la configuración correspondiente. Por ejemplo:

```bash
PYTHONPATH=src python3 -m sia_tp1 --config results/config_level_02_bfs.json --search \
  --video output/videos/aenigma_soko_01/bfs.mp4
```

Para generar de una vez los videos de BFS, DFS, Greedy y A* con las dos
heurísticas acordadas:

```bash
python3 scripts/generate_videos.py levels/level_03.txt \
  --output-dir output/videos/level3
```

El runner restaura el contenido original de `config.json` incluso si una
ejecución falla.

Las configuraciones de esta primera comparación limitan la búsqueda a
1.000.000 de nodos expandidos y 25 segundos. Se conservan como snapshots de
los experimentos históricos; para ejecuciones nuevas se usa el runner del
siguiente apartado.

## Runner de experimentos

`scripts/run_experiments.py` ejecuta los casos sin modificar `config.json` y
genera un CSV con una fila por repetición más otro CSV resumido. Cada repetición
se ejecuta en un proceso nuevo para que las cachés de las heurísticas comiencen
vacías y no alteren la comparación temporal.

Los seis casos obligatorios sobre un nivel:

```bash
python3 scripts/run_experiments.py levels/level_03.txt \
  --suite core \
  --repetitions 10 \
  --output results/level_03_core.csv
```

Para incluir las cuatro combinaciones con las heurísticas extendidas:

```bash
python3 scripts/run_experiments.py levels/aenigma_soko_03.txt \
  --suite all \
  --repetitions 10 \
  --output results/aenigma_soko_03_all.csv
```

También se pueden pasar varios niveles en una sola ejecución. Los límites por
defecto son 1.000.000 de expansiones y 25 segundos; se cambian con
`--max-expanded-nodes` y `--timeout-seconds`. Si no se indica `--output`, se
crea un nombre con fecha y hora. Un archivo existente sólo se reemplaza usando
`--overwrite`.

El archivo resumido agrega `_summary` al nombre solicitado. Los tiempos se
resumen mediante mediana, mínimo y máximo. Los estados `cutoff` se cuentan por
separado y nunca se convierten en soluciones de costo cero.

## Generación de gráficos

`scripts/generate_plots.py` toma el CSV crudo, conserva las mediciones de cada
repetición y genera los cinco análisis requeridos:

- costo de solución mediante el promedio de las ejecuciones exitosas;
- nodos expandidos y máxima frontera mediante el promedio —estas métricas
  estructurales resultaron deterministas en la comparación principal—;
- tiempo promedio con una barra de error de una desviación estándar, calculada
  sobre las diez repeticiones;
- matriz de resultados con cantidades de `success`, `cutoff` y `failure`.

Para la comparación principal, manteniendo el orden de dificultad elegido:

```bash
python3 scripts/generate_plots.py results/comparacion_principal.csv \
  --output-dir results/plots/comparacion_principal \
  --level-order level_03 level_02 level_04
```

Los gráficos se guardan como PNG. Se puede usar `--format svg` para obtener
imágenes vectoriales apropiadas para la presentación. Cuando los valores de
una métrica difieren por un factor de 20 o más, el eje usa automáticamente una
escala logarítmica. Un cutoff nunca se interpreta como una solución de costo
cero: queda excluido del gráfico de costo y aparece explícitamente en la
matriz de resultados.

## Índice actual

| Fecha | Nivel | Algoritmo | Heurística | Resultado | Costo | Expandidos | Máx. frontera | Tiempo (s) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 2026-08-21 | `levels/level_02.txt` | BFS | `null` | éxito | 78 | 30.375 | 866 | 0,157919 |
| 2026-08-21 | `levels/level_02.txt` | A* | `minimum_matching_manhattan` | éxito | 78 | 30.315 | 1.015 | 0,214426 |
| 2026-08-21 | `levels/level_02.txt` | A* | `shortest_push_access` | éxito | 78 | 26.823 | 1.199 | 0,896522 |

El nivel corresponde al tablero `soko 01` (lid 200) de la fuente consultada.
El detalle de los tres casos está en
[`2026-08-21_comparacion_level_02.md`](2026-08-21_comparacion_level_02.md).

También se registró una matriz de diez ejecuciones sobre el nivel difícil
[`aenigma_soko_03.txt`](2026-08-21_comparacion_aenigma_soko_03.md): BFS, DFS,
las cuatro heurísticas con Greedy y las cuatro con A*.
