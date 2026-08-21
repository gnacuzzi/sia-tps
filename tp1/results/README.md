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
- **Máxima frontera:** máximo de nodos pendientes simultáneamente; es la
  medida más útil para comparar el uso de memoria.
- **Solución:** camino de movimientos desde el estado inicial al final.
- **Tiempo de procesamiento:** segundos que tardó la búsqueda. Depende de la
  máquina, por lo que se compara con cautela.

También se guardan los movimientos y empujes, límites de la ejecución y la
ruta al GIF cuando existe.

## Cómo repetir una prueba

Desde la carpeta `tp1`, usá la configuración correspondiente. Por ejemplo:

```bash
PYTHONPATH=src python3 -m sia_tp1 --config results/config_level_02_bfs.json --search \
  --gif output/videos/aenigma_soko_01/bfs.gif
```

Las configuraciones de esta primera comparación limitan la búsqueda a
1.000.000 de nodos expandidos y 25 segundos. Para una nueva prueba, se crea
una configuración equivalente, una ficha `AAAA-MM-DD_<nivel>_<caso>.md` y
una fila en `registro.csv`.

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
