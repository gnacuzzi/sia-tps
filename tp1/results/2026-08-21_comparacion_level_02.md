# Comparación de búsquedas: `levels/level_02.txt`

Fecha: 2026-08-21  
Nivel: `levels/level_02.txt` (`soko 01`, lid 200)  
Modelo de costo: unitario  
Límites: 1.000.000 nodos expandidos y 25 segundos

| Algoritmo | Heurística | Resultado | Costo | Nodos expandidos | Frontera final | Máxima frontera | Tiempo | Movimientos | Empujes | GIF |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BFS | `null` | Éxito (`success`) | 78 | 30.375 | 19 | 866 | 0,157919 s | 78 | 17 | `output/videos/aenigma_soko_01/bfs.gif` |
| A* | `minimum_matching_manhattan` | Éxito (`success`) | 78 | 30.315 | 37 | 1.015 | 0,214426 s | 78 | 17 | `output/videos/aenigma_soko_01/astar_minimum_matching_manhattan.gif` |
| A* | `shortest_push_access` | Éxito (`success`) | 78 | 26.823 | 190 | 1.199 | 0,896522 s | 78 | 17 | `output/videos/aenigma_soko_01/astar_shortest_push_access.gif` |

## Solución

Los tres casos encontraron exactamente el mismo camino óptimo: costo 78,
78 movimientos y 17 empujes. La secuencia completa, desde el estado inicial
al final, se encuentra en
[`level_02_solucion_78_movimientos.md`](level_02_solucion_78_movimientos.md).

## Configuraciones para repetir los casos

- BFS: `config_level_02_bfs.json`
- A* con matching Manhattan: `config_level_02_astar_minimum_matching_manhattan.json`
- A* con acceso al próximo empuje: `config_level_02_astar_shortest_push_access.json`

Los tiempos son una medición de esta máquina y pueden variar levemente entre
ejecuciones; las métricas de nodos y el costo son las medidas principales para
la comparación.
