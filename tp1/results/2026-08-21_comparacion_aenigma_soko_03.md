# Comparación de búsquedas: `aenigma_soko_03`

Fecha: 2026-08-21  
Nivel: `levels/aenigma_soko_03.txt` (`soko 03`)  
Modelo de costo: unitario; cada movimiento del jugador cuesta 1  
Límites por caso: 1.000.000 nodos expandidos o 25 segundos

El resultado `cutoff` no significa que el tablero sea insoluble: indica que
la búsqueda no encontró la meta dentro de los límites establecidos.

| Algoritmo | Heurística | Resultado | Costo | Expandidos | Frontera final | Máx. frontera | Tiempo | Movimientos | Empujes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BFS | — | Corte: máximo de nodos | — | 1.000.000 | 258.826 | 258.828 | 10,784093 s | — | — |
| DFS | — | Corte: máximo de nodos | — | 1.000.000 | 336 | 2.007 | 6,330902 s | — | — |
| Greedy | `minimum_matching_manhattan` | Éxito | 111 | 942 | 167 | 170 | 0,026251 s | 111 | 34 |
| Greedy | `shortest_push_access` | Corte: tiempo | — | 625.632 | 439.235 | 439.235 | 25,000015 s | — | — |
| Greedy | `deadlock_aware_reverse_push_matching` | Éxito | 98 | 337 | 124 | 125 | 0,008448 s | 98 | 34 |
| Greedy | `pair_pattern_database_matching` | Éxito | 98 | 337 | 124 | 125 | 0,039467 s | 98 | 34 |
| A* | `minimum_matching_manhattan` | Corte: tiempo | — | 724.127 | 190.080 | 195.649 | 25,000026 s | — | — |
| A* | `shortest_push_access` | Corte: tiempo | — | 581.487 | 256.483 | 257.809 | 25,000012 s | — | — |
| A* | `deadlock_aware_reverse_push_matching` | Corte: tiempo | — | 862.207 | 171.009 | 176.430 | 25,155284 s | — | — |
| A* | `pair_pattern_database_matching` | Corte: tiempo | — | 728.058 | 149.134 | 155.216 | 25,640586 s | — | — |

## Lectura de la comparación

Las dos heurísticas basadas en pushes inversos permitieron resolver el nivel
con Greedy y encontraron una solución 13 movimientos más corta que Greedy con
Manhattan (98 frente a 111). No prueba optimalidad: Greedy ordena solamente
por `h(n)`, sin sumar el costo ya recorrido `g(n)`.

`pair_pattern_database_matching` combina el matching inverso con una base de
patrones de pares de cajas. Detecta bloqueos que requieren considerar ambas
cajas simultáneamente, pero en este nivel no eleva el valor inicial (ambas
heurísticas dan 28) ni cambia la primera ruta de Greedy. A* mantiene el costo
óptimo como objetivo, pero sigue generando un estado por cada paso del jugador;
las cuatro variantes alcanzan el tiempo límite. La siguiente mejora
recomendada sería usar macro-movimientos de empuje con una PDB de tres cajas o
con bloqueos dinámicos adicionales.

## Soluciones encontradas

### Greedy con `minimum_matching_manhattan`

Costo 111; 34 empujes.

```text
UP UP DOWN DOWN DOWN DOWN UP UP LEFT LEFT LEFT RIGHT RIGHT RIGHT RIGHT RIGHT
LEFT LEFT LEFT LEFT LEFT LEFT DOWN LEFT UP LEFT UP RIGHT DOWN RIGHT UP UP DOWN
DOWN RIGHT RIGHT UP RIGHT RIGHT UP UP UP RIGHT UP LEFT UP LEFT DOWN RIGHT DOWN
LEFT UP LEFT DOWN RIGHT RIGHT DOWN DOWN DOWN DOWN RIGHT RIGHT RIGHT RIGHT DOWN
RIGHT UP UP UP RIGHT UP LEFT LEFT DOWN RIGHT DOWN DOWN LEFT LEFT LEFT DOWN LEFT
DOWN LEFT DOWN UP UP UP RIGHT RIGHT RIGHT RIGHT DOWN RIGHT DOWN DOWN DOWN LEFT
LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN LEFT UP LEFT UP RIGHT
```

### Greedy con `deadlock_aware_reverse_push_matching`

Costo 98; 34 empujes.
GIF: `output/videos/aenigma_soko_03/greedy_deadlock_aware_reverse_push_matching.gif`

```text
UP UP UP DOWN DOWN DOWN DOWN DOWN DOWN UP UP UP LEFT LEFT LEFT RIGHT RIGHT
RIGHT RIGHT RIGHT RIGHT RIGHT DOWN RIGHT UP UP UP RIGHT UP LEFT LEFT DOWN RIGHT
DOWN DOWN DOWN DOWN DOWN DOWN LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT DOWN LEFT
UP LEFT UP RIGHT DOWN RIGHT RIGHT RIGHT RIGHT UP UP UP UP UP UP UP UP RIGHT UP
LEFT UP LEFT DOWN RIGHT DOWN LEFT UP LEFT DOWN RIGHT RIGHT DOWN DOWN DOWN DOWN
LEFT LEFT LEFT LEFT DOWN LEFT UP LEFT UP RIGHT DOWN RIGHT UP UP
```

### Greedy con `pair_pattern_database_matching`

Costo 98; 34 empujes. Encontró exactamente la misma secuencia que Greedy con
`deadlock_aware_reverse_push_matching`, por lo que no se repite el camino.
