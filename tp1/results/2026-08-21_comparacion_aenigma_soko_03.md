# Comparación de búsquedas: `aenigma_soko_03`

Fecha: 2026-08-21  
Nivel: `levels/aenigma_soko_03.txt` (`soko 03`)  
Modelo de costo: unitario; cada movimiento del jugador cuesta 1  
Límites por caso: 1.000.000 nodos expandidos o 25 segundos

El resultado `cutoff` no significa que el tablero sea insoluble: indica que
la búsqueda no encontró la meta dentro de los límites establecidos.

| Algoritmo | Heurística | Resultado | Costo | Expandidos | Frontera final | Máx. frontera | Tiempo | Movimientos | Empujes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BFS | — | Corte: máximo de nodos | — | 1.000.000 | 258.826 | 258.828 | 10,225394 s | — | — |
| DFS | — | Corte: máximo de nodos | — | 1.000.000 | 336 | 2.007 | 6,037729 s | — | — |
| Greedy | `minimum_matching_manhattan` | Éxito | 111 | 942 | 167 | 170 | 0,022756 s | 111 | 34 |
| Greedy | `shortest_push_access` | Corte: tiempo | — | 619.411 | 434.920 | 434.920 | 25,000050 s | — | — |
| Greedy | `deadlock_aware_reverse_push_matching` | Éxito | 98 | 337 | 124 | 125 | 0,007720 s | 98 | 34 |
| A* | `minimum_matching_manhattan` | Corte: tiempo | — | 734.874 | 189.053 | 195.649 | 25,000022 s | — | — |
| A* | `shortest_push_access` | Corte: tiempo | — | 580.818 | 256.456 | 257.692 | 25,000063 s | — | — |
| A* | `deadlock_aware_reverse_push_matching` | Corte: tiempo | — | 904.003 | 169.710 | 176.430 | 25,000049 s | — | — |

## Lectura de la comparación

La nueva heurística permitió resolver el nivel con Greedy y encontró una
solución 13 movimientos más corta que Greedy con Manhattan (98 frente a 111).
No prueba optimalidad: Greedy ordena solamente por `h(n)`, sin sumar el costo
ya recorrido `g(n)`.

La heurística es admisible para A* porque su matching cuenta solamente
empujes mínimos y cada empuje cuesta al menos una acción. Sin embargo, A*
sigue generando un estado por cada paso del jugador; por eso las tres
variantes alcanzan el tiempo límite. La siguiente mejora recomendada sería
usar macro-movimientos de empuje, sin alterar esta heurística.

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
