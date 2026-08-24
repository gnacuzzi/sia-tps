# TP1 - Sokoban

Motor de busqueda para resolver niveles de Sokoban (BFS, DFS, Greedy y A*).

El ejercicio 1 (8-puzzle) esta en `ej1.md`.

## Como correrlo

Necesita Python 3.9+. Para las animaciones y los graficos usa Pillow y matplotlib.

Desde la carpeta `tp1`, armo un entorno virtual e instalo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Uso el venv porque en mac el `pip` del sistema esta bloqueado y tira error.
Con el venv activado ya anda el comando:

```bash
sia-tp1 --config config.json --search
```

Si no querés instalar nada, se puede correr directo:

```bash
PYTHONPATH=src python3 -m sia_tp1 --config config.json --search
```

## El config.json

Es donde se elige que correr. Ejemplo:

```json
{
  "level_file": "levels/level_01.txt",
  "algorithm": "bfs",
  "heuristic": null,
  "cost_model": "unit",
  "limits": {
    "max_expanded_nodes": null,
    "timeout_seconds": null
  },
  "seed": 0
}
```

- `algorithm`: bfs, dfs, greedy o astar.
- `heuristic`: va en null para bfs y dfs. Para greedy y astar hay que poner una:
  minimum_matching_manhattan, shortest_push_access,
  deadlock_aware_reverse_push_matching o pair_pattern_database_matching.
- `limits`: se puede cortar por cantidad de nodos o por tiempo (o null para que no corte).

## Los niveles

Son archivos de texto en `levels/`. Los simbolos:

```
#  pared
_  piso
.  objetivo
$  caja
@  jugador
*  caja arriba de un objetivo
+  jugador arriba de un objetivo
```

Ejemplo (`levels/level_01.txt`):

```
#######
#@_$_.#
#######
```

## Que imprime

Cuando corre la busqueda muestra el resultado (success, failure o cutoff), los
nodos expandidos, los de la frontera, el maximo de la frontera y el tiempo. Si
encuentra solucion tambien tira el costo y el camino de movimientos.

## Graficos

Los graficos que usamos para comparar los metodos ya estan generados en
`results/plots/comparacion_principal/`. Se abren directo con cualquier visor de
imagenes:

```
solution_cost.png       costo de la solucion por metodo
expanded_nodes.png      nodos expandidos
max_frontier.png        maximo de la frontera
elapsed_time_repetitions.png  tiempo (con desvio)
outcome_matrix.png      exito / cutoff / fallo
```

Para volver a generarlos a partir del CSV:

```bash
python3 scripts/generate_plots.py results/comparacion_principal.csv --output-dir results/plots/comparacion_principal
```

Quedan como PNG en la carpeta que pongas en `--output-dir` (o SVG con `--format svg`).

## Experimentos

El CSV con los resultados sale de correr:

```bash
python3 scripts/run_experiments.py levels/level_03.txt --suite core --repetitions 10 --output results/level_03_core.csv
```

Todo lo generado queda en `results/`.

## Videos

Cada corrida puede guardar la animacion de la solucion:

```bash
sia-tp1 --config config.json --search --gif output/solucion.gif
sia-tp1 --config config.json --search --video output/solucion.mp4
```

O generar un video de todos los metodos sobre un nivel:

```bash
python3 scripts/generate_videos.py levels/level_03.txt --output-dir output/videos/level3
```

## Tests

```bash
pytest
```
