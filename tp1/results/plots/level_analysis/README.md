# Gráficos para el análisis narrativo de niveles

Todos los gráficos se generan desde
`results/comparacion_principal_summary.csv` mediante:

```bash
python3 scripts/generate_level_analysis_plots.py
```

## Uso sugerido

### `difficulty_dimensions.png`

Abre el bloque de resultados. Muestra que las tres dimensiones elegidas no
ordenan los niveles de la misma manera:

- costo óptimo de BFS como longitud del plan;
- nodos expandidos por BFS como trabajo de búsqueda;
- peor costo dividido por el costo de BFS como sensibilidad al método.

### `path_length_vs_search.png`

Compara los niveles base e intermedio. El intermedio requiere más movimientos,
pero BFS explora menos estados. Sirve para sostener que un camino largo no
implica automáticamente una búsqueda más grande.

### `hard_level_sensitivity.png`

Caso de estudio del nivel difícil. Compara BFS, DFS, Greedy+MMM y A*+MMM con
dos escalas normalizadas:

```text
calidad = costo / costo BFS
trabajo = expandidos / expandidos BFS
```

La línea en 1 representa a BFS. Este gráfico debería acompañarse con el video
de DFS frente a A* o Greedy+MMM.

### `quality_work_tradeoff.png`

Síntesis de los seis métodos en los tres niveles. El extremo inferior izquierdo
representa menor costo y menos expansiones. Las escalas son logarítmicas y BFS
queda en `(1, 1)`.

### `astar_spa_vs_mmm.png`

Compara SPA contra MMM dentro de A*. Una razón menor que 1 favorece a SPA; una
razón mayor que 1 significa que SPA consume más. El costo de solución no se
muestra porque ambas variantes producen exactamente el mismo costo en los tres
niveles.

## Recomendación de selección

Para no volver a llenar la presentación de gráficos, usar como máximo:

1. `difficulty_dimensions.png` para justificar los niveles;
2. `path_length_vs_search.png` para comparar base e intermedio;
3. `hard_level_sensitivity.png` como caso principal;
4. `astar_spa_vs_mmm.png` si se conserva una diapositiva dedicada a las
   heurísticas.

`quality_work_tradeoff.png` puede utilizarse como síntesis final o quedar como
respaldo.
