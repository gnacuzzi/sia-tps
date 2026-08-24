# Ejercicio 1 — 8-puzzle

## 1. Formulación del problema

Según la clasificación de agentes y ambientes, el 8-puzzle es un ambiente:

- Totalmente observable.
- Determinístico.
- Secuencial.
- Estático.
- Discreto.
- Conocido.
- Individual.

### Estructura de estado

El tablero se representa como una tupla inmutable de nueve posiciones en orden fila-columna, utilizando `0` para el espacio vacío.

Por ejemplo, el estado inicial:

```text
5 7 3
8 2 _
1 6 4
```

se representa como:

```text
(5, 7, 3, 8, 2, 0, 1, 6, 4)
```

Esta representación permite:

- Comparar estados directamente.
- Usar estados como claves de conjuntos o diccionarios.
- Detectar estados repetidos eficientemente.
- Encontrar la fila y columna de cada ficha.
- Separar claramente el estado del nodo de búsqueda.

El estado contiene únicamente la configuración del tablero. El nodo de búsqueda es una estructura diferente:

```text
Nodo = (
    estado,
    padre,
    acción aplicada,
    profundidad,
    costo acumulado
)
```

Un mismo estado puede aparecer en distintos nodos si fue alcanzado mediante caminos diferentes.

### Estado inicial

```text
5 7 3
8 2 _
1 6 4
```

### Acciones posibles

Las acciones son:

```text
ARRIBA
ABAJO
IZQUIERDA
DERECHA
```

Cada acción representa el movimiento del espacio vacío en una dirección.

Una acción es aplicable solamente si el espacio vacío permanece dentro del tablero. De manera equivalente, puede pensarse que se mueve una ficha adyacente hacia el espacio vacío.

### Modelo de transición

Aplicar una acción intercambia la posición del espacio vacío con la ficha numerada ubicada en la dirección elegida.

### Función de costo

Cada movimiento tiene costo unitario:

```text
c(s, a, s') = 1
```

Por lo tanto, el costo de una solución coincide con su cantidad de movimientos.

### Condición de solución

La consigna acepta tres tableros objetivo:

```text
G1            G2            G3

1 2 3         3 2 1         3 6 _
4 5 6         6 5 4         2 5 8
7 8 _         _ 8 7         1 4 7
```

La condición de solución es:

```text
goal(s) = 1  si s ∈ {G1, G2, G3}
goal(s) = 0  en otro caso
```

### Espacio de estados

El espacio de estados contiene todas las configuraciones alcanzables desde el estado inicial.

Aunque existen `9!` permutaciones posibles de las nueve posiciones, solamente la mitad son alcanzables desde una configuración determinada debido a la paridad del 8-puzzle:

```text
9! / 2 = 181 440
```

Por este motivo es importante detectar estados repetidos durante la búsqueda.

---

## 2. Primera heurística: fichas fuera de lugar

Para un estado `s` y un objetivo `G`, se cuenta cuántas fichas numeradas no ocupan la posición que les corresponde en ese objetivo.

El espacio vacío no se cuenta.

```text
h_fuera(s, G) = Σ [ pos_s(i) ≠ pos_G(i) ]   para i = 1..8
```

Como la consigna admite tres objetivos, la heurística toma el mínimo:

```text
h_fuera(s) = min { h_fuera(s, G) : G ∈ {G1, G2, G3} }
```

### Justificación de admisibilidad

Cada acción mueve exactamente una ficha numerada.

Por lo tanto, un movimiento puede colocar correctamente como máximo una ficha que estaba fuera de lugar.

Si existen `k` fichas fuera de lugar respecto de un objetivo, serán necesarios al menos `k` movimientos para alcanzarlo:

```text
h_fuera(s, G) ≤ h*(s, G)
```

Al tomar el mínimo entre los tres objetivos:

```text
h_fuera(s) ≤ min { h*(s, G) : G ∈ {G1, G2, G3} }
```

Por lo tanto, la heurística nunca sobreestima el costo real de alcanzar alguno de los objetivos.

Además:

- Devuelve cero en cualquier estado objetivo.
- Devuelve un valor positivo en cualquier estado no objetivo.
- Es consistente, porque un movimiento puede cambiar su valor como máximo en una unidad.

En consecuencia, es una heurística admisible no trivial.

---

## 3. Segunda heurística: suma de distancias Manhattan

La distancia Manhattan entre dos posiciones se define como:

```text
d_M((f1, c1), (f2, c2)) = |f1 - f2| + |c1 - c2|
```

Para cada ficha se calcula la distancia entre su posición actual y la posición que debería ocupar en el objetivo.

El espacio vacío no se cuenta.

Para un objetivo `G`:

```text
h_manhattan(s, G) = Σ d_M( pos_s(i), pos_G(i) )   para i = 1..8
```

Como hay tres objetivos posibles:

```text
h_manhattan(s) = min { h_manhattan(s, G) : G ∈ {G1, G2, G3} }
```

### Justificación de admisibilidad

Cada movimiento desplaza exactamente una ficha una posición horizontal o vertical.

Por lo tanto, una acción puede reducir la suma de distancias Manhattan como máximo en una unidad.

Si la suma Manhattan respecto de un objetivo vale `k`, serán necesarios al menos `k` movimientos para alcanzar ese objetivo:

```text
h_manhattan(s, G) ≤ h*(s, G)
```

La heurística ignora que otras fichas pueden bloquear el recorrido. Esta relajación solamente puede hacer que la estimación sea menor que el costo real, nunca mayor.

Al tomar el mínimo entre los tres objetivos:

```text
h_manhattan(s) ≤ min { h*(s, G) : G ∈ {G1, G2, G3} }
```

Por lo tanto, la heurística es admisible.

También es consistente, porque para dos estados vecinos `s` y `s'` se cumple:

```text
h_manhattan(s) ≤ 1 + h_manhattan(s')
```

### Dominancia

Para cualquier ficha fuera de lugar, su distancia Manhattan es al menos uno.

Por lo tanto:

```text
h_manhattan(s) ≥ h_fuera(s)
```

La distancia Manhattan domina a la cantidad de fichas fuera de lugar.

Esto significa que ambas son admisibles, pero Manhattan suele brindar más información y permite que A* expanda menos nodos.

---

## 4. Evaluación del estado inicial

Los valores de las heurísticas para el estado inicial son:

| Objetivo | Fichas fuera de lugar | Distancia Manhattan |
|---|---:|---:|
| G1 | 7 | 15 |
| G2 | 8 | 17 |
| G3 | 7 | 13 |
| Mínimo | **7** | **13** |

Por lo tanto:

```text
h_fuera(s0)     = 7
h_manhattan(s0) = 13
```

Una búsqueda BFS independiente permitió verificar que la solución óptima del estado inicial tiene costo `23`.

En consecuencia:

```text
h_fuera(s0) = 7 ≤ 13 = h_manhattan(s0) ≤ 23 = h*(s0)
```

Esto confirma, para la instancia de ejemplo, que ambas heurísticas subestiman el costo óptimo real.

---

## 5. Análisis de paridad

En un tablero 3×3, la paridad de la cantidad de inversiones se conserva después de cada movimiento.

Para la instancia de la consigna:

| Estado | Paridad |
|---|---|
| Estado inicial | Impar |
| G1 | Par |
| G2 | Impar |
| G3 | Par |

Por lo tanto, desde el estado inicial solamente es posible alcanzar G2.

Los objetivos G1 y G3 pertenecen a otra componente del espacio de estados.

Si se realiza esta comprobación antes de iniciar la búsqueda, las heurísticas pueden calcularse únicamente respecto de G2:

```text
Fichas fuera de lugar: 8
Distancia Manhattan:   17
Costo óptimo real:     23
```

Así se obtienen estimaciones más informativas:

```text
8 ≤ 17 ≤ 23
```

La comprobación de paridad es una optimización válida, pero no es necesaria para que las heurísticas definidas sobre los tres objetivos sean admisibles.

---

## 6. Métodos de búsqueda seleccionados

### Breadth-First Search

BFS no utiliza heurística.

Expande primero los nodos de menor profundidad. Como todas las acciones tienen costo `1`, la profundidad de un nodo coincide con su costo acumulado.

Por lo tanto, BFS es:

- Completa, porque el factor de ramificación es finito.
- Óptima, porque el costo de todas las acciones es uniforme.

BFS resulta útil como línea base para comprobar el costo óptimo, aunque puede consumir una cantidad considerable de memoria.

### A* con fichas fuera de lugar

A* ordena la frontera mediante:

```text
f(n) = g(n) + h(n)
```

Utilizando la cantidad de fichas fuera de lugar, A* conserva completitud y optimalidad porque:

- El factor de ramificación es finito.
- Todos los costos son mayores que cero.
- La heurística es admisible.

Esta combinación resulta útil como primera búsqueda informada y como referencia para comparar heurísticas.

### A* con distancia Manhattan

Esta sería la elección principal.

También conserva completitud y optimalidad, pero Manhattan domina a la cantidad de fichas fuera de lugar.

Por ese motivo, generalmente permite descartar antes los caminos poco prometedores y expandir menos nodos.

Ante dos nodos con igual valor de:

```text
f(n) = g(n) + h(n)
```

se elegiría primero el nodo con menor `h(n)`, siguiendo el criterio indicado en la teórica.

### Greedy con distancia Manhattan

Greedy ordena la frontera utilizando solamente:

```text
f(n) = h(n)
```

La distancia Manhattan sería la heurística más conveniente porque ofrece más información que la cantidad de fichas fuera de lugar.

Greedy puede encontrar rápidamente una solución, pero no considera el costo acumulado del camino. Por lo tanto:

- No garantiza encontrar una solución óptima.
- Puede preferir un estado aparentemente cercano al objetivo al que se llegó mediante un camino muy costoso.

Se utilizaría como comparación para estudiar el intercambio entre velocidad y calidad de la solución.

### Depth-First Search

DFS no utiliza heurística y expande primero los nodos de mayor profundidad.

No sería la elección principal para este problema porque:

- No garantiza encontrar la solución de menor costo.
- Puede recorrer caminos innecesariamente largos.
- Su resultado depende fuertemente del orden de generación de las acciones.

Su principal ventaja es el menor consumo de memoria respecto de BFS.

### IDA* como alternativa

Si el consumo de memoria de A* fuera demasiado elevado, podría utilizarse IDA* con distancia Manhattan.

IDA*:

- Mantiene las garantías de completitud y optimalidad de A* bajo las mismas condiciones.
- Requiere menos memoria.
- Puede expandir repetidamente los mismos estados.

---

## 7. Selección final

Los métodos elegidos serían:

1. **BFS sin heurística**, como referencia desinformada que garantiza la solución óptima.
2. **A\* con fichas fuera de lugar**, para evaluar una primera heurística admisible.
3. **A\* con distancia Manhattan**, como método principal, debido a que Manhattan domina a la primera heurística.
4. **Greedy con distancia Manhattan**, para comparar rapidez contra optimalidad.

DFS no sería utilizado como método principal porque el objetivo es minimizar la cantidad de movimientos y DFS no ofrece esa garantía.

Si existieran restricciones importantes de memoria, se podría agregar IDA* con distancia Manhattan.

---

## 8. Comparación propuesta

Para analizar los métodos se deberían registrar:

- Resultado de la búsqueda.
- Costo de la solución.
- Cantidad de nodos expandidos.
- Cantidad de nodos en frontera al finalizar.
- Máximo tamaño alcanzado por la frontera.
- Tiempo de procesamiento.
- Camino solución.

La comparación principal sería:

| Método | Heurística | ¿Óptimo? | Resultado esperado |
|---|---|---:|---|
| BFS | Ninguna | Sí | Referencia óptima, alto uso de memoria |
| A* | Fichas fuera de lugar | Sí | Menos expansiones que BFS |
| A* | Manhattan | Sí | Mejor búsqueda informada propuesta |
| Greedy | Manhattan | No | Posible solución rápida, no necesariamente mínima |

La hipótesis principal es que A* con distancia Manhattan encontrará una solución óptima expandiendo menos nodos que BFS y que A* con fichas fuera de lugar.
