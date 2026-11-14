# Dataset de funciones Python

Esta especificación fue discutida y aprobada para el ejercicio 2. Define el
dataset y su representación, pero no selecciona la arquitectura ni los
hiperparámetros del VAE final.

## Familia de programas

Se usa código Python sintético de longitud fija para conservar una evaluación
objetiva y un costo experimental compatible con un VAE implementado desde
cero. Cada fuente tiene la forma:

```python
def f(x): return (x+47)*83
```

Los dos operadores pertenecen a `{+, -, *}` y las dos constantes enteras a
`{0, ..., 99}`. Las constantes se alinean a dos columnas; por ejemplo, el 3 se
escribe como ` 3`. Por eso todas las fuentes tienen 26 caracteres y existen:

```text
3 * 100 * 3 * 100 = 90.000 fuentes posibles
```

El dataset versionado toma 2.700 fuentes sin repetición y las divide de manera
reproducible, con semilla 17, en:

| Split | Total | Casos por cada par de operadores |
|---|---:|---:|
| entrenamiento | 1.800 | 200 |
| validación | 450 | 50 |
| test | 450 | 50 |

Así, los nueve pares de operadores están igualmente representados en cada
split. El CSV se encuentra en `data/python_functions.csv`.

## Representación one-hot

El vocabulario contiene los 25 caracteres que pueden aparecer en la plantilla.
Cada una de las 26 posiciones se codifica con un vector one-hot de longitud 25:

```text
26 posiciones * 25 caracteres = 650 valores
```

El decoder no interpreta esos 650 valores como bits independientes. Los separa
en 26 grupos de 25 logits y aplica un softmax independiente en cada posición.
La reconstrucción usa la suma de las 26 entropías cruzadas categóricas,
promediada sobre las muestras del lote. En consecuencia, cada posición elige
exactamente un carácter al decodificar por `argmax`.

Esta representación evita el problema de la codificación binaria ASCII: aquí
la distancia entre los bits de dos caracteres no introduce una semejanza
arbitraria. El modelo aprende directamente qué carácter es probable en cada
posición. El costo es aumentar la salida de 182 bits ASCII a 650 valores, pero
la estructura de la pérdida coincide con la estructura real del dato.

El round-trip `fuente -> one-hot -> fuente` es exacto. El softmax agrupado, la
entropía cruzada y su integración con el AE y el VAE fueron comprobados mediante
diferencias finitas, incluyendo el término KL del VAE.

## Criterios de evaluación

Las muestras generadas se juzgan con propiedades separadas:

1. **Compila:** `compile(source, ..., "exec")` acepta la fuente sin ejecutarla.
2. **Pertenece a la gramática:** el AST satisface exactamente la plantilla,
   operadores y rango de constantes aprobados.
3. **Es nueva sintácticamente:** no coincide con ninguna de las 2.700 fuentes
   guardadas en entrenamiento, validación o test.
4. **Es nueva semánticamente:** su firma exacta `(a, b)`, donde
   `f(x) = a*x + b`, no aparece en ninguno de los tres splits.

La novedad semántica es más exigente que comparar texto: expresiones diferentes
pueden representar la misma función. El dataset guardado contiene 1.786 firmas
semánticas distintas. El universo completo contiene 41.524, por lo que quedan
39.738 comportamientos posibles fuera de todos los splits.

Compilar no basta para pertenecer a la familia; por ejemplo, una función con
división puede ser Python válido y quedar fuera de la gramática. Del mismo modo,
una fuente puede ser sintácticamente nueva pero calcular una función ya vista.
