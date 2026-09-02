# Métodos de selección usados en el TP2

Este documento explica los métodos de selección disponibles en el TP2 de algoritmos genéticos. En la implementación se usan en dos momentos distintos:

1. `genetic.parent_selection`: selección de padres que van a cruzarse para generar hijos.
2. `genetic.survival.selection`: selección de individuos que pasan a la siguiente generación, aplicada dentro de una estrategia de supervivencia.

Ambos lugares comparten el mismo contrato de configuración:

```json
{
  "method": "nombre_del_metodo",
  "params": {}
}
```

El algoritmo siempre selecciona individuos maximizando `fitness`. Para este TP el error principal es `NMSE` y el fitness se calcula como:

```text
fitness = max(epsilon, 1 - NMSE)
```

Por lo tanto, un individuo con menor error tiene mayor fitness y es considerado mejor por los métodos de selección.

## Métodos disponibles

Los métodos implementados son:

| Método | Parámetros |
| --- | --- |
| `elite` | `{}` |
| `roulette` | `{}` |
| `universal` | `{}` |
| `ranking` | `{}` |
| `boltzmann` | `initial_temperature`, `final_temperature`, `decay_rate` |
| `tournament_deterministic` | `tournament_size` |
| `tournament_probabilistic` | `threshold` |

Todos los métodos devuelven exactamente la cantidad de individuos solicitada. Salvo la selección elite cuando se usa para supervivencia con un pedido menor o igual al tamaño del conjunto, los métodos pueden seleccionar al mismo individuo más de una vez. Esto es especialmente importante en selección de padres: un buen individuo puede participar en más de una cruza.

## Presión selectiva

La presión selectiva indica cuánto favorece el método a los individuos más aptos frente al resto de la población.

Una presión alta acelera la explotación de las mejores soluciones encontradas. El riesgo es perder diversidad demasiado temprano y caer en convergencia prematura. Una presión baja conserva más diversidad y exploración, pero puede hacer que la evolución avance más lento porque individuos mediocres siguen teniendo muchas oportunidades.

Los métodos del TP permiten regular este equilibrio:

- `elite` tiene presión muy alta.
- `roulette` depende directamente de la diferencia de fitness.
- `universal` es parecido a `roulette`, pero con menor varianza en la muestra.
- `ranking` controla la presión usando posiciones en el ranking en lugar de fitness crudo.
- `boltzmann` cambia la presión con la generación mediante una temperatura.
- `tournament_deterministic` aumenta su presión al aumentar `tournament_size`.
- `tournament_probabilistic` aumenta su presión al acercar `threshold` a `1`.

## Elite

La selección elite ordena todos los individuos por fitness descendente y toma primero a los mejores.

Pasos:

1. Calcula el fitness de cada individuo.
2. Ordena la población desde el mayor fitness hasta el menor.
3. Devuelve los primeros `count` individuos del ranking.
4. Si `count` es mayor que el tamaño del conjunto, vuelve a recorrer el ranking desde el mejor.

Ejemplo conceptual:

```text
Población ordenada: A(0.95), B(0.80), C(0.40)
count = 2
Seleccionados: A, B
```

Si se pidieran cinco individuos:

```text
Seleccionados: A, B, C, A, B
```

Ventajas:

- Conserva de forma directa las mejores soluciones.
- Es determinística y fácil de interpretar.
- Funciona muy bien en supervivencia aditiva, porque evita perder buenos individuos ya encontrados.

Desventajas:

- Tiene presión selectiva alta.
- Si se usa como selección de padres puede reducir mucho la diversidad.
- No permite que individuos de menor fitness participen salvo que se pidan más individuos que el tamaño del conjunto.

Uso típico en este TP:

```json
{
  "method": "elite",
  "params": {}
}
```

En la configuración por defecto se usa en `genetic.survival.selection`, junto con supervivencia aditiva. Eso significa que la próxima generación se forma eligiendo los mejores individuos entre padres e hijos.

## Ruleta

La selección por ruleta elige individuos con probabilidad proporcional a su fitness. Cada individuo ocupa una porción de una ruleta cuyo tamaño depende de su fitness.

La probabilidad de seleccionar al individuo `i` es:

```text
P(i) = fitness(i) / suma_fitness
```

Pasos:

1. Calcula el fitness de todos los individuos.
2. Suma todos los fitness.
3. Asigna a cada individuo una probabilidad proporcional a su fitness.
4. Para cada selección, genera un número aleatorio y elige el individuo cuyo intervalo contiene ese número.
5. Repite el proceso hasta obtener `count` individuos.

Ejemplo conceptual:

```text
A fitness = 0.50
B fitness = 0.30
C fitness = 0.20
Suma = 1.00

P(A) = 50 %
P(B) = 30 %
P(C) = 20 %
```

Ventajas:

- Mantiene una posibilidad de selección para individuos no óptimos.
- Introduce azar, lo que ayuda a conservar diversidad.
- Es simple y se relaciona directamente con la aptitud medida.

Desventajas:

- Si un individuo tiene fitness mucho mayor que el resto, puede dominar la selección.
- Si los fitness son muy parecidos, la selección se vuelve casi aleatoria.
- Es sensible a la escala del fitness crudo.

Uso típico:

```json
{
  "method": "roulette",
  "params": {}
}
```

En este TP puede usarse tanto para padres como para supervivencia. Es una buena opción cuando se quiere una presión intermedia y se confía en que la escala del fitness representa bien la diferencia entre individuos.

## Universal

La selección universal estocástica usa la misma idea de pesos proporcionales al fitness que la ruleta, pero reduce la varianza de la muestra.

En lugar de girar la ruleta de forma independiente `count` veces, se elige un único punto inicial aleatorio y luego se usan punteros equidistantes.

Pasos:

1. Calcula los pesos usando el fitness de cada individuo.
2. Suma todos los pesos.
3. Divide la suma total por `count` para obtener la distancia entre punteros.
4. Elige un inicio aleatorio dentro del primer intervalo.
5. Coloca `count` punteros separados por la misma distancia.
6. Selecciona los individuos alcanzados por esos punteros.

Fórmula del paso:

```text
step = suma_fitness / count
```

Ejemplo conceptual:

```text
suma_fitness = 1.00
count = 4
step = 0.25
inicio = 0.10
punteros = 0.10, 0.35, 0.60, 0.85
```

Ventajas:

- Mantiene probabilidades proporcionales al fitness.
- Produce muestras más estables que la ruleta común.
- Evita que el azar de muchas tiradas independientes distorsione demasiado la proporción esperada.

Desventajas:

- Sigue dependiendo de la escala del fitness.
- Si un individuo tiene mucho fitness, también puede dominar.
- Es menos simple de explicar que la ruleta, aunque el comportamiento esperado es parecido.

Uso típico:

```json
{
  "method": "universal",
  "params": {}
}
```

Conviene cuando se quiere conservar el criterio proporcional de la ruleta, pero con una distribución más regular de los seleccionados.

## Ranking

La selección por ranking no usa el valor bruto de fitness como peso. Primero ordena los individuos por fitness y luego asigna pesos según la posición en ese ranking.

En la implementación del TP, si hay `N` individuos, el mejor recibe peso `N - 1`, el siguiente `N - 2`, y así sucesivamente hasta que el último recibe `0`. Luego se aplica una selección tipo ruleta sobre esos pesos.

Pasos:

1. Ordena la población de mayor a menor fitness.
2. Asigna pesos por posición, no por valor absoluto de fitness.
3. Aplica selección ponderada con esos pesos.
4. Devuelve `count` individuos.

Ejemplo conceptual con cuatro individuos:

```text
Ranking: A, B, C, D
Pesos:   3, 2, 1, 0
```

En ese caso `A` tiene más probabilidad que `B`, `B` más que `C`, y `D` no queda seleccionado por peso propio.

Ventajas:

- Reduce la sensibilidad a escalas raras del fitness.
- Evita que un individuo extremadamente bueno domine sólo por una diferencia numérica grande.
- Es útil cuando importa más el orden relativo que la magnitud exacta de la diferencia.

Desventajas:

- Pierde información sobre cuánto mejor es un individuo respecto de otro.
- En esta implementación el último del ranking recibe peso `0`.
- Puede ser demasiado conservador si las diferencias reales de fitness son informativas.

Uso típico:

```json
{
  "method": "ranking",
  "params": {}
}
```

Conviene cuando las diferencias absolutas de fitness pueden ser engañosas o cuando se busca una presión más controlada que en ruleta.

## Boltzmann

La selección de Boltzmann transforma el fitness usando una temperatura `T`. Después aplica selección ponderada sobre los valores transformados.

La idea es controlar la presión selectiva a lo largo de las generaciones. Con temperatura alta, las diferencias de fitness pesan menos y hay más exploración. Con temperatura baja, las diferencias se amplifican y la selección favorece más a los mejores individuos.

La temperatura por generación se calcula como:

```text
T(g) = final_temperature + (initial_temperature - final_temperature) * exp(-decay_rate * g)
```

Donde:

- `g` es la generación actual.
- `initial_temperature` es la temperatura inicial.
- `final_temperature` es la temperatura mínima final.
- `decay_rate` controla qué tan rápido baja la temperatura.

Los pesos se calculan de manera estable como:

```text
peso(i) = exp((fitness(i) - max_fitness) / T)
```

Restar `max_fitness` no cambia la relación relativa de probabilidades y evita problemas numéricos por valores exponenciales demasiado grandes.

Pasos:

1. Calcula la temperatura de la generación actual.
2. Obtiene el fitness de todos los individuos.
3. Calcula pesos exponenciales escalados por temperatura.
4. Aplica selección ponderada sobre esos pesos.
5. Devuelve `count` individuos.

Ejemplo de configuración:

```json
{
  "method": "boltzmann",
  "params": {
    "initial_temperature": 1.0,
    "final_temperature": 0.1,
    "decay_rate": 0.2
  }
}
```

Validaciones:

- `initial_temperature` debe ser mayor que `0`.
- `final_temperature` debe ser mayor que `0`.
- `initial_temperature` debe ser mayor o igual que `final_temperature`.
- `decay_rate` debe ser mayor o igual que `0`.

Ventajas:

- Permite más exploración al principio y más explotación al final.
- La presión selectiva cambia gradualmente.
- Es útil cuando se quiere evitar convergencia prematura en generaciones tempranas.

Desventajas:

- Requiere calibrar tres parámetros.
- Una temperatura inicial demasiado baja se comporta como una selección muy agresiva desde el comienzo.
- Una temperatura final demasiado alta puede dejar poca presión selectiva al final.
- Un `decay_rate` mal elegido puede enfriar demasiado rápido o demasiado lento.

En el TP, Boltzmann recibe la generación actual tanto al seleccionar padres como al seleccionar supervivientes. Por eso su comportamiento puede cambiar durante toda la corrida.

## Torneo determinístico

El torneo determinístico elige varios individuos al azar y selecciona el mejor de ese subconjunto.

Pasos:

1. Toma `tournament_size` individuos distintos de la población.
2. Compara sus fitness.
3. Elige el individuo de mayor fitness.
4. Repite el torneo hasta obtener `count` seleccionados.

Ejemplo conceptual:

```text
tournament_size = 3
Participantes: B(0.70), F(0.40), A(0.90)
Ganador: A
```

Parámetro:

- `tournament_size`: cantidad de participantes de cada torneo.

Configuración:

```json
{
  "method": "tournament_deterministic",
  "params": {
    "tournament_size": 3
  }
}
```

Validaciones:

- `tournament_size` debe ser al menos `2`.
- `tournament_size` no puede superar el tamaño del conjunto desde el que se selecciona.

Efecto del parámetro:

- Si `tournament_size` es pequeño, hay más azar y menor presión selectiva.
- Si `tournament_size` es grande, aumenta la probabilidad de que aparezca un individuo muy bueno en cada torneo y sube la presión selectiva.
- Si `tournament_size` fuera igual al tamaño de la población, siempre ganaría el mejor individuo global.

Ventajas:

- No depende de la escala numérica del fitness, sólo de comparaciones.
- Es fácil controlar la presión selectiva con `tournament_size`.
- Es robusto cuando las diferencias absolutas de fitness son pequeñas o difíciles de calibrar.

Desventajas:

- Puede seleccionar muchas veces a los mejores si el torneo es grande.
- Si el torneo es muy chico, puede avanzar lento.
- No usa la magnitud exacta de las diferencias de fitness.

Uso por defecto en este TP:

```json
"parent_selection": {
  "method": "tournament_deterministic",
  "params": {
    "tournament_size": 3
  }
}
```

Esto significa que los padres se eligen mediante torneos de tres individuos. Es una elección razonable para padres porque controla la presión sin depender de la escala exacta del fitness.

## Torneo probabilístico

El torneo probabilístico toma dos individuos al azar. Luego elige el mejor con probabilidad `threshold` y el peor con probabilidad `1 - threshold`.

Pasos:

1. Toma dos individuos distintos de la población.
2. Identifica cuál tiene mayor fitness y cuál menor.
3. Genera un número aleatorio.
4. Si el número cae dentro de `threshold`, selecciona el mejor.
5. Si no, selecciona el peor.
6. Repite hasta obtener `count` individuos.

Ejemplo conceptual:

```text
threshold = 0.75
Participantes: A(0.90), B(0.40)

P(seleccionar A) = 75 %
P(seleccionar B) = 25 %
```

Configuración:

```json
{
  "method": "tournament_probabilistic",
  "params": {
    "threshold": 0.75
  }
}
```

Validaciones:

- `threshold` debe estar entre `0.5` y `1.0`.

Efecto del parámetro:

- `threshold = 0.5`: el torneo no favorece al mejor; equivale a elegir casi al azar entre los dos.
- `threshold = 1.0`: siempre gana el mejor; equivale a un torneo determinístico de tamaño `2`.
- Valores intermedios permiten regular la presión selectiva.

Ventajas:

- Controla la presión con un único parámetro.
- Conserva una probabilidad explícita de seleccionar individuos peores.
- Ayuda a mantener diversidad sin abandonar por completo la preferencia por mejores fitness.

Desventajas:

- Sólo compara de a dos individuos.
- Puede ser más lento que un torneo determinístico grande si se necesita explotación fuerte.
- Igual que otros torneos, no usa la magnitud exacta de la diferencia de fitness.

## Uso en selección de padres

En cada generación, el motor pide `offspring_count` padres. Como las cruzas toman padres de a pares y generan dos hijos, `offspring_count` debe ser par.

Flujo:

1. Se evalúa la población actual.
2. Se seleccionan `offspring_count` padres con `genetic.parent_selection`.
3. Se toman de a pares.
4. Cada par pasa por cruza.
5. Cada hijo pasa por mutación.
6. Los hijos resultantes se evalúan.

La selección de padres puede repetir individuos. Esto permite que un individuo con buen fitness aporte material genético a varios hijos dentro de la misma generación.

Configuración por defecto:

```json
"parent_selection": {
  "method": "tournament_deterministic",
  "params": {
    "tournament_size": 3
  }
}
```

## Uso en supervivencia

Después de crear y evaluar los hijos, el TP forma la siguiente generación con una estrategia de supervivencia. La estrategia define de qué conjunto se selecciona y `survival.selection` define cómo se selecciona.

Hay dos estrategias:

| Estrategia | Conjunto candidato |
| --- | --- |
| `additive` | población actual + hijos |
| `exclusive` | principalmente hijos; si faltan lugares, se completan con padres seleccionados |

## Supervivencia aditiva

En supervivencia aditiva se unen la población actual y los hijos. Luego se seleccionan `population_size` individuos de ese conjunto combinado.

Si `P` es el tamaño de población y `K` es la cantidad de hijos:

```text
candidatos = P padres + K hijos
nueva_generación = seleccionar P candidatos
```

Ventajas:

- Puede conservar individuos muy buenos de generaciones anteriores.
- Reduce el riesgo de perder el mejor individuo por azar.
- Combinada con `elite`, produce una supervivencia fuertemente elitista.

Desventajas:

- Puede reducir el recambio generacional.
- Puede bajar la diversidad si siempre sobreviven los mismos individuos.

Configuración por defecto:

```json
"survival": {
  "strategy": "additive",
  "selection": {
    "method": "elite",
    "params": {}
  }
}
```

Con esta configuración, la siguiente generación queda formada por los mejores individuos entre padres e hijos.

## Supervivencia exclusiva

En supervivencia exclusiva, los hijos tienen prioridad para formar la nueva generación.

Hay dos casos:

1. Si la cantidad de hijos `K` es mayor que `P`, se seleccionan `P` individuos entre los hijos.
2. Si `K` es menor o igual que `P`, entran todos los hijos y se seleccionan `P - K` individuos de la población anterior para completar la generación.

Ejemplo:

```text
P = 100
K = 40

Entran 40 hijos.
Se seleccionan 60 padres para completar.
```

Si en cambio:

```text
P = 100
K = 140

Se seleccionan 100 individuos entre los 140 hijos.
No pasan padres directamente.
```

Ventajas:

- Produce mayor recambio de población.
- Favorece la exploración generacional.
- Evita que una generación vieja domine durante demasiado tiempo.

Desventajas:

- Puede perder buenos individuos previos si no hay elitismo externo.
- Puede ser más inestable que la supervivencia aditiva.
- Depende mucho de la calidad de los hijos generados.

Configuración:

```json
"survival": {
  "strategy": "exclusive",
  "selection": {
    "method": "elite",
    "params": {}
  }
}
```

## Comparación rápida

| Método | Tipo | Presión selectiva | Depende de escala de fitness | Conserva diversidad |
| --- | --- | --- | --- | --- |
| `elite` | Determinístico | Alta | No | Baja |
| `roulette` | Probabilístico proporcional | Variable | Sí | Media |
| `universal` | Probabilístico proporcional estable | Variable | Sí | Media |
| `ranking` | Probabilístico por orden | Media controlada | No | Media |
| `boltzmann` | Probabilístico con temperatura | Cambia con generaciones | Sí, luego de escalar | Media-alta al inicio, menor al final |
| `tournament_deterministic` | Torneo | Regulable por tamaño | No | Depende del tamaño |
| `tournament_probabilistic` | Torneo | Regulable por umbral | No | Depende del umbral |

## Recomendaciones para experimentar

Para selección de padres:

- `tournament_deterministic` con tamaño chico o medio suele ser una base sólida.
- `roulette` o `universal` sirven si el fitness está bien distribuido y se quiere selección proporcional.
- `ranking` es útil si la escala del fitness produce comportamientos raros.
- `boltzmann` conviene si se quiere mucha exploración al principio y más explotación al final.

Para supervivencia:

- `additive` con `elite` conserva mejor las mejores soluciones encontradas.
- `exclusive` aumenta el recambio y puede ayudar cuando la población se estanca.
- Usar métodos probabilísticos en supervivencia puede mantener diversidad, pero aumenta el riesgo de perder individuos buenos.

Una combinación conservadora es:

```json
"parent_selection": {
  "method": "tournament_deterministic",
  "params": {
    "tournament_size": 3
  }
},
"survival": {
  "strategy": "additive",
  "selection": {
    "method": "elite",
    "params": {}
  }
}
```

Esta es la configuración por defecto del TP: padres por torneo determinístico y supervivencia aditiva elitista.
