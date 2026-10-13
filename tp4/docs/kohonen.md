# Implementación de Kohonen

El algoritmo está en `src/sia_tp4/kohonen.py` y el runner experimental en
`src/sia_tp4/kohonen_analysis.py`. Quedaron preparados, pero no se ejecutaron
todavía sobre los creadores ni se eligieron hiperparámetros finales.

## Entradas y entrenamiento

- Usa las 15 variables de `BASE_FEATURES`, incluida
  `total_engagement_per_1000`.
- Ajusta z-score una sola vez y conserva las medias y los desvíos para asignar
  datos futuros con la misma transformación.
- Construye una grilla rectangular `k x k`; cada neurona tiene un vector de 15
  pesos.
- Inicializa por defecto con muestras del dataset para reducir el riesgo de
  unidades muertas. También permite una inicialización uniforme entre el
  mínimo y el máximo observados de cada variable para poder compararlas.
- Para cada muestra busca la BMU por distancia euclídea y actualiza su
  vecindario con `w <- w + eta * (x - w)`.
- La tasa y el radio usan un descenso lineal entre los valores inicial y final.
  Dejarlos iguales permite ensayar un valor constante.
- El orden puede mezclarse en cada época y queda reproducible mediante una
  semilla.

## Convención del radio

Las diapositivas escriben el vecindario con distancia estrictamente menor que
el radio, pero la explicación oral dice que en una grilla rectangular `R = 1`
incluye las cuatro vecinas pegadas. La implementación usa distancia menor o
igual que el radio. Por lo tanto, con `R = 1` se actualizan la ganadora y sus
vecinas ortogonales existentes; las diagonales quedan afuera.

Esta convención queda cubierta por una prueba unitaria pequeña para una grilla
`3 x 3`.

## Resultados disponibles

El modelo permite obtener, sin volver a entrenarlo:

- la BMU de una muestra;
- las coordenadas asignadas a todas las muestras;
- el mapa de hits u ocupación;
- la matriz U, calculada como el promedio de distancias a las vecinas
  ortogonales;
- el perfil completo de cada neurona, en z-scores o en unidades originales;
- el mapa de una variable individual para caracterizar regiones.

## Decisiones que siguen abiertas

Cada archivo JSON representa exactamente una corrida. El archivo inicial está
en `configs/kohonen/base.json` y contiene un solo valor para cada campo de
`KohonenConfig`: grilla, épocas, tasas, radios, inicialización, mezcla y semilla.
El parser rechaza listas de corridas, campos desconocidos y valores que no
respetan las restricciones del algoritmo.

Durante los experimentos se copiará ese archivo para crear configuraciones
separadas. Por ejemplo, `01-grid-3.json` y `02-grid-4.json` diferirán sólo en la
grilla y el radio inicial. Cada una se ejecutará con su propio `--config`; por
defecto, el nombre de la corrida define un directorio de salida diferente. Así
queda claro qué cambió entre corridas y se evitan productos cartesianos
implícitos de hiperparámetros.

La configuración base usa provisionalmente una grilla `4 x 4`. Las 150 épocas
equivalen a 7500 presentaciones para las 50 filas: toman como referencia
exploratoria las `500 * N` iteraciones de la clase con `N = 15` entradas. El
radio inicial es el diámetro de la grilla, por lo que inicialmente puede
cubrirla completa, y decrece hasta 1. Estos valores son puntos de partida, no
una configuración seleccionada.

Cada corrida genera asignaciones, mapa de hits, matriz U, prototipos, mapas de
las 15 variables, grilla con identificadores y followers promedio usados sólo
como baseline. También crea un resumen y un informe individual. La comparación
entre informes se implementará al comenzar el experimento, cuando estén
definidas las variantes concretas.

La elección no se hará con una única cifra: se analizarán ocupación, unidades
muertas, matriz U, mapas por variable, estabilidad e interpretación comercial.
