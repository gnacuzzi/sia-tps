# Implementación de Kohonen

El algoritmo está en `src/sia_tp4/kohonen.py` y el runner experimental en
`src/sia_tp4/kohonen_analysis.py`. La implementación se validó y ya se ejecutó
el primer barrido de tamaños de grilla sobre los creadores. Todavía no se
eligieron los hiperparámetros finales.

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
- el error de cuantización, como distancia media de cada muestra a su BMU;
- la matriz U, calculada como el promedio de distancias a las vecinas
  ortogonales;
- el perfil completo de cada neurona, en z-scores o en unidades originales;
- el mapa de una variable individual para caracterizar regiones.

## Validación sintética

Antes de aplicar la red a los creadores se ejecutó un caso controlado con cuatro
nubes bidimensionales separadas. Las etiquetas verdaderas sólo se usaron para
validar el resultado y nunca entraron al entrenamiento.

La validación asignó las 120 muestras, reprodujo exactamente los pesos con la
misma semilla, obtuvo pureza 1,000 en las neuronas ocupadas y dejó las muestras
del mismo grupo mucho más cerca en la grilla que las de grupos diferentes
(0,645 frente a 2,802). El máximo error entre un centro conocido y su prototipo
más cercano fue 0,153.

El script reproducible está en `scripts/validate_kohonen.py`, su configuración
única en `configs/kohonen/synthetic.json` y el informe con gráficos y CSV en
`docs/kohonen-sintetico/README.md`. Estas métricas aprovechan etiquetas conocidas
y no se trasladan como criterio de selección al dataset real.

## Protocolo experimental

Cada archivo JSON representa exactamente una corrida. El archivo inicial está
en `configs/kohonen/base.json` y contiene un solo valor para cada campo de
`KohonenConfig`: grilla, épocas, tasas, radios, inicialización, mezcla y semilla.
El parser rechaza listas de corridas, campos desconocidos y valores que no
respetan las restricciones del algoritmo.

Durante los experimentos se copia ese archivo para crear configuraciones
separadas. Cada una se ejecuta con su propio `--config`; por
defecto, el nombre de la corrida define un directorio de salida diferente. Así
queda claro qué cambió entre corridas y se evitan productos cartesianos
implícitos de hiperparámetros.

La configuración base usa provisionalmente una grilla `4 x 4`. Las 150 épocas
equivalen a 7500 presentaciones para las 50 filas: toman como referencia
exploratoria las `500 * N` iteraciones de la clase con `N = 15` entradas. El
radio inicial es el diámetro de la grilla, por lo que inicialmente puede
cubrirla completa, y decrece hasta 1. Estos valores son puntos de partida, no
una configuración seleccionada.

Cada corrida individual genera asignaciones, mapa de hits, matriz U,
prototipos, mapas de las 15 variables, grilla con identificadores y followers
promedio usados sólo como baseline. También crea un resumen y un informe
individual.

El primer barrido está definido por 35 JSON en `configs/kohonen/01-grid/`:
`k = 2, ..., 8`, con semillas `0, ..., 4`. Las épocas, las tasas, el radio final,
la inicialización y la mezcla se mantienen fijos. El radio inicial es el
diámetro de cada grilla; cambia sólo para conservar la misma cobertura relativa
al comparar tamaños distintos.

`scripts/experiment_kohonen_grid.py` ejecuta esos archivos y genera el informe
en `docs/kohonen-experimentos/01-grid/README.md`. Se grafican por separado el
error de cuantización, las unidades muertas, la concentración máxima de hits y
las unidades ocupadas, usando media y desvío de las cinco semillas.

Para poder inspeccionar las grillas sin mostrar 35 figuras casi iguales, el
informe incluye un mapa de hits y una matriz U por cada `k`. La corrida mostrada
es la que queda más cerca de la mediana del error de cuantización entre sus
cinco semillas; la elección es reproducible y evita privilegiar siempre una
semilla fija. Los valores numéricos de esas matrices también quedan en CSV.

El error de cuantización no se presentó en clase como regla única de selección.
Se incorpora como métrica auxiliar porque usa la misma distancia entre muestra
y BMU del algoritmo. Siempre disminuye al representar los datos con más
neuronas, por lo que se interpreta junto con la ocupación y los mapas, sin crear
un puntaje combinado arbitrario.

La primera corrida deja `k = 4` y `k = 5` como candidatos. En `k = 6`, las
unidades muertas suben de 9,6 % a 21,1 %, mientras que la concentración máxima
sólo baja de 8,8 % a 8,4 %. Los tamaños mayores continúan reduciendo el error de
cuantización, pero fragmentan más el mapa. Los dos candidatos deben continuar a
la comparación de los demás hiperparámetros antes de fijar uno.

### Interacción entre tamaño y tasa constante

El segundo experimento vuelve a incluir `k = 2, ..., 8` y cruza cada tamaño con
`η = 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9`, siempre constante, usando cinco
semillas. Son 245 corridas independientes definidas en
`configs/kohonen/02-rate/`. Esto permite verificar que un tamaño descartado en
el primer barrido no mejore sustancialmente con otra tasa.

Las tasas altas aumentaron el error y la dispersión. La región útil quedó entre
`0.05` y `0.2`, pero depende del tamaño. Para comparar tasa constante contra
decreciente se conservan tres puntos: `k = 4, η = 0.05`; `k = 5, η = 0.2`; y
`k = 6, η = 0.1`. El tercero funciona como control de la interacción y no como
ganador. Los resultados y gráficos están en
`docs/kohonen-experimentos/02-rate/README.md`.

### Tasa constante frente a decreciente

La comparación completa repite los 245 pares de tamaño, tasa inicial y semilla
con disminución lineal hasta el 10 % de la tasa inicial, y los enfrenta con las
245 tasas constantes. Se reentrenan las 490 redes mediante el mismo pipeline;
el radio decrece de igual manera en ambos grupos.

El descenso hace aprovechables tasas iniciales altas en `k = 4` y `k = 5`, pero
no gana para toda la grilla. En `k = 6`, las tasas constantes bajas conservan
mejor ocupación. Para los siguientes experimentos se mantienen regiones de
`k = 4`, `k = 5` y `k = 6` con alternativas constantes y decrecientes, sin fijar
todavía un único schedule. La evidencia completa está en
`docs/kohonen-experimentos/03-rate-schedule/README.md`.

La elección no se hace con una única cifra: se analizan ocupación, unidades
muertas, matriz U, mapas por variable, estabilidad e interpretación comercial.
