# Análisis de los QR con una red de Hopfield

## Representación

Los cinco códigos QR se cargan como matrices binarias de 37×37 y se convierten
a representación bipolar: negro `+1` y blanco `-1`. Cada patrón se aplana en
un vector de 1369 componentes. La carga es determinista y no usa semillas ni
hiperparámetros.

## Matriz de Gram

Se calcula `G = XXᵀ` usando enteros de 64 bits y se normaliza dividiendo por
1369. La diagonal normalizada vale 1 porque cada patrón se compara consigo
mismo. Para patrones bipolares, un valor fuera de la diagonal cercano a cero
indicaría mayor cercanía a la ortogonalidad.

- producto interno normalizado medio fuera de la diagonal: **0.670**;
- mínimo absoluto: **0.576** en
  **beverages–wine**, **dessert–wine**;
- máximo absoluto: **0.860** entre
  **food** y **wine**.

![Matriz de Gram normalizada](generated/matriz_gram_normalizada.svg)

Todos los productos internos son positivos y altos. Por lo tanto, estos QR no
son cercanos a ser ortogonales: comparten muchos más píxeles con el mismo signo
que con signo opuesto. Esta observación anticipa interferencia al almacenar los
cinco, pero por sí sola no demuestra el comportamiento dinámico; la prueba de
recuperación que aparece debajo comprueba esa hipótesis.

## Balance de píxeles

- **beverages**: 710 negros, 659 blancos (51.9 % negros).
- **dessert**: 720 negros, 649 blancos (52.6 % negros).
- **food**: 710 negros, 659 blancos (51.9 % negros).
- **specials**: 704 negros, 665 blancos (51.4 % negros).
- **wine**: 706 negros, 663 blancos (51.6 % negros).

El balance es cercano a 50 % en los cinco casos. Por eso, los productos
internos altos no se explican simplemente por una mayoría global de blanco o
negro, sino por posiciones coincidentes entre los QR, incluidas sus estructuras
compartidas. No se modifica ni se centra la entrada: se conserva la
codificación bipolar pedida para Hopfield.

## Recuperación sin ruido

Se entrenó una única red mediante la regla de Hebb almacenando simultáneamente
los cinco vectores. Luego cada QR original se usó como consulta, sin modificar
ningún píxel. El límite de 100 iteraciones funciona sólo como
resguardo y no fue alcanzado. Esta prueba tampoco usa semillas.

- patrones que permanecieron estables: **0/5**;
- recuperaciones correctas: **0/5**;
- estados finales distintos encontrados: **1**;
- destinos: **beverages→espurio**, **dessert→espurio**, **food→espurio**, **specials→espurio**, **wine→espurio**.

La primera actualización lleva cada consulta al estado espurio y la segunda
confirma que ese estado no cambia; por eso todas las filas registran dos
iteraciones.

- **beverages**: `spurious_fixed`, 2 iteraciones, 225 píxeles de error, energía -1789.7 → -2172.9.
- **dessert**: `spurious_fixed`, 2 iteraciones, 225 píxeles de error, energía -1801.8 → -2172.9.
- **food**: `spurious_fixed`, 2 iteraciones, 71 píxeles de error, energía -2077.4 → -2172.9.
- **specials**: `spurious_fixed`, 2 iteraciones, 103 píxeles de error, energía -1982.5 → -2172.9.
- **wine**: `spurious_fixed`, 2 iteraciones, 91 píxeles de error, energía -2041.5 → -2172.9.

En los cinco recorridos la energía disminuye durante la primera actualización
y permanece constante al confirmar el punto fijo. Esta observación describe
estas trayectorias concretas; no se usa para cambiar la regla de actualización.

![Recuperación sin ruido](generated/recuperacion_sin_ruido.svg)

![Energía sin ruido](generated/energia_sin_ruido.svg)

El chequeo reproducible confirma que ninguno de los QR originales es un punto
fijo cuando se almacenan los cinco juntos. Todos convergen al mismo estado
fijo espurio. Por lo tanto, un barrido de ruido sobre esta red no puede
interpretarse aisladamente como robustez frente a perturbaciones: ya existe
interferencia con 0 % de ruido. El barrido siguiente conserva igualmente los
cinco patrones, como exige la consigna, y esta línea de base se mantiene visible
al interpretar sus resultados.

## Consultas deterioradas

Se comparan tres funciones sugeridas en clase sobre los mismos niveles y
semillas:

1. **Flip uniforme:** se eligen exactamente `round(nivel × 1369)` posiciones
   sin reemplazo y se invierte su signo.
2. **Borrado uniforme:** se elige la misma cantidad de posiciones de manera
   uniforme y se las lleva a blanco (`-1`). Un píxel ya blanco no cambia.
3. **Borrado hacia bordes:** también lleva las posiciones elegidas a blanco,
   pero su peso de selección decrece linealmente con la distancia al borde. En
   una imagen 37×37 el peso sin normalizar es `19 - distancia_al_borde`.

El diseño usa niveles de 0 %, 5 %, 10 %, 15 %, 20 %, 25 %, 30 %, 35 %, 40 %, 45 %, 50 %, 55 %, 60 %, 65 %, 70 %, 75 %, 80 %, 85 %, 90 %, 95 %, 100 %,
semillas 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 y un máximo de
100 iteraciones como resguardo. El 0 % corresponde a las
cinco consultas deterministas ya analizadas. Cada uno de los
19 niveles entre 5 % y 95 % evalúa los cinco QR con
10 semillas. El 100 % se ejecuta una vez porque deteriorar
todos los píxeles produce la misma consulta con cualquier semilla. En total se
ejecutan 2865 consultas con ruido más las cinco consultas base.

El nivel es **nominal**: indica cuántas posiciones se seleccionan. En el flip
todas cambian; en los borrados sólo cambian efectivamente las que eran negras.
Por eso también se informa el error real de entrada y no se supone que tres
ruidos con el mismo porcentaje nominal tengan igual cantidad de alteraciones.

Para las curvas se calcula primero cada métrica sobre los cinco QR de una
semilla y después se muestran media y desvío estándar entre semillas. Las
barras están acotadas al rango válido de cada métrica. No se conectan semillas
como si fueran una secuencia temporal y no se elige un nivel de ruido óptimo.
Cuando el desvío es cero, la barra coincide con el marcador y no tiene longitud
visible; esto ocurre en toda la curva de recuperación.

- **Flip uniforme**: a 100 % nominal modifica en promedio 1369.0 píxeles; recuperación 0.0 %, error final 1226.0 píxeles. No hubo recuperaciones correctas en ningún nivel positivo.
- **Borrado uniforme**: a 100 % nominal modifica en promedio 710.0 píxeles; recuperación 0.0 %, error final 1226.0 píxeles. No hubo recuperaciones correctas en ningún nivel positivo.
- **Borrado hacia bordes**: a 100 % nominal modifica en promedio 710.0 píxeles; recuperación 0.0 %, error final 1226.0 píxeles. No hubo recuperaciones correctas en ningún nivel positivo.

En **2304** corridas el destino fue el mismo espurio de
0 %; en **561** fue su inverso y en
**0** apareció otro destino. Con flip, el inverso pasa de
84 % al 50 % a
100 % al 55 %, y conserva ese
dominio hasta el 100 %. Con borrado, la transición aparece mucho más tarde: al
95 % el inverso representa
52 % de las corridas uniformes
y 56 % de las sesgadas hacia los
bordes; al 100 % alcanza
100 % en ambos casos. Esto
explica el salto del error final sin confundirlo con una recuperación correcta.

![Recuperación frente al ruido](generated/recuperacion_vs_ruido.svg)

![Error final frente al ruido](generated/error_final_vs_ruido.svg)

![Daño efectivo de entrada](generated/error_entrada_vs_ruido.svg)

![Tipos de resultado](generated/resultados_vs_ruido.svg)

Se observaron **0** corridas con algún aumento de energía entre
las 2865 evaluadas. Las trayectorias completas quedan en CSV para
no seleccionar conclusiones a partir de un único ejemplo.

## Comparación visual controlada

Para mostrar cómo actúa cada función sin elegir casos por su resultado, se usa
la primera etiqueta en el orden del dataset (**beverages**), el
nivel positivo central configurado (**50 %**)
y la menor semilla (**0**). La selección se fija antes de
mirar el estado final.

![Consultas con ruido controlado](generated/consultas_ruido_controlado.svg)

![Energía de los casos controlados](generated/energia_ruido_controlado.svg)

La recuperación base es 0/5 y ninguna consulta deteriorada recuperó su objetivo:
en esta red, agregar ruido no corrige la interferencia que ya existe con los
cinco patrones originales. Por eso, la robustez sólo se evalúa después sobre
subconjuntos que sí son memorias estables sin ruido.

## Limitaciones observadas e hipótesis de capacidad

### Qué muestran los resultados actuales

La red se entrenó con los cinco QR simultáneamente, tal como pide la consigna.
Sin embargo, ninguno de los cinco patrones originales permanece estable al ser
usado como consulta sin ruido: todos convergen al mismo punto fijo espurio.
Además, ninguna de las 2865 consultas deterioradas recupera su QR
objetivo. La mayor parte converge al mismo espurio encontrado sin ruido y, en
los casos restantes, a su inverso.

Este comportamiento es coherente con la interferencia anticipada por la matriz
de Gram. Los productos internos normalizados fuera de la diagonal están entre
0.576 y 0.860, con
media 0.670: los QR comparten una fracción
importante de sus píxeles y, por lo tanto, no constituyen patrones cercanos a
la ortogonalidad. Al sumar sus productos externos, la regla de Hebb introduce
términos cruzados que modifican los mínimos de energía de la red.

En las trayectorias evaluadas, la energía descendió o se mantuvo; este resultado
describe las consultas observadas y no se extrapola como garantía para cualquier
estado inicial bajo actualización sincrónica. Tampoco garantiza que el estado
alcanzado sea una memoria deseada. En este caso, el punto fijo espurio es un
atractor válido de la dinámica, aunque no coincida con ninguno de los cinco QR
almacenados.

### Capacidad teórica frente a capacidad efectiva

En clase se presentó como referencia una capacidad del orden de 0.15 N
patrones para una red de N neuronas. Para estos QR, N = 1369, lo que
daría aproximadamente 205 patrones bajo
condiciones favorables. Ese valor es una guía teórica, no una garantía
independiente de los datos: supone patrones suficientemente separados y no
contempla por sí solo la fuerte correlación observada aquí.

Por eso, que la red falle con solo cinco patrones no contradice esa referencia.
Este experimento principal sólo permite afirmar que la capacidad efectiva es
menor que cinco para este conjunto y esta regla de Hebb; por sí solo no indica
cuántos QR sí pueden almacenarse. El barrido exhaustivo de la sección siguiente
responde esa pregunta sin extrapolarla a otras familias de patrones.

### Contraste de capacidad sin ruido

El [barrido exhaustivo de capacidad](../hopfield-qr-capacidad/README.md)
entrena una red para cada uno de los 31 subconjuntos no vacíos de los cinco QR.
Los cinco patrones individuales y los diez pares son completamente estables;
en cambio, ninguna combinación de tres, cuatro o cinco conserva uno de sus
originales como punto fijo. La capacidad efectiva observada es entonces de dos
QR para esta familia, codificación, regla de aprendizaje y dinámica, no una
capacidad universal de una red de 1369 neuronas.

El par más correlacionado también es estable, por lo que el peor producto
interno aislado no explica el corte. Los 15 subconjuntos válidos se evalúan en
el [barrido de robustez](../hopfield-qr-robustez/README.md), sin confundir el
efecto del ruido con una falla que ya existe en la memoria sin perturbar.

## Archivos generados

- `generated/matriz_gram.csv`: productos internos crudos;
- `generated/matriz_gram_normalizada.csv`: productos internos divididos por 1369;
- `generated/similitud_pares.csv`: los diez pares y sus píxeles diferentes;
- `generated/balance_pixeles.csv`: cantidad de módulos negros y blancos;
- `generated/recuperacion_sin_ruido.csv`: resultado de cada consulta original;
- `generated/energia_sin_ruido.csv`: energía y error en cada iteración;
- `generated/corridas_ruido.csv`: una fila por consulta deteriorada;
- `generated/energia_corridas_ruido.csv`: energía de todas las iteraciones;
- `generated/resumen_ruido.csv`: medias y desvíos entre semillas;
- `generated/resumen_ruido_por_qr.csv`: agregados sin ocultar diferencias entre QR;
- `generated/casos_ruido_controlado.csv`: casos de la comparación visual;
- `generated/matriz_gram_normalizada.svg`: visualización del Gram;
- `generated/recuperacion_sin_ruido.svg`: objetivos, consultas y estados finales;
- `generated/energia_sin_ruido.svg`: trayectorias de energía;
- cuatro gráficos agregados para comparar ruido;
- `generated/consultas_ruido_controlado.svg` y
  `generated/energia_ruido_controlado.svg`: casos fijados sin selección por
  resultado.
