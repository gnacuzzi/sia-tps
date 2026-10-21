# Análisis inicial de los QR para Hopfield

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
interferencia con 0 % de ruido. El paso siguiente debe separar este efecto
estudiando subconjuntos de uno a cinco patrones.

## Archivos generados

- `generated/matriz_gram.csv`: productos internos crudos;
- `generated/matriz_gram_normalizada.csv`: productos internos divididos por 1369;
- `generated/similitud_pares.csv`: los diez pares y sus píxeles diferentes;
- `generated/balance_pixeles.csv`: cantidad de módulos negros y blancos;
- `generated/recuperacion_sin_ruido.csv`: resultado de cada consulta original;
- `generated/energia_sin_ruido.csv`: energía y error en cada iteración;
- `generated/matriz_gram_normalizada.svg`: visualización del Gram;
- `generated/recuperacion_sin_ruido.svg`: objetivos, consultas y estados finales;
- `generated/energia_sin_ruido.svg`: trayectorias de energía.
