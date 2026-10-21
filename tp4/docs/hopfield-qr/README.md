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
que con signo opuesto. Esto anticipa interferencia al almacenar los cinco,
pero todavía no demuestra qué ocurrirá durante la recuperación; eso debe
comprobarse entrenando la red en la etapa siguiente.

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

## Archivos generados

- `generated/matriz_gram.csv`: productos internos crudos;
- `generated/matriz_gram_normalizada.csv`: productos internos divididos por 1369;
- `generated/similitud_pares.csv`: los diez pares y sus píxeles diferentes;
- `generated/balance_pixeles.csv`: cantidad de módulos negros y blancos;
- `generated/matriz_gram_normalizada.svg`: visualización única del Gram.
