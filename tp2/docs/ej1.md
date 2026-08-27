# Ejercicio 1 - Representación de una imagen mediante caracteres ASCII

## 1. ¿Qué representa un individuo?

Un individuo representa una solución candidata completa: un mapa de `N x N` caracteres ASCII. La población contiene muchos mapas completos diferentes y cada uno puede transformarse en una imagen para evaluar qué tan bien aproxima el objetivo.

Los padres son dos individuos, es decir, dos mapas completos. Se eligen mediante el método de selección configurado a partir de su fitness y del componente aleatorio del método. No tienen que ser mapas "cercanos" entre sí ni ocupar posiciones vecinas dentro de la población.

## 2. ¿Cuál es el genotipo del individuo?

El genotipo es la matriz de `N x N` caracteres que codifica la solución. Para aplicar algunos operadores también puede interpretarse como una secuencia ordenada de `N²` posiciones, siempre conservando una conversión inequívoca entre el índice de la secuencia y las coordenadas `(fila, columna)`.

## 3. ¿Qué es un gen y qué valores puede tomar su alelo?

Cada celda es un gen y el carácter almacenado en ella es su alelo. El gen ubicado en el locus `(fila, columna)` representa qué carácter se muestra en esa celda. Su alelo puede ser cualquiera de los caracteres de un alfabeto ASCII definido previamente, por ejemplo:

```text
[" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]
```

El alfabeto exacto forma parte del diseño: puede ordenarse aproximadamente desde caracteres de menor densidad visual hasta caracteres de mayor densidad. Todo alelo elegido de ese conjunto produce una celda válida.

## 4. ¿Cuál es el fenotipo?

El fenotipo es la imagen que vemos al renderizar el mapa ASCII. Se obtiene acomodando los caracteres del genotipo en sus filas y columnas y dibujándolos con una configuración visual concreta, como fuente monoespaciada, tamaño, espaciado, color del texto y color de fondo.

El genotipo contiene los símbolos y sus posiciones; el fenotipo contiene su apariencia observable. Dos configuraciones de renderizado diferentes podrían producir imágenes distintas a partir del mismo mapa de caracteres, por lo que estos parámetros visuales deben permanecer fijos durante una ejecución.

## 5. ¿Cómo generamos la población inicial?

Generamos aleatoriamente una población de `P` individuos. Para cada individuo creamos un mapa de `N x N` y elegimos de manera independiente el alelo de cada celda entre los caracteres del alfabeto permitido.

Todos los caracteres tienen la misma probabilidad de ser elegidos.

## 6. ¿Cómo medimos qué tan buena es una aproximación?

Como la entrada es una imagen y no un mapa ASCII conocido, no existe de antemano un carácter objetivo para cada posición. Se divide la imagen original en `N x N` regiones y, para cada celda del individuo, se observa qué tan bien los trazos del carácter representan las líneas, zonas claras y zonas oscuras de la región correspondiente. La aptitud del individuo reúne estas semejanzas para medir qué tan parecida es la imagen ASCII completa a la imagen original: cuanto mayor sea la semejanza visual, mayor será el fitness.

Este problema tiene la particularidad de que cada celda puede evaluarse en gran medida de forma independiente. 

## 7. ¿Cómo cruzamos dos individuos?

Se cruzan los dos mapas respetando la correspondencia entre posiciones. Para cada celda del hijo, el carácter heredado debe provenir de esa misma posición en alguno de los dos padres. De esta forma, nunca se cruzan directamente caracteres ubicados en regiones diferentes de la imagen.

El método de cruza elegido determina qué posiciones se heredan de cada padre. Como ambos padres tienen la misma estructura de `N x N` y todos sus alelos pertenecen al alfabeto permitido, los hijos conservan un tamaño y una estructura válidos.

## 8. ¿Cómo mutamos un individuo?

Después de la cruza, se selecciona al azar una celda del hijo y se reemplaza su carácter actual por otro carácter permitido. Para asegurar que exista un cambio efectivo, el nuevo alelo debe ser diferente del que el hijo tenía en esa posición.

El nuevo carácter podría coincidir con el que tenía el padre del cual no fue heredada esa celda. Aun así se considera una mutación, porque la referencia es el genotipo actual del hijo después de la cruza: si su alelo cambia en la etapa de mutación, ocurrió una mutación.

## 9. ¿Cómo seleccionamos padres y formamos la siguiente generación?

Para seleccionar padres podemos usar Ruleta: los individuos con mayor fitness tienen mayor probabilidad de reproducirse, pero los demás conservan alguna posibilidad. Esto combina aptitud y azar y evita que la selección sea completamente determinística.

Después de generar los hijos, la nueva generación puede formarse mediante cualquiera de las dos estrategias vistas en clase:

- En la supervivencia aditiva, se seleccionan `N` individuos entre la población actual y los hijos. Esto permite conservar buenas soluciones anteriores, aunque puede producir un recambio menor.
- En la supervivencia exclusiva, la nueva generación se forma prioritariamente con los hijos según la cantidad generada. Esto produce un mayor recambio, aunque puede perder buenos individuos de la generación anterior.

No se puede afirmar teóricamente que una sea siempre mejor para este problema. La elección depende del equilibrio deseado entre conservar soluciones buenas y renovar la población.

## 10. ¿Cuándo termina la ejecución?

La ejecución debe tener una cantidad máxima de generaciones `Gmax` para asegurar que termine. No conviene exigir que la imagen ASCII sea idéntica a la original, porque esa representación puede no tener la capacidad de reproducirla perfectamente y la búsqueda podría extenderse demasiado.

La cota exacta no puede determinarse sólo de forma teórica: es un hiperparámetro relacionado con el tamaño del mapa, el alfabeto y el presupuesto disponible. Además de esa cota, el algoritmo puede terminar antes si alcanza una semejanza considerada suficiente o si el mejor fitness deja de mejorar de manera relevante durante cierta cantidad de generaciones.
