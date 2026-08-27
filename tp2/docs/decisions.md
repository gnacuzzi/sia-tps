# Decisiones de diseño - Ejercicio 2

## 1. Representación genética

### Individuo y cromosoma

Un individuo representa una solución candidata completa. Está representado por un cromosoma y mantiene asociado el resultado de su evaluación, como error y fitness.

El cromosoma es la representación genética del individuo y contiene exactamente `T` genes, uno por cada triángulo utilizado para aproximar la imagen objetivo. Al renderizar el cromosoma sobre el canvas se obtiene el fenotipo del individuo: una imagen candidata.

### Genes, loci y alelos

Cada gen representa un triángulo completo. Contiene sus tres vértices y un color `RGBA`: `R`, `G` y `B` determinan el color, mientras que `A` determina su opacidad.

Un alelo es una configuración concreta del triángulo, por ejemplo:

```text
Triangle(
    vertices=((x1, y1), (x2, y2), (x3, y3)),
    color=(r, g, b, a)
)
```

El dominio de alelos está formado por todas las combinaciones permitidas de posiciones de los vértices, componentes de color y opacidad. El locus es la posición que ocupa cada triángulo dentro del cromosoma de `T` genes.

## 2. Validez de un individuo

Un individuo válido debe contener exactamente la cantidad `T` de triángulos recibida como input. Cada triángulo debe cumplir las siguientes condiciones:

- sus tres vértices se encuentran dentro de los límites del canvas;
- sus vértices definen un área estrictamente positiva, por lo que no pueden estar repetidos ni ser colineales;
- sus componentes `R`, `G` y `B` se encuentran dentro del rango permitido y su canal alfa cumple `A > 0`;
- todos sus valores pueden ser interpretados por el renderer sin correcciones implícitas.

Los colores válidos son todos los que pueden representarse en el espacio de color elegido; no tienen que pertenecer necesariamente a la paleta observada en la imagen objetivo. Los triángulos degenerados se consideran inválidos porque no aportan una superficie visible y desperdiciarían uno de los `T` genes disponibles.

Inicialmente tampoco se permite `A = 0`, ya que produciría un triángulo completamente invisible. Se admite transparencia parcial, pero se exige opacidad positiva para intentar aprovechar todos los triángulos disponibles. Esta restricción podrá revisarse si los resultados muestran que permitir triángulos invisibles aporta alguna ventaja.

## 3. Fenotipo y orden de renderizado

El fenotipo es la imagen que se obtiene al renderizar los `T` triángulos del individuo sobre un canvas. Los triángulos se dibujan siguiendo el orden de sus loci en el cromosoma. Este orden forma parte de la solución porque, cuando existen superposiciones y transparencia, intercambiar dos triángulos puede cambiar la imagen resultante.

El canvas es blanco por defecto, pero su color se define en `config.json`. Durante una corrida se mantiene fijo. El posible efecto del color de fondo sobre la calidad y la convergencia queda abierto como pregunta experimental.

## 4. Función de error y fitness

La primera aproximación compara la imagen generada con la imagen objetivo píxel por píxel. Ambas se expresan en RGB y con las mismas dimensiones. La métrica primaria es el error cuadrático medio normalizado:

```text
NMSE = promedio((objetivo - generado)²) / 255²
```

El NMSE está acotado entre `0` y `1`: un error `0` indica que las imágenes son idénticas y un valor mayor representa una diferencia mayor. Como los métodos de selección deben maximizar la aptitud, se utiliza:

```text
fitness = max(epsilon, 1 - NMSE)
```

El fitness queda así acotado, es positivo y aumenta cuando disminuye el error. Se registran tanto NMSE como fitness para mantener separada la medida interpretable del error y la transformación utilizada por el algoritmo.

NMSE se adopta como primer enfoque para completar y validar el motor con una métrica simple, determinística y reproducible. Después se verificará si su evolución coincide con la calidad visual observada; si no fuera suficiente, podrá compararse con otras medidas de similitud.

## 5. Mutación

### Unidad de mutación

Se implementan inicialmente las dos variantes requeridas:

- En la mutación de un gen, cuando ocurre la mutación se selecciona un único triángulo del cromosoma.
- En la mutación multigen uniforme, cada triángulo tiene de manera independiente una probabilidad `Pm` de ser mutado.

Ambas variantes utilizan la misma operación para modificar el alelo del triángulo seleccionado.

### Cambio aplicado a un alelo

El alelo es la configuración completa de un triángulo. Al mutarlo se selecciona una de sus propiedades y se cambia su valor: puede modificarse uno de sus vértices, uno de los componentes `R`, `G` o `B`, o el canal alfa `A`.

El valor resultante debe ser diferente del actual y mantener el triángulo dentro del dominio válido: vértices dentro del canvas, área positiva, canales de color dentro de rango y `A > 0`. La magnitud o distribución exacta del cambio se parametriza para poder distinguir entre una perturbación local y un reemplazo más amplio.

## 6. Cruza

La primera implementación cruza únicamente en límites de triángulos. Cada gen se hereda como una unidad completa de uno de los padres, sin mezclar un vértice con un color ni propiedades semánticamente incompatibles.

Se implementan cruce de un punto y cruce uniforme sobre la secuencia de `T` genes. En ambos casos, el triángulo que ocupa un locus del hijo procede de ese mismo locus en uno de los padres. Esto mantiene siempre la cantidad de triángulos y su estructura válida.

Como decisión abierta queda evaluar una cruza interna alineada por propiedad, en la que las coordenadas se combinen con sus coordenadas correspondientes, cada canal de color con el mismo canal y alfa únicamente con alfa.

## 7. Población inicial

La población inicial contiene `P` individuos generados aleatoriamente. Cada individuo posee exactamente `T` triángulos válidos. Sus vértices, colores y valores alfa se muestrean dentro de sus dominios permitidos; si se genera un triángulo degenerado, se vuelve a generar.

La semilla aleatoria forma parte de la configuración y se registra para poder reproducir la misma población inicial y comparar corridas bajo condiciones controladas.

Existe una única imagen objetivo. Los `P` individuos producen `P` fenotipos candidatos diferentes, que se renderizan y comparan contra ese mismo objetivo para calcular sus fitness.

## 8. Criterios de corte y resultado final

La ejecución termina cuando se cumple el primero de los criterios de corte habilitados:

- **Cantidad máxima de generaciones:** `max_generations` establece una cota obligatoria y garantiza que la ejecución termine.
- **Error objetivo:** si se configura `target_nmse`, la ejecución puede finalizar al encontrar un individuo cuyo NMSE sea menor o igual a ese valor.
- **Estancamiento por contenido:** la ejecución puede finalizar si el mejor NMSE no mejora al menos `min_improvement` durante una ventana de `patience` generaciones.
- **Tiempo máximo:** `max_seconds` puede utilizarse como límite de seguridad cuando exista un presupuesto temporal.

Los valores exactos se definen en `config.json`. No se exige una coincidencia perfecta con la imagen objetivo, porque la representación mediante una cantidad fija de triángulos puede no tener capacidad para reproducirla exactamente.

Durante la evolución se renderizan los fenotipos necesarios para calcular el fitness. Además, se mantiene explícitamente el mejor individuo encontrado desde el comienzo de la ejecución. Cuando se activa un criterio de corte, el resultado final es ese `best-so-far`, aunque no pertenezca a la última población. A partir de su cromosoma se generan la imagen de salida y la enumeración final de triángulos.

## 9. Hipótesis iniciales

Estas hipótesis orientan los primeros experimentos, pero no se consideran conclusiones:

1. Las imágenes simples y de baja resolución permitirán observar mejoras de fitness más rápido que las imágenes con mayor detalle.
2. Aumentar la cantidad de triángulos incrementará la capacidad de representación, pero también agrandará el espacio de búsqueda y puede retrasar la convergencia bajo un presupuesto fijo.
3. Una presión de selección alta puede mejorar rápidamente el fitness inicial, pero también reducir la diversidad y provocar convergencia prematura.
4. Una mutación demasiado baja puede impedir escapar de soluciones estancadas, mientras que una demasiado alta puede dificultar la convergencia.
5. La supervivencia aditiva tenderá a conservar mejor las soluciones alcanzadas, mientras que la exclusiva producirá un mayor recambio de la población; ninguna se asume superior antes de experimentar.
6. NMSE representará razonablemente la calidad de imágenes simples, pero puede no coincidir completamente con la semejanza visual percibida en imágenes más complejas.
7. El color del canvas puede modificar la cantidad y el tipo de triángulos necesarios para aproximar una imagen.

## 10. Decisiones abiertas

Las siguientes decisiones se mantienen explícitamente abiertas para revisarlas a partir de evidencia:

- comprobar si NMSE representa adecuadamente la calidad visual o si hace falta otra medida de similitud;
- comparar el canvas blanco con otros colores de fondo;
- evaluar una cruza interna alineada por propiedad frente a la cruza inicial de triángulos completos;
- determinar si la mutación de un alelo funciona mejor mediante cambios locales, reemplazos amplios o una combinación de ambos;
- calibrar la probabilidad y la magnitud de mutación;
- revisar la restricción `A > 0` si permitir triángulos invisibles mostrara alguna ventaja;
- analizar el efecto del orden de los triángulos y de los operadores sobre las capas del fenotipo.

Estas decisiones abiertas no impiden completar la primera implementación: definen posibles preguntas para las fases experimentales posteriores.
