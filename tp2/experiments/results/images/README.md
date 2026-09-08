# Imágenes representativas

Esta carpeta conserva los `best.png` usados para inspeccionar visualmente los
resultados y preparar la presentación.

## Campañas formales

Las carpetas `resolution`, `capacity`, `selection`, `crossover`, `mutation` y
`survival` contienen una imagen representativa por condición. Para cada grupo
de cinco semillas se eligió la corrida cuyo NMSE final corresponde a la
mediana; la semilla elegida aparece en el nombre del archivo.

La carpeta `validation` contiene las imágenes representativas de la
configuración ganadora a 1000 generaciones para Colombia, señal e ícono.

La carpeta `showcase` contiene las corridas extendidas realizadas usando la
semilla mediana seleccionada en la validación. Son una corrida por objetivo, no
la mediana de cinco corridas largas.

## Exploración de triángulos y generaciones

La carpeta `triangle-count-exploration` contiene los `best.png` de los estudios
de 4000 y 5000 generaciones para señal e ícono. Todas esas corridas usaron la
semilla 303:

- 4000 generaciones: 25, 50 y 100 triángulos.
- 5000 generaciones: 50 y 100 triángulos.

Estas imágenes documentan una exploración con una sola semilla y no tienen el
mismo peso estadístico que las campañas formales de cinco semillas.
