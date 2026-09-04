# TP2 — Estudio comparativo para la exposición

## Propósito

El estudio separa exploración, convergencia y calidad final sin ejecutar todas
las combinaciones posibles. Cada condición se repite con cinco semillas fijas y
con el mismo presupuesto para que la diferencia observada pueda atribuirse al
operador que se está comparando.

## Métricas y criterio de lectura

- **NMSE histórico:** el menor error alcanzado hasta cada generación. Es la
  medida de calidad final dentro de una misma imagen.
- **AUC normalizada:** el promedio de `NMSE histórico / NMSE inicial`. Menor es
  mejor y representa cuánto tarda, en conjunto, en mejorar.
- **Generación al 90 %:** primera generación que reduce el error inicial al 10
  %, si se logra. Resume velocidad de convergencia.
- **Diversidad:** variación media entre individuos; permite detectar convergencia
  prematura.
- **Tiempo y evaluaciones:** controlan que se compare con el mismo costo.

Los NMSE absolutos no se comparan entre objetivos distintos: una señal y un
ícono tienen distinta dificultad. Para cada objetivo se comparan condiciones
con el mismo número de triángulos, población, hijos y generaciones.

## Etapas

1. Perfil sintético de los siete métodos de selección (el torneo determinístico
   se prueba con tamaños 2 y 5). Mide presión sin el costo del renderizado.
2. Selección de padres sobre la bandera, manteniendo cruza uniforme, mutación
   multigénica local y supervivencia aditiva con elite.
3. Cruza uniforme frente a un punto para los dos selectores ganadores, sobre
   bandera y señal.
4. Refinamiento local/aditivo frente a exploración global/aditiva y
   global/exclusiva, con el mejor selector y la mejor cruza.
5. Validación de la configuración ganadora en bandera, señal e ícono, seguida
   de una demostración extendida con la semilla mediana y checkpoints visuales.

## Nota sobre elite

El perfil mide la implementación real. En este motor, al pedir tantos padres
como individuos para una selección elite, se devuelven todos una vez antes de
repetir el mejor; por eso su presión como **selector de padres** no equivale al
elitismo de la supervivencia. La presión de preservación principal del protocolo
proviene de la supervivencia aditiva con elite.

## Limitación importante

Una mejora de NMSE en la imagen reducida no garantiza fidelidad perfecta a
tamaño original. En las diapositivas se deben acompañar los números con los
checkpoints y la imagen final renderizada a tamaño original.
