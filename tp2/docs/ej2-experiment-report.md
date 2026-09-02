# Ejercicio 2 - Informe de corridas exploratorias

Este informe documenta cuatro ejecuciones reproducibles del motor genético que
aproxima imágenes mediante triángulos RGBA. Las configuraciones se versionan en
`tp2/experiments/configs/`; los resultados e imágenes generados viven en
`.context/tp2-experiments/` y se regeneran al ejecutar cada corrida.

## Objetivo de las corridas

Las corridas no buscan todavía encontrar los hiperparámetros definitivos. Buscan
responder tres preguntas básicas:

1. ¿El motor completa el ciclo de evolución y reduce el error?
2. ¿Cómo cambia el resultado al aumentar los triángulos y el presupuesto de
   búsqueda en una imagen simple?
3. ¿Qué ocurre con imágenes de estructura distinta y con otra combinación de
   operadores?

Todos los experimentos usan una semilla fija, por lo que se pueden repetir.
Cada uno registra la configuración efectiva, las métricas por generación, el
mejor cromosoma y el PNG resultante.

## Metodología

El motor trabaja con un cromosoma de triángulos. Sus vértices están normalizados
en `[0, 1]`, cada triángulo tiene color RGBA y se renderiza respetando el orden
del cromosoma. La evaluación usa NMSE contra la imagen objetivo redimensionada:

```text
NMSE = promedio((objetivo - generado)^2) / 255^2
fitness = 1 - NMSE
```

Por eso, un NMSE menor es mejor. La cantidad de evaluaciones esperada es
`P + G x K`: se evalúan `P` individuos iniciales y `K` hijos en cada una de las
`G` generaciones.

## Configuraciones evaluadas

| Corrida | Objetivo | P / K / generaciones | Triángulos | Operadores |
| --- | --- | --- | --- | --- |
| 1 | Bandera de Colombia | 20 / 20 / 50 | 5 | Torneo determinístico, cruza uniforme, mutación multigénica local, supervivencia aditiva + elite. |
| 2 | Bandera de Colombia | 50 / 50 / 100 | 20 | Igual que 1, con más representación y presupuesto. |
| 3 | Señal de doble sentido | 50 / 50 / 100 | 30 | Igual que 2. |
| 4 | Ícono de formas redondeadas | 50 / 50 / 100 | 30 | Boltzmann, cruza de un punto, single-gene global y supervivencia exclusiva + ranking. |

Las configuraciones están en `tp2/experiments/configs/`.

## Resultados numéricos

| Corrida | NMSE inicial mejor | Mejor NMSE histórico | Mejora relativa del error | Fitness mejor | Diversidad inicial -> final | Tiempo | Evaluaciones |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.244884 | 0.167694 | 31.5% | 0.832306 | 0.333664 -> 0.001012 | 0.07 s | 1,020 |
| 2 | 0.229181 | 0.069014 | 69.9% | 0.930986 | 0.333866 -> 0.003465 | 0.91 s | 5,050 |
| 3 | 0.126467 | 0.052098 | 58.8% | 0.947902 | 0.336543 -> 0.004911 | 1.33 s | 5,050 |
| 4 | 0.103726 | 0.084161 | 18.9% | 0.915839 | 0.331665 -> 0.054807 | 1.30 s | 5,050 |

La mejora relativa es `(NMSE inicial - mejor NMSE) / NMSE inicial`. No se deben
comparar los NMSE absolutos entre imágenes como si midieran únicamente
dificultad: cada objetivo tiene distinta resolución de trabajo, fondo y
distribución de colores. La comparación más útil es el descenso dentro de cada
corrida bajo una misma configuración.

## Salida visual

### Corrida 1 - bandera, 5 triángulos

Resultado generado localmente:
`.context/tp2-experiments/runs/evolution-seed-11/best.png`.

Con solamente cinco triángulos, el motor baja el error pero no puede expresar
las tres franjas horizontales de la bandera con precisión. La representación es
demasiado pequeña y las capas aleatorias cubren porciones grandes del canvas.

### Corrida 2 - bandera, 20 triángulos

Resultado generado localmente:
`.context/tp2-experiments/runs/evolution-seed-12/best.png`.

El NMSE baja mucho más que en la corrida 1. Hay cuatro veces más genes y cinco
veces más evaluaciones, por lo que el algoritmo puede ajustar mejor colores y
coberturas. Aun así, la imagen no reconstruye claramente las franjas: reducir
el promedio de error de píxeles no obliga a descubrir la estructura semántica de
una bandera.

### Corrida 3 - señal, 30 triángulos

Resultado generado localmente:
`.context/tp2-experiments/runs/evolution-seed-13/best.png`.

La señal tiene grandes zonas blancas y amarillas. Esto permite un NMSE bajo,
pero no significa que las flechas negras ni el borde se hayan reconstruido. Es
un ejemplo de por qué el fitness numérico debe acompañarse de inspección visual.

### Corrida 4 - ícono, operadores alternativos

Resultado generado localmente:
`.context/tp2-experiments/runs/evolution-seed-14/best.png`.

Esta corrida usa operadores más exploratorios. Conserva bastante más diversidad
al final (0.0548 frente a aproximadamente 0.004 en las corridas elitistas), pero
reduce menos el error durante el mismo presupuesto. El mejor individuo apareció
en la generación 87; luego la población final fue ligeramente peor. El motor lo
conservó como mejor histórico, que es el comportamiento correcto.

## Qué enseña cada resultado

### 1. El ciclo genético funciona

Las cuatro corridas llegan al límite de generaciones, producen exactamente la
cantidad esperada de evaluaciones y mejoran respecto de su población inicial.
También escriben todos los artefactos solicitados: `best.png`, `triangles.json`,
`metrics.csv`, `metadata.json` y `config.effective.json`.

### 2. Más triángulos aportan capacidad, pero encarecen la búsqueda

En la bandera, pasar de 5 a 20 triángulos y de 1,020 a 5,050 evaluaciones bajó
el NMSE de 0.167694 a 0.069014. El tiempo creció de 0.07 s a 0.91 s. Esto ilustra
el compromiso principal del problema:

- más triángulos permiten describir más detalles;
- pero agregan posiciones, colores, alfas y orden de capas que el algoritmo debe
  descubrir;
- por eso suelen requerir mayor población, más generaciones o mejores operadores.

### 3. La convergencia elitista actual pierde diversidad muy rápido

Las corridas 1 a 3 usan torneo determinístico para padres y supervivencia
aditiva con elite. La diversidad cae de aproximadamente 0.33 a menos de 0.005.
Eso explica que el mejor y el promedio final sean casi iguales: la población se
volvió muy parecida. Esta presión selectiva acelera mejoras al principio, pero
favorece convergencia prematura.

### 4. Explorar más evita el colapso, pero no garantiza mejoras rápidas

La corrida 4 combina Boltzmann, cruza de un punto, mutación global y
supervivencia exclusiva. Conserva diversidad, pero el salto global de un único
gen y el recambio completo de la población no aprovecharon tan bien las buenas
capas halladas. El resultado muestra el equilibrio central de un AG:

- **explotación:** conservar y refinar buenas soluciones;
- **exploración:** mantener variantes para poder encontrar regiones nuevas del
  espacio de búsqueda.

No hay un selector o estrategia universalmente mejor; se comparan bajo el mismo
presupuesto y con varias semillas.

## Por qué las imágenes todavía se ven abstractas

El motor sí está optimizando el NMSE, pero la búsqueda es difícil:

1. La inicialización elige posiciones y colores al azar, sin mirar la imagen
   objetivo. Las primeras capas no parten de contornos o colores relevantes.
2. Un cromosoma de 20 triángulos tiene muchos grados de libertad: 60 coordenadas
   de vértices, 80 canales RGBA y un orden de renderizado significativo.
3. El fitness es global. Un triángulo aparentemente bueno puede dejar de serlo
   cuando cambia una capa posterior transparente.
4. Una mutación local altera sólo una propiedad y puede necesitar muchas
   generaciones para mover o recolorear una capa de forma útil.
5. NMSE promedia todos los píxeles. Puede mejorar cubriendo grandes regiones
   con colores cercanos al promedio, aun si faltan bordes y figuras importantes.

Esto no invalida la solución: explica por qué el problema pide experimentar y
no sólo implementar operadores. Una imagen simple es el punto de partida
correcto, pero necesita un presupuesto de búsqueda y ajustes acordes.

## Próximos experimentos recomendados

1. Repetir cada configuración con varias semillas y reportar media y dispersión;
   una sola corrida no mide robustez.
2. Para la bandera, aumentar generaciones antes de cambiar todos los operadores
   a la vez. Mantener la misma imagen permite aislar efectos.
3. Comparar supervivencia aditiva contra exclusiva con la misma selección y
   mutación, observando NMSE y diversidad.
4. Probar una menor presión de selección, por ejemplo torneo de tamaño 2 o
   ranking, si la diversidad colapsa temprano.
5. Como mejora de implementación futura, inicializar colores a partir de muestras
   de la imagen objetivo próximas a cada triángulo. No cambia la definición del
   AG, pero da una población inicial mucho más informada.
6. Considerar una métrica que complemente NMSE, por ejemplo una basada en bordes,
   si la calidad visual sigue sin coincidir con el error numérico.

## Reproducción local

Los resultados están bajo `.context/tp2-experiments/runs/`. Para ejecutar una
configuración luego de crear un entorno con las dependencias del proyecto:

```bash
cd tp2
PYTHONPATH=src .venv/bin/python -m sia_tp2 \
  --config experiments/configs/01-simple-5-triangles.json evolve
```

Durante estas pruebas se detectó una limitación de empaquetado: una instalación
no editable resuelve las rutas relativas de las imágenes desde el directorio de
la librería instalada, no desde `tp2/`. Ejecutar desde el código fuente con
`PYTHONPATH=src`, como arriba, evita el problema. Es un aspecto a corregir antes
de entregar el programa como paquete instalable.
