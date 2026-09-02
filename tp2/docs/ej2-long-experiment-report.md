# Ejercicio 2 - Corrida larga de convergencia visual

Este es un segundo informe, independiente de `ej2-experiment-report.md`. La
configuración se versiona en `tp2/experiments/configs/`; los artefactos crudos
se regeneran localmente en `.context/tp2-long-experiments/`.

## Propósito

La primera serie mostró que el motor disminuye NMSE, pero con presupuestos
cortos las imágenes seguían siendo abstractas. Esta corrida aumenta mucho el
presupuesto de búsqueda sobre el objetivo más adecuado para validación visual:
la bandera de Colombia. La hipótesis es que sus tres regiones grandes, colores
planos y bordes horizontales pueden ser aproximados por suficientes triángulos.

## Configuración

| Parámetro | Valor |
| --- | ---: |
| Objetivo | `assets/targets/01_simple.png` - bandera de Colombia |
| Tamaño de trabajo | `32 x 21` píxeles |
| Triángulos por individuo | 60 |
| Población `P` | 100 |
| Hijos por generación `K` | 100 |
| Generaciones | 3,000 |
| Selección de padres | Torneo determinístico, tamaño 2 |
| Cruza | Uniforme, probabilidad 0.9, intercambio 0.5 |
| Mutación | Multigénica uniforme, probabilidad 0.1 por triángulo, delta local |
| Supervivencia | Aditiva con elite |
| Semilla | 20260902 |

La configuración completa está en
`tp2/experiments/configs/05-flag-long-60-triangles.json`.

Se redujo el tamaño del torneo de 3 a 2 respecto de la primera serie para bajar
la presión selectiva. Se aumentaron triángulos, población y generaciones para
dar más capacidad de representación y más oportunidades a las mutaciones de
refinarla. La supervivencia aditiva con elite conserva las buenas capas ya
encontradas.

## Resultado final

| Métrica | Resultado |
| --- | ---: |
| Motivo de corte | Máximo de generaciones |
| Generación final | 3,000 |
| Mejor generación | 2,997 |
| Evaluaciones | 300,100 |
| Tiempo del motor | 136.95 s |
| NMSE inicial mejor | 0.179392 |
| NMSE final mejor | 0.000816 |
| Mejora relativa del error | 99.545% |
| Fitness final | 0.999184 |
| Diversidad inicial -> final | 0.333472 -> 0.009788 |

La cantidad de evaluaciones coincide con `P + G x K = 100 + 3000 x 100`.
Esto muestra que se evaluó la población inicial y cada hijo producido, sin
saltar etapas del ciclo genético.

## Progreso medido

| Generación | Mejor NMSE | NMSE medio | Diversidad | Evaluaciones | Tiempo acumulado |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.179392 | 0.215953 | 0.333472 | 100 | 0.04 s |
| 500 | 0.001966 | 0.001996 | 0.013227 | 50,100 | 21.62 s |
| 1,000 | 0.001282 | 0.001289 | 0.011290 | 100,100 | 43.67 s |
| 1,500 | 0.001097 | 0.001102 | 0.008552 | 150,100 | 68.01 s |
| 2,000 | 0.001009 | 0.001011 | 0.007292 | 200,100 | 90.83 s |
| 2,500 | 0.000893 | 0.000898 | 0.007817 | 250,100 | 113.64 s |
| 3,000 | 0.000816 | 0.000819 | 0.009788 | 300,100 | 136.95 s |

La mejora fuerte ocurre antes de la generación 500: el motor identifica muy
rápido las tres zonas de color. Después continúa refinando lentamente los
límites y los colores. Entre las generaciones 500 y 3000 el NMSE vuelve a bajar
58.5%, aunque la ganancia visual sea más sutil.

## Evidencia visual

### Imagen objetivo

![Bandera objetivo](../assets/targets/01_simple.png)

### Generación 500

Checkpoint generado localmente:
`.context/tp2-long-experiments/runs/evolution-seed-20260902/checkpoints/generation-000500.png`.

Ya se distinguen amarillo, azul y rojo en el orden correcto. Persisten bordes
diagonales y capas superpuestas porque los triángulos todavía cubren regiones
que no pertenecen a su franja.

### Generación 1,500

Checkpoint generado localmente:
`.context/tp2-long-experiments/runs/evolution-seed-20260902/checkpoints/generation-001500.png`.

Las franjas se estabilizan y el color dominante de cada tercio es más cercano al
objetivo. El algoritmo está explotando individuos similares: mejor y promedio
ya tienen NMSE muy próximo.

### Generación 2,500

Checkpoint generado localmente:
`.context/tp2-long-experiments/runs/evolution-seed-20260902/checkpoints/generation-002500.png`.

Los cambios posteriores son ajustes finos. Se conserva una diversidad pequeña,
pero no nula, que permite seguir mejorando sin destruir la estructura hallada.

### Mejor individuo final

Resultado generado localmente:
`.context/tp2-long-experiments/runs/evolution-seed-20260902/best.png`.

La salida reproduce visualmente la estructura principal de la bandera: las tres
franjas horizontales y sus colores. Aún se ven artefactos triangulares en los
bordes y zonas con transparencia; son esperables porque la imagen se construye
mediante 60 triángulos aleatorios y no con rectángulos alineados.

## Por qué esta corrida sí converge visualmente

1. **El objetivo es simple.** Tres colores planos y franjas grandes son mucho
   más fáciles de aproximar que letras, contornos finos o curvas.
2. **Hay capacidad suficiente.** Sesenta triángulos permiten que algunas capas
   cubran grandes regiones y otras corrijan sectores restantes.
3. **Hay presupuesto de búsqueda.** Se realizaron 300,100 evaluaciones, frente
   a 1,020 y 5,050 de las dos corridas de bandera anteriores.
4. **La presión selectiva es moderada.** El torneo de tamaño 2 explota las
   soluciones buenas sin eliminar de inmediato todas las alternativas.
5. **La mutación es local y multigénica.** Una vez encontradas las franjas, las
   capas pueden desplazarse y recolorearse gradualmente en vez de reiniciarse.

Esta es una demostración útil de que el algoritmo no sólo ejecuta los
operadores: el fitness guía una mejora observable de la solución.

## Interpretación correcta del fitness casi perfecto

El fitness `0.999184` se calculó a resolución de trabajo `32 x 21`, no a los
560 x 368 píxeles originales. El resultado final se re-renderiza al tamaño
original usando las coordenadas normalizadas. Por eso los pequeños bordes
diagonales y artefactos que se ven en el PNG grande tienen poco peso, o incluso
pueden no verse, en la comparación de baja resolución.

Esto no es un error del motor: es una decisión de rendimiento. Pero para exigir
mayor fidelidad visual en la imagen final habría que aumentar
`working_max_side`, por ejemplo a 64 o 96, y aceptar que cada evaluación será
más cara. También se necesitarían más triángulos o más generaciones para
conservar la calidad lograda a esa resolución.

## Artefactos generados

La carpeta de la corrida contiene:

```text
.context/tp2-long-experiments/runs/evolution-seed-20260902/
  best.png
  triangles.json
  metrics.csv
  metadata.json
  config.effective.json
  checkpoints/generation-000500.png
  checkpoints/generation-001000.png
  checkpoints/generation-001500.png
  checkpoints/generation-002000.png
  checkpoints/generation-002500.png
  checkpoints/generation-003000.png
```

`triangles.json` permite inspeccionar la solución como lista de triángulos y
`metrics.csv` permite graficar NMSE, fitness y diversidad por generación.

## Siguiente paso sugerido

Antes de aumentar todo a la vez, repetiría esta misma corrida con
`working_max_side: 48` o `64` y la misma semilla. Así se puede medir cuánto
aumenta el tiempo y cuánto mejoran los bordes al comparar el resultado final a
una resolución más exigente. Después convendría probar la señal de tránsito,
que exige contornos y flechas negras, una prueba más dura que la bandera.
