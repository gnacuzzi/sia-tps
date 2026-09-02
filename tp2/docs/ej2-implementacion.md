# Ejercicio 2 - Diseño teórico e implementación

Esta guía relaciona la consigna del TP2 con el código actual del repositorio.
El objetivo es recibir una imagen y aproximarla mediante una cantidad fija de
triángulos de color uniforme, renderizados sobre un canvas. El algoritmo
genético busca reducir la diferencia entre esa imagen generada y la imagen
objetivo.

## 1. Qué problema resuelve el algoritmo

La entrada propia del problema es:

- una imagen objetivo;
- una cantidad `T` de triángulos.

El resto son hiperparámetros del algoritmo genético, como tamaño de población,
métodos de selección, probabilidades de cruza y mutación, y criterios de corte.
El resultado es una imagen aproximada, la enumeración de sus triángulos y las
métricas de la corrida.

La representación no pretende copiar los píxeles uno a uno: describe la imagen
mediante una lista corta de primitivas geométricas. Por eso puede interpretarse
como una forma de compresión con pérdida: se conserva una aproximación visual y
se pierde detalle que no puede expresarse con los `T` triángulos disponibles.

## 2. Representación genética

### Individuo, cromosoma, gen y alelos

Un **individuo** es una imagen candidata completa. Su **cromosoma** es una
secuencia ordenada de exactamente `T` triángulos. Cada triángulo es un **gen**:

```text
TriangleGene(
    vertices=((x1, y1), (x2, y2), (x3, y3)),
    color=(R, G, B, A),
)
```

Los alelos de ese gen son sus tres vértices y sus cuatro componentes RGBA. Las
coordenadas están normalizadas en `[0, 1]`; los canales RGB y alfa usan enteros
en `[0, 255]`. El alfa permite transparencia y el rango admisible se configura
por corrida.

El orden de los genes importa. El renderer dibuja el primer triángulo, luego el
segundo y así sucesivamente. Si se superponen y son translúcidos, cambiar el
orden altera la composición de colores y, por lo tanto, el fenotipo.

### Validez

Todo individuo debe conservar `T` genes. Cada triángulo debe tener sus vértices
dentro del canvas, área estrictamente positiva y color RGBA válido. En
particular, el canal alfa es positivo para no ocupar un gen con un triángulo
invisible. Esta validación hace que cruza y mutación no puedan producir una
solución geométricamente inválida.

### Fenotipo

El **fenotipo** es el bitmap producido al dibujar el cromosoma sobre el canvas
configurado. Es lo que se compara con la imagen de entrada; no se compara el
vector de genes directamente contra la foto.

## 3. Fitness: cómo se evalúa una aproximación

La imagen objetivo se convierte a RGB y se reduce, preservando su relación de
aspecto, a un tamaño de trabajo. Cada individuo se renderiza a ese mismo tamaño.
Para cada píxel y canal se calcula su diferencia al cuadrado:

```text
NMSE = promedio((objetivo - generado)^2) / 255^2
fitness = max(epsilon, 1 - NMSE)
```

`NMSE` es un error normalizado: `0` representa imágenes idénticas y valores más
altos representan peor aproximación. Como los selectores maximizan, el motor
usa `fitness`, que crece cuando el NMSE disminuye y se mantiene positivo gracias
a `epsilon`.

Esta métrica es simple, determinista y permite comparar corridas. No garantiza
que dos imágenes con igual NMSE se perciban igual de parecidas; por eso se
deben mirar también las imágenes generadas y contrastar experimentos.

## 4. Ciclo evolutivo

1. Se crea una población inicial de `P` cromosomas aleatorios válidos.
2. Se renderiza y evalúa cada individuo.
3. Se seleccionan `K` padres con el selector configurado.
4. Se agrupan los padres de a dos; cada par produce dos hijos por cruza.
5. Cada hijo pasa por mutación y luego se evalúa.
6. Se construye la próxima generación mediante la estrategia de supervivencia.
7. Se registran métricas y se conserva el mejor individuo histórico.
8. El ciclo se repite hasta que se active un criterio de corte.

Mantener el mejor histórico es importante: aun si una estrategia exclusiva lo
descarta de la población actual, el resultado final no pierde la mejor imagen
encontrada hasta ese momento.

## 5. Selección de padres y sobrevivientes

El mismo módulo permite aplicar los selectores tanto a padres como, cuando la
estrategia lo necesita, al conjunto de candidatos a sobrevivir.

| Método | Idea | Efecto principal |
| --- | --- | --- |
| Elite | Ordena por fitness y toma los mejores. | Mucha explotación; puede reducir diversidad rápido. |
| Ruleta | Muestrea con probabilidad proporcional al fitness. | Combina azar y preferencia por los mejores. |
| Universal | Usa un inicio aleatorio y punteros equiespaciados sobre la ruleta. | Tiene menos variabilidad que repetir ruletas independientes. |
| Ranking | Reemplaza fitness por el rango en el orden de calidad. | Evita que diferencias extremas de fitness dominen la elección. |
| Boltzmann | Pesa con `exp(fitness / T)` y una temperatura que decae. | Más exploración con temperatura alta; más presión selectiva al enfriarse. |
| Torneo determinístico | Sortea varios participantes y elige al mejor. | La presión aumenta con el tamaño del torneo. |
| Torneo probabilístico | Entre dos, elige al mejor con probabilidad `threshold`. | Mantiene exploración incluso frente a un candidato claramente mejor. |

## 6. Cruza

Los dos métodos implementados operan en límites de genes completos. Por lo
tanto, un hijo siempre hereda un triángulo completo ubicado en el mismo locus de
uno de sus padres; no combina, por ejemplo, el color de un triángulo con los
vértices de otro.

- **Un punto:** elige un corte interno del cromosoma. Un hijo hereda el prefijo
  del primer padre y el sufijo del segundo; el otro hijo recibe la combinación
  inversa. Es útil si se espera que bloques consecutivos del orden de capas
  formen una estructura que conviene preservar.
- **Uniforme:** para cada locus, intercambia los genes entre padres con una
  probabilidad configurable. Es útil cuando no hay motivo para creer que los
  genes vecinos deban heredarse juntos y se quiere mezclar capas con más
  libertad.

La consigna exige al menos dos métodos de cruza; estos dos satisfacen ese mínimo.
No se implementaron cruza de dos puntos ni anular, que son alternativas, no un
requisito adicional.

## 7. Mutación

La mutación cambia un triángulo sin romper su validez. Para un gen elegido se
elige uniformemente una de siete propiedades: uno de sus tres vértices, uno de
los tres canales RGB o alfa. Si una propuesta hace degenerado al triángulo, se
vuelve a intentar.

- **Single-gene:** con probabilidad `Pm` por hijo, se selecciona exactamente un
  triángulo para mutar. Es una perturbación suave: resulta apropiada cuando ya
  hay buenas aproximaciones y se quiere refinarlas.
- **Multigene uniforme:** cada triángulo muta independientemente con
  probabilidad `Pm`. Puede modificar varios genes del mismo individuo y aporta
  exploración cuando la población se estanca o está muy concentrada.

Además hay dos escalas de cambio de alelo:

- `local_delta`: modifica posición, color o alfa dentro de un delta acotado;
  favorece ajuste fino;
- `global_resample`: vuelve a muestrear el valor en todo su dominio; permite
  saltos grandes y mayor exploración.

Las dos primeras variantes satisfacen el mínimo de dos métodos de mutación de
la consigna. La implementación no contiene un cronograma temporal explícito de
mutación no uniforme; el modo local o global se mantiene fijo durante cada
corrida.

## 8. Supervivencia

- **Aditiva:** une población actual e hijos y selecciona `P` individuos del
  conjunto combinado. Conserva buenas soluciones previas y es más elitista si
  se combina con selección elite.
- **Exclusiva:** si hay más de `P` hijos, selecciona `P` entre ellos. Si hay
  `K <= P` hijos, incorpora esos `K` y completa los lugares restantes con
  individuos de la población anterior. Genera un recambio mayor que la aditiva.

Ambas estrategias solicitadas por la consigna están implementadas.

## 9. Criterios de corte y métricas

Los criterios habilitados se combinan con OR; el primero que se cumpla detiene
la corrida:

- máximo de generaciones, obligatorio;
- NMSE objetivo opcional;
- estancamiento: no obtener una mejora acumulada significativa durante una
  cantidad de generaciones configurada;
- tiempo máximo opcional.

Por generación se pueden registrar NMSE y fitness mejor, medio, mediano y su
desvío estándar, cantidad de evaluaciones, tiempo transcurrido y diversidad
genotípica. La diversidad es la distancia absoluta promedio entre los alelos
normalizados de todos los pares de individuos: `0` indica cromosomas idénticos.

## 10. Dónde está cada parte en el código

| Responsabilidad | Archivo | Implementación relevante |
| --- | --- | --- |
| Contrato y validación de configuración | `src/sia_tp2/config.py` | Lee JSON estricto, valida dominios y parámetros de cada operador. |
| Modelo genético | `src/sia_tp2/domain/model.py` | Define `TriangleGene` e `Individual` inmutables y sus invariantes. |
| Población inicial | `src/sia_tp2/domain/initialization.py` | Genera triángulos válidos y reproducibles a partir de la semilla. |
| Render y carga de imágenes | `src/sia_tp2/domain/renderer.py` | Usa Pillow para convertir, escalar y dibujar capas RGBA. |
| Fitness | `src/sia_tp2/domain/fitness.py` | Calcula NMSE y fitness. |
| Cruza y mutación | `src/sia_tp2/domain/operators.py` | Implementa un punto, uniforme, single-gene y multigene. |
| Selección | `src/sia_tp2/ga/selection.py` | Implementa elite, ruleta, universal, ranking, Boltzmann y torneos. |
| Supervivencia | `src/sia_tp2/ga/survival.py` | Implementa reemplazo aditivo y exclusivo. |
| Motor genérico | `src/sia_tp2/ga/engine.py` | Ejecuta el ciclo evolutivo y conserva el mejor histórico. |
| Integración del problema | `src/sia_tp2/workflow.py` | Conecta configuración, renderer, fitness, operadores y reporte. |
| Resultados | `src/sia_tp2/reporting/serialization.py` | Escribe archivos de cada corrida. |
| Interfaz de línea de comandos | `src/sia_tp2/cli.py` | Ofrece `inspect-config`, `render-random` y `evolve`. |

## 11. Artefactos y ejecución

Cada corrida crea un directorio nuevo y guarda, como mínimo:

```text
config.effective.json  # configuración exacta usada
metadata.json          # corte, semilla, mejor generación y tamaños
metrics.csv            # historial de métricas
triangles.json         # cromosoma del mejor individuo
best.png               # imagen generada
```

Los ejemplos de configuración están en `configs/`. Para instalar y ejecutar una
corrida corta desde `tp2/`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/sia-tp2 --config configs/smoke.json evolve
```

`configs/phase3-smoke.json` ejercita otra combinación de operadores: Boltzmann,
cruza de un punto, mutación single-gene con remuestreo global y supervivencia
exclusiva.

## 12. Cobertura frente a la consigna y pendiente de entrega

El motor implementa los siete selectores solicitados, ambas supervivencias, dos
cruzas y dos mutaciones. Usa Pillow solamente para manejo de imágenes; el
algoritmo genético se implementa en el repositorio, como exige la consigna.

Para completar el entregable todavía hace falta transformar este material en:

- un `README` de TP2 con las instrucciones de instalación y ejecución (el actual
  todavía sólo tiene el título);
- una presentación que explique decisiones y resultados;
- experimentos reproducibles y su análisis: imágenes simples, distintas
  cantidades de triángulos, configuraciones de selección/mutación/supervivencia
  y sus curvas de error y diversidad.

Los tests unitarios ya cubren el contrato, los operadores, el motor y el flujo
de salida. Antes de presentar conviene instalar las dependencias de desarrollo,
ejecutar esos tests y guardar corridas de referencia para respaldar las
conclusiones experimentales.
