# Implementación reutilizable y validación

## Qué resuelve cada archivo

| Archivo | Responsabilidad |
|---|---|
| `src/sia_tp3/models.py` | Perceptrón simple, red multicapa, activaciones, forward, Rosenblatt, backpropagation y guardado/carga |
| `src/sia_tp3/optimizers.py` | Descenso básico, Momentum, eta adaptativo, RMSProp y Adam |
| `src/sia_tp3/training.py` | Entrenamiento online, mini-batch o batch y métricas al terminar cada época, sin conocer el dataset |
| `src/sia_tp3/experiments.py` | Holdout estratificado reproducible para fraude y dígitos sin cambiar los loaders development/test |
| `src/sia_tp3/fraud_experiment.py` | Configuración, ejecución y artefactos del baseline del ejercicio 1 |
| `src/sia_tp3/metrics.py` | Matriz de confusión y métricas estándar de clasificación globales, por clase y macro |
| `src/sia_tp3/validation.py` | Datos sintéticos de la consigna, configuración y evidencia de las corridas |
| `configs/validation.json` | Arquitecturas, inicialización, semillas, tasas, orden y criterios de aceptación |
| `scripts/plot_validation.py` | Gráficos a partir de resultados guardados, sin volver a entrenar |
| `scripts/plot_fraud_experiment.py` | Curvas completas, logarítmicas y ampliadas de training/validation para fraude |
| `tests/` | Cuentas manuales, gradientes numéricos, persistencia, determinismo y aprendizaje |

El código de AND/XOR no está dentro de las neuronas: los modelos reciben matrices.
Para fraude y dígitos se reutilizan `Perceptron`, `MultilayerPerceptron` y `fit`.
Lo que cambiará será la preparación de los datos, la arquitectura, los
hiperparámetros y el protocolo de evaluación.

## Métricas de clasificación

`classification_metrics` implementa las convenciones de la clase 12.2: filas
reales, columnas predichas y cálculo one-vs-rest para cada clase. Devuelve
accuracy global y precision, recall, F1, TPR y FPR por clase; TPR y recall son
deliberadamente iguales. Los promedios macro dan el mismo peso a cada clase
evaluable, independientemente de cuántas muestras tenga.

La lista `labels` es obligatoria. De ese modo una clase ausente en los datos,
como el 8 en training, conserva su fila y columna y sus métricas sin denominador
quedan como `NaN` en lugar de aparentar un resultado válido. El módulo recibe
clases enteras: el umbral de fraude y el `argmax` de dígitos pertenecen al
protocolo de cada ejercicio y no quedan ocultos dentro de las métricas.

## Contrato de los datos

- `X`: matriz de forma `(cantidad_de_muestras, cantidad_de_entradas)`.
- `y`: matriz de forma `(cantidad_de_muestras, cantidad_de_salidas)`, incluso
  cuando haya una sola salida. Se rechazan vectores 1D para evitar broadcasting
  que podría producir errores silenciosos.
- Pesos por capa: `(neuronas_destino, neuronas_origen)`.
- Bias por capa: `(neuronas_destino,)`.
- `predict(X)` devuelve valores continuos para identidad, tanh y logística;
  solo el escalón devuelve clases directamente.
- El motor no normaliza, divide ni modifica los datasets recibidos. Esas
  decisiones pertenecen a cada ejercicio y se tomarán sobre su conjunto de desarrollo.

Se usa `float64` y operaciones matriciales de NumPy. La red admite cualquier
cantidad de entradas, capas y salidas. Se verificó también el cálculo de
gradientes de una red con varias salidas y la forma de salida de una red
`[784, 8, 10]`; eso no equivale a haber entrenado el clasificador de dígitos.

## Modelos y reglas

`Perceptron(input_size, activation=...)` representa una sola neurona.
`MultilayerPerceptron(architecture, activations=...)` recibe una activación
por capa de pesos. El perceptrón simple reutiliza el mismo cálculo matricial
como caso de una capa; no hay fórmulas diferentes en el runner de validación.

| Activación | Predicción | Derivada |
|---|---|---|
| `step` | +1 si h ≥ 0; −1 si h < 0 | No se deriva: actualización de Rosenblatt |
| `linear` | h | 1 |
| `tanh` | tanh(βh) | β(1 − salida²) |
| `logistic` | 1 / (1 + exp(−2βh)) | 2β · salida · (1 − salida) |

La logística respeta la parametrización de clase 10.2. Está disponible como
opción reutilizable para probabilidades, pero no se seleccionó todavía el
modelo del ejercicio de fraude. Su cálculo evita overflow y su derivada está
contrastada numéricamente. No se agregaron funciones de activación ajenas a
los materiales recibidos.

El escalón solo se permite en una neurona simple; usarlo en una capa oculta
impediría entrenar mediante las derivadas de backpropagation.

Para los modelos diferenciables, `loss` es:

$$
L=\frac{1}{2N}\sum_{\mu=1}^N\sum_{j=1}^{K}
(\hat y_j^{(\mu)}-y_j^{(\mu)})^2.
$$

`gradients` devuelve las derivadas de **esa** función, calculadas antes de
modificar los pesos. El código define delta con el signo del gradiente
`(obtenido − esperado) · derivada` y luego **resta** `eta * gradiente`.
Es algebraicamente equivalente al apunte, que usa el signo de corrección
`(esperado − obtenido)` y **suma** el ajuste.

El MSE registrado promedia sobre muestras y salidas: `MSE = 2L/K`.
En la validación todas las redes tienen una salida, por lo que `MSE = 2L`.
`fit` calcula el gradiente medio del grupo usado en cada actualización. Con
`batch_size=1` entrena online; con un valor intermedio usa mini-batches; con
`batch_size=len(X)` o `None` usa todo training como un batch. Esta decisión es
independiente del optimizador: el lote decide **con qué muestras se calcula el
gradiente** y el optimizador decide **cómo ese gradiente modifica los parámetros**.

## Optimizadores de clase 12.1

Todos actualizan pesos y bias; ambos son parámetros aprendibles. Se mantienen
las fórmulas y la ubicación de epsilon mostradas en clase:

| Clase | Actualización implementada | Estado que conserva |
|---|---|---|
| `GradientDescent` | $\theta\leftarrow\theta-\eta g_t$ | Ninguno |
| `Momentum` | $\Delta\theta_t=-\eta g_t+\alpha\Delta\theta_{t-1}$ | Cambio anterior |
| `AdaptiveLearningRate` | Suma $a$ a eta tras K disminuciones consecutivas; resta una fracción $b\eta$ tras K aumentos | Loss anterior y rachas |
| `RMSProp` | $S_t=\gamma S_{t-1}+(1-\gamma)g_t^2$; divide por $\sqrt{S_t+\epsilon}$ | Promedio de gradientes cuadrados |
| `Adam` | Momentos primero y segundo con corrección $1-\beta_1^t$ y $1-\beta_2^t$ | Ambos momentos y cantidad de pasos |

Adam usa los valores de referencia de la diapositiva como defaults:
`eta=0.001`, `beta1=0.9`, `beta2=0.999` y `epsilon=1e-8`. En RMSProp se exigen
`gamma` y `epsilon`, y en eta adaptativo se exigen incremento, fracción de
reducción y paciencia, porque la clase explica la estrategia pero no impone
una única configuración para el TP.

El perceptrón escalón conserva la regla de Rosenblatt online. No acepta estos
optimizadores basados en gradientes ni lotes mayores que uno porque el escalón
no tiene la derivada que necesitan; las comparaciones corresponden a los
modelos diferenciables.

## Entrenamiento y reproducibilidad

Los pesos se inicializan uniformemente en `[-init_scale, init_scale]` y los
bias en cero. Los pesos de distintas neuronas son aleatorios para romper la
simetría; no se asigna la solución conocida de AND ni de XOR al entrenar.
Los ejemplos manuales con todos los parámetros nulos se mantienen en tests
como comprobación separada de una sola actualización.

La inicialización y el orden usan generadores locales de NumPy, con semilla
explícita. Misma configuración, semilla y entorno produce la misma trayectoria.
Se documentan las versiones de Python y NumPy; no se promete igualdad de bits
entre versiones o plataformas diferentes.

Se registra la época cero y se evalúa todo el conjunto con parámetros fijos
al terminar cada época. No se mezclan predicciones de distintos momentos del
entrenamiento para calcular accuracy o MSE.

Criterios de aceptación, fijados antes de la corrida:

- AND: MSE = 0 y cuatro clases correctas.
- Lineal y tanh: MSE ≤ 10⁻⁶ sobre las 50 muestras.
- XOR, ambas arquitecturas: MSE ≤ 10⁻³ y cuatro clases correctas.
- Cada caso tiene un máximo de épocas. Alcanzarlo sin cumplir los criterios
  produce `converged=false`, y el comando termina con código 1 si algún caso falla.

En AND se mantiene el orden de la consigna. Los modelos diferenciables
barajan las mismas muestras al comenzar cada época; la semilla permite
repetir exactamente ese orden en el mismo entorno.

## Configuración y resultados

`validation.json` contiene solo cuatro campos globales: `seeds`,
`sample_count`, `input_interval` y `cases`. Cada caso declara todos los campos
que aparecen en el archivo de referencia; se rechazan claves desconocidas,
nombres duplicados, dimensiones incompatibles y valores de entrenamiento inválidos.
Las rutas de CLI son relativas al directorio desde el que se invoca el comando.

Cada corrida conserva:

- `config.json` y `environment.json`.
- `summary.csv` y `summary.json`, con aprobación/fallo por caso y semilla.
- Por caso/semilla: `initial.npz`, `model.npz`, `data.npz`, `history.csv` y
  `predictions.csv`.

El CLI exige una carpeta nueva o vacía para no pisar corridas anteriores.
El modelo se guarda sin pickle y puede recargarse para predecir o seguir
entrenando. Guarda parámetros y arquitectura, no el estado del generador de
barajado ni la época: continuar con `fit` inicia un nuevo tramo y no promete
reproducir una corrida ininterrumpida.

## Ejemplo de uso independiente de la validación

```python
import numpy as np
from sia_tp3 import MultilayerPerceptron, fit

X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float)
y = np.array([[1], [1], [-1], [-1]], dtype=float)
model = MultilayerPerceptron([2, 2, 1], activations=['tanh', 'tanh'], seed=0)
history = fit(model, X, y, learning_rate=0.03, max_epochs=10000,
              target_mse=0.001, shuffle=True, seed=0,
              require_bipolar_accuracy=True)
print(model.predict(X))
model.save('xor.npz')
restored = MultilayerPerceptron.load('xor.npz')
```

Para cambiar optimizador y modalidad sin cambiar el modelo:

```python
from sia_tp3 import Adam

optimizer = Adam()  # usa los valores de referencia de clase
history = fit(model, X, y, optimizer=optimizer, batch_size=2,
              max_epochs=10000, target_mse=0.001,
              shuffle=True, seed=0)
```

No se pasa `learning_rate` junto con `optimizer`: la tasa pertenece al objeto
optimizador. Cada corrida debe crear una instancia nueva para no compartir
Momentum o momentos acumulados entre modelos.

`fit` acepta además `validation_data=(X_validation, y_validation)`. Esas
muestras se predicen al terminar cada época y agregan `validation_mse` al
historial, pero no generan gradientes, no actualizan parámetros y no participan
del criterio de convergencia:

```python
history = fit(
    model, X_train, y_train,
    learning_rate=0.01,
    batch_size=32,
    max_epochs=500,
    target_mse=0,
    shuffle=True,
    seed=0,
    validation_data=(X_validation, y_validation),
)
```

## Carga y separación de los datasets reales

`data.py` expone los dos loaders. Fraude se estandariza después del split;
dígitos conserva los píxeles recibidos en `[0,1]`:

```python
from pathlib import Path
from sia_tp3 import load_digits_train_test, load_fraud_train_test

data_dir = Path('data')

fraud = load_fraud_train_test(data_dir / 'fraud_dataset.csv',
                              test_fraction=0.2, seed=0)

digits_exercise_2 = load_digits_train_test(
    data_dir / 'digits.csv', data_dir / 'digits_test.csv')

digits_exercise_3 = load_digits_train_test(
    data_dir / 'digits.csv', data_dir / 'digits_test.csv',
    additional_train_path=data_dir / 'more_digits.csv')
```

En fraude, `X_train` y `X_test` contienen las nueve variables de entrada
estandarizadas con la media y el desvío calculados exclusivamente sobre
training. `y_train`/`y_test` contienen la probabilidad de BigModel con forma
`(N, 1)` y conservan su escala original.
`flagged_fraud` no aparece entre las entradas ni como objetivo; se usa para
estratificar el split y solo se expone como `flagged_fraud_test` para la futura
evaluación final. La semilla vuelve reproducible la partición. El objeto
`fraud.standardizer` permite transformar entradas futuras y guardar sus
parámetros mediante `save`; debe conservarse junto con el modelo entrenado.

En dígitos, las imágenes quedan como matrices `float32` de forma `(N, 784)` y
las etiquetas como enteros de forma `(N,)`. `digits_test.csv` se carga siempre
como test externo. Para el ejercicio 3, el archivo adicional se concatena solo
al training. La codificación de las diez salidas se decidirá junto con el
modelo y no forma parte de la carga.

## Holdout temporal disponible para experimentos

`load_fraud_experiment_split` y `load_digits_experiment_split` conservan el
test externo y derivan training/validation del conjunto de desarrollo. Ambos
devuelven los índices internos para registrar exactamente la partición.

Estos helpers implementan **un único corte reproducible**. Son infraestructura
disponible, no el protocolo obligatorio de todos los ejercicios. El análisis
final de generalización del ejercicio 1 usa test reservado más 5-fold
estratificado mediante `scripts/analyze_fraud_generalization_extension.py`;
allí el estandarizador se ajusta nuevamente dentro de cada fold. Para los
ejercicios 2 y 3 todavía debe elegirse entre holdout y k-fold según el costo de
entrenar el MLP.

En fraude se leen primero los valores crudos, se reserva test, se separa
validation y sólo entonces se ajusta `Standardizer` con training interno. Sus
parámetros se reutilizan en validation y test. `flagged_fraud` sólo participa
en la estratificación y queda disponible para evaluar umbrales en validation y
test; nunca aparece en las entradas ni como objetivo entrenable.

```python
from sia_tp3 import load_fraud_experiment_split

data = load_fraud_experiment_split(
    'data/fraud_dataset.csv',
    test_fraction=0.2,
    validation_fraction=0.2,
    test_seed=0,
    validation_seed=0,
)
```

## Runner del ejercicio de fraude

`configs/fraud-learning.json` declara el protocolo `learning`: compara lineal y
logístico entrenando ambos con las 7500 muestras, sin crear validation ni test.
El runner también admite el protocolo `generalization`, pero exige exactamente
un modelo ya seleccionado y recién allí crea las tres particiones. Ese modo fue
el baseline inicial de holdout; no debe confundirse con el protocolo 5-fold
usado en el informe final del ejercicio 1.

El comando para la comparación obligatoria es:

```bash
python scripts/run_fraud_experiment.py \
  --config configs/fraud-learning.json \
  --data data/fraud_dataset.csv \
  --output output/fraud-learning-01
```

Los gráficos se generan después, leyendo los historiales sin volver a entrenar:

```bash
python scripts/plot_fraud_experiment.py output/fraud-learning-01
```

El script guarda las curvas completas en escala normal y logarítmica y un CSV
de diagnóstico. En `learning` grafica sólo training; en `generalization` agrega
validation. Nunca vuelve a entrenar para producir una figura.

Los valores del JSON son una propuesta para revisar, no resultados ni una
selección final.

Las arquitecturas y técnicas base corresponden a clases 10.1, 10.2 y 11; los
optimizadores corresponden a clase 12.1. El ejercicio 1 ya realizó su selección
experimental con 5-fold. Siguen pendientes la configuración y comparación de
los ejercicios 2 y 3. Los loaders mantienen development/test y
`experiments.py` ofrece un holdout para sus futuros runners; también puede
implementarse k-fold si su costo se justifica. No se utilizó `digits_test.csv`
para ninguna elección de este desarrollo.
