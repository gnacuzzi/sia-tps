# Guía del código: configuración, multicapa y casos borde

Esta guía explica cómo una configuración se convierte en una corrida, cómo se
usan las neuronas del perceptrón multicapa y qué decisiones tomó la
implementación frente a casos borde. También desarrolla dos piezas centrales
del clasificador de dígitos: **softmax con entropía cruzada** y el proceso de
**data augmentation**.

La explicación está organizada por bloques de código. La intención no es
comentar cada línea, sino entender qué responsabilidad tiene cada bloque y cómo
se conecta con el siguiente.

## 1. Mapa de una corrida

Una configuración JSON es una receta. No ejecuta nada por sí misma: un runner
la lee, valida sus campos y entrega cada decisión al componente correspondiente.

En la evaluación final de dígitos el recorrido es:

```text
configuración JSON
        ↓
validación de la configuración
        ↓
carga de development
        ↓
conversión de etiquetas a one-hot
        ↓
creación del MultilayerPerceptron
        ↓
entrenamiento por épocas y mini-batches
        ↓
guardado del modelo y del historial
        ↓
apertura y evaluación única de test
```

El runner que conecta estas etapas es
[`digit_final_evaluation.py`](../src/sia_tp3/digit_final_evaluation.py). La red y
el algoritmo de entrenamiento permanecen separados en
[`models.py`](../src/sia_tp3/models.py) y
[`training.py`](../src/sia_tp3/training.py).

Esta separación permite usar el mismo motor para XOR, fraude y dígitos sin que
el modelo necesite saber qué representan sus entradas.

## 2. Cómo la configuración afecta una corrida

La configuración final del ejercicio 3 es
[`17-final-evaluation.json`](ejercicio3/17-final-evaluation.json):

```json
{
  "protocol": "final",
  "architecture": [784, 128, 10],
  "activations": ["tanh", "softmax"],
  "loss": "categorical_cross_entropy",
  "l2_lambda": 0.0,
  "beta": 1.0,
  "init_scale": 0.5,
  "optimizer": {
    "name": "adam",
    "learning_rate": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8
  },
  "batch_size": 128,
  "epochs": 300,
  "model_seed": 0,
  "shuffle_seed": 0,
  "shuffle": true
}
```

La lectura práctica es:

> Crear una red que reciba 784 píxeles, tenga 128 neuronas ocultas y produzca
> diez probabilidades. Entrenarla con Adam en grupos de 128 imágenes durante
> 300 épocas.

### 2.1 Parámetros que definen el modelo

| Parámetro | Efecto |
|---|---|
| `architecture` | Define la cantidad de entradas, neuronas por capa y salidas. |
| `activations` | Elige la transformación aplicada después de cada suma ponderada. |
| `loss` | Define qué error intenta reducir la red. |
| `beta` | Controla la pendiente de `tanh` y logística; no modifica softmax en este código. |
| `init_scale` | Determina el intervalo uniforme de los pesos iniciales. |
| `l2_lambda` | Agrega o no una penalización por pesos grandes. |
| `model_seed` | Hace reproducibles los pesos iniciales. |

El runner utiliza esos valores al construir la red:

```python
model = MultilayerPerceptron(
    config["architecture"],
    activations=config["activations"],
    beta=config["beta"],
    seed=config["model_seed"],
    init_scale=config["init_scale"],
    loss=config.get("loss", "mse"),
    l2_lambda=config.get("l2_lambda", 0.0),
)
```

### 2.2 Parámetros que definen el entrenamiento

| Parámetro | Efecto |
|---|---|
| `optimizer` | Decide cómo transformar gradientes en cambios de parámetros. |
| `learning_rate` | Controla el tamaño de esos cambios. |
| `learning_rate_schedule` | Cambia el learning rate a partir de ciertas épocas. |
| `batch_size` | Indica cuántas muestras participan en cada actualización. |
| `epochs` | Fija la cantidad de pasadas completas por development. |
| `shuffle` | Decide si se cambia el orden de las muestras en cada época. |
| `shuffle_seed` | Hace reproducible ese orden. |

En la configuración final, el learning rate cambia así:

```text
épocas   1–150 → 0,01
épocas 151–220 → 0,003
épocas 221–300 → 0,001
```

Las correcciones son mayores al principio y más pequeñas al final.

### 2.3 Parámetros que definen el protocolo

`protocol: "final"` no cambia una ecuación de la red. Cambia el procedimiento
experimental: se entrena con todo development y el test reservado se abre una
sola vez después de terminar el entrenamiento.

Por eso no todos los campos del JSON llegan al modelo. Algunos configuran la
red, otros el entrenamiento y otros el protocolo experimental.

## 3. Cómo se representan y usan las neuronas

La arquitectura actual es:

```text
784 entradas → 128 neuronas ocultas → 10 neuronas de salida
```

Los 784 valores de entrada no son parámetros entrenables: son los píxeles de
una imagen. Los parámetros entrenables son pesos y bias.

Para una neurona oculta `j`:

\[
h_j = \sum_{i=1}^{784} w_{ji}x_i+b_j
\]

\[
a_j=\tanh(\beta h_j)
\]

La neurona recibe todos los píxeles porque la red es densa. El código no le
asigna de antemano el significado de "borde", "curva" o "parte de un cinco".
Si aparece una representación de ese estilo, surge de los pesos aprendidos.

Las matrices creadas para `[784, 128, 10]` son:

```text
W0: (128, 784)  — una fila de 784 pesos por neurona oculta
b0: (128,)      — un bias por neurona oculta

W1: (10, 128)   — una fila de 128 pesos por salida
b1: (10,)       — un bias por clase
```

La cantidad total de parámetros es:

```text
128 × 784 + 128 = 100.480
 10 × 128 +  10 =   1.290
Total             = 101.770
```

Todas las neuronas participan para todas las imágenes. Esta implementación no
usa dropout, poda ni selección dinámica de neuronas.

## 4. Forward propagation

El avance por todas las capas se implementa con un único bloque vectorizado:

```python
outputs = [X]
for w, b, name in zip(self.weights, self.biases, self.activations):
    h = outputs[-1] @ w.T + b
    activation = self._activate(h, name)
    outputs.append(activation)
```

Para un mini-batch de 128 imágenes, las formas son:

```text
X                         (128, 784)
X @ W0.T + b0             (128, 128)
tanh(...)                 (128, 128)
capa oculta @ W1.T + b1   (128, 10)
softmax(...)              (128, 10)
```

Cada fila de la última matriz corresponde a una imagen y contiene diez valores,
uno por dígito.

## 5. Softmax

### 5.1 El problema que resuelve

Antes de softmax, las diez salidas son números sin una escala probabilística
obligatoria. Por ejemplo:

```text
logits = [1,2; 0,1; 3,0]
```

No suman uno y pueden ser negativos. Softmax convierte esos valores en una
distribución:

\[
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
\]

Para tres logits `[2, 1, 0]`, aproximadamente:

```text
exp(logits) = [7,389; 2,718; 1]
suma        = 11,107
softmax     = [0,665; 0,245; 0,090]
```

Los resultados quedan entre cero y uno y suman uno. La clase predicha es la de
mayor valor:

```python
predicted_label = np.argmax(probabilities)
```

### 5.2 Implementación numéricamente estable

El código no calcula directamente `exp(h)`, porque un logit muy grande podría
producir overflow. Primero resta el máximo de cada muestra:

```python
shifted = h - np.max(h, axis=1, keepdims=True)
exponentials = np.exp(shifted)
return exponentials / exponentials.sum(axis=1, keepdims=True)
```

Restar la misma constante a todos los logits no modifica softmax:

\[
\operatorname{softmax}(z)=\operatorname{softmax}(z-c)
\]

Por ejemplo, para `[1002, 1001, 1000]` el máximo es `1002`, por lo que el
código calcula:

```text
[1002, 1001, 1000] - 1002 = [0, -1, -2]
```

Se obtiene la misma distribución sin calcular `exp(1002)`. Restar `1000`
también conservaría matemáticamente el resultado y produciría `[2, 1, 0]`,
pero **no es lo que hace nuestro código**: se resta el máximo para asegurar que
el mayor argumento de la exponencial sea siempre cero y que todos los demás
sean negativos.

### 5.3 Qué representa cada salida

En dígitos, la posición tiene significado fijo:

```text
índice 0 → dígito 0
índice 1 → dígito 1
...
índice 9 → dígito 9
```

Una salida como:

```text
[0,01; 0,02; 0,03; 0,04; 0,05; 0,78; 0,02; 0,01; 0,02; 0,02]
```

se clasifica como `5` porque el índice 5 contiene el máximo.

En nuestro código esto es exactamente **softmax seguido de argmax**:

```python
outputs = model.predict(X)                 # ya contienen softmax
predicted_labels = np.argmax(outputs, axis=1)
```

`softmax` no elige la clase: transforma los logits en valores normalizados.
`argmax` realiza la elección tomando la posición del valor mayor.

Como la exponencial es creciente y todas las salidas se dividen por la misma
suma, softmax conserva el orden de los logits:

```text
logits                = [2;     1;     0]
softmax               = [0,665; 0,245; 0,090]
argmax en ambos casos = índice 0
```

Por eso, si sólo interesara la clase final, hacer `argmax` sobre los logits daría
la misma clase. Softmax sigue siendo importante en nuestro entrenamiento porque
produce la distribución que recibe la entropía cruzada y permite medir cuánto
valor se asignó a la clase correcta.

No conviene interpretar siempre `0,78` como una probabilidad perfectamente
calibrada de que la imagen sea un cinco. En esta implementación funciona como
score normalizado para comparar las clases; la calibración no fue un objetivo
del experimento.

## 6. Entropía cruzada categórica

### 6.1 Entropía y entropía cruzada no son exactamente lo mismo

La **entropía** describe cuánta incertidumbre tiene una distribución. Por
ejemplo, una distribución muy repartida entre las clases tiene más
incertidumbre que una concentrada en una sola clase.

En este entrenamiento no calculamos la entropía de la predicción por sí sola.
Usamos **entropía cruzada**, que compara dos distribuciones:

- `y`: la distribución objetivo, construida con one-hot;
- `p`: la distribución predicha por softmax.

Su pregunta práctica es:

> ¿Cuánta probabilidad le dio el modelo a la clase que sabemos que era la
> correcta?

### 6.2 Objetivo one-hot

La etiqueta entera se convierte antes del entrenamiento. Para un cinco:

```text
etiqueta = 5
one-hot  = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```

La pérdida para una muestra es:

\[
L=-\sum_k y_k\log(p_k)
\]

Como sólo una posición de `y` vale uno, queda:

\[
L=-\log(p_{\text{clase correcta}})
\]

Si la clase correcta es cinco:

```text
probabilidad asignada al 5 = 0,90 → loss ≈ 0,105
probabilidad asignada al 5 = 0,50 → loss ≈ 0,693
probabilidad asignada al 5 = 0,01 → loss ≈ 4,605
```

La pérdida castiga con fuerza estar muy seguro de una respuesta incorrecta.

No se usa para elegir directamente la clase predicha. Se usa durante training
para producir un número que representa el error y, a partir de él, calcular en
qué dirección deben cambiar los pesos.

### 6.3 Ejemplo completo de uso durante training

Supongamos tres clases y que la respuesta correcta es la clase 1:

```text
logits producidos por la red = [2; 1; 0]
softmax                      = [0,665; 0,245; 0,090]
objetivo one-hot             = [0;     1;     0]
```

La suma completa sería:

\[
L=-\left(0\log(0{,}665)+1\log(0{,}245)+0\log(0{,}090)\right)
\]

Los términos multiplicados por cero desaparecen:

\[
L=-\log(0{,}245)\approx 1{,}407
\]

La red predijo la clase 0, pero la correcta era la 1. La pérdida es relativamente
alta porque sólo asignó `0,245` a la clase correcta.

Si después de aprender produjera:

```text
softmax nuevo = [0,05; 0,90; 0,05]
```

entonces:

\[
L=-\log(0{,}90)\approx 0{,}105
\]

La pérdida bajó. El entrenamiento busca justamente modificar los pesos para
reducir el promedio de esta pérdida sobre el batch.

### 6.4 Implementación de la pérdida

```python
safe = np.clip(
    predictions,
    np.finfo(np.float64).tiny,
    1.0,
)
loss = -np.mean(np.sum(y * np.log(safe), axis=1))
```

`clip` evita calcular `log(0)`, que produciría infinito. Sólo protege el cálculo
numérico de la pérdida; no cambia la clase elegida.

### 6.5 Por qué softmax y entropía cruzada se usan juntas

Softmax produce una distribución entre clases y la entropía cruzada mide cuánta
masa recibió la clase correcta. Además, sus derivadas se simplifican:

```python
delta = outputs[-1] - y
```

Es decir:

\[
\delta=p-y
\]

Ejemplo para tres clases:

```text
predicción = [0,70; 0,20; 0,10]
objetivo   = [0;    1;    0]
delta      = [0,70; -0,80; 0,10]
```

La segunda salida necesita aumentar, mientras que la primera y la tercera
necesitan disminuir. Backpropagation usa ese delta para calcular qué parte del
error corresponde a cada peso y a cada neurona oculta.

La clase `MultilayerPerceptron` exige que softmax y entropía cruzada aparezcan
juntas. Una configuración que incluya sólo una de las dos se rechaza antes de
entrenar.

## 7. Backpropagation y actualización

Después del forward se recorren las capas en sentido inverso:

```python
for layer in range(len(self.weights) - 1, -1, -1):
    dw[layer] = delta.T @ outputs[layer] / len(X)
    dw[layer] += self.l2_lambda * self.weights[layer]
    db[layer] = delta.mean(axis=0)

    if layer:
        delta = (
            delta @ self.weights[layer]
            * self._derivative(
                outputs[layer],
                self.activations[layer - 1],
            )
        )
```

El bloque realiza tres tareas:

1. Calcula el gradiente de los pesos y bias de la capa actual.
2. Propaga la responsabilidad del error hacia la capa anterior.
3. Agrega el gradiente de L2 cuando `l2_lambda` es mayor que cero.

Los gradientes no modifican directamente el modelo. Se entregan al optimizador;
en la configuración final, Adam mantiene promedios móviles del gradiente y de
su cuadrado antes de decidir el cambio de cada parámetro.

## 8. Data augmentation

### 8.1 Para qué se usa

Data augmentation genera variaciones pequeñas de las imágenes durante el
entrenamiento. Busca que la red no dependa de que un dígito esté exactamente en
la misma posición y orientación que en el CSV.

No agrega archivos nuevos ni modifica el dataset original. Cada mini-batch se
copia y se transforma en memoria antes de calcular sus gradientes.

La configuración final indica:

```json
"augmentation": {
  "name": "translation_rotation",
  "max_shift": 1,
  "translation_probability": 0.5,
  "max_angle_degrees": 4,
  "rotation_probability": 0.5,
  "seed": 0
}
```

Para cada imagen y en cada aparición durante el entrenamiento hay dos decisiones
aleatorias independientes:

- 50 % de probabilidad de trasladarla hasta un píxel;
- 50 % de probabilidad de rotarla un ángulo entre `-4°` y `4°`.

Aproximadamente, las combinaciones esperadas son:

```text
25 % sin transformación
25 % sólo trasladadas
25 % sólo rotadas
25 % trasladadas y rotadas
```

Son proporciones esperadas, no cantidades exactas por batch.

### 8.2 Dónde entra en el entrenamiento

El runner construye una función parcialmente configurada:

```python
batch_transform = partial(
    random_translate_rotate_images,
    max_shift=augmentation["max_shift"],
    translation_probability=augmentation["translation_probability"],
    max_angle_degrees=augmentation["max_angle_degrees"],
    rotation_probability=augmentation["rotation_probability"],
)
```

`fit` la aplica sólo al batch de entradas:

```python
batch_X, batch_y = X[indices], y[indices]
batch_X = batch_transform(batch_X, transform_rng)
dw, db = model._gradients(batch_X, batch_y)
```

Las etiquetas no cambian: un cinco trasladado o rotado sigue siendo un cinco.

Las métricas al final de la época se calculan sobre `X` original, no sobre una
versión aumentada aleatoria. Así se puede comparar la evolución de las épocas
sobre una referencia estable.

### 8.3 Traslación

Para cada imagen seleccionada se elige un desplazamiento `(dy, dx)` dentro de:

```text
-max_shift ≤ dy ≤ max_shift
-max_shift ≤ dx ≤ max_shift
```

Se excluye `(0, 0)`, porque una imagen seleccionada para trasladarse debe moverse
realmente.

La imagen destino empieza llena de ceros:

```python
translated = np.zeros(
    (len(indices), 28, 28),
    dtype=X.dtype,
)
translated[:, destination_y, destination_x] = (
    source_images[indices, source_y, source_x]
)
```

Los píxeles que salen por un borde se pierden y las posiciones nuevas reciben
fondo cero. No reaparecen del lado opuesto.

### 8.4 Rotación

Cada imagen seleccionada recibe un ángulo uniforme dentro del intervalo
configurado:

```python
angles = rng.uniform(
    -max_angle_degrees,
    max_angle_degrees,
    size=len(selected),
)
```

La implementación usa mapeo inverso: para cada píxel de la imagen destino busca
qué posición de la imagen original le corresponde. Como esa posición puede caer
entre cuatro píxeles, utiliza interpolación bilineal para combinarlos.

```text
destino
   ↓ transformación inversa
coordenada decimal en la imagen original
   ↓
promedio ponderado de sus cuatro vecinos
```

Si la coordenada cae fuera de la imagen, su aporte es cero. Esto mantiene un
fondo negro y evita envolver la imagen.

### 8.5 Reproducibilidad y ausencia de mutación

La semilla de augmentation crea un generador independiente del que mezcla las
muestras:

```python
rng = np.random.default_rng(shuffle_seed)
transform_rng = np.random.default_rng(batch_transform_seed)
```

Por eso se pueden cambiar el orden de los datos y las transformaciones de forma
controlada. Con las mismas semillas y configuración, la corrida reproduce las
mismas decisiones aleatorias.

Las funciones comienzan con una copia:

```python
output = X.copy()
```

Por lo tanto, no modifican las imágenes originales almacenadas en development.

## 9. Casos borde y decisiones de implementación

Esta sección está escrita con el formato útil para una defensa oral:
**situación, decisión e implementación**.

### 9.1 El total de muestras no es divisible por `batch_size`

**Situación:** con 1000 imágenes y batches de 128 quedan 104 imágenes al final.

**Decisión:** no descartarlas ni rellenar el batch. Se procesa un último batch
de 104 muestras.

```python
for start in range(0, len(X), batch_size):
    indices = order[start:start + batch_size]
```

El slicing devuelve automáticamente las muestras restantes. `_gradients`
divide por `len(X)` del batch recibido, por lo que el último gradiente se
promedia usando 104 y no 128.

### 9.2 Una clase no se divide exactamente entre los folds

**Situación:** once ejemplos de una clase no pueden repartirse en cinco folds
iguales.

**Decisión:** usar folds cuyos tamaños difieren como máximo en uno y conservar
todas las muestras.

```python
np.array_split(shuffled, fold_count)
```

El reparto de esa clase queda `3, 2, 2, 2, 2`. También se exige que cada clase
tenga por lo menos `fold_count` muestras; de lo contrario, la configuración se
rechaza.

### 9.3 Una clase está ausente

**Situación:** el training original del ejercicio 2 no contiene el dígito 8.

**Decisión:** conservar explícitamente las diez clases en la matriz de confusión
y marcar como `NaN` las métricas que no se pueden calcular.

```python
def _ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")
```

`NaN` significa "no evaluable". Informar cero afirmaría incorrectamente que la
métrica fue medida y dio cero. Los promedios macro usados para comparar modelos
ignoran valores no finitos y, durante development, se informan además las clases
presentes y ausentes.

### 9.4 Dos fuentes contienen la misma imagen

**Situación:** `digits.csv` y `more_digits.csv` pueden solaparse.

**Decisión:** deduplicar antes de separar training y validation.

- Imagen idéntica con la misma etiqueta: conservar una representación y guardar
  la trazabilidad de ambas filas.
- Imagen idéntica con etiquetas diferentes: detener la corrida.
- Duplicado interno dentro de una misma fuente: detener la corrida.

Deduplicar antes del split evita que una copia aparezca en training y otra en
validation, lo que produciría una estimación demasiado optimista.

### 9.5 Una feature tiene desvío estándar cero

**Situación:** una columna constante produciría una división por cero al
estandarizar.

**Decisión:** usar escala uno para esa columna.

```python
scale = np.where(std == 0, 1.0, std)
```

Como todos sus valores coinciden con la media, la columna transformada queda en
cero y no introduce información artificial.

### 9.6 Una transformación sale del borde de la imagen

**Situación:** una traslación o rotación puede pedir píxeles fuera de `28 × 28`.

**Decisión:** usar fondo cero y no envolver los píxeles hacia el borde opuesto.

Esta elección representa mejor una imagen movida dentro de un lienzo negro.

### 9.7 Softmax recibe logits muy grandes

**Situación:** `exp(1000)` puede desbordar la representación numérica.

**Decisión:** restar el mayor logit antes de aplicar la exponencial. La
distribución resultante no cambia.

### 9.8 La entropía cruzada intentaría calcular `log(0)`

**Situación:** una probabilidad puede redondearse numéricamente a cero.

**Decisión:** limitarla inferiormente al menor `float64` positivo antes de
aplicar el logaritmo.

### 9.9 Hay un empate entre clases

**Situación:** dos salidas tienen exactamente el mismo valor máximo.

**Decisión efectiva:** `np.argmax` elige el primer índice máximo. Por ejemplo,
un empate entre las clases 1 y 2 se resuelve a favor de la clase 1.

Es un comportamiento heredado de NumPy, no una regla específica agregada por
el proyecto. Los empates exactos son poco frecuentes con pesos continuos.

### 9.10 La configuración o los datos son inválidos

**Situación:** dataset vacío, valores no finitos, dimensiones incompatibles,
batch demasiado grande, activaciones incompatibles o parámetros fuera de rango.

**Decisión:** fallar antes de entrenar en vez de corregir silenciosamente la
configuración. Esto permite conocer exactamente qué experimento se ejecutó.

También se exige una carpeta de salida nueva o vacía para no mezclar o pisar la
evidencia de corridas anteriores.

## 10. Resumen para explicar oralmente

Una respuesta compacta que conecta todo el proceso sería:

> La configuración separa decisiones del modelo, del entrenamiento y del
> protocolo experimental. Para dígitos construimos una red densa
> `[784, 128, 10]`: todas las neuronas ocultas procesan cada imagen y las diez
> salidas representan las clases. Softmax normaliza los logits y la entropía
> cruzada penaliza la probabilidad asignada a la clase correcta; juntas producen
> el delta simple `p - y` para backpropagation. Durante training aplicamos
> traslaciones y rotaciones aleatorias sólo sobre copias de cada mini-batch. La
> implementación conserva batches y folds incompletos, mantiene visibles las
> clases ausentes, evita divisiones y logaritmos inválidos, deduplica imágenes
> antes del split y rechaza configuraciones ambiguas o incompatibles.
