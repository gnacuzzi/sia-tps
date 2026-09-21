# Ejercicio de validación — fórmulas y cálculos

Fuente: página 2 del [enunciado](Enunciado%20TP3%20-%202Q%202026.pdf).
Estos ejercicios no se presentan: se usan para comprobar las implementaciones.

Este apunte prepara la revisión paso a paso. La consigna fija los problemas,
pero no detalla las reglas de aprendizaje ni la notación. Usamos bias aditivo,
actualización por muestra y error cuadrático para los modelos diferenciables.
Estas convenciones se contrastaron con las clases 10 y 11 recibidas.
Las referencias están en [material de clases](material-clases.md).
La implementación está en `src/sia_tp3/` y los resultados de entrenamiento
en [el informe de validación](resultados-validacion.md). Los pesos elegidos
a mano de este apunte siguen siendo ejemplos didácticos independientes.

### Equivalencias con la notación de la cátedra

| En este apunte | En las diapositivas | Significado |
|---|---|---|
| $y^{(\mu)}$ | $\zeta^\mu$ | Salida esperada para la muestra $\mu$ |
| $\hat y^{(\mu)}$ | $O^\mu$ | Salida obtenida |
| $g$ | $\theta$ | Función de activación |
| $b$ | $w_0$, con $x_0=1$ | Bias aditivo |
| $a_j^{(\ell)}$ | $V_j^m$ | Salida de la neurona $j$ de una capa intermedia |
| $L$ | $M$ | Índice de la última capa |
| $N$ | $p$ | Cantidad de muestras |

En la formalización inicial de la clase 10.1, el umbral se llama $u$ y se
resta; por lo tanto, $b=-u$. No confundir ese umbral con $\theta$, que en
las diapositivas nombra la función de activación.

## 1. La cuenta común a todos los perceptrones

Cada neurona recibe entradas, las multiplica por pesos y suma un sesgo o bias:

$$
h = \sum_{i=1}^{d} w_i x_i + b,
\qquad \hat y = g(h).
$$

| Símbolo | Significado |
|---|---|
| $x_i$ | Componente de la entrada |
| $w_i$ | Peso que se aprende |
| $b$ | Bias que se aprende; desplaza el umbral de activación |
| $h$ | Suma ponderada antes de aplicar la activación |
| $g$ | Función de activación |
| $y$ | Salida esperada |
| $\hat y$ | Salida calculada |
| $\eta>0$ | Tasa de aprendizaje: tamaño del ajuste |

Para dos entradas: $h=w_1x_1+w_2x_2+b$.
Si escribimos $h=\sum_i w_i x_i-u$ como en la clase, la equivalencia es $b=-u$.
También se puede absorber el bias en un peso usando $x_0=1$ y $w_0=b$.

Una **época** recorre todas las muestras. En actualización por muestra, los
pesos cambian después de procesar cada una.

## 2. Perceptrón simple escalón: AND

Usamos $-1$ como falso y $+1$ como verdadero. Respetamos el orden del enunciado:

| $x_1$ | $x_2$ | $y$ |
|---:|---:|---:|
| -1 | 1 | -1 |
| 1 | -1 | -1 |
| -1 | -1 | -1 |
| 1 | 1 | 1 |

### Predicción

$$
\hat y = g(h)=
\begin{cases}
+1 & h\geq 0,\\
-1 & h<0.
\end{cases}
$$

Elegimos explícitamente $g(0)=+1$ para que los cálculos sean reproducibles,
igual que en la formulación bipolar de la clase 10.1, página 30.
La página 15 introduce una salida $0/1$ y el pseudocódigo de la página 42
también usa esa codificación, con condición estricta. Para este ejercicio
seguimos la consigna bipolar y la convención de la página 30; no mezclamos
ambas versiones.

### Aprendizaje

$$
e=y-\hat y,
\qquad w_i^{\mathrm{nuevo}}=w_i+\eta e x_i,
\qquad b^{\mathrm{nuevo}}=b+\eta e.
$$

Si acierta, $e=0$ y no modifica nada. Si falla, $e$ vale $+2$ o $-2$
porque las salidas son bipolares. Coincide con la clase 10.1, páginas 33–34:
si falla, $\Delta w_i=2\eta y x_i$; si acierta, $\Delta w_i=0$.

Esta es la regla del perceptrón; no se obtiene derivando la función escalón.

### Una actualización a mano

Tomamos $w_1=w_2=b=0$, $\eta=0.1$ y la primera muestra $x=(-1,1)$, $y=-1$:

$$
h=0,\quad \hat y=+1,\quad e=-1-1=-2.
$$

$$
w_1'=0+0.1(-2)(-1)=0.2,\quad
w_2'=0+0.1(-2)(1)=-0.2,\quad
b'=0+0.1(-2)=-0.2.
$$

Al volver a evaluar esa muestra, $h=0.2(-1)-0.2(1)-0.2=-0.6$ y
$\hat y=-1$. Corrigió esa muestra; todavía hay que revisar las otras tres.

### Una solución conocida para comprobar las predicciones

Con $w_1=w_2=1$ y $b=-1$:

| Entrada | $h=x_1+x_2-1$ | $\hat y$ |
|---|---:|---:|
| $(-1,1)$ | -1 | -1 |
| $(1,-1)$ | -1 | -1 |
| $(-1,-1)$ | -3 | -1 |
| $(1,1)$ | 1 | 1 |

Estos pesos se eligieron a mano. No son el resultado de una corrida.
La frontera $x_1+x_2-1=0$ separa los puntos: AND es linealmente separable.
La comprobación de entrenamiento será obtener cero errores en una evaluación
de las cuatro muestras con los mismos pesos finales.

## 3. Perceptrón simple lineal: ajustar y = x

La consigna propone, como ejemplo, 50 muestras. Para concretarlo podemos usar
50 valores equiespaciados en $[-1,1]$ y definir $y^{(\mu)}=x^{(\mu)}$.
El intervalo es una elección del apunte, no un requisito del enunciado.

### Predicción y error

$$
g(h)=h,\qquad \hat y=wx+b,
\qquad E_\mu=\frac12(y-\hat y)^2.
$$

El cuadrado evita que errores positivos y negativos se cancelen.
El factor $1/2$ simplifica la derivada. Para informar el ajuste sobre $N$ muestras:

$$
\operatorname{MSE}=\frac1N\sum_{\mu=1}^{N}
\left(y^{(\mu)}-\hat y^{(\mu)}\right)^2.
$$

La clase 10.2, página 12, escribe el error del conjunto como
$E=\sum_\mu E_\mu=\frac12\sum_\mu(y^{(\mu)}-\hat y^{(\mu)})^2$.
Entonces $E=\frac N2\operatorname{MSE}$. No son el mismo número, aunque
para un conjunto fijo tienen los mismos mínimos. Una tolerancia para uno
no puede copiarse directamente al otro.

### Aprendizaje por descenso del gradiente

$$
\frac{\partial E_\mu}{\partial w}=-(y-\hat y)x,
\qquad \frac{\partial E_\mu}{\partial b}=-(y-\hat y).
$$

Restar el gradiente da:

$$
w'=w+\eta(y-\hat y)x,
\qquad b'=b+\eta(y-\hat y).
$$

La forma coincide con la regla anterior, pero ahora la salida es continua y la
actualización sí deriva del error cuadrático.

Estas ecuaciones usan $E_\mu$: procesamos **una muestra** y actualizamos,
como el algoritmo online de la clase 10.2, página 21. Si usáramos el error
total $E$ para una actualización batch, sumaríamos los gradientes de todas
las muestras calculados con los mismos pesos. Promediar en vez de sumar
cambia la escala del gradiente y requiere tenerlo en cuenta al elegir $\eta$.

### Ejemplo

Para $x=y=0.5$, $w=b=0$ y $\eta=0.1$:

$$
\hat y=0,\quad e=0.5,\quad w'=0.025,\quad b'=0.05.
$$

La nueva predicción para esa muestra es $0.025(0.5)+0.05=0.0625$,
más cerca de $0.5$. La solución exacta del conjunto es $w=1$, $b=0$.
Eso sirve como referencia para comprobar el ajuste; no implica que una
cantidad finita de actualizaciones alcance exactamente esos valores.

## 4. Perceptrón simple no lineal: ajustar y = tanh(x)

Podemos usar las mismas 50 entradas y tomar $y^{(\mu)}=\tanh(x^{(\mu)})$.
Elegimos como activación la misma función que queremos ajustar:

$$
h=wx+b,\qquad \hat y=\tanh(h),
\qquad \tanh(h)=\frac{e^h-e^{-h}}{e^h+e^{-h}}.
$$

Su derivada es:

$$
g'(h)=1-\tanh^2(h)=1-\hat y^2.
$$

Con $E_\mu=\frac12(y-\hat y)^2$, la regla de la cadena da:

$$
\delta=(y-\hat y)(1-\hat y^2),
\qquad w'=w+\eta\delta x,
\qquad b'=b+\eta\delta.
$$

La diferencia con el lineal es la derivada de la activación. Cuando la salida
está cerca de $-1$ o $+1$, esa derivada es pequeña: la neurona está saturada
y los ajustes se reducen aunque exista error.

### Ejemplo

Para $x=0.5$, $y=\tanh(0.5)\approx0.462117$, $w=b=0$ y $\eta=0.1$:

$$
h=0,\quad \hat y=0,\quad \delta\approx0.462117,
\quad w'\approx0.023106,\quad b'\approx0.046212.
$$

También acá $w=1$, $b=0$ representa exactamente la función objetivo.
Si se usa $g(h)=\tanh(\beta h)$, hay que incluir el factor $\beta$ en
la derivada: $g'(h)=\beta(1-g(h)^2)$. En este apunte usamos $\beta=1$.

## 5. Perceptrón multicapa: XOR

| $x_1$ | $x_2$ | $y$ |
|---:|---:|---:|
| -1 | 1 | 1 |
| 1 | -1 | 1 |
| -1 | -1 | -1 |
| 1 | 1 | -1 |

XOR devuelve $+1$ cuando las entradas son distintas. Sus clases ocupan
esquinas opuestas: ninguna recta puede separarlas. Por eso un perceptrón
simple escalón no puede clasificar correctamente los cuatro puntos.

Una red multicapa incorpora representaciones intermedias. Las activaciones
ocultas deben ser no lineales: componer transformaciones afines solamente
produce otra transformación afín.

### Propagación hacia adelante

Usamos vectores columna. $a^{(0)}=x$, y para cada capa $\ell$:

$$
h^{(\ell)}=W^{(\ell)}a^{(\ell-1)}+b^{(\ell)},
\qquad a^{(\ell)}=\tanh(h^{(\ell)}).
$$

La tangente hiperbólica se aplica componente a componente. La última capa
produce $\hat y=a^{(L)}$. Para clasificar usamos $+1$ si $\hat y\geq0$
y $-1$ en otro caso. El entrenamiento usa la salida continua, sin umbralizar.

### Arquitectura [2, 2, 1]

$$
\begin{aligned}
a_1^{(1)}&=\tanh(w_{11}^{(1)}x_1+w_{12}^{(1)}x_2+b_1^{(1)}),\\
a_2^{(1)}&=\tanh(w_{21}^{(1)}x_1+w_{22}^{(1)}x_2+b_2^{(1)}),\\
\hat y&=\tanh(w_{11}^{(2)}a_1^{(1)}+w_{12}^{(2)}a_2^{(1)}+b_1^{(2)}).
\end{aligned}
$$

Tiene 9 parámetros: $2\cdot2+2$ en la capa oculta y $1\cdot2+1$ en la salida.

### Arquitectura [2, 3, 2, 1]

$$
\begin{aligned}
a^{(1)}&=\tanh(W^{(1)}x+b^{(1)}),\\
a^{(2)}&=\tanh(W^{(2)}a^{(1)}+b^{(2)}),\\
\hat y&=\tanh(W^{(3)}a^{(2)}+b^{(3)}).
\end{aligned}
$$

| Capa | Dimensión de $W$ | Dimensión de $b$ | Parámetros |
|---|---|---|---:|
| Primera oculta | $3\times2$ | $3\times1$ | 9 |
| Segunda oculta | $2\times3$ | $2\times1$ | 8 |
| Salida | $1\times2$ | $1\times1$ | 3 |
| Total | | | 20 |

### Aprendizaje: backpropagation

Para una muestra usamos $E_\mu=\frac12(y-\hat y)^2$. Definimos $\delta$
con el signo de corrección, de modo que se **suma** en la actualización.

En la salida:

$$
\delta^{(L)}=(y-a^{(L)})\odot\left(1-(a^{(L)})^2\right).
$$

En cada capa oculta, recorriendo hacia atrás:

$$
\delta^{(\ell)}=
\left((W^{(\ell+1)})^T\delta^{(\ell+1)}\right)
\odot\left(1-(a^{(\ell)})^2\right).
$$

$\odot$ significa multiplicación componente a componente. El error de salida
se distribuye hacia atrás según los pesos y la derivada de cada neurona.

Luego actualizamos:

$$
W^{(\ell)\prime}=W^{(\ell)}+
\eta\delta^{(\ell)}(a^{(\ell-1)})^T,
\qquad
b^{(\ell)\prime}=b^{(\ell)}+\eta\delta^{(\ell)}.
$$

Hay que calcular **todos los deltas antes de modificar los pesos**: deben
corresponder a la misma pasada hacia adelante. En el multicapa no conviene
inicializar todas las neuronas iguales, porque aprenderían de forma simétrica.

Estas expresiones son la versión matricial de la clase 11, página 51.
Incluimos un bias entrenable por neurona en todas las capas no pertenecientes
a la entrada para permitir desplazamientos de cada activación; esta opción
aparece en las páginas 63–70. Lo escribimos separado de $W$ para que se vea
qué se actualiza. La transcripción de clase 11 aclara que también se puede
representar así, en lugar de agregar una entrada constante a cada capa.
La inicialización aleatoria pequeña recomendada en las páginas 57–58 se
definirá al implementar; los ceros de los ejemplos simples son solo didácticos.

### Ejemplo de predicción para [2, 2, 1]

Para visualizar que la arquitectura puede representar XOR, elegimos a mano:

$$
W^{(1)}=\begin{pmatrix}1&-1\\-1&1\end{pmatrix},\quad
b^{(1)}=\begin{pmatrix}-1\\-1\end{pmatrix},\quad
W^{(2)}=\begin{pmatrix}1&1\end{pmatrix},\quad b^{(2)}=1.
$$

| Entrada | $h^{(1)}$ | $a^{(1)}$ aprox. | $h^{(2)}$ aprox. | $\hat y$ aprox. | Clase |
|---|---|---|---:|---:|---:|
| $(-1,1)$ | $(-3,1)$ | $(-0.9951,0.7616)$ | 0.7665 | 0.6449 | 1 |
| $(1,-1)$ | $(1,-3)$ | $(0.7616,-0.9951)$ | 0.7665 | 0.6449 | 1 |
| $(-1,-1)$ | $(-1,-1)$ | $(-0.7616,-0.7616)$ | -0.5232 | -0.4802 | -1 |
| $(1,1)$ | $(-1,-1)$ | $(-0.7616,-0.7616)$ | -0.5232 | -0.4802 | -1 |

Clasifica los cuatro puntos, pero su error cuadrático no es cero. Con pesos
finitos, $\tanh$ no alcanza exactamente $\pm1$: hay que distinguir acierto
de clasificación de cercanía a las salidas objetivo.

Este ejemplo por sí solo no valida backpropagation ni demuestra convergencia.
Los tests del motor ahora comparan una actualización manual de `[2,2,1]` y
gradientes numéricos de todos los parámetros de ambas arquitecturas. El
informe de resultados presenta por separado el entrenamiento desde pesos aleatorios.

## 6. Comprobaciones de la implementación

| Caso | Comprobación |
|---|---|
| AND | Cuatro predicciones correctas y actualización manual coincidente |
| Lineal | Ajuste cercano a $y=x$ y MSE pequeño |
| No lineal | Ajuste cercano a $\tanh(x)$ y derivada incluida en la actualización |
| XOR | Cuatro clases correctas; forward y backpropagation manuales coincidentes en ambas arquitecturas |

Se fijó MSE ≤ 10⁻⁶ para lineal y no lineal, y MSE ≤ 10⁻³ más cuatro clases
correctas para XOR. Los máximos de épocas y condiciones iniciales reproducibles
están en `configs/validation.json`. En XOR, una corrida fallida no prueba que la arquitectura sea
incapaz de resolverlo: también influyen inicialización y tasa de aprendizaje.

El orden de revisión será AND, lineal, no lineal, XOR `[2, 2, 1]` y XOR
`[2, 3, 2, 1]`. El siguiente cálculo pendiente es completar una época de AND
con los pesos iniciales y la tasa usados arriba.
