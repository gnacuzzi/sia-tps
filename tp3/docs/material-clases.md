# Clases del TP3: referencias para trabajar

Los cinco PDFs están directamente en `docs/`, como pidió el equipo. Las tres
transcripciones se conservaron localmente en `docs/transcripciones/` y se
excluyeron de Git por pedido del equipo. Se copiaron sin modificar
el contenido de los adjuntos y se comprobó igualdad de bytes.
Los acentos de los nombres de PDF se guardaron en forma Unicode compuesta;
se conservó la extensión `.PDF` del archivo de multicapa.

Las referencias a páginas usan la numeración física del PDF, comenzando en 1.
Se revisó el texto de los cinco PDFs y se inspeccionaron visualmente las
diapositivas con fórmulas de perceptrón simple, lineal/no lineal y multicapa.
Las transcripciones se consultaron en los pasajes sobre actualización,
inicialización, bias, alcance del TP y aprendizaje/generalización; no se
corrigió su texto ni se hizo una edición completa de ellas.

## Índice y relación con el TP

| Material | Contenido que usamos | Referencias |
|---|---|---|
| [Machine Learning](Clase7-MachineLearning.pdf) | Aprender como ajustar parámetros libres; tipos de aprendizaje | Páginas 4 y 12 |
| [Optimización no lineal](Clase7-OptimizacionNOLineal.pdf) | Minimizar una función de error, dirección de descenso y tamaño de paso | Páginas 14, 22–26; métodos estocásticos en 41–44 |
| [Perceptrón simple escalón](Clase10.1-Perceptrón%20Simple%20Escalón.pdf) | Excitación, activación, bias y regla de Rosenblatt | Páginas 15, 30, 33–34, 42–45 |
| [Perceptrón lineal y no lineal](Clase10.2-PerceptrónLinealyNoLineal.pdf) | Identidad, error cuadrático, derivadas, online, tanh y logística | Páginas 8, 12–14, 19–26 |
| [Perceptrón multicapa](Clase11-PerceptrónMulticapa.PDF) | Forward, regla de la cadena, backpropagation, modos de entrenamiento e inicialización | Páginas 13, 38–58; bias en 63–70 |
| Transcripción de clase 7 (solo local) | Intuición de optimización, gradiente y tasa de aprendizaje | Complemento explicativo |
| Transcripción de clase 10 (solo local) | Online, ajuste de funciones, destilación y diferencia entre aprendizaje y generalización | Complemento explicativo |
| Transcripción de clase 11 (solo local) | Bias separado, simetría, análisis de resultados y uso de train/validación/test | Complemento explicativo |

Los PDFs de clase 10 dicen «Primer Cuatrimestre 2026» en la portada; el de
clase 11 y la consigna corresponden al segundo. Se usa el material que envió
el equipo, sin cambiar portadas ni sustituir archivos por otras versiones.

## Qué se mantiene y qué se aclaró en el apunte

1. **Escalón bipolar.** La consigna usa $-1,+1$. La clase 10.1 incluye esa
   versión en la página 30 y una versión $0,1$ en la página 15 y en su
   pseudocódigo. Elegimos la bipolar con salida $+1$ en cero; la regla de
   actualización coincide con las páginas 33–34.
2. **Notación.** Nuestro $y$ equivale a $\zeta$, $\hat y$ a $O$ y $g$ a
   $\theta$. El bias es $b=w_0=-u$. Se añadió una tabla para leer el apunte
   y las diapositivas sin confundir la activación $\theta$ con el umbral $u$.
3. **Error y actualización.** El error sumado de clase es $E=\frac12\sum e^2$.
   El ejemplo online deriva el término de una muestra, $E_\mu=\frac12 e_\mu^2$.
   El MSE que usamos para informar es $\frac1N\sum e^2$. Se explicita la
   escala y se diferencia acumular gradientes de actualizar tras cada dato.
4. **No lineal.** La clase 10.2 usa $\tanh(\beta h)$ y su derivada
   $\beta(1-\tanh^2(\beta h))$. El apunte fija $\beta=1$ para validar
   $y=\tanh(x)$; no es un valor impuesto para todo el TP.
5. **Multicapa.** La página 51 de clase 11 confirma los deltas y su signo.
   Separar bias y pesos está permitido en la explicación oral. Incluimos
   bias por neurona para dar flexibilidad a cada capa; no es una exigencia
   universal ni aumenta los tamaños de las arquitecturas pedidos por la consigna.
6. **Inicialización.** La clase recomienda pesos aleatorios pequeños para
   evitar simetría. Los pesos nulos del apunte solo simplifican ejemplos de
   una neurona; no se usarán como inicialización uniforme del multicapa.

## Base para los ejercicios posteriores

- La explicación de clase 10 confirma que fraude consiste primero en aproximar
  las probabilidades del modelo grande. Un perceptrón escalón no resuelve ese
  objetivo. El umbral de clasificación se estudia después, en generalización.
- La logística de clase 10.2 se escribe $g(h)=1/(1+e^{-2\beta h})$,
  con derivada $2\beta g(h)(1-g(h))$. Si luego se elige para fraude,
  habrá que mantener esa parametrización explícita; todavía no se eligió.
- Clase 11 distingue online, mini-batch y batch. Comenzamos por online para
  seguir las cuentas manuales; eso no decide qué modo será mejor para dígitos.
- La consigna y la explicación de clase 11 reservan `digits_test.csv` para
  evaluación final. Los ajustes de pesos y de hiperparámetros se hacen usando
  particiones del conjunto de desarrollo, no el test final.
- Optimización incluye momentum, AdaGrad y Adam. Que aparezcan en clase no
  significa que ya se hayan seleccionado o implementado. Antes de comparar
  optimizadores se documentará su fórmula exacta y sus hiperparámetros.
- La clase 11 pide analizar las curvas de pérdida y explicar las decisiones
  propias. No basta con presentar un número final sin discutir convergencia
  ni justificar las alternativas comparadas.

Las reglas ya están implementadas y contrastadas con cuentas manuales y
gradientes numéricos. Ver [implementación](implementacion.md) y
[resultados de validación](resultados-validacion.md). La selección de modelos
e hiperparámetros para los datasets reales sigue pendiente.
