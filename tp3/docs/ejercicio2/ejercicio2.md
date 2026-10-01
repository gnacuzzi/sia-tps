# Ejercicio 2: clasificación de dígitos

## El problema

CompanyX busca digitalizar números escritos a mano. Para eso entrenamos un
perceptrón multicapa que recibe los 784 píxeles de una imagen y debe elegir uno
de los diez dígitos posibles.

La consigna plantea dos preguntas: **cómo evaluar el desempeño del sistema** y
**qué variantes explorar para encontrar una solución**. Nuestro objetivo no fue
quedarnos con la mejor corrida aislada, sino construir una configuración que
fuera buena, estable y defendible sobre datos no vistos.

## 1. ¿Cómo evaluamos el desempeño?

Usamos `digits.csv` para todo el desarrollo del modelo y reservamos
`digits_test.csv` como representación del mundo real. Durante la búsqueda,
separamos `digits.csv` en training y validation de forma estratificada. El test
se abrió una sola vez, después de congelar todos los hiperparámetros.

El análisis inicial reveló una limitación importante: en development hay muy
pocos ejemplos del dígito 5 y **ningún ejemplo del 8**. Por eso la accuracy no
era suficiente. Un modelo podía acertar la mayoría de las imágenes y, aun así,
ignorar una clase minoritaria.

Elegimos como métrica principal el **macro-F1**, que calcula el F1 de cada clase
y luego les da a todas el mismo peso. Lo complementamos con:

- accuracy global;
- precision, recall y F1 por dígito;
- F1 específico del 5, por ser la clase presente más escasa;
- matriz de confusión;
- MSE de training y validation;
- variación entre semillas y particiones;
- tiempo de entrenamiento y cantidad de parámetros.

También observamos las curvas de aprendizaje. Si training y validation fueran
mal y quedaran en una meseta, hablaríamos de underfitting. Si training siguiera
mejorando mientras validation se estanca, habría evidencia de overfitting. En
nuestros mejores modelos no observamos underfitting: aprendieron casi por
completo las clases disponibles. Sí apareció una brecha moderada entre training
y validation, compatible con cierto overfitting.

## 2. ¿Cómo buscamos la solución?

La búsqueda se organizó como un embudo. Primero usamos una partición 80/20 fija
para comparar alternativas bajo las mismas condiciones. Las configuraciones
prometedoras se repitieron con cinco semillas. Cuando vimos que el orden de los
finalistas podía cambiar según la partición, los comparamos mediante validación
cruzada estratificada de cinco folds.

No realizamos un producto cartesiano de todos los hiperparámetros porque habría
sido costoso y difícil de interpretar. En cambio, cada etapa respondía una
pregunta y conservaba sólo las familias competitivas para la siguiente.

### Learning rate, batch y épocas

Las primeras pruebas mostraron que una tasa demasiado pequeña aprendía muy
lentamente y que una tasa excesiva podía producir oscilaciones o hacer que la
red dejara de reconocer el dígito 5. Además, el learning rate no podía elegirse
independientemente del batch: cambiar el tamaño del batch modifica tanto el
ruido del gradiente como la cantidad de actualizaciones por época.

Para el primer gráfico se modificó únicamente el learning rate. La configuración
de referencia fue:

```text
arquitectura       [784, 32, 10]
activaciones       tanh y logística
función de costo   MSE
optimizador        descenso básico
batch              64
épocas             200
partición          80/20 estratificada, validation_seed=0
inicialización     escala 0.5, model_seed=0
variable           learning rate: 0.001, 0.01, 0.1, 0.3, 1, 3 y 10
```

![Comparación inicial de learning rates](results/analysis-01-02/learning-rates.png)

Probamos mini-batches entre 32 y 1024, además de las modalidades online y full
batch. Los batches muy grandes producían curvas más suaves, pero no mejoraban
la generalización y necesitaban muchas más épocas. Online realizaba demasiadas
actualizaciones y full batch aprendía con demasiada lentitud. El mini-batch de
128 ofreció el mejor compromiso entre desempeño, estabilidad y tiempo.

La comparación consolidada de batch y learning rate mantuvo fija la red
`[784,32,10]`, las activaciones, MSE, descenso básico y la partición 80/20. Se
variaron conjuntamente el batch y la tasa porque sus efectos están
relacionados. Los pares competitivos se repitieron con `model_seed=0,1,2,3,4`.
Para batches 32, 64 y 128 se usaron 200 épocas; para 512 y 1024 se extendió el
horizonte a 600 y 1200 épocas para compensar su menor cantidad de
actualizaciones por época.

![Comparación consolidada de batch y learning rate](results/analysis-01-02/batch-eta-comprehensive-seeds.png)

Después de seleccionar la arquitectura y Adam se verificó que la elección del
batch siguiera siendo válida para esa combinación. En esta segunda comparación
se mantuvieron fijos `[784,128,10]`, Adam, `eta=0.003`, las activaciones, MSE y
el split. Se compararon batch 64 y 128 durante 200 épocas y batch 512 durante
600; cada finalista se repitió con cinco semillas.

![Finalistas de batch con Adam](results/analysis-01-02/adam-batch-finalists.png)

Por último comparamos las tres modalidades: online, mini-batch y full batch.
La arquitectura `[784,128,10]`, Adam, MSE y la partición permanecieron fijos,
pero cada modalidad necesitó una tasa y un horizonte adecuados a su frecuencia
de actualización: online usó `eta=0.001` y 20 épocas; mini-batch 128,
`eta=0.003` y 200; full batch, `eta=0.003` y 3000. Online y mini-batch se
repitieron con cinco semillas; full batch se descartó luego de la corrida
exploratoria por costo y menor desempeño.

Antes de fijar esas tasas, los pilotos compararon online con
`eta=0.0001, 0.0003, 0.001, 0.003` durante cinco épocas y full batch con
`eta=0.003, 0.01, 0.03` durante 3000. El gráfico deja visible por qué cada
modalidad necesitó una escala y un horizonte propios.

![Learning rates de online y full batch](results/analysis-01-02/adam-extreme-modalities-learning-rates.png)

La comparación estable posterior se concentró en online y mini-batch, las dos
modalidades que justificaban repetirse con cinco semillas. Full batch quedó
como referencia exploratoria: obtuvo menor macro-F1 y tardó más que ambas.

![Online frente a mini-batch](results/analysis-01-02/adam-extreme-modalities-finalists.png)

No hicimos una búsqueda independiente de cantidad de épocas. Usamos hasta 200
y comprobamos que las curvas de validation ya habían llegado a una región casi
estacionaria. Aumentar las épocas podía seguir reduciendo el error de training,
pero no aportaba una mejora sistemática de validation.

### Arquitectura

Comparamos redes con una capa oculta de 16, 32, 64, 128 y 256 neuronas. El
desempeño mejoró claramente hasta 64; 128 quedó cerca y 256 presentó problemas
de optimización bajo la receta inicial. También probamos una segunda capa
oculta. Las redes profundas ajustaban mejor training, pero no mejoraban
validation y aumentaban el costo, por lo que la profundidad adicional no se
justificaba.

En la comparación de ancho sólo se modificó la cantidad de neuronas ocultas:

```text
arquitecturas      [784,16,10], [784,32,10], [784,64,10],
                   [784,128,10] y [784,256,10]
activaciones       tanh y logística
función de costo   MSE
optimizador        descenso básico, eta=2
batch              128
épocas             200
partición          80/20 estratificada, validation_seed=0
semillas           model_seed=0, shuffle_seed=0
```

![Mejor resultado según el ancho](results/analysis-01-02/architecture-width-best.png)

Para estudiar profundidad conservamos la primera capa de 64 neuronas y
comparamos `[784,64,10]`, `[784,64,16,10]`, `[784,64,32,10]` y
`[784,64,64,10]`. Se mantuvieron descenso básico con `eta=2`, batch 128, 200
épocas, el mismo split y la misma semilla. Las configuraciones tienen una
cantidad de parámetros relativamente cercana, por lo que la diferencia se
puede asociar principalmente a agregar una segunda transformación oculta y, en
dos casos, un cuello de botella.

![Comparación de profundidad](results/analysis-01-02/architecture-depth.png)

Este resultado todavía no cerraba la arquitectura. Al cruzar arquitecturas y
optimizadores descubrimos una interacción importante: una arquitectura que
funcionaba bien con un mecanismo podía ser inestable con otro. En particular,
la red de 128 neuronas se volvió competitiva al combinarla con Adam y una tasa
menor.

### Mecanismos de optimización

Comparamos descenso básico, Momentum, tasa adaptativa, RMSProp y Adam. Para ser
justos, buscamos una región de learning rate apropiada para cada optimizador.
Momentum y algunas configuraciones de descenso básico llegaban a ignorar el 5.
Los candidatos consistentes fueron:

- `[784,64,10]` con RMSProp y `eta=0.01`;
- `[784,128,10]` con Adam y `eta=0.003`.

La primera comparación de optimizadores se hizo con arquitectura
`[784,64,10]`, batch 128, 200 épocas, MSE, activaciones `tanh` y logística,
split 80/20 fijo y cinco semillas para los finalistas. Lo que cambiaba era el
mecanismo y su tasa previamente seleccionada: GD (`eta=2` y `3`), adaptativo
(`eta0=1`), RMSProp (`eta=0.01`) y Adam (`eta=0.001` y `0.01`). Momentum se
descartó antes de esta repetición porque prácticamente no reconocía el 5.

![Estabilidad de los optimizadores finalistas](results/analysis-01-02/optimizer-finalists-stability.png)

La interacción entre arquitectura y optimizador fue uno de los principales
hallazgos del estudio: elegir primero la arquitectura y después el optimizador,
sin volver a cruzarlos, habría llevado a una conclusión incompleta.

En el gráfico de interacción se mantuvieron batch 128, 200 épocas, MSE,
activaciones, split y semilla 0. Se cruzaron cinco arquitecturas
representativas —32, 64, 128 y 256 neuronas, más `[784,64,64,10]`— con el
adaptativo (`eta0=1`), RMSProp (`eta=0.01`) y Adam (`eta=0.01`). Después se
ajustó localmente la tasa en las combinaciones que parecían fallar por
optimización, como 128 + Adam.

![Interacción entre arquitectura y optimizador](results/analysis-01-02/architecture-optimizer-interaction.png)

Los cuatro finalistas de esa interacción se repitieron con cinco semillas,
manteniendo batch 128, 200 épocas y el split. Este paso mostró que el máximo
aislado de 128 + RMSProp no era estable y que 128 + Adam con `eta=0.003` era la
combinación más consistente.

![Finalistas de arquitectura y optimizador](results/analysis-01-02/architecture-optimizer-finalists.png)

## 3. Confirmación de los finalistas

El holdout fijo fue útil para explorar, pero al cambiar la partición 80/20 el
orden entre Adam y RMSProp cambiaba. Por eso realizamos validación cruzada de
cinco folds sobre cuatro combinaciones representativas de 64 y 128 neuronas con
Adam y RMSProp. No repetimos toda la búsqueda: los experimentos anteriores ya
habían descartado regiones claramente inferiores.

En esta confirmación se mantuvieron fijos batch 128, 200 épocas, activaciones,
MSE, inicialización y orden de barajado con semilla 0. Se construyeron cinco
folds estratificados idénticos para los cuatro candidatos: cada muestra quedó
una vez en validation y cuatro veces en training. Sólo variaron la arquitectura
y el optimizador con su tasa correspondiente.

| Configuración | Macro-F1 | F1 del 5 | Tiempo por fold | Parámetros |
|---|---:|---:|---:|---:|
| 64 + Adam | 0.9426 ± 0.0029 | 0.8240 ± 0.0297 | 7.89 s | 50,890 |
| 64 + RMSProp | 0.9512 ± 0.0066 | **0.8660 ± 0.0286** | **7.33 s** | **50,890** |
| **128 + Adam** | **0.9544 ± 0.0080** | 0.8657 ± 0.0472 | 11.61 s | 101,770 |
| 128 + RMSProp | 0.8990 ± 0.0550 | 0.3729 ± 0.4653 | 10.61 s | 101,770 |

![Resumen de validación cruzada](results/analysis-01-02/cross-validation-summary.png)

Adam con 128 neuronas obtuvo el mayor macro-F1, ganó cuatro de los cinco folds
frente a RMSProp con 64 neuronas y tuvo el mejor peor caso. La diferencia fue
pequeña y no alcanza para afirmar una superioridad universal. RMSProp-64 quedó
como una alternativa eficiente: logró un F1 del 5 prácticamente idéntico con
la mitad de parámetros y menor tiempo. Como la métrica principal era el
macro-F1, seleccionamos Adam-128.

La configuración congelada fue:

```text
arquitectura       [784, 128, 10]
activaciones       tanh y logística
función de costo   MSE
optimizador        Adam
learning rate      0.003
batch              128
épocas             200
```

## 4. Evaluación final en producción

Una vez tomada la decisión, entrenamos una red nueva con las 12,449 muestras de
`digits.csv` y la evaluamos una única vez sobre las 2,497 imágenes de
`digits_test.csv`.

La configuración ya estaba congelada: `[784,128,10]`, `tanh`, salida
logística, MSE, Adam con `eta=0.003`, batch 128, 200 épocas e inicialización y
barajado con semilla 0. En esta etapa no se comparó ningún hiperparámetro. La
única diferencia respecto de la búsqueda es que se entrenó con todo
`digits.csv`, sin separar validation ni elegir un checkpoint.

![Curvas del entrenamiento final](results/analysis-01-02/final-training-curves.png)

| Métrica | Test |
|---|---:|
| Accuracy | **0.8654** |
| Macro-F1 sobre los diez dígitos | **0.8193** |
| F1 del dígito 5 | **0.8796** |
| MSE | **0.01998** |

La accuracy de 86.54 % parece mucho peor que los resultados de validation, pero
la matriz de confusión explica gran parte de la caída: test contiene 243 ochos
y el modelo nunca predice esa clase. No podía aprenderla porque no había visto
ningún 8 durante el desarrollo. Los ochos fueron confundidos principalmente
con 3, 2, 9, 4 y 5.

![Resultado final por clase](results/analysis-01-02/final-test-confusion-and-f1.png)

Como diagnóstico, si se consideran únicamente las nueve clases que sí estaban
presentes durante el aprendizaje, la accuracy es 0.9587 y el macro-F1 es
0.9103. Las métricas oficiales siguen siendo las calculadas sobre las diez
clases; este análisis sólo permite entender la causa del error.

## Conclusión

El estudio encontró una red estable y competitiva para la información
disponible. La mejor configuración fue una arquitectura `[784,128,10]` con
Adam, learning rate `0.003`, batch 128 y 200 épocas. La decisión no surgió de
una única corrida: se estudiaron learning rate, batch, ancho, profundidad,
optimizadores y sus interacciones, y los finalistas se confirmaron con cinco
folds.

La principal conclusión, sin embargo, no es que haga falta una red más grande o
más épocas. El límite dominante es el conjunto de datos: ningún ajuste de
hiperparámetros puede enseñar una clase completamente ausente. Para mejorar el
resultado real hay que incorporar ejemplos representativos del dígito 8 y
volver a entrenar. Ese es justamente el punto de partida del ejercicio 3, donde
CompanyX entrega `more_digits.csv` y exige alcanzar al menos 98 % de accuracy.

## Qué llevar a la presentación

Para contar el ejercicio en aproximadamente seis o siete minutos, alcanza con
mostrar cinco ideas:

1. **Problema y datos:** clasificación de diez dígitos, con el 5 escaso y el 8
   ausente en development.
2. **Evaluación:** macro-F1 como métrica principal, métricas por clase y test
   reservado hasta el final.
3. **Embudo experimental:** learning rate y batch; arquitectura; optimizador e
   interacciones; confirmación con semillas y cinco folds.
4. **Elección:** Adam-128 supera por poco a RMSProp-64, que queda como
   alternativa más eficiente.
5. **Resultado y transición:** 86.54 % de accuracy en test, explicado en gran
   medida por el 0 % de recall del 8; el ejercicio 3 incorpora nuevos datos para
   resolver esa limitación.

El detalle completo de grillas, corridas, decisiones intermedias y evidencia
reproducible se conserva en [`ejercicio2-extenso.md`](ejercicio2-extenso.md).
