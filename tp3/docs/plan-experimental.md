# Plan experimental de los ejercicios obligatorios

> **Estado del documento:** plan histórico. Los tres ejercicios obligatorios ya
> fueron ejecutados y sus resultados finales están en `ejercicio1.md`,
> `ejercicio2/ejercicio2.md` y `ejercicio3/ejercicio3-resumen.md`. Los checkboxes
> de abajo conservan la secuencia de planificación original y no representan el
> estado vigente del TP. La variante posterior se documenta en
> `rmsprop64/README.md`.

Este documento registra el plan con el que se organizó el TP3. Las variantes
opcionales se habilitaron cuando ayudaban a responder una pregunta concreta o
cuando las curvas mostraban un problema que las justificaba.

## 1. Protocolo acordado: aprendizaje antes de generalización

El ejercicio 1 tiene dos etapas distintas. Primero se compara la capacidad del
perceptrón lineal y no lineal usando **las 7500 muestras de fraude para
entrenar**, tal como aclaran el enunciado y la clase. En esa etapa no existe
validation ni test: sólo se estudian aprendizaje, underfitting y saturación.

Después de seleccionar el tipo de perceptrón comienza la generalización. El
protocolo finalmente usado para fraude reserva test y aplica 5-fold
estratificado sobre development:

```text
7500 transacciones
        |
        +---- test reservado: 1500; evaluación final
        |
        +---- development: 6000
                    |
                    +---- 5 folds de 1200
                          cada vuelta: 4800 training + 1200 validation
```

El helper de validation temporal implementado anteriormente sigue disponible
como infraestructura para hacer un único holdout reproducible. No fue el
protocolo definitivo del ejercicio 1: k-fold reduce la dependencia de una sola
partición, y el estandarizador se vuelve a ajustar dentro de cada vuelta.

En los ejercicios 2 y 3, `digits_test.csv` ya viene separado. El protocolo final
usó un holdout estratificado para el embudo de búsqueda y 5-fold para confirmar
los modelos congelados; no se creó un tercer CSV ni se modificaron los
originales.

Reglas obligatorias:

- [x] Usar las 7500 muestras de fraude para la comparación inicial de los dos
  perceptrones.
- [x] Verificar en el ejercicio 1 que test no elija arquitectura, learning rate, optimizador,
  épocas ni umbral durante la generalización.
- [x] Verificar que todas las configuraciones de fraude comparadas usen las
  mismas particiones internas.
- [x] En la generalización de fraude, hacer la división interna **antes** de ajustar el
   estandarizador. Media y desvío se calculan sólo con training interno y luego
   se aplican a validation y test.
- [x] En dígitos, dividir `digits.csv`; `digits_test.csv` nunca aporta ejemplos a
   training o validation.
- [x] Evaluar test de fraude con modelo y umbral ya congelados.
- [x] Implementar el holdout temporal en `experiments.py` como una alternativa
  reutilizable, sin imponerlo a todos los runners.

Estado de implementación del protocolo:

- [x] Mantener intactos los loaders training/test existentes.
- [x] Crear una división temporal estratificada y reproducible.
- [x] Devolver los índices de training y validation para registrar la partición.
- [x] Dividir fraude antes de ajustar el estandarizador.
- [x] Aplicar a validation y test los parámetros aprendidos de training interno.
- [x] Mantener `digits_test.csv` fuera de la división temporal.
- [x] Cubrir reproducción, disjunción y ausencia de leakage con tests.
- [x] Integrar aprendizaje y generalización de fraude; el análisis final de
  generalización usa el runner extendido con 5-fold.
- [ ] Elegir e integrar el protocolo de validation de los ejercicios 2 y 3.

Infraestructura del baseline del ejercicio 1:

- [x] Extender `fit` para registrar MSE de validation sin entrenar con ella.
- [x] Crear `configs/fraud-learning.json` con valores visibles y editables.
- [x] Crear el modo `learning`, que no divide las 7500 muestras.
- [x] Conservar el modo `generalization` para después de seleccionar el modelo.
- [x] Crear gráficos de las curvas completas sin volver a entrenar.
- [ ] Revisar y aprobar en equipo la configuración propuesta.
- [x] Ejecutar la primera corrida de sanidad.

## 2. Método para no generar millones de corridas

No se hará el producto cartesiano de todas las alternativas. Cada ejercicio
seguirá un embudo:

1. [ ] **Sanidad:** una configuración comprueba que el pipeline y el modelo funcionan.
2. [ ] **Comparaciones obligatorias:** se cambia una dimensión por vez.
3. [ ] **Diagnóstico:** se interpretan las curvas disponibles; sólo hay
   validation en las etapas de generalización.
4. [ ] **Corrección:** se modifica únicamente lo relacionado con el problema visto.
5. [ ] **Finalistas:** se prueban las pocas combinaciones prometedoras y sus
   interacciones.
6. [ ] **Robustez:** se repiten los finalistas con varias semillas.
7. [ ] **Congelamiento:** se elige con validation y se registra toda la configuración.
8. [ ] **Test final:** se abre una vez y no se vuelve a ajustar el modelo.

En todas las corridas se deben guardar configuración, semilla, muestras usadas,
loss disponibles por época, métricas, cantidad de épocas, criterio de
finalización y tiempo de entrenamiento.

## 3. Cómo leer las curvas

| Training | Validation | Diagnóstico probable | Siguiente decisión |
|---|---|---|---|
| Altos y todavía bajando | Altos y bajando | Falta de convergencia | Más épocas o mejor optimización |
| Altos y en meseta | Altos y en meseta | Underfitting | Más capacidad o activación adecuada |
| Bajo | Mucho más alto o empeora | Overfitting | Early stopping, L2 o más datos |
| Oscila o diverge | Oscila o diverge | Entrenamiento inestable | Bajar learning rate o revisar optimizador/lote |
| Bajos y cercanos | Bajo | Buen ajuste | Confirmar con varias semillas |
| Muy variables entre semillas | Variable | Sensibilidad a inicialización | Revisar inicialización u optimización |

No se aplicará regularización porque “suele ayudar”. Primero debe existir
evidencia de overfitting. Tampoco se aumentará capacidad si la loss todavía
desciende: en ese caso puede faltar convergencia, no neuronas.

En la primera etapa de fraude sólo existe la curva de training. Allí hay
evidencia de underfitting o saturación si el error queda alto y estable aun
cuando la optimización ya convergió, especialmente si el otro perceptrón logra
un error claramente menor sobre exactamente las mismas 7500 muestras. Sin datos
separados no se diagnostica overfitting ni generalización.

## 4. Ejercicio 1: fraude con perceptrón simple

### 4.1 Preguntas obligatorias

- [x] ¿El perceptrón lineal aprende a aproximar la probabilidad de BigModel?
- [x] ¿El perceptrón no lineal aprende mejor?
- [x] ¿Alguno muestra underfitting o saturación de capacidad?
- [x] ¿Cuál tiene mejor potencial de generalización?
- [x] ¿Qué estrategia de datos y métricas se utiliza?
- [x] ¿Qué modelo y qué umbral de fraude se recomiendan?

### 4.2 Preparar los datos — obligatorio y separado por etapa

- [x] En aprendizaje, usar las 7500 muestras y ajustar el `Standardizer` con las
  7500 porque todas pertenecen a training.
- [x] En generalización, reservar test y formar cinco folds estratificados con
  development mediante `flagged_fraud`.
- [x] En cada vuelta, ajustar `Standardizer` sólo con los cuatro folds de
  training y aplicarlo al fold de validation; reajustarlo con development para
  el modelo final y aplicarlo sin recalcular a test.
- [x] Entrenar contra `big_model_fraud_probability`.
- [x] No incorporar `flagged_fraud` a las entradas ni al objetivo entrenable.

Los modos son explícitos en la configuración para impedir que una comparación
de aprendizaje se confunda con un estudio de generalización.

### 4.3 Comparar capacidad lineal y no lineal — obligatorio

Primera comparación controlada:

- [x] Perceptrón lineal.
- [x] Perceptrón no lineal con una activación compatible con probabilidades.
- [x] Las mismas 7500 muestras, semilla, inicialización comparable, optimizador,
  batch y máximo de épocas.
- [x] MSE contra la probabilidad de BigModel como medida primaria de aprendizaje.
- [x] Curvas completas de training.

Resultados del diagnóstico:

- [x] Se comprobó que las curvas habían convergido antes de interpretar sus
  mesetas.
- [x] Se identificó un piso de error considerablemente mayor para el modelo
  lineal.
- [x] Se eligió el perceptrón logístico para la etapa de generalización por su
  mejor capacidad de aproximación sobre los mismos datos.

### 4.4 Afinar el entrenamiento — sólo si el diagnóstico lo requiere

Si una curva sigue bajando, oscila o diverge, se ajustarán épocas o learning
rate antes de atribuir el resultado a la capacidad del perceptrón. No se hará
una grilla de hiperparámetros si la comparación inicial ya converge de forma
estable.

La extensión profundizó este control comparando descenso básico, Momentum, eta
adaptativo, RMSProp y Adam; varios learning rates; tamaños de lote; épocas y
semillas. Esas pruebas no cambian la capacidad estructural del perceptrón, pero
permitieron comprobar que la meseta observada no se explicaba simplemente por
una mala elección de optimización.

### 4.5 Elegir modelo y umbral — obligatorio

El tipo de perceptrón se selecciona por su potencial de aprendizaje con las
7500 muestras. Después, su configuración de generalización se elige con
validation considerando:

- [x] MSE de training y validation.
- [x] Distancia entre ambas curvas.
- [x] Estabilidad entre semillas.
- [x] Épocas y costo hasta converger.

Después, sin reentrenar para cada umbral, se convierten sus probabilidades de
validation en fraude/no fraude y se calculan matriz de confusión, accuracy,
precision, recall/TPR, F1 y FPR para distintos umbrales.

- [x] Se examinó cómo bajar el umbral aumenta el recall y también puede aumentar
  los falsos positivos.
- [x] Se examinó cómo subirlo reduce falsos positivos a costa de dejar pasar más
  fraudes.
- [x] El umbral se eligió con predicciones out-of-fold y métricas sobre
  `flagged_fraud`, sin apoyarse sólo en accuracy.
- [x] Se distinguió entre imitar la probabilidad de BigModel y clasificar la
  etiqueta real de fraude.

La recomendación debe explicitar el compromiso entre dejar pasar un fraude y
bloquear una operación legítima. Modelo, transformación, hiperparámetros y
umbral quedan congelados antes de evaluar test.

## 5. Ejercicio 2: dígitos con perceptrón multicapa

### 5.1 Mínimos obligatorios

El enunciado exige comparar variantes de:

- [ ] Learning rate.
- [ ] Arquitectura.
- [ ] Mecanismo de optimización.

También se debe monitorear loss por época, explicar cómo se evalúa el desempeño
y analizar las variantes realizadas.

### 5.2 Preparar los datos — obligatorio

- [x] Disponer de un helper para dividir `digits.csv` en un holdout estratificado
  reproducible si se elige ese protocolo.
- [ ] Elegir y documentar para el ejercicio 2 si se usará ese holdout o k-fold,
  considerando el costo de entrenar el multicapa.
- [x] Mantener los píxeles en `[0,1]`.
- [ ] Definir la codificación de las diez salidas.
- [x] Mantener `digits_test.csv` fuera de la división temporal.

La clase 8 está ausente de `digits.csv`: no aparecerá honestamente ni en
training ni en validation. No se inventarán ochos ni se tomarán de test. Esta
limitación debe quedar visible en el análisis del ejercicio 2.

### 5.3 Construir un baseline — obligatorio

Se comienza con una red sencilla de una capa oculta, bias, una activación de
clase, MSE, descenso básico y un único tamaño de lote. El baseline debe verificar
que la codificación, las diez salidas, backpropagation y las métricas funcionan;
no pretende ser la configuración final.

### 5.4 Comparar learning rate — obligatorio

Se mantienen fijos arquitectura, optimizador, batch e inicialización. Una
primera escala posible para descenso básico es `0.001`, `0.01` y `0.1`; son
candidatos del equipo, no valores impuestos por la cátedra.

- [ ] Muy lento y estable: considerar una tasa mayor.
- [ ] Oscila o diverge: reducirla.
- [ ] Training mejora pero validation empeora: diagnosticar overfitting, no agregar
  épocas automáticamente.

### 5.5 Comparar arquitectura — obligatorio

Con una tasa razonable se comparan capacidades crecientes. Candidatos iniciales,
a confirmar antes de ejecutar, podrían ser `[784,16,10]`, `[784,32,10]`,
`[784,64,10]` y `[784,32,16,10]`.

- [ ] Training alto y estable: aumentar capacidad o revisar activación.
- [ ] Training bajo y validation alto: la red puede estar sobreajustando.
- [ ] Una arquitectura mayor sin mejora consistente: conservar la más simple.
- [ ] Para las arquitecturas finalistas se vuelve a comprobar el learning rate,
  porque la tasa conveniente puede cambiar con la arquitectura.

### 5.6 Comparar optimización — obligatorio

Se comparan las variantes implementadas:

- [ ] Descenso básico.
- [ ] Momentum.
- [ ] Eta adaptativo.
- [ ] RMSProp.
- [ ] Adam.

No se exigirá el mismo learning rate a todos, porque sus actualizaciones tienen
escalas distintas. Como búsqueda inicial se pueden considerar `0.001`, `0.01`
y `0.1` para descenso/Momentum, y `0.0001`, `0.001` y `0.01` para RMSProp/Adam.
Son propuestas que deberán aprobarse antes de correr la grilla. Adam parte de
los valores de referencia de clase; Momentum puede comenzar con `alpha=0.8` y
`0.9`.

Primero se descartan divergencias y configuraciones que claramente no aprenden.
Después se repiten los finalistas con las mismas semillas, comparando curvas,
métricas, épocas y costo.

### 5.7 Seleccionar el finalista del ejercicio 2 — obligatorio

Validation decide usando accuracy, matriz de confusión, precision, recall y F1
por clase, F1 macro, curvas y estabilidad. La ausencia del 8 debe interpretarse
por separado. El modelo se congela, pero todavía no se abre test: primero se
seleccionará también el finalista del ejercicio 3.

## 6. Ejercicio 3: más datos y accuracy mayor o igual al 98 %

### 6.1 Aislar el efecto de los datos — obligatorio

El primer experimento reutiliza exactamente la configuración finalista del
ejercicio 2 y agrega `more_digits.csv` sólo al conjunto de desarrollo. Se vuelve
a formar training/validation estratificado y se mantienen arquitectura,
optimizador, learning rate, lote, activación, inicialización y criterio de corte.

Antes de entrenar se revisan balance, presencia de la clase 8, duplicados y
rango de píxeles. La comparación controlada permite atribuir la primera
diferencia principalmente a la incorporación de datos.

### 6.2 Comparar ejercicios 2 y 3 — obligatorio

Sobre validation se comparan accuracy, F1 macro, métricas por clase, matriz de
confusión, curvas y variación entre semillas.

- [ ] Si mejora manteniendo todo igual, los datos adicionales explican la mejora.
- [ ] Si mejora especialmente el 8, se obtiene evidencia del efecto de tener
  ejemplos de esa clase.
- [ ] Si training empeora pero validation mejora, puede haber mejor generalización.
- [ ] Si ambos errores quedan altos, persiste underfitting.
- [ ] Si training es bueno y validation malo, hay overfitting.

### 6.3 Iterar hasta el objetivo — obligatorio

Si todavía no se alcanza 98 %, el orden es:

1. [ ] Comprobar si terminó de converger.
2. [ ] Ajustar learning rate u optimizador.
3. [ ] Ajustar arquitectura.
4. [ ] Ajustar máximo de épocas.
5. [ ] Aplicar técnicas condicionadas al diagnóstico.

Ante underfitting se puede aumentar capacidad, mejorar activación/optimización o
dar más épocas si la curva aún baja. Ante overfitting se habilitan early
stopping, L2 o data augmentation. No se mezclan todas las técnicas en la primera
corrida: cada cambio debe tener una comparación que permita explicar su efecto.

### 6.4 Evaluación final — obligatorio

Una vez elegidos con validation los finalistas de los ejercicios 2 y 3, ambos
se evalúan en una única etapa sobre el mismo `digits_test.csv`. Cada configuración
usa test una sola vez y no se modifica después de conocer el resultado. Allí se
informa si el ejercicio 3 alcanza accuracy mayor o igual al 98 % y se compara
con el ejercicio 2.

## 7. Inventario de variantes y prioridad

### Obligatorias

| Estado | Variante | Dónde se exige o necesita |
|---|---|---|
| [x] | Perceptrón lineal vs. no lineal | Ejercicio 1 |
| [x] | Aprendizaje, generalización, métricas y umbral | Ejercicio 1 |
| [ ] | Learning rate | Ejercicio 2 |
| [ ] | Arquitectura | Ejercicio 2 |
| [ ] | Mecanismo de optimización | Ejercicio 2 |
| [x] | Curvas por época | Aplicado en ejercicio 1; pendiente en 2 y 3 |
| [ ] | Incorporar datos nuevos y aislar su efecto | Ejercicio 3 |
| [ ] | Comparar ejercicios 2 y 3 | Ejercicio 3 |
| [ ] | Accuracy mayor o igual al 98 % | Ejercicio 3 |
| [x] | Implementar holdout reproducible | Alternativa disponible del protocolo común |
| [x] | Usar validation para seleccionar y abrir test al final | Aplicado en ejercicio 1; pendiente en 2 y 3 |

### Importantes pero secundarias

- [ ] Máximo de épocas y criterio de finalización.
- [ ] Online, mini-batch y batch.
- [ ] Función de activación.
- [ ] Inicialización de pesos.
- [ ] Varias semillas.
- [ ] Parámetros internos de cada optimizador.
- [ ] Tiempo de entrenamiento.

Se estudian sobre configuraciones finalistas o cuando son necesarias para
explicar un comportamiento; no se cruzan todas desde el comienzo.

### Condicionales al diagnóstico

- [ ] Aumentar capacidad.
- [ ] Cambiar profundidad.
- [ ] Cambiar activación.
- [ ] Early stopping.
- [ ] Regularización L2.
- [ ] Data augmentation.
- [ ] Estrategia adaptativa de learning rate.

### Opcionales

- [ ] Entrenar sin bias.
- [ ] Losses alternativas.
- [ ] Activaciones adicionales como ReLU.
- [ ] Construir o eliminar features de fraude.
- [ ] Calibración de probabilidades.
- [ ] Robustez frente a ruido.
- [ ] Métodos de interpretabilidad.
- [ ] Comparación extensa de inicializaciones.
- [ ] Evaluar si k-fold se justifica para los MLP de ejercicios 2 y 3 frente al
  costo menor de un holdout estratificado.
- [ ] Múltiples estrategias de data augmentation.

Los opcionales no comienzan hasta que estén resueltos los requisitos del
ejercicio correspondiente. Si alguno se incorpora, debe responder una hipótesis
explícita y no sólo sumar corridas al informe.
