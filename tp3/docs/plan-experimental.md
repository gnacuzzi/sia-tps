# Plan experimental de los ejercicios obligatorios

Este documento organiza lo que falta del TP3. La prioridad es completar y
analizar correctamente los tres ejercicios obligatorios; las variantes
opcionales sólo se habilitan cuando ayudan a responder una pregunta concreta o
cuando las curvas muestran un problema que las justifica.

## 1. Protocolo acordado: validation temporal

Los loaders públicos continúan entregando **training y test**. No se crea un
tercer CSV ni se modifica el test externo. Al comenzar cada experimento, el
runner separará reproduciblemente una parte de training como **validation
temporal**:

```text
datos de desarrollo del loader
            |
            +---- training interno: ajusta pesos y transformaciones
            |
            +---- validation temporal: compara configuraciones y umbrales

test del loader: permanece cerrado hasta congelar la configuración
```

La palabra “temporal” no significa improvisada: se guardarán semilla, índices y
proporción para poder repetir exactamente cada corrida. Significa que validation
se deriva del training para los experimentos y no pasa a ser un archivo fuente
independiente.

Reglas obligatorias:

- [ ] Verificar que test no elija arquitectura, learning rate, optimizador,
  épocas ni umbral durante las corridas.
- [ ] Verificar que todas las configuraciones comparadas usen la misma
  partición interna.
- [x] En fraude, hacer la división interna **antes** de ajustar el
   estandarizador. Media y desvío se calculan sólo con training interno y luego
   se aplican a validation y test.
- [x] En dígitos, dividir `digits.csv`; `digits_test.csv` nunca aporta ejemplos a
   training o validation.
- [ ] Evaluar test una sola vez por modelo final ya congelado.
- [x] Implementar la división temporal en `experiments.py`; los runners de cada
   ejercicio deberán reutilizarla en lugar de crear otra partición.

Estado de implementación del protocolo:

- [x] Mantener intactos los loaders training/test existentes.
- [x] Crear una división temporal estratificada y reproducible.
- [x] Devolver los índices de training y validation para registrar la partición.
- [x] Dividir fraude antes de ajustar el estandarizador.
- [x] Aplicar a validation y test los parámetros aprendidos de training interno.
- [x] Mantener `digits_test.csv` fuera de la división temporal.
- [x] Cubrir reproducción, disjunción y ausencia de leakage con tests.
- [ ] Integrar estos helpers en los runners de los ejercicios 1, 2 y 3.

## 2. Método para no generar millones de corridas

No se hará el producto cartesiano de todas las alternativas. Cada ejercicio
seguirá un embudo:

1. [ ] **Sanidad:** una configuración comprueba que el pipeline y el modelo funcionan.
2. [ ] **Comparaciones obligatorias:** se cambia una dimensión por vez.
3. [ ] **Diagnóstico:** se interpretan curvas de training y validation.
4. [ ] **Corrección:** se modifica únicamente lo relacionado con el problema visto.
5. [ ] **Finalistas:** se prueban las pocas combinaciones prometedoras y sus
   interacciones.
6. [ ] **Robustez:** se repiten los finalistas con varias semillas.
7. [ ] **Congelamiento:** se elige con validation y se registra toda la configuración.
8. [ ] **Test final:** se abre una vez y no se vuelve a ajustar el modelo.

En todas las corridas se deben guardar configuración, semilla, índices de la
partición, loss de training y validation por época, métricas, cantidad de
épocas, criterio de finalización y tiempo de entrenamiento.

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

## 4. Ejercicio 1: fraude con perceptrón simple

### 4.1 Preguntas obligatorias

- [ ] ¿El perceptrón lineal aprende a aproximar la probabilidad de BigModel?
- [ ] ¿El perceptrón no lineal aprende mejor?
- [ ] ¿Alguno muestra underfitting o saturación de capacidad?
- [ ] ¿Cuál tiene mejor potencial de generalización?
- [ ] ¿Qué estrategia de datos y métricas se utiliza?
- [ ] ¿Qué modelo y qué umbral de fraude se recomiendan?

### 4.2 Preparar los datos — obligatorio

- [x] Mantener el test actual reservado en el helper de partición.
- [x] Separar temporalmente el training actual en training interno y validation,
  estratificando mediante `flagged_fraud` para conservar aproximadamente el
  desbalance en ambos.
- [x] Ajustar `Standardizer` sólo con el training interno.
- [x] Transformar validation y test con esos mismos parámetros.
- [ ] Entrenar contra `big_model_fraud_probability`.
- [x] No incorporar `flagged_fraud` a las entradas ni al objetivo entrenable.

El loader actual estandariza antes de que exista esta división interna. Por eso
el runner de fraude deberá poder obtener las entradas sin estandarizar, dividir
y recién entonces ajustar el `Standardizer`; dividir la matriz ya estandarizada
filtraría estadísticas de validation.

### 4.3 Comparar capacidad lineal y no lineal — obligatorio

Primera comparación controlada:

- [ ] Perceptrón lineal.
- [ ] Perceptrón no lineal con una activación compatible con probabilidades.
- [ ] Mismo split, semilla, inicialización comparable, optimizador, batch y máximo
  de épocas;
- [ ] MSE contra la probabilidad de BigModel como medida primaria de aprendizaje.
- [ ] Curvas de training y validation.

Decisiones:

- [ ] Si la loss sigue bajando al alcanzar el máximo, aumentar épocas o mejorar la
  optimización antes de declarar underfitting.
- [ ] Si queda alta y estable, revisar capacidad o activación.
- [ ] Si el lineal queda alto y el no lineal baja, continuar con el no lineal.
- [ ] Si ambos son equivalentes en validation, preferir el modelo más simple salvo
  que otra métrica relevante lo contradiga.

### 4.4 Afinar el entrenamiento — obligatorio y acotado

Para ambos modelos se probará una cantidad pequeña de learning rates que cubra
un valor bajo, uno intermedio y uno alto razonable. Las épocas se elegirán con
las curvas, no sólo comparando el último número. Los finalistas se repetirán con
varias semillas.

Momentum, eta adaptativo, RMSProp, Adam y los tres tamaños de lote quedan como
variantes secundarias en este ejercicio. Se prueban si descenso básico no
converge bien o si permiten responder mejor la comparación, no como una grilla
completa desde el comienzo.

### 4.5 Elegir modelo y umbral — obligatorio

El modelo se elige con validation considerando:

- [ ] MSE de training y validation.
- [ ] Distancia entre ambas curvas.
- [ ] Estabilidad entre semillas.
- [ ] Épocas y costo hasta converger.

Después, sin reentrenar para cada umbral, se convierten sus probabilidades de
validation en fraude/no fraude y se calculan matriz de confusión, accuracy,
precision, recall/TPR, F1 y FPR para distintos umbrales.

- [ ] Recall insuficiente: considerar bajar el umbral.
- [ ] FPR demasiado alto: considerar subirlo.
- [ ] Accuracy alta con recall malo: no alcanza para recomendar el umbral.
- [ ] Buena imitación de BigModel pero mala detección real: analizar la relación
  entre BigModel y `flagged_fraud`; no atribuirlo automáticamente a TinyModel.

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

- [x] Dividir temporalmente `digits.csv` en training interno y validation,
  estratificando las clases disponibles.
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
| [ ] | Perceptrón lineal vs. no lineal | Ejercicio 1 |
| [ ] | Aprendizaje, generalización, métricas y umbral | Ejercicio 1 |
| [ ] | Learning rate | Ejercicio 2 |
| [ ] | Arquitectura | Ejercicio 2 |
| [ ] | Mecanismo de optimización | Ejercicio 2 |
| [ ] | Curvas por época | Diagnóstico de los tres ejercicios |
| [ ] | Incorporar datos nuevos y aislar su efecto | Ejercicio 3 |
| [ ] | Comparar ejercicios 2 y 3 | Ejercicio 3 |
| [ ] | Accuracy mayor o igual al 98 % | Ejercicio 3 |
| [x] | Implementar validation temporal | Protocolo común |
| [ ] | Usar validation para seleccionar y abrir test al final | Protocolo común |

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
- [ ] K-fold en lugar del holdout temporal fijo.
- [ ] Múltiples estrategias de data augmentation.

Los opcionales no comienzan hasta que estén resueltos los requisitos del
ejercicio correspondiente. Si alguno se incorpora, debe responder una hipótesis
explícita y no sólo sumar corridas al informe.
