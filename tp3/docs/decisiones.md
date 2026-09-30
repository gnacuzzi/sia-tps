# Registro de decisiones del TP3

Por pedido del equipo, toda decisión debe explicar **qué**, **por qué** y
**cómo se aplicó**, e indicar si ya está aplicada o sigue pendiente.
Este registro se actualiza junto con los cambios. Una elección didáctica no
debe presentarse como requisito de la cátedra ni como resultado experimental.

Las secciones conservan la evolución del trabajo. Una propuesta pendiente no
reemplaza una decisión del equipo: los cambios de criterio requieren acuerdo
explícito y deben registrarse como tales.

## Organización y alcance

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| Crear `tp3/` independiente | Mantener la estructura del monorepo y no mezclar trabajos | Aplicada; raíz con README y contexto, material y apuntes en `docs/` |
| Documentar en español y enlazar desde el README general | Continuidad con TP1 y TP2 y acceso sencillo | Aplicada en los README y apuntes |
| Comenzar por el desarrollo de validación | El usuario pidió ver las fórmulas y revisarlas juntos | Aplicada en `validacion.md`; aún sin implementación de modelos |
| Incluir los cuatro casos y las dos arquitecturas de XOR | Cubrir el ejercicio completo de la página 2 | Aplicada al apunte; falta el cálculo manual de aprendizaje del multicapa |
| Conservar copia exacta de la consigna | Tener la fuente junto al desarrollo | Aplicada en `docs/Enunciado TP3 - 2Q 2026.pdf` |
| Separar ejemplos manuales de corridas | Evitar confundir representación del problema con aprendizaje validado | Aplicada en README y apunte; no se reporta convergencia experimental |
| Contrastar con las teóricas antes de implementar | Regla general del repositorio: mantenerse en contenidos de cátedra | Realizado con los cinco PDFs recibidos y pasajes pertinentes de las transcripciones; referencias en `material-clases.md` |

## Convenciones y ejemplos de validación

Todas estas elecciones están aplicadas **al apunte**, no a un programa de entrenamiento.

| Decisión | Motivo | Cómo se aplicó |
|---|---|---|
| Codificación bipolar y orden original de muestras | Es lo indicado para AND y XOR | Tablas con valores $-1,+1$ en el orden del enunciado |
| Bias aditivo $h=\sum_i w_ix_i+b$ | Explicitar el desplazamiento sin ambigüedad de signos | Misma convención en todos los modelos; equivalencia $b=w_0=-u$ documentada tras revisar la notación de clase |
| Escalón con $g(0)=+1$ | Resolver el caso de borde en los cálculos manuales | Usado en la primera actualización de AND |
| Error con signo $e=y-\hat y$ | Mantener actualizaciones con signo positivo de corrección | $w'=w+\eta ex$ para escalón y lineal; se aclara la escala bipolar $\pm2$ |
| Actualización por muestra | Permitir seguir cada cálculo y compararlo luego con código | Las fórmulas modifican parámetros tras una muestra; no se eligió aprendizaje por lotes |
| $\eta=0.1$ y pesos/bias nulos en los ejemplos simples | Facilitar las cuentas, sin afirmar que sean hiperparámetros óptimos | Solo ejemplos de una actualización; no se extiende inicialización nula al multicapa |
| AND con pesos manuales $(1,1)$ y bias $-1$ como referencia | Disponer de una solución fácil de comprobar | Tabla de cuatro predicciones; no se presenta como pesos aprendidos |
| Proponer 50 entradas equiespaciadas en $[-1,1]$ | Concretar el ejemplo de 50 muestras sugerido por la consigna con valores moderados | Propuesta para lineal y no lineal; datos aún no generados |
| Error por muestra $E=\frac12(y-\hat y)^2$ en modelos diferenciables | El factor $1/2$ simplifica la derivación | Derivadas de lineal, tanh y backpropagation; MSE sin ese factor para informar ajuste |
| Activación identidad para ajustar $y=x$ | Caso lineal que tiene solución conocida | Referencia $w=1,b=0$ |
| Activación tanh con $\beta=1$ para ajustar $y=\tanh(x)$ | Coincidir con el ejemplo de la consigna y tener solución conocida | Derivada $1-\hat y^2$; se explica el factor adicional si cambia $\beta$ |
| Tanh en capas ocultas y salida del ejemplo XOR | Introducir no linealidad y salida compatible con codificación bipolar | Forward y backpropagation para ambas arquitecturas; activación y derivadas contrastadas con clases 10.2 y 11 |
| Vectores columna y matrices destino × origen | Evitar transposiciones ambiguas | Dimensiones explícitas de pesos, bias y producto exterior de actualización |
| Definir $\delta$ con signo de corrección | Mantener coherencia con $e=y-\hat y$ | Actualizaciones suman $\eta\delta a^T$; todos los deltas usan pesos previos |
| Umbralizar XOR en cero solo para clasificar | La salida continua es necesaria para derivar y medir el error | Se distingue accuracy de MSE; tanh no alcanza exactamente $\pm1$ con pesos finitos |
| Dar pesos manuales de XOR `[2,2,1]` | Mostrar numéricamente cómo representa XOR una capa oculta | Tabla comprobada con Python; no valida el algoritmo de aprendizaje |
| No fijar todavía tolerancias, épocas o inicialización aleatoria | Deben acordarse al implementar y justificarse | Pendientes explícitos; no se declararon entrenamientos exitosos |

## Datos recibidos para los demás ejercicios

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| Guardar CSV y loader juntos en `data/`, y el PDF en `docs/` | Separar datos de documentación y conservar el loader cerca de los archivos que usa | Aplicada; solo cambió su ubicación respecto del ZIP |
| Mantener nombres y bytes originales | Preservar trazabilidad y evitar alteraciones silenciosas | Aplicada; igualdad SHA-256 comprobada con las entradas del ZIP |
| No duplicar el ZIP dentro del TP | Ya están extraídos sus contenidos útiles | Aplicada; el adjunto original permanece en su ubicación de origen |
| Registrar las discrepancias de nombres | ZIP y consigna difieren en fraude y datos adicionales | Aplicada en `data/README.md`; correspondencias marcadas como inferidas |
| Reservar `big_model_fraud_probability` como objetivo de destilación | El ejercicio pide imitar BigModel y la documentación identifica sus salidas | Criterio documentado; entrenamiento pendiente |
| Excluir `flagged_fraud` del entrenamiento | Prohibición explícita de la documentación del dataset | Regla documentada; aún no existe pipeline que la aplique |
| Reservar `digits_test.csv` para evaluación final | La consigna impide usarlo para ajustar parámetros e hiperparámetros | Regla documentada; no se hicieron divisiones ni entrenamientos |
| Posponer limpieza, normalización y selección de variables | Requieren EDA y corresponden a ejercicios posteriores | Los archivos están intactos; revisar encabezados y contar filas no constituye un EDA completo |

## Verificación realizada y límites

- Se leyó el enunciado completo y se inspeccionó visualmente la página de validación.
- Se contrastó la organización con los archivos locales de TP1 y TP2.
- Se recalcularon las cuatro salidas del ejemplo manual de XOR con Python.
- Se revisaron el loader y la documentación de fraude; se verificaron las copias
  del ZIP y se contaron registros de los CSV.
- No se entrenaron redes, no se ejecutó el loader y no se validaron todavía
  gradientes contra una implementación.


## Revisión a partir de las clases recibidas

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| PDFs directamente en `docs/` y transcripciones en `docs/transcripciones/` | Pedido del usuario y separación del material por formato | Aplicada; ocho copias verificadas byte a byte |
| Conservar las versiones provistas aunque algunas portadas digan primer cuatrimestre | Son las fuentes que el equipo indicó para este TP | Aplicada; diferencia de portada aclarada en `material-clases.md` |
| Guardar acentos de nombres PDF en Unicode compuesto | Mantener nombres legibles y enlaces locales consistentes | Aplicada solo al nombre; contenido de los adjuntos intacto |
| Usar PDFs para las fórmulas y transcripciones como complemento | Varias fórmulas son imágenes y las transcripciones tienen errores de reconocimiento | Aplicada: lectura de texto, inspección visual de fórmulas y consulta de pasajes pertinentes |
| Agregar referencias por página y equivalencias de símbolos | Poder justificar cada fórmula y compararla con la clase | Aplicada en `material-clases.md` y `validacion.md` |
| Cambiar la letra del umbral de la explicación de equivalencia a $u$ | En clase $\theta$ nombra la activación, no el umbral | Aplicada; los cálculos siguen iguales, $b=-u$ |
| Mantener escalón bipolar y $g(0)=+1$ | Coincide con la consigna y la página 30 de clase 10.1 | Aplicada; se explica la diferencia con el pseudocódigo $0/1$ |
| Explicitar error total, error por muestra y MSE | Evitar mezclar gradientes, tolerancias o escalas de tasa de aprendizaje | Aplicada: $E=\sum E_\mu=(N/2)\operatorname{MSE}$ |
| Mantener online para la primera validación | Es el algoritmo de clase 10.2 y permite cotejar una muestra a mano | Aplicada al desarrollo; batch/mini-batch no seleccionados para experimentos |
| Mantener bias separado y por neurona | Hace visible cada actualización y permite desplazar las activaciones | Aplicada a las fórmulas; clase 11 permite separar pesos y bias |
| No agregar optimizadores por el solo hecho de aparecer en clase | En esa etapa el objetivo era entender y verificar las reglas básicas | Decisión histórica de la validación inicial; luego el equipo pidió implementar Momentum, eta adaptativo, RMSProp y Adam. La selección sigue pendiente |
| No copiar el pseudocódigo literalmente donde alterna convenciones | Debemos respetar la consigna y mantener coherencia matemática | Diferencias documentadas; originales preservados |

Se confirmó que las reglas de Rosenblatt, lineal, tanh y backpropagation del
apunte corresponden al contenido provisto. Esta revisión sustenta las fórmulas;
no reemplaza la futura verificación numérica de una implementación.

## Explicación visual de la validación

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| Tres visualizaciones interactivas en la conversación | El equipo pidió gráficos para comprender las fórmulas | Fragmentos de explicación en `tp3/output/visualizaciones/`; no son el motor de entrenamiento del TP |
| AND con ocho muestras procesadas en el orden original | Mostrar dos épocas, incluyendo actualizaciones y pasos sin cambio | Simulación de Rosenblatt con pesos/bias iniciales cero y tasa 0.1, como el apunte |
| Mantener la suma muy próxima a cero como cero en la demostración AND | Evitar que el redondeo de JavaScript cambie el escalón en cálculos decimales exactos | Tolerancia numérica de 1e-12 solo en esta demostración; no es una decisión del futuro motor |
| Sombrear la predicción y distinguir objetivos por círculo/cuadrado | Separar visualmente lo que predice el modelo de la respuesta correcta | Fondo suave para regiones; formas y etiquetas para clases, sin depender solo del color |
| Peso y bias manuales compartidos para lineal y tanh | Ver cómo cambia la respuesta al cambiar la activación | Dos gráficos y MSE sobre 50 muestras equiespaciadas en [-1,1]; no se simula entrenamiento de estos modelos |
| Rangos de exploración w en [-2,2] y b en [-1,1] | Incluir pendiente inversa, solución exacta y desplazamientos moderados | Controles con paso 0.05; inicio didáctico w=0.3, b=0.25, no hiperparámetros elegidos para entrenar |
| Mostrar XOR en el espacio original y en el de activaciones ocultas | Explicar qué aporta la capa oculta | Se reutilizan los pesos manuales `[2,2,1]` del apunte y se distingue que C y D coinciden tras la transformación |
| Usar `[2,2,1]` para la primera explicación gráfica de XOR | Es la arquitectura más pequeña de las dos solicitadas y permite dibujar ambas activaciones en un plano | La validación de `[2,3,2,1]` sigue incluida en el apunte y pendiente de desarrollo manual |
| Figuras adaptables y controles nativos | Permitir explorar con teclado y leer en pantallas pequeñas | Ejes con nombres, cifras visibles y redibujado según el ancho disponible |

Verificación de las visualizaciones: interacciones probadas en Chrome sin
excepciones de JavaScript; AND termina con 4/4 aciertos tras ocho muestras,
los dos MSE son cero al fijar w=1 y b=0, y las cuatro entradas de XOR reciben
su clase correcta. Se comprobó ausencia de desborde horizontal a 360 px y se
inspeccionaron capturas de los gráficos. Estos controles verifican la
explicación interactiva, no una implementación completa del TP.

## Página de lectura

- Se reunieron las explicaciones y los tres gráficos existentes en
  `tp3/validacion-visual.html`, porque el equipo pidió una página para leer
  de corrido. Se exportó con el wrapper de la skill de visualizaciones para
  conservar estilos e interacciones fuera de la conversación.
- Se agregaron navegación por tema, fórmulas y decisiones junto a los ejemplos.
  Se conservaron los cálculos y el alcance de las visualizaciones originales.
- Es una página local; la biblioteca D3 se carga desde CDN y necesita conexión
  a internet. No se publicó ni se subieron datos.

## Motor reutilizable implementado

Esta sección describe el estado actual. Las menciones anteriores a ausencia de
código corresponden a la etapa inicial de preparación teórica, ya superada.

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| Implementar el motor real antes de los ejercicios con datasets | El usuario aclaró que la validación debe probar las herramientas reutilizables | `src/sia_tp3/` implementado; cinco casos y tres semillas ejecutados |
| Python y NumPy, sin librerías de redes neuronales | Mantener continuidad con TP1/TP2 y hacer explícitas las fórmulas de clase | Dependencia de ejecución NumPy; pytest y Matplotlib separados como extras |
| Perceptrón simple como caso de una sola capa del motor | Evitar fórmulas duplicadas que pudieran funcionar en validación pero diferir en los ejercicios | `Perceptron` reutiliza `MultilayerPerceptron`; Rosenblatt tiene una rama propia |
| Activaciones identidad, tanh, escalón y logística de clase | Cubrir las herramientas de la consigna y disponer de salidas probabilísticas para el próximo ejercicio | Logística con factor 2β; no se decidió aún usarla como mejor modelo de fraude |
| Rechazar escalón en capas ocultas | No tiene la derivada necesaria para el backpropagation implementado | Validación del constructor; escalón solo para una neurona simple |
| Matrices 2D y float64, con bias separado | Evitar errores de broadcasting y facilitar la comprobación numérica | Validación de dimensiones y valores finitos en la API; pesos destino × origen |
| Capas y cantidad de salidas configurables | Reutilizar el motor para dígitos sin cambiar las fórmulas | Forward y gradientes admiten varias salidas; forma `[784,8,10]` comprobada, sin entrenar dígitos |
| Pesos uniformes en [-0.5,0.5], bias inicial cero | Romper la simetría entre neuronas con pesos pequeños, manteniendo un punto de partida simple para los desplazamientos | `init_scale` configurable; no se asignaron soluciones a mano al entrenamiento |
| Mantener beta=1 en la validación | Coincide con el objetivo tanh(x) y con los ejemplos manuales | Configuración explícita; derivadas también probadas con beta distinto de 1 |
| Gradientes con signo matemático, luego restar el ajuste | Facilitar cotejo con derivadas numéricas | Equivalencia con el delta de corrección del apunte documentada en `implementacion.md` |
| Loss = media por muestra de ½ suma sobre salidas; MSE = media sobre todos los elementos | Diferenciar la función derivada de la métrica informada, incluso con varias salidas | Se documenta MSE=2·loss/cantidad_de_salidas; tests numéricos incluyen varias salidas |
| Entrenamiento online con épocas y barajado reproducible | Corresponde al algoritmo de clase y permite revisar una muestra a la vez | AND conserva orden; modelos diferenciables barajan cada época con generador local y semilla explícita |
| Tres semillas 0,1,2 | Detectar sensibilidad de la optimización sin convertir la validación en un estudio estadístico | 15 corridas finales; no se promete convergencia para toda inicialización |
| η=0.1 para AND, 0.05 para lineal/tanh y 0.1 para XOR profundo | Valores iniciales moderados para validar las fórmulas | Configurados y comprobados, sin afirmar optimalidad |
| Cambiar solo η de XOR [2,2,1] de 0.1 a 0.03 | Dos semillas se estancaron con 0.1; entre 0.01, 0.03 y 0.05 solo 0.03 cumplió todos los criterios en las tres semillas | Primera corrida y tabla comparativa preservadas; tolerancias y límites no relajados |
| MSE≤1e-6 en ajustes de funciones y MSE≤1e-3 más 100% accuracy en XOR | Exigir precisión cuantitativa además del signo correcto | Umbrales fijados antes de ejecutar; AND exige error cero |
| Límites de 100, 1000 y 10000 épocas para AND, funciones y XOR respectivamente | Evitar corridas sin límite y permitir mayor tiempo a la optimización no convexa | El CLI devuelve fallo si se alcanza el máximo sin cumplir el objetivo |
| Evaluar con pesos fijos al finalizar cada época | No mezclar predicciones obtenidas con distintos estados del modelo | Historial incluye época cero y métricas posteriores a cada recorrido |
| Comprobar gradientes por diferencias finitas | Converger en un ejemplo no demuestra que backpropagation esté bien implementado | Todos los pesos y bias de ambas arquitecturas XOR, modelos simples y red multisalida contrastados |
| Mantener XOR escalón como control negativo | Evitar una validación que declare éxito falso en un problema no separable | Test alcanza su máximo de épocas y conserva errores |
| Guardar pesos iniciales/finales, métricas, config y entorno | Poder revisar y repetir los resultados | NPZ sin pickle, CSV y JSON por corrida; recarga y continuación de parámetros comprobadas |
| No sobrescribir directorios de corridas con contenido | Preservar fallos y comparaciones anteriores | CLI rechaza una carpeta de salida no vacía |
| Graficar en un script separado a partir de los archivos guardados | Reproducir el análisis sin reentrenar y seguir la separación de responsabilidades de TP2 | `scripts/plot_validation.py`; seed 0 predeterminada para fronteras, las tres para curvas y ajustes |
| Conservar resumen e imágenes fuera de output | La regla existente del repo ignora output | Informe, CSV e imágenes en `docs/resultados-validacion/`; corridas detalladas locales en output |
| Reutilizar paquetes locales en el entorno de esta ejecución | NumPy y Matplotlib ya estaban instalados | `.venv` creada con `--system-site-packages`; README incluye instalación normal y entorno registrado |
| Mantener datasets reales intactos y no abrir todavía el test para entrenar | El alcance actual es validar la base que usarán esos ejercicios | No se normalizó ni entrenó fraude o dígitos; protocolos de evaluación pendientes |

Resultado: 31 pruebas automatizadas aprobadas y 15/15 corridas de la
configuración final aprobadas. El informe distingue resultados reales,
fallos de la configuración inicial y limitaciones de la validación.


## Publicación en main

- Se excluyen `docs/transcripciones/` mediante `.gitignore`, por pedido del
  equipo; se conservan localmente. Las referencias públicas ya no las enlazan
  como archivos disponibles en el repositorio.
- Se versionan explícitamente los tres fragmentos fuente de la guía en
  `output/visualizaciones/`, aunque la regla general ignore `output/`, para
  conservar las fuentes editables junto a la página exportada. Las corridas
  temporales y el entorno virtual siguen excluidos.
- Los cambios se separan en material/datos, motor y tests, resultados y
  gráficos, y documentación visual. Se respeta el estilo de commits y las
  identidades de coautoría humana que ya aparecen en el historial.

## Carga y separación training/test

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| Mantener los loaders con training y test, sin un tercer dataset permanente | Pedido explícito del usuario para la etapa de carga y preferencia por no agregar una partición fija | Aplicada en `src/sia_tp3/data.py`; no existe un tercer archivo o retorno de validation |
| Usar 20 % de fraude como test y semilla 0 por defecto en el protocolo de generalización | Disponer de una partición reproducible y configurable cuando solo existe un CSV | Aplicada después de la comparación inicial que usa las 7500 filas juntas; produce 6000 filas de training y 1500 de test |
| Estratificar fraude mediante `flagged_fraud` | Conservar aproximadamente el desbalance original en ambos subconjuntos | Aplicada únicamente al reparto; la columna no entra en `X` ni es objetivo del modelo y solo se devuelve para test |
| Entrenar fraude contra `big_model_fraud_probability` | Es el objetivo de destilación indicado por la documentación | Aplicada; `y_train` y `y_test` tienen una salida continua |
| Usar `digits_test.csv` completo como test externo | Es el conjunto que la consigna reserva para medir generalización | Aplicada; no se extraen muestras de ese archivo para training |
| Concatenar `more_digits.csv` únicamente al training del ejercicio 3 | Aislar el efecto de disponer de más datos sin alterar el test | Aplicada mediante `additional_train_path` opcional |
| Mantener imágenes en `float32` y etiquetas como enteros | Coincidir con el loader recibido y posponer la codificación hasta definir la salida de la red | Aplicada; formas verificadas contra los cuatro CSV reales |
| Posponer inicialmente normalización, estandarización y codificación | Esas transformaciones debían decidirse después del análisis de los datos | Aplicada durante los puntos 1–3; reemplazada para fraude por la decisión del punto 4 |
| No usar test para seleccionar configuraciones | Test debe representar la evaluación final | Vigente; la selección futura usará validation temporal derivada de training |

## Clases 12 y 13 y revisión del plan (28 de septiembre de 2026)

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| Incorporar los tres PDFs y las dos transcripciones recibidas | Completar el material que sustenta las siguientes etapas | Copias exactas en `docs/` y `docs/transcripciones/`; índice actualizado en `material-clases.md` |
| Versionar únicamente las nuevas transcripciones de clases 12.2 y 13 | El pedido actual incluye esos adjuntos y reemplaza la exclusión previa para ellos | Excepciones explícitas en `.gitignore`; las transcripciones anteriores siguen excluidas |
| Trabajar de a un punto y revisarlo antes de continuar | Pedido del equipo para comprender y justificar cada paso | Primero se vuelve a comprobar el motor con los casos de validación |
| Analizar las 7500 transacciones en el EDA de fraude | La consigna actualizada pide usar todas las muestras en la primera etapa de aprendizaje | Aplicado; el EDA fue recalculado y el split queda para el protocolo posterior de generalización |
| Incorporar validation temporal para los experimentos | Permite elegir hiperparámetros, diagnosticar generalización y fijar umbrales sin consultar repetidamente test | Aplicada en `experiments.py`: se deriva reproduciblemente de training, expone sus índices y no modifica los CSV ni el contrato de los loaders anteriores. Falta integrarla en cada runner experimental |
| Justificar preprocesamiento y regularización con evidencia | Seguir el hilo experimental de clases 12.2 y 13 | EDA sobre training antes de transformar; regularización solo ante un diagnóstico de sobreajuste; todavía no implementados |
| Mantener el motor y la configuración de validación actuales | La nueva ejecución pasó los 38 tests y las 15 corridas sin cambios | Punto 1 verificado; se recargaron los 15 modelos y se reprodujeron sus métricas; evidencia en `resultados-validacion.md` |

## Punto 3: análisis de training

En fraude, el EDA usa las 7500 transacciones sin split porque coincide con la
primera etapa de aprendizaje del ejercicio 1. En dígitos, `digits.csv` sigue
siendo training y `digits_test.csv` el test externo. No se introduce validation
en el EDA porque no se comparan modelos; los experimentos posteriores sí usan
la validation temporal implementada en `experiments.py`. Sólo para dígitos se
comparan inputs exactos entre training y test, sin usar etiquetas ni métricas.
El punto 3 del plan es el EDA; no debe confundirse con el ejercicio 3 de la
consigna, que incorpora los datos adicionales.

El commit inicial aplicó siete criterios de análisis. Durante la revisión del
punto se confirmaron además dos decisiones explícitas: comprobar solapamiento
exacto entre particiones sin utilizar etiquetas de test y conservar por ahora
variables, candidatos a outlier y píxeles constantes. Los resultados medidos
están en `eda-training.md` y los artefactos reproducibles en `eda-training/`.

| ID | Criterio o decisión | Motivo y aplicación | Alternativa considerada | Estado |
|---|---|---|---|---|
| EDA-1 | Datasets incluidos ahora | Fraude y `digits.csv`; dejar `more_digits.csv` para la incorporación de nuevos datos y conservar una referencia del ejercicio 2 | Empezar sólo por fraude, o analizar también `more_digits.csv` por separado | Aplicado en `scripts/analyze_training.py` |
| EDA-2 | Formato y herramientas | Script Python con NumPy/Matplotlib, tablas CSV e informe Markdown con gráficos; reutilizar las herramientas y organización del TP | Notebook con código y explicación | Aplicado en `scripts/analyze_training.py` |
| EDA-3 | Descripción de distribuciones y escalas | Tipos, unidades, faltantes, no finitos, cantidad de valores distintos, mínimo/máximo, media, mediana, desvío, cuartiles y percentiles 1/99; histogramas y boxplots. En dígitos: conteos por clase, rango de píxeles y ejemplos de imágenes | Resumen reducido a rangos, cuartiles, histogramas y balance | Aplicado en `scripts/analyze_training.py` |
| EDA-4 | Criterio para señalar outliers numéricos | Regla del boxplot: valores fuera de [Q1 − 1,5×IQR; Q3 + 1,5×IQR], con IQR = Q3 − Q1; revisar el significado de cada variable. Son candidatos, no errores confirmados ni una regla para eliminar registros o píxeles | Sólo percentiles y gráficos sin regla de outliers | Aplicado en `scripts/analyze_training.py` |
| EDA-5 | Balance real de fraude | Contar `flagged_fraud` en las 7500 filas sólo para describir proporciones, sin incorporarlo a entradas/objetivos ni usarlo para seleccionar variables. Analizar BigModel como probabilidad continua, sin fijar umbral | Mantener la etiqueta real oculta y describir sólo la distribución de probabilidades de BigModel; no llamarla balance de clases reales | Aplicado en `scripts/analyze_training.py` |
| EDA-6 | Diagnóstico de variables problemáticas | Revisar constantes, duplicados exactos, valores incompatibles con su significado y correlaciones lineales entre entradas y con BigModel. Correlación baja no implica irrelevancia. En dígitos: píxeles constantes e imágenes duplicadas, sin matriz de 784×784 | Calidad básica y constantes; posponer correlaciones | Aplicado en `scripts/analyze_training.py` |
| EDA-7 | Qué hacer ante hallazgos | Registrar evidencia y alternativas; decidir los tratamientos en el punto 4. Conservar originales y no imputar, eliminar, reescalar, balancear ni crear variables durante el EDA | Revisar y decidir el tratamiento de cada hallazgo antes de continuar el análisis | Aplicado en `scripts/analyze_training.py` |
| EDA-8 | Comprobar duplicados exactos entre training y test de dígitos | Detectar data leakage sin utilizar test para ajustar el modelo; se comparan únicamente los vectores de imagen y no se interpretan etiquetas, distribuciones ni métricas de test. En fraude no corresponde este control durante el EDA porque no hay split | Posponer cualquier apertura de test hasta la evaluación final | Confirmado en la revisión y aplicado; no se encontraron solapamientos en dígitos |
| EDA-9 | Conservar todas las entradas y candidatos IQR | Los outliers observados pueden ser casos reales y los 97 píxeles constantes sólo describen bordes sin variación; no hay evidencia de que eliminarlos mejore el modelo | Eliminar outliers, variables de baja correlación o píxeles constantes antes de entrenar | Confirmado en la revisión; se mantienen las nueve entradas de fraude y los 784 píxeles |

Base de las decisiones: consigna, página 4 (explorar documentación, rangos y
limpieza), clase 12.2 (distribuciones, balance, boxplots y escalas) y reglas
vigentes de `data/README.md`. La regla IQR, los percentiles y las correlaciones
son criterios seleccionados de EDA descriptivo estándar; no se presentan como parámetros
exigidos por la cátedra. Umbrales, gráficos particulares y tratamientos que
dependan de los hallazgos se justificarán al revisarlos con el equipo.


### Aplicación y resultados del punto 3

- Fraude convierte y analiza las entradas y probabilidades de las 7500 filas.
  La etiqueta real se lee únicamente para el conteo descriptivo; no entra en
  las correlaciones ni en el entrenamiento. La API pública y el motor no cambian.
- Dígitos usa `_load_digit_file` con `digits.csv`. De `digits_test.csv` sólo
  convierte los vectores de imagen para compararlos exactamente con training;
  no interpreta sus etiquetas ni calcula estadísticas. No abre
  `more_digits.csv`. Una imagen inválida detendría el control de integridad.
- Convenciones de cálculo explícitas del script: desvío descriptivo `ddof=0`,
  percentiles con interpolación lineal, Pearson por pares finitos y duplicados
  de entradas exactas. Faltantes/NaN e infinitos se informan por separado;
  los estadísticos usan valores finitos, sin modificar el dataset original.
- Presentación: histogramas con 30 intervalos para fraude y 40 para píxeles;
  eje Y logarítmico en el histograma de píxeles para ver el fondo dominante.
  Los boxplots conservan unidades originales y ejes independientes. Hasta tres
  imágenes por clase, muestreadas con semilla 0, con sus filas registradas.
  Estas son convenciones de cálculo y visualización, no hiperparámetros de
  entrenamiento ni requisitos atribuidos a la cátedra.
- Se verificaron 7500 transacciones y 12449 imágenes de training. No se
  detectaron faltantes/NaN, infinitos ni duplicados exactos en esos conjuntos.
  Tampoco aparecieron imágenes exactas compartidas entre training y test de
  dígitos. No hay alertas en los controles semánticos implementados.
- Fraude: 869 etiquetas positivas (11,59 %) y escalas muy diferentes entre
  variables. Las probabilidades de BigModel se describen como valores
  continuos; no se fija un umbral ni se comparan con la etiqueta real.
- Dígitos: clase 8 ausente, 271 muestras del 5, píxeles ya en [0,1] y 97 píxeles
  constantes en cero. Los conteos por clase se corroboraron directamente con
  `label` del CSV original, independientemente del loader.
- La regla IQR marca los trazos no negros como candidatos al aplicarla a todos
  los píxeles, porque Q1=Q3=0. Este hallazgo muestra por qué el criterio no debe
  convertirse automáticamente en una regla de limpieza.
- **Pendiente de discusión en el punto 4:** escalado de fraude y rango de
  entrada de dígitos. Se decidió conservar variables, candidatos IQR y píxeles
  constantes como referencia inicial. No se borraron, imputaron, balancearon
  ni transformaron datos; no se incorporaron ejemplos adicionales ni se
  evaluaron modelos.

## Punto 4: preprocesamiento

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| Estandarizar las nueve entradas de fraude | Sus escalas originales son muy diferentes y pueden producir actualizaciones desbalanceadas durante el entrenamiento. La clase 13 distingue este caso del min-max usado para adecuar una salida al intervalo de su activación | Aplicada en `data.py` mediante `Standardizer` |
| Ajustar el estandarizador sólo con `X_train` | Evitar que la distribución de test se filtre al entrenamiento | La separación ocurre antes del ajuste; `X_test` se transforma con la media y el desvío de training |
| No transformar `big_model_fraud_probability` ni `flagged_fraud` | La primera ya es una probabilidad en `[0,1]` y la segunda no es entrenable | Ambas salidas conservan sus valores originales |
| No aplicar min-max adicional al objetivo del modelo logístico | La logística tiene imagen `(0,1)` y `big_model_fraud_probability` ya pertenece a `[0,1]`; no hay que cambiar su intervalo | El modelo aprende directamente las probabilidades provistas por BigModel. Si se eligiera `tanh`, esta decisión debería revisarse porque su imagen es `(-1,1)` |
| Mantener los píxeles de dígitos en `[0,1]` | Ya están reescalados; volver a dividir por 255 reduciría incorrectamente su magnitud | El loader de dígitos permanece sin cambios |
| Conservar variables, candidatos IQR y los 97 píxeles constantes | No hay evidencia experimental que justifique eliminarlos | No se filtran columnas ni filas durante el preprocesamiento |
| Guardar los parámetros del estandarizador | Las entradas futuras deben recibir exactamente la misma transformación que training | `Standardizer.save/load` persiste medias, desvíos y orden de variables sin pickle |

Si una columna de training fuera constante, su escala se guarda como 1 para
centrarla en cero sin dividir por cero. En el dataset real de fraude no se
detectaron columnas constantes.

## Punto 5: métricas de clasificación

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| Usar filas reales y columnas predichas en la matriz | Mantener la convención explícita de la clase 12.2 | Aplicada en `metrics.py` |
| Calcular métricas one-vs-rest por clase | Extender TP, TN, FP y FN al problema multiclase de dígitos | Aplicada para precision, recall, F1, TPR y FPR |
| Informar accuracy global y promedios macro | Accuracy sola puede ocultar clases minoritarias; macro da igual peso a cada clase evaluable | Aplicada en `ClassificationReport` |
| Exigir la lista completa de clases | Evitar que clases ausentes, especialmente el 8, desaparezcan de la matriz | `labels` es obligatorio y también valida predicciones desconocidas |
| Marcar divisiones por cero como indefinidas | No confundir ausencia de evidencia con rendimiento igual a cero o uno | Se devuelve `NaN` para la métrica afectada y el promedio macro usa sólo valores definidos |
| Mantener la conversión de salidas fuera de las métricas | Poder comparar umbrales de fraude sin reimplementar fórmulas y usar `argmax` en dígitos | Las funciones reciben únicamente clases reales y predichas |
| Exponer recall y TPR | La consigna pide ambos nombres aunque representan la misma fórmula | Ambos campos contienen el mismo valor y se prueba esa igualdad |

## Punto 6: optimizadores y frecuencia de actualización

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| Implementar descenso básico, Momentum, eta adaptativo, RMSProp y Adam | Son las variantes desarrolladas en clase 12.1 y el equipo decidió disponer de todas antes de elegir cuáles comparar | Aplicada en `optimizers.py`; todavía no se eligió una ganadora para los ejercicios |
| Separar optimizador de tamaño de lote | Responden preguntas diferentes: el lote forma el gradiente y el optimizador transforma ese gradiente en un cambio de parámetros | `fit` recibe cualquier `Optimizer` y un `batch_size` independiente |
| Hacer configurables online, mini-batch y batch | La modalidad es una variable experimental que debe probarse, no quedar fijada al implementar el optimizador | `batch_size=1` es online, un valor intermedio es mini-batch y `None` o la cantidad total de muestras es batch |
| Promediar los gradientes dentro de cada lote | Evita que el tamaño del paso crezca sólo por incluir más muestras y coincide con la loss media del motor | `_gradients` divide por la cantidad de muestras del lote antes de actualizar |
| Mantener Rosenblatt exclusivamente online y con descenso básico | El escalón no tiene la derivada requerida por Momentum, RMSProp o Adam; inventar una extensión mezclaría algoritmos no presentados | `fit` rechaza otro optimizador o `batch_size` mayor que uno para activación `step` |
| Usar en Adam los valores de referencia de la diapositiva | Clase 12.1 sí presenta `eta=0.001`, `beta1=0.9`, `beta2=0.999` y `epsilon=1e-8` | Son defaults explícitos y pueden modificarse por experimento |
| Exigir los hiperparámetros no fijados por clase | Momentum, RMSProp y eta adaptativo requieren elecciones que no deben quedar escondidas como si fueran obligatorias | `alpha`, `gamma`, `epsilon`, incremento, reducción y paciencia se pasan al construir cada optimizador |
| Interpretar “consistentemente” como K épocas consecutivas | La diapositiva propone parametrizar la consistencia, pero no fija una única regla | Eta adaptativo compara loss de épocas contiguas y usa `patience=K`; suma `increase_by` o multiplica por `1-decrease_fraction` |
| Conservar epsilon dentro de la raíz en RMSProp y fuera en Adam | Es la diferencia escrita en las fórmulas de clase y altera el cálculo | Cubierta por cuentas manuales en `test_optimizers.py` |
| No seleccionar todavía optimizador ni tamaño de lote final | La implementación no aporta evidencia de cuál generaliza o converge mejor en cada dataset | Pendiente de experimentos controlados sobre training; test continúa reservado |

## Infraestructura del baseline de fraude

| Decisión | Motivo | Aplicación y estado |
|---|---|---|
| Registrar MSE de validation dentro de `fit` | Medir ambas curvas con exactamente los mismos pesos al final de cada época | Aplicada mediante `validation_data`; validation no genera gradientes ni interviene en el criterio de convergencia |
| Comparar primero activación lineal y logística | El ejercicio exige perceptrón lineal y no lineal; logística conserva una salida compatible con probabilidades | La corrida `output/fraud-baseline-01` queda sólo como sanidad histórica porque utilizó un split |
| Usar descenso básico como baseline | Comenzar con la actualización más directa antes de atribuir diferencias a optimizadores avanzados | Utilizado en la corrida de sanidad; no se lo selecciona todavía como optimizador final |
| Mantener los hiperparámetros en JSON | Revisarlos y modificarlos sin esconder decisiones dentro del runner | La sanidad usó eta 0,01, mini-batch 32, máximo 500 épocas, inicialización 0,5 y semillas 0; son valores iniciales, no una configuración final aprobada |
| Usar `target_mse=0` en la corrida de sanidad | Obtener la curva completa hasta el máximo en vez de detenerla con una tolerancia aún no justificada | La sanidad recorrió las 500 épocas; cero no constituye un objetivo realista ni una conclusión experimental |
| Separar aprendizaje de generalización en el ejercicio 1 | El enunciado y la aclaración de clase indican que la comparación de capacidad usa todas las muestras; la manipulación de datos corresponde a la etapa posterior | `protocol=learning` usa las 7500 filas del CSV, sin validation/test; `protocol=generalization` exige un solo modelo seleccionado y recién entonces divide los datos |
| No evaluar generalización durante la comparación inicial | Underfitting y saturación se observan mediante el error de training; sin muestras separadas no se concluye overfitting ni generalización | El modo `learning` guarda únicamente historial y predicciones de training |
| Limitar la primera etapa a lo obligatorio del ejercicio 1 | El enunciado exige comenzar comparando el perceptrón lineal y el no lineal respecto del aprendizaje; no exige una grilla completa de todos los hiperparámetros para este ejercicio | Primero se hará una comparación controlada con iguales condiciones de entrenamiento. Los valores del JSON son un punto de partida modificable, no candidatos que deban cruzarse todos desde el inicio |
| Graficar desde historiales guardados | Separar entrenamiento de análisis y evitar repetir una corrida para cambiar una figura | `plot_fraud_experiment.py` genera las curvas completas normal y logarítmica; muestra sólo training en aprendizaje y agrega validation en generalización |
