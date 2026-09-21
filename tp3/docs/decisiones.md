# Registro de decisiones del TP3

Por pedido del equipo, toda decisión debe explicar **qué**, **por qué** y
**cómo se aplicó**, e indicar si ya está aplicada o sigue pendiente.
Este registro se actualiza junto con los cambios. Una elección didáctica no
debe presentarse como requisito de la cátedra ni como resultado experimental.

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
| No agregar optimizadores por el solo hecho de aparecer en clase | El objetivo actual es entender y verificar las reglas básicas | Momentum, AdaGrad y Adam siguen pendientes de selección y justificación |
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
