# TP5 - Plan de trabajo y aprendizaje

Fecha de entrega y presentacion: **17 de noviembre de 2026**.

Este plan no supone que hay que entender toda la matematica antes de empezar.
La idea es avanzar en capas: entender una pieza pequena, implementarla, verificarla
y recien entonces agregar la siguiente. Las pausas marcadas como **PING-PONG** son
parte del trabajo, no demoras: no se avanza hasta poder explicar con palabras
propias que entra, que sale y que error se esta minimizando.

## Estado actual - Etapa 8 iniciada el 14 de noviembre

> **Corrección de proceso:** la gramática, el split y la representación que se
> enumeran abajo ya fueron discutidos y aprobados por la usuaria. Las
> arquitecturas, pesos KL y corridas de código anteriores siguen siendo pilotos
> exploratorios del asistente: no seleccionan el modelo final ni reemplazan el
> embudo experimental que se acordará con la usuaria antes de iniciar las
> corridas del VAE final.

### Decisiones aprobadas de la Etapa 7

- Familia de plantilla fija:
  `def f(x): return (x <op1> c1) <op2> c2`.
- Operadores `{+, -, *}` y constantes enteras de 0 a 99: universo de 90.000
  fuentes posibles.
- Split estratificado por los nueve pares de operadores: 1.800 train, 450
  validación y 450 test. Cada par aporta 200/50/50 casos.
- Una generación se evaluará por compilación, pertenencia AST, ausencia textual
  de los 2.700 casos guardados y novedad semántica de su firma exacta `(a,b)`.
- El dataset guardado contiene 1.786 firmas semánticas; el universo contiene
  41.524 y deja 39.738 comportamientos fuera de todos los splits.
- Cada una de las 26 posiciones se representa mediante un one-hot sobre un
  vocabulario de 25 caracteres: 650 valores por fuente.
- El decoder producirá 26 distribuciones separadas mediante softmax agrupado y
  la reconstrucción se medirá como la suma de 26 entropías cruzadas, promediada
  sobre el lote. Esta combinación ya tiene gradientes numéricos verificados en
  el AE y en el VAE completo con KL.
- El control de capacidad del AE comparó anchuras 64, 128 y 256 con latente 16
  bajo un protocolo prefijado. Solo `650-256-16-256-650` superó 100 % exacto en
  train, 99 % en validación y 100 % de gramática durante tres controles.
- Al abrir test una vez, el AE seleccionado reconstruyó 448/450 fuentes
  (`99,56 %`); las 450 compilaron y pertenecieron a la gramática. Esta evidencia
  habilita el embudo variacional, pero no selecciona todavía el VAE final.

### Completado

- Etapas 0-2: reglas, dataset, parser, autoencoder mínimo y gradientes
  numéricos verificados.
- Etapa 3: búsqueda de arquitectura y optimización cerrada mediante un embudo
  reproducible, sin elegir retrospectivamente checkpoints favorables.
- Configuración final: `35 -> 45 -> 2 -> 45 -> 35`, BCE, descenso de gradiente
  con tasa 0.1 y 2.000 épocas full-batch.
- Verificación: reconstrucción exacta de los 32 patrones en las diez semillas
  0-9.
- Corrida final: semilla 0, 2.000 épocas completas, pesos, historia, métricas,
  reconstrucciones y coordenadas latentes guardadas en
  `results/ae-final-bce-width45-seed0/`.
- Publicación: curva, reconstrucciones, mapa latente, métricas y coordenadas
  finales copiadas a `docs/` para que el informe no dependa de `results/`, que
  está ignorado por Git.
- Mapa latente generado y discutido: cualquier par puede entrar al decoder,
  pero solo las regiones aprendidas tienen respaldo empírico; los ejes no poseen
  semántica fija.
- Generación: el segmento `m-n` fue descartado porque sus interpolaciones se
  binarizan como patrones existentes. El punto medio `a-x` produce un patrón
  nuevo, a distancia de Hamming 1 de `c`, y quedó congelado como caso final.
- Etapa 4 cerrada: la utilidad reproducible carga el modelo, genera siete puntos
  del recorrido `a-x`, verifica pertenencia y distancia de Hamming, y guarda
  datos más una figura original en `docs/`.
- Ejercicio 1A cerrado: no reabrir arquitectura, optimización, semillas ni
  selección del segmento latente sin una hipótesis nueva que lo justifique.
- El entrenamiento ya admite entradas y objetivos distintos, y conserva el
  comportamiento anterior cuando no se indican objetivos explícitos.
- El ruido bit-flip es reproducible y el DAE genera una corrupción nueva en
  cada época manteniendo siempre el patrón limpio como objetivo.
- Se ejecutó el primer piloto con `p_train = 0.10`, arquitectura
  `35 -> 45 -> 2 -> 45 -> 35`, BCE, descenso con tasa 0.1, semilla de modelo 0
  y 2.000 épocas. La evaluación usó 3.200 corrupciones independientes con
  `p_test = 0.10`.
- El piloto todavía no elimina ruido en promedio: la entrada tuvo 3,470 errores
  por patrón y la salida 3,581, una mejora media de -0,111. Recuperó exactamente
  el 28,1 % de los casos, mejoró el 50,6 % y empeoró el 39,1 %.
- El control limpio del mismo modelo produjo 1,875 errores medios y 13 de 32
  reconstrucciones exactas. Por eso el resultado no demuestra aún falta de
  capacidad de la arquitectura: primero hay que separar presupuesto y
  optimización insuficientes del efecto del cuello latente.
- Al extender únicamente el presupuesto de 2.000 a 10.000 épocas, con las
  mismas semillas y exactamente la misma evaluación, la salida bajó a 3,275
  errores medios frente a 3,470 de entrada: una mejora de 0,195. La recuperación
  exacta subió de 28,1 % a 34,0 %, pero el control limpio permaneció en 1,875
  errores medios y bajó de 13 a 12 patrones exactos.
- Más épocas corrigieron la mejora media, pero no estabilizaron el aprendizaje:
  varias letras que eran fáciles a las 2.000 épocas pasaron a estar entre las
  difíciles a las 10.000. No se justifica seguir aumentando el presupuesto sin
  revisar primero la variabilidad de las actualizaciones con ruido y tasa 0.1.
- Manteniendo las 10.000 épocas y bajando solamente la tasa de 0.1 a 0.03, el
  error de salida cayó a 1,304 frente a 3,470 de entrada. La mejora media fue
  2,166, la recuperación exacta 70,5 % y las 32 letras mejoraron en promedio.
  El control limpio alcanzó 0,219 errores medios y 27 de 32 patrones exactos.
- Los tres pilotos y su protocolo quedaron documentados en
  `docs/dae/ejercicio_1b.md`. La evidencia apunta a inestabilidad de optimización en
  los primeros intentos, no a una falta demostrada de capacidad arquitectónica.
- El modelo congelado se evaluó en `p_test = 0`, `0.05`, `0.10`, `0.20` y
  `0.30`, con 3.200 casos independientes por nivel. Los errores medios de salida
  fueron 0,219; 0,464; 1,304; 5,219 y 9,498, respectivamente.
- La recuperación es fuerte hasta `p = 0.10`, sigue siendo positiva pero se
  degrada en `p = 0.20`, y en `p = 0.30` queda una mejora media menor a un
  píxel. La figura y los agregados se publicaron en
  `docs/dae/pilot/noise-curve-seed0.*`.
- El refinamiento local comparó tasas `0.01`, `0.03` y `0.05` sobre la misma
  inicialización y los mismos 16.000 casos. `0.03` produjo el menor error de
  salida en los cinco niveles y la mayor recuperación exacta hasta `p = 0.20`.
  Se selecciona provisionalmente, todavía sin declararla final por tratarse de
  una sola semilla. La tabla completa está en
  `docs/dae/experiments/rate-comparison-seed0.csv` y `docs/dae/ejercicio_1b.md`.
- La tasa `0.03` se confirmó con semillas 0, 1 y 2, variando inicialización y
  ruido de entrenamiento. En `p = 0.10` el error de salida fue
  1,258 ± 0,040 y la recuperación exacta 73,4 ± 2,6 %. La forma de la curva
  se conservó en las tres corridas. Los resultados por semilla y agregados están
  en `docs/dae/experiments/confirmation-seeds012.csv` y
  `docs/dae/experiments/confirmation-aggregate.csv`.
- Se comparó una corrupción por letra (`K = 1`) contra cuatro (`K = 4`) durante
  10.000 actualizaciones y tres semillas. `K = 4` fue peor y mucho más variable:
  en `p = 0.10` produjo 2,582 ± 1,389 errores frente a 1,258 ± 0,040 con
  `K = 1`. Se conserva una corrupción por letra. La comparación completa está en
  `docs/dae/experiments/corruptions-comparison.csv` y `docs/dae/ejercicio_1b.md`.
- El tamiz arquitectónico comparó el baseline `35-45-2-45-35`, una variante
  ancha `35-60-2-60-35` y una profunda `35-45-15-2-15-45-35`. Las dos variantes
  mejoraron con semilla 0; la profunda ganó en ruido medio y pasó a confirmación.
- En tres semillas, la profunda redujo el error respecto del baseline de 1,258
  a 1,013 en `p = 0.10` y de 5,056 a 4,389 en `p = 0.20`. La recuperación exacta
  subió de 73,4 % a 79,9 % y de 37,8 % a 49,8 %, respectivamente. Se selecciona
  `35-45-15-2-15-45-35` como arquitectura del DAE. Los agregados están en
  `docs/dae/experiments/architecture-comparison.csv`.
- Sobre la arquitectura profunda se comparó entrenamiento exclusivamente
  ruidoso contra un lote con 32 entradas limpias y 32 corrupciones nuevas por
  época. La mezcla mejoró los cinco niveles en tres semillas: la recuperación
  exacta subió de 90,6 % a 99,0 % en `p = 0`, de 88,2 % a 96,7 % en `p = 0.05`
  y de 79,9 % a 88,4 % en `p = 0.10`. Se selecciona el lote limpio + ruidoso.
  La comparación está en `docs/dae/experiments/clean-noisy-comparison.csv`.
- Se incorporó un schedule exponencial de tasa `0.03 -> 0.003`. En tres semillas
  redujo el error medio en los cinco niveles y casi eliminó la dispersión en
  `p = 0.10` (0,659 ± 0,002 errores), pero la recuperación exacta allí pasó de
  88,4 % a 87,8 %. Se selecciona el schedule por error y estabilidad, dejando
  explícito el trade-off. Los datos están en
  `docs/dae/experiments/schedule-comparison.csv`.
- Se comparó el latente bidimensional contra uno de dimensión 5, manteniendo el
  protocolo estable. El latente 5 redujo el error de 0,659 a 0,400 en
  `p = 0.10`, de 4,037 a 3,073 en `p = 0.20` y de 8,664 a 7,538 en `p = 0.30`.
  La recuperación exacta subió a 98,3 % en `p = 0.05` y 90,4 % en `p = 0.10`.
  Se selecciona `35-45-15-5-15-45-35`; la comparación está en
  `docs/dae/experiments/latent-comparison.csv`.
- Con el latente 5 se comparó entrenar solo con `p_train = 0.10` contra un lote
  limpio + `0.05` + `0.10` + `0.20`. En tres semillas, la mezcla subió la
  recuperación exacta a 98,7 % en `p = 0.05`, 91,7 % en `p = 0.10`, 57,1 % en
  `p = 0.20` y 21,2 % en `p = 0.30`. Se selecciona el entrenamiento multinivel;
  los datos están en `docs/dae/experiments/multilevel-comparison.csv`.
- Sobre el protocolo multinivel se comparó `tanh` contra Leaky ReLU en las capas
  ocultas. En tres semillas, Leaky ReLU subió la recuperación exacta de 91,7 % a
  93,8 % en `p = 0.10`, de 57,1 % a 64,4 % en `p = 0.20` y de 21,2 % a 28,0 %
  en `p = 0.30`. Se selecciona Leaky ReLU; la comparación está en
  `docs/dae/experiments/activation-comparison.csv`.
- El tamiz de optimizadores del DAE mostró que Adam supera al descenso con
  schedule. El refinamiento local comparó tasas `0.001`, `0.002`, `0.003` y
  `0.005`; luego se confirmaron `0.003` y `0.005` con semillas 0, 1 y 2.
  Adam `0.005` redujo el error medio en toda la curva y alcanzó 96,34 % exacto
  en `p = 0.10`, 75,47 % en `p = 0.20` y 37,09 % en `p = 0.30`. Se selecciona
  Adam `0.005`. Los resultados están en `docs/dae/experiments/adam-rate-screen-seed0.csv`
  y `docs/dae/experiments/adam-rate-confirmation.csv`.
- La configuración seleccionada se confirmó finalmente con diez semillas sin
  reabrir los hiperparámetros. La recuperación exacta fue 99,06 ± 0,14 % en
  `p = 0.05`, 96,06 ± 0,47 % en `p = 0.10`, 74,65 ± 1,24 % en `p = 0.20` y
  36,58 ± 1,31 % en `p = 0.30`. En limpio alcanzó 100 % en las diez semillas.
  La baja dispersión permite congelar formalmente el DAE. Los resultados están
  en `docs/dae/final/confirmation-seeds.csv` y
  `docs/dae/final/confirmation-aggregate.csv`.
- Se publicó la curva agregada de diez semillas y un análisis reproducible de
  casos. En `p = 0.10` el error baja de 3,514 a 0,151; en `p = 0.20`, de 7,010
  a 1,715; y en `p = 0.30`, de 10,505 a 6,059, aunque allí 16,93 % de los casos
  empeoran. Los ejemplos fácil, difícil y fallido se eligen mediante reglas
  fijadas, no por inspección visual. Los artefactos están en
  `docs/dae/final/`.
- El ejercicio 1B queda cerrado respecto de la consigna: se distorsionaron las
  entradas en cinco niveles y la capacidad de eliminar ruido se estudió con
  error antes/después, recuperación exacta, casos empeorados, diez semillas y
  ejemplos representativos. La conclusión y las limitaciones están redactadas
  en `docs/dae/ejercicio_1b.md`.
- Se completó el ping-pong 6A y la introducción de 6B: `mu` ubica el centro de
  la distribución de una entrada, `log_var` controla su dispersión y la loss de
  reconstrucción y KL ejercen fuerzas opuestas que deben equilibrarse.
- Para aislar el mecanismo se creó un dataset de juguete de seis barras binarias
  `3x3`. No es el dataset nuevo exigido por el ejercicio 2 ni condiciona su
  elección posterior.
- Se agregó un `VariationalAutoencoder` separado del AE/DAE congelado, con
  encoder compartido y dos ramas lineales de igual dimensión para `mu` y
  `log_var`. El primer checkpoint mantuvo deliberadamente `z = mu` y reconstruyó
  sin errores las seis barras en 92 épocas con la semilla 0.
- Se incorporó la reparametrización
  `z = mu + exp(0.5 * log_var) * epsilon`, todavía sin KL. El muestreo usa un
  generador con semilla separada y puede recibir un `epsilon` explícito para
  hacer comparaciones controladas.
- Con `epsilon` fijo y no nulo, los gradientes de reconstrucción de ambas ramas
  coincidieron con diferencias finitas. La discrepancia máxima fue menor que
  `4e-10`; los gradientes de `log_var` son ahora distintos de cero.
- Con `epsilon = 0` se recupera exactamente `z = mu` y el gradiente de
  `log_var` vuelve a cero, como exige la derivación. Cinco pruebas cubren formas,
  fórmula, gradientes, checkpoint determinista y muestreo reproducible.
- Se agregó la KL diagonal cerrada y su peso configurable. La loss total y sus
  gradientes completos coincidieron con diferencias finitas; incluso con
  `epsilon = 0`, KL actualiza ahora la rama `log_var`.
- El entrenador específico del VAE reutiliza el mismo `epsilon` en predicción,
  loss y backward de cada época y registra total, reconstrucción y KL por
  separado. Nueve pruebas cubren el mecanismo y su reproducibilidad.
- El primer piloto usa `kl_weight = 0.01`, Adam `0.01`, 2.000 épocas y semilla
  de modelo 0. En las últimas 100 épocas obtuvo BCE `0,0128`, KL sin ponderar
  `5,856`, `0,0033` errores medios y `5,99/6` patrones exactos. Es una prueba de
  funcionamiento, no una selección final del peso KL.
- El VAE puede guardarse y cargarse preservando distribución, decoder y peso
  KL. El piloto quedó almacenado en `results/vae-toy-kl001-seed0/model.npz`.
- Se muestrearon 1.000 puntos reproducibles de `N(0,I)` y se decodificaron sin
  usar el encoder. El `58,7 %` produjo exactamente una de las seis barras; la
  distancia de Hamming mínima media fue `0,62`, el máximo fue 3 y aparecieron
  las seis clases válidas.
- Las 1.000 muestras generaron 46 patrones binarios distintos y mostraron sesgo
  hacia la barra vertical central. Esto es consistente con un peso KL pequeño:
  demuestra generación desde el prior, pero también que la alineación no es
  perfecta. La grilla, el CSV y el resumen están en `docs/vae/`.
- Como piloto exploratorio se construyó una gramática de 900 funciones Python
  aritméticas de 24 caracteres. Cada carácter usa siete bits ASCII, dando 168
  features binarias. El split reproducible contiene 720 fuentes de train y 180
  de test; el CSV está en `data/python_functions.csv`.
- Las tres métricas del dataset se mantienen separadas: compilación sin
  ejecución, pertenencia del AST a la gramática y novedad sintáctica respecto
  de train. Las 900 fuentes originales pasan las dos primeras.
- El AE control `168-64-8-64-168` reconstruyó exactamente train en 2.301 épocas.
  En test reconstruyó 175 de 180 fuentes; las 180 compilaron y pertenecieron a
  la gramática, descartando una falta de capacidad base.
- Un tamiz exploratorio de KL comparó `0.01`, `0.1` y `1.0` con semilla 0. `0.1` conservó
  reconstrucción media exacta en train/test y elevó la pertenencia del prior a
  `89,6 %`; `0.01` obtuvo `50,8 %` y `1.0` obtuvo `99,4 %` a costa de reducir
  la reconstrucción exacta a 149/720 y 20/180.
- El piloto señaló `kl_weight = 0.1` como candidato y se lo ejecutó con semillas 0, 1 y 2. Sobre
  1.000 puntos fijos del prior, `90,63 ± 3,76 %` perteneció a la gramática y
  `15,93 ± 1,27 %` fue válido y no visto en train. Las tres semillas
  reconstruyeron exactamente train y test en modo `z=mu`.
- La primera muestra válida y nueva por una regla prefijada fue
  `def f(x): return (x+1)+3`. También se conservaron casos compilables fuera de
  la familia e inválidos. En una interpolación objetiva, ocho de nueve puntos
  pertenecieron a la gramática y el punto medio no compiló.
- Existe un borrador exploratorio del ejercicio 2 en `docs/ejercicio_2.md`; no
  debe tratarse como informe aprobado hasta rehacer juntos las decisiones.

### Próximo paso

1. Mantener congeladas las configuraciones finales del AE y del DAE.
2. Conservar las corridas de código solo como pilotos exploratorios; no hay VAE
   final congelado.
3. Diseñar con la usuaria un embudo del VAE que produzca evidencia suficiente
   sobre reconstrucción, regularización, validez, diversidad, novedad y
   estabilidad. No iniciar esas corridas antes de acordar sus factores,
   métricas, reglas de selección y cantidad de semillas.

### Extensiones opcionales del DAE

Solo se retoman después del trabajo obligatorio y sin modificar la conclusión
principal:

- matriz de confusión por nivel, asignando salidas incorrectas al patrón limpio
  más cercano por distancia de Hamming y registrando empates;
- análisis del latente de dimensión 5 mediante distancias en el espacio original
  y PCA exclusivamente como visualización;
- evaluación con ruido estructurado, como filas, columnas o bloques;
- una nueva búsqueda de arquitectura u optimización únicamente si se formula
  una hipótesis nueva y se la trata como extensión separada.

La explicación conceptual y la especificación de la figura quedaron en
`docs/ejercicio_1a.md`, sección **Movimiento en el espacio latente**.

No realizar `git push` desde el asistente. Los pushes quedan exclusivamente a
cargo de la usuaria.

## Bitácora histórica del avance

Los checkpoints siguientes se conservan para reconstruir por qué se tomaron las
decisiones. Sus textos de “próximo paso” describen el estado de ese momento y no
reemplazan el checkpoint actual anterior.

### Estado inicial - 7 de noviembre

La **Etapa 0** quedo cerrada conceptualmente. Se tomaron estas decisiones:

- Se pueden usar NumPy, Matplotlib y herramientas generales, pero no una
  libreria que implemente o entrene el autoencoder por nosotros. Si aparece un
  enunciado 2026 actualizado, se vuelve a verificar esta interpretacion.
- TP5 tendra su propio paquete. Se reutilizaran y adaptaran las partes necesarias
  del MLP del TP3 (capas densas, activaciones, backpropagation, optimizadores y
  persistencia), sin importar directamente el paquete `sia_tp3` ni modificarlo.
- El autoencoder se entrena completo: para una entrada `x`, el objetivo es esa
  misma `x`; el error de reconstruccion actualiza encoder y decoder.
- El encoder comprime los 35 pixeles a `(z1, z2)` y el decoder reconstruye los
  35 pixeles a partir de esos dos valores.
- El criterio de cumplimiento sera el peor caso: cada uno de los 32 patrones
  debe tener como maximo un pixel incorrecto despues de umbralizar la salida. Un
  promedio menor que uno no alcanza si algun patron supera ese limite.
- Los hiperparametros viviran en archivos JSON y un runner generico guardara
  configuracion, semilla, historia, metricas, pesos y figuras por corrida. No se
  editaran constantes dentro del script para distinguir experimentos.

La **Etapa 1** tambien quedo cerrada el 7 de noviembre:

- El cargador lee directamente `data/font.h` y convierte cada fila hexadecimal
  en cinco bits, del bit mas significativo (izquierda) al menos significativo.
- Se obtienen imagenes de forma `(32, 7, 5)` y patrones aplanados de forma
  `(32, 35)`, con valores exclusivamente binarios.
- Diez pruebas verifican conversiones conocidas, cantidad y orden de caracteres,
  forma, valores y orientacion izquierda-derecha.
- La grilla `docs/font-original.png` fue inspeccionada visualmente y las letras
  se ven correctamente orientadas.

Proximo paso exacto: comenzar la Etapa 2 adaptando el MLP y demostrar que puede
sobreajustar un solo sprite; todavia no se elige la arquitectura final.

La **Etapa 2** quedo cerrada en su version minima el 7 de noviembre:

- Se adapto el nucleo del MLP del TP3 a un autoencoder con `encode`, `decode` y
  acceso explicito a la capa latente.
- La arquitectura didactica provisional es `35 -> 15 -> 2 -> 15 -> 35`, con
  `tanh` en las capas ocultas, codigo latente lineal y salida logistica.
- Se usa MSE inicialmente por ser una loss ya implementada y facil de verificar;
  esto no la convierte todavia en la eleccion final frente a BCE.
- Backpropagation coincidio con gradientes numericos en una red diminuta.
- Con Adam y tasa `0.01`, una unica letra `a` paso de 16 pixeles incorrectos a
  cero en 12 epocas. Su codigo latente aprendido fue aproximadamente
  `(1.742, -1.581)` para esa semilla.
- Este resultado solo demuestra que la implementacion supera una prueba basica
  de funcionamiento. No demuestra que la arquitectura pueda aprender los 32
  patrones ni que esos hiperparametros sean finales.

Proximo paso exacto: crear una corrida base reproducible sobre los 32 patrones
manteniendo inicialmente la misma arquitectura, loss y optimizador, para que el
unico cambio conceptual sea la cantidad de muestras.

### Checkpoint de cierre - 7 de noviembre

Se completo la calibracion minima de optimizadores:

- Se implementaron y probaron descenso, Momentum, RMSProp y Adam.
- La calibracion inicial ejecuto 24 corridas de 2.000 epocas sobre las
  arquitecturas ancla de 15 y 35 neuronas, con semilla 0. No hubo divergencias.
- Las tasas muy pequenas siguieron mejorando, pero quedaron descartadas del
  tamiz por baja eficiencia dentro del presupuesto, no por incapacidad.
- Se detecto interaccion arquitectura-tasa: GD 1.0 y Momentum 0.2 fueron
  inestables en la ancla de 15, pero continuaron mejorando en la de 35.
- Se ejecutaron seis corridas de refinamiento. GD 0.3 fue el compromiso mas
  estable; GD 0.5 y Momentum 0.1 no aportaron una mejora consistente.
- Los perfiles que pasan al tamiz son GD 0.3, Momentum 0.01, RMSProp 0.01 y
  Adam 0.01. Son representantes provisionales, no hiperparametros finales.
- El hilo experimental y las tablas quedaron documentados en
  `docs/ejercicio_1a.md`.

Proximo paso exacto al retomar:

1. Definir una configuracion de tamiz con A0-A4, los cuatro perfiles anteriores,
   semilla 0 y presupuesto fijo de 2.000 epocas.
2. Verificar en modo seco que sean 20 corridas.
3. Ejecutar sin figuras ni modelos individuales.
4. Analizar estado final y trayectoria; no seleccionar por el mejor checkpoint.
5. Promover solo combinaciones justificadas a semillas 1 y 2.

El commit local de esta etapa es
`c802fe7 feat(tp5): agregar calibracion de optimizadores`. No realizar `git push`
desde el asistente; los pushes quedan exclusivamente a cargo de la usuaria.

### Avance del cruce grueso

El tamiz A0-A4 por los cuatro perfiles con semilla 0 ya fue ejecutado: 20
corridas, 2.000 epocas, 9 segundos y ninguna divergencia. Ninguna combinacion
cumplio aun el maximo de un pixel. A0 y A1 quedaron detras; A2, A3 y A4 con GD
o RMSProp llegaron a maximo 2. Adam fue inestable en A4 y Momentum quedo
dominado en esta semilla.

Proximo paso exacto:

1. La confirmacion con semillas 1 y 2 ya fue ejecutada para siete combinaciones.
   A3 + GD obtuvo maximos 2, 2 y 1 y definio la region principal.
2. Se refinaron localmente las tasas de GD, RMSProp y Adam sobre A3. Esto mostro
   que Adam 0.003 y RMSProp 0.03 merecian controles acotados, aunque GD siguio
   siendo la opcion principal.
3. El tamiz local comparo anchos 30 y 45. Ancho 45 + GD 0.3 cumplio el requisito
   en semilla 0; el ancho 30 no mejoro A3.
4. La confirmacion local dio, para ancho 45 + GD 0.3, maximos 1, 2 y 1 en las
   semillas 0, 1 y 2. Cumple en dos de tres y queda como finalista principal.
   Ancho 45 + RMSProp 0.03 cumple en una de tres y queda como control secundario.

Proximo paso exacto al retomar:

1. La prueba de presupuesto ya se ejecuto: ancho 45 + GD 0.3 con semillas 0, 1
   y 2 a 4.000 epocas. Las losses bajaron y no hubo deterioro, pero las metricas
   discretas quedaron iguales que a 2.000: maximos 1, 2 y 1.
2. El refinamiento de capacidad tambien se ejecuto a 4.000 epocas. Ancho 50 dio
   maximos 1, 2 y 2; ancho 60 dio 1, 2 y 3. Ambos cumplen solo en una de tres
   semillas y no superan al ancho 45.
3. La comparacion profunda tambien se ejecuto. D1
   (`35 -> 35 -> 15 -> 2 -> 15 -> 35 -> 35`) fue inestable y cumple en una de
   tres. D2 (`35 -> 45 -> 20 -> 2 -> 20 -> 45 -> 35`) dio maximos 1, 1 y 2 y
   cumple en dos de tres, igual que ancho 45, pero con mas parametros.
4. Se eligio ancho 45 sobre D2 por parsimonia: mismo numero de cumplimientos con
   aproximadamente 50 % menos parametros. D2 queda como evidencia de que la
   profundidad puede funcionar, no como finalista principal.
5. Se implemento BCE, se valido su gradiente numericamente y se calibro la tasa
   sin reutilizar automaticamente la de MSE. GD 0.01 fue lento, 0.3 inestable y
   0.03/0.1 viables.
6. Ancho 45 + BCE + GD 0.1 termino con reconstruccion exacta de los 32 patrones
   en las diez semillas 0-9 a 2.000 epocas. Las semillas 5-9 se agregaron solo
   como verificacion posterior, sin reabrir hiperparametros. La Etapa 3 queda
   cerrada; no se justifica ampliar el barrido.

Proximo paso exacto:

1. La corrida final reproducible ya fue ejecutada con semilla 0 y 2.000 epocas
   completas. Guarda modelo, metricas, historia, curva, reconstrucciones y
   coordenadas latentes; reconstruye exactamente los 32 patrones.
2. Hacer el ping-pong de lectura del mapa latente: que significan distancia,
   vecindad e interpolacion, y que conclusiones no permite sacar.
3. Elegir justificadamente un punto nuevo del espacio latente y decodificarlo.
4. Con eso cerrar el ejercicio 1A antes de comenzar el denoising autoencoder.

Figura pendiente para el informe:

- crear una visualizacion propia del recorrido entre dos codigos latentes, no
  copiar el diseño de la referencia;
- mostrar los extremos y varios valores de `t` para
  `z(t) = (1-t) z_a + t z_b`;
- para cada punto, decodificar, umbralizar y mostrar el sprite `7x5`;
- marcar objetivamente si el patrón coincide con uno de entrenamiento o si es
  nuevo;
- acompañar la tira con los valores de `t` y, si no sobrecarga la figura, la
  distancia de Hamming al patrón conocido más cercano;
- usar el segmento `a-x` ya identificado como candidato, cuyo punto medio es
  nuevo y queda a un píxel de `c`; no seguir buscando una salida más atractiva
  después de congelar esta elección.

La figura debe enseñar que moverse continuamente en el latente no implica que
la salida binarizada cambie de manera continua: puede permanecer igual durante
parte del trayecto y cambiar al cruzar el umbral de uno o varios píxeles.

## 1. Que pide el TP

### Parte 1A - Autoencoder basico

- Leer los 32 sprites binarios de `data/font.h`, de 7 filas por 5 columnas.
- Implementar encoder y decoder sin usar una libreria que construya o entrene la
  red neuronal.
- Comprimir los 35 pixeles a un espacio latente de dimension 2.
- Reconstruir los 32 patrones con **como maximo un pixel incorrecto por patron**.
- Comparar y justificar arquitecturas y tecnicas de optimizacion.
- Graficar los 32 codigos latentes en dos dimensiones.
- Elegir un punto nuevo del espacio latente y mostrar la letra/sprite que genera
  el decoder.

### Parte 1B - Denoising autoencoder (DAE)

- Entrenar una arquitectura conveniente con una entrada distorsionada y el
  patron limpio como salida esperada.
- Evaluar distintos niveles de ruido y medir cuanto del original recupera.

### Parte 2 - Variational autoencoder (VAE)

- Elegir o construir un dataset nuevo.
- Implementar el encoder variacional, el muestreo mediante reparametrizacion y
  la perdida de reconstruccion mas regularizacion KL.
- Generar muestras nuevas y justificar que pertenecen a la familia aprendida.

La explicacion oral de 2026 agrega dos precisiones que no aparecen en el PDF de
2023: **no usar emojis** y construir el VAE incrementalmente. Proponer codigo
simple fue un desafio opcional del docente, no un requisito. Tambien aclara que
se pueden usar herramientas auxiliares, pero no una libreria que implemente la
red por nosotros. Conviene confirmar estas reglas si se publica una version
actualizada del enunciado.

## 2. Criterios que vamos a conservar durante todo el TP

1. **Una sola novedad por paso.** No cambiar arquitectura, optimizador, loss y
   tasa de aprendizaje simultaneamente.
2. **Prueba pequena antes de corrida larga.** Primero una muestra, luego las 32,
   luego varias semillas.
3. **Cada corrida debe responder una pregunta.** Si no sabemos que decision
   permitiria tomar, no se ejecuta.
4. **Separar probabilidad de pixel de pixel decidido.** La salida sigmoidea es
   continua; para contar pixeles incorrectos se aplica un umbral (inicialmente
   0.5) y se compara con el patron binario.
5. **Reproducibilidad.** Toda corrida guarda configuracion, semilla, historia de
   loss, metricas, tiempo y artefactos relevantes.
6. **Seleccionar con varias semillas.** Una semilla sirve para depurar; una
   conclusion necesita repeticiones.
7. **Documentar al decidir, no al final.** Cada etapa deja una nota breve con
   hipotesis, resultado, interpretacion y proximo paso.

## 3. La matematica minima, en el orden en que la necesitaremos

No hace falta estudiar toda la teoria de informacion de una vez.

1. **Autoencoder:** `x -> encoder -> z -> decoder -> x_hat`. El encoder aprende
   una representacion de 35 valores en solo 2; el decoder intenta recuperar
   `x`. Se necesita entender propagacion hacia adelante, funcion de perdida y
   regla de la cadena/backpropagation.
2. **DAE:** el mecanismo no cambia. Solo cambia el par de entrenamiento:
   `entrada = x_con_ruido`, `objetivo = x_limpio`. Si el objetivo tambien fuera
   ruidoso, la red aprenderia a copiar ruido.
3. **VAE, paso 1:** en vez de producir directamente `z`, el encoder produce dos
   vectores: `mu` y `log_var`.
4. **VAE, paso 2:** se toma `epsilon ~ N(0, I)` y se calcula
   `z = mu + exp(0.5 * log_var) * epsilon`. Esto separa el azar de los parametros
   entrenables y permite retropropagar.
5. **VAE, paso 3:** la perdida suma reconstruccion y
   `KL(q(z|x) || N(0,I))`. Reconstruccion pide conservar informacion; KL pide un
   espacio latente continuo y sampleable. El peso de KL controla la tension
   entre ambos objetivos.

Antes de programar el VAE se escribiran en papel las dimensiones de cada tensor
y los gradientes de `mu` y `log_var`. Luego se validaran numericamente en un caso
minusculo mediante diferencias finitas.

## 4. Etapas de implementacion

### Etapa 0 - Alinear reglas y preparar el terreno

**Objetivo:** que sepamos exactamente que esta permitido y como mediremos exito.

- Revisar el enunciado, el cierre de la transcripcion y las diapositivas
  relevantes.
- Confirmar si la version vigente mantiene: NumPy y Matplotlib permitidos, red
  neuronal propia, emojis excluidos.
- Definir estructura del proyecto, configuraciones y formato de resultados.
- Decidir si reutilizamos conceptos/codigo general del MLP del TP3. Reutilizar
  activaciones u optimizadores propios puede ahorrar trabajo, pero el TP5 debe
  conservar una interfaz clara para encoder, decoder y luego las dos ramas del
  VAE.

**PING-PONG 0 - alcance y reglas**

Preguntas a poder contestar antes de seguir:

- ¿Por que la salida esperada del autoencoder es igual a su entrada?
- ¿Que significa que el cuello de botella tenga dimension 2?
- ¿La catedra exige implementar tambien los optimizadores o solo la red?
- ¿Que criterio exacto usaremos para afirmar que se aprendieron los 32 patrones?

**Salida:** reglas anotadas y esqueleto reproducible, todavia sin entrenar.

### Etapa 1 - Entender y probar el dataset

**Objetivo:** eliminar cualquier ambiguedad de bits, filas y orientacion.

- Implementar un parser pequeno para `font.h` o trasladar sus constantes de
  forma verificable.
- Convertir cada entero de fila a 5 bits y aplanar cada sprite a 35 valores.
- Mostrar una grilla con los 32 patrones y sus caracteres.
- Agregar pruebas sobre forma `(32, 35)`, valores binarios y algunos patrones
  conocidos. Verificar visualmente izquierda/derecha para no entrenar letras
  espejadas.

**PING-PONG 1 - representacion**

- Convertir juntos una fila hexadecimal a cinco bits.
- Explicar por que aplanar 7x5 no destruye informacion para una MLP, aunque si
  pierde la nocion explicita de vecindad.
- Diferenciar cantidad de muestras (32) de cantidad de features (35).

**Salida:** cargador probado y figura de los datos originales.

### Etapa 2 - Autoencoder didactico minimo

**Objetivo:** verificar forward y backward sin que el dataset o la arquitectura
oculten errores.

- Implementar una red simetrica muy pequena con bottleneck 2.
- Hacer una pasada forward e imprimir dimensiones y rangos.
- Intentar sobreajustar primero **un solo sprite**. Si no puede copiar una
  muestra, no tiene sentido entrenar 32.
- Validar gradientes de una red diminuta con diferencias finitas.
- Graficar loss y reconstruccion durante el entrenamiento.

La primera loss candidata sera coherente con pixeles binarios (sigmoide de
salida y binary cross-entropy), pero antes de fijarla compararemos su derivada
con MSE, que es mas intuitiva. No se elige por costumbre: se deja escrito que
cambia en la señal de error.

**PING-PONG 2 - forward, loss y backprop**

- Seguir un ejemplo desde `x` hasta `x_hat` indicando dimensiones.
- Calcular a mano la contribucion de un pixel a la loss.
- Explicar por que el umbral para contar errores no se usa dentro del
  entrenamiento (no es derivable).

**Salida:** un sprite reconstruido correctamente y gradientes validados.

### Etapa 3 - Autoencoder sobre los 32 patrones

**Objetivo:** alcanzar el criterio obligatorio sin hacer un barrido ciego.

La arquitectura y el optimizador pueden interactuar: una red profunda puede ser
dificil para descenso basico y funcionar con Adam, por ejemplo. Por eso no se
hara primero un estudio exhaustivo de arquitectura con un unico optimizador ni
un producto cartesiano ciego de todos los hiperparametros.

Orden de experimentacion acordado:

1. **Calibracion minima de optimizadores.** Copiar y probar los optimizadores de
   la materia: descenso, Momentum, RMSProp y Adam. Sobre dos arquitecturas ancla
   (`35 -> 15 -> 2 -> 15 -> 35` y `35 -> 35 -> 2 -> 35 -> 35`), probar pocas
   tasas logaritmicamente separadas con una semilla. Esta etapa solo descarta
   tasas divergentes o evidentemente lentas; no elige un ganador.
2. **Cruce grueso arquitectura x optimizador.** Comparar inicialmente:
   - `A0: 35 -> 2 -> 35`, control sin capas ocultas intermedias;
   - `A1: 35 -> 10 -> 2 -> 10 -> 35`, red angosta;
   - `A2: 35 -> 20 -> 2 -> 20 -> 35`, capacidad intermedia;
   - `A3: 35 -> 35 -> 2 -> 35 -> 35`, red ancha;
   - `A4: 35 -> 20 -> 10 -> 2 -> 10 -> 20 -> 35`, red profunda.
   Cada arquitectura se cruza con los optimizadores que sobrevivan la
   calibracion y con tres semillas. Las activaciones, loss, modalidad full-batch
   y presupuesto maximo de actualizaciones se mantienen fijos.
3. **Descarte temprano con evidencia.** Un optimizador puede retirarse antes del
   cruce completo si, despues de probar tasas razonables en las dos arquitecturas
   ancla y mas de una semilla, diverge o queda claramente dominado en error y
   costo. Una mala corrida aislada no alcanza para descartarlo. Cada descarte se
   documenta con la comparacion que lo respalda.
4. **Refinamiento dirigido de arquitectura.** Si aparece una frontera entre un
   ancho que falla y uno que funciona, probar valores intermedios. Si el
   rendimiento sigue mejorando al aumentar el ancho, explorar tambien por encima
   del rango inicial. Si la profundidad ayuda, probar uno o dos perfiles internos
   adicionales. Cuando sea util, comparar redes anchas y profundas con cantidades
   de parametros parecidas.
5. **Refinamiento de combinaciones.** Si hay interaccion fuerte, conservar y
   refinar pares arquitectura-optimizador. Si el ranking de arquitecturas casi no
   cambia entre optimizadores, usar el mas estable para el refinamiento final.
6. **Epocas y schedule al final.** Estudiar cantidad de epocas o disminucion de
   tasa sobre pocas combinaciones finalistas, no sobre todo el espacio. Evaluar
   el modelo de la ultima epoca de cada presupuesto; no elegir retrospectivamente
   el mejor checkpoint de una unica semilla.
7. **Confirmacion.** Repetir los finalistas con cinco semillas y preferir la
   arquitectura mas simple que cumpla de forma estable, salvo que una alternativa
   algo mayor demuestre mejor estabilidad o costo.

Metricas obligatorias por corrida:

- maximo de pixeles incorrectos en un patron;
- cantidad de patrones con 0, 1 y mas de 1 errores;
- promedio de pixeles incorrectos;
- loss continua final y trayectoria de entrenamiento;
- tiempo y cantidad de actualizaciones.

No alcanza con que el error promedio sea menor que uno: el requisito dice que
el **peor patron** debe tener como maximo un pixel incorrecto.

**PING-PONG 3 - decidir experimentos**

- Mirar la corrida base y formular una hipotesis antes de tocar parametros.
- Elegir juntos que variable cambiar y que resultado apoyaria o refutaria la
  hipotesis.
- Separar “no tiene capacidad” de “tiene capacidad pero no optimiza bien”.

**Salida:** configuracion final justificada y tabla compacta de alternativas.

### Etapa 4 - Interpretar y generar con el autoencoder

**Objetivo:** no limitarse a que la loss baje.

- Graficar los 32 puntos `(z1, z2)` etiquetados por caracter.
- Observar vecinos, zonas vacias y posibles valores latentes extremos.
- Decodificar una grilla regular del plano para ver como cambia la salida.
- Elegir puntos entre dos letras cercanas y fuera de los codigos exactos.
- Definir “nueva letra” como una salida binarizada que no coincide con ninguno
  de los 32 patrones, mostrando tambien sus probabilidades sin umbral.

**PING-PONG 4 - generacion**

- ¿Por que interpolar dos codigos no garantiza una letra valida en un AE comun?
- ¿Que evidencia permite decir que la salida es nueva y no una copia?
- ¿Que diferencia hay entre una salida novedosa y una salida simplemente mala?

**Salida:** mapa latente, grilla generativa y al menos un caso discutido.

### Etapa 5 - Denoising autoencoder — COMPLETADA

**Objetivo:** medir recuperacion, no mostrar solo ejemplos favorables.

- Empezar con ruido salt-and-pepper/bit-flip, natural para datos binarios.
- En cada epoca generar corrupciones nuevas en vez de memorizar una unica copia
  ruidosa.
- Entrenar con `(x_con_ruido, x_limpio)`.
- Evaluar en corrupciones no usadas durante la actualizacion de pesos.
- Comparar varios niveles de probabilidad de flip, incluyendo 0 como control.
- Reportar error de entrada y error de salida; la mejora es la diferencia entre
  ambos. Incluir promedio, dispersion y ejemplos faciles/dificiles.
- Si el tiempo alcanza, probar ruido estructurado (bloques o filas) como
  extension, sin mezclarlo con la conclusion principal.

**PING-PONG 5 - protocolo de ruido**

- Distinguir ruido de entrenamiento y de evaluacion.
- Decidir que niveles son informativos despues de un piloto corto.
- Explicar por que a ruido extremo puede ser imposible identificar el original.

**Salida:** curva capacidad de recuperacion vs. ruido y ejemplos representativos.

**Cierre:** se entregaron los dos elementos de salida con cinco niveles de ruido,
diez semillas y casos elegidos mediante reglas reproducibles. La matriz de
confusión, el análisis latente y el ruido estructurado permanecen como
extensiones opcionales; no bloquean la Etapa 6.

### Etapa 6 - VAE de juguete, todavia sin dataset final — COMPLETADA

**Objetivo:** aprender el mecanismo variacional por partes, tal como indico el
docente.

Subpasos que deben funcionar por separado:

1. [x] Autoencoder determinista sobre un conjunto diminuto.
2. [x] Encoder con dos salidas, `mu` y `log_var`, aun usando `z = mu`.
3. [x] Agregar `epsilon` y reparametrizacion, con KL temporalmente en cero.
4. [x] Verificar los gradientes de ambas ramas.
5. [x] Agregar KL con un peso pequeno y registrar reconstruccion y KL por
   separado.
6. [x] Muestrear `z ~ N(0,I)` y pasar por el decoder.

**PING-PONG 6A - probabilidad sin formulas largas**

- Diferenciar “un punto z” de “una distribucion de posibles z”.
- Explicar que representan `mu`, varianza y `epsilon`.

**PING-PONG 6B - loss del VAE**

- Explicar la tension reconstruccion vs. KL con un ejemplo extremo para cada
  termino.
- Derivar juntos la reparametrizacion y revisar los gradientes numericos.

**Salida:** VAE minimo probado, con curvas separadas de reconstruccion y KL.

**Cierre:** los seis subpasos funcionan por separado. El backward completo fue
validado con diferencias finitas, las curvas separan reconstrucción y KL, y
1.000 muestras del prior fueron evaluadas con una regla objetiva. La tasa de
validez de `58,7 %` no se interpreta como óptima: expone el efecto del peso KL
pequeño y deja la comparación de pesos para el VAE final.

### Etapa 7 - Elegir y preparar el dataset nuevo — COMPLETADA

**Objetivo:** elegir algo evaluable y terminable, no solo llamativo.

Criterios de eleccion:

- representacion fija y pequena;
- suficientes muestras y variacion para aprender una distribucion;
- criterio objetivo o al menos claro de validez;
- generacion visible/evaluable;
- preparacion que no consuma mas tiempo que el modelo;
- sin emojis.

Opciones para discutir:

- **sprites geometricos parametrizados:** riesgo bajo, generacion visual y
  pertenencia facil de justificar;
- **glifos o simbolos sinteticos distintos de `font.h`:** riesgo medio y buena
  continuidad visual, cuidando que sea realmente un dataset nuevo;
- **codigo de un lenguaje minimo:** interesante pero de riesgo alto; exige una
  codificacion fija y validar sintaxis/semantica. Solo elegirlo si ya esta
  estable el VAE y queda margen.

**PING-PONG 7 - decision del dataset**

Armar una tabla corta con valor pedagogico, costo, riesgo y forma de evaluar.
No implementar el dataset definitivo hasta tomar esta decision conscientemente.

**Salida:** dataset versionado, visualizacion/ejemplos, reglas de pertenencia y
division reproducible si corresponde.

**Cierre:** están acordados la plantilla, el universo de 90.000 funciones, los
criterios de novedad y el split balanceado 1.800/450/450. La representación
aprobada es one-hot por carácter (`26 x 25 = 650`), con 26 softmax agrupados y
entropía cruzada categórica por posición. El round-trip es exacto y los
gradientes de esta salida fueron verificados por diferencias finitas tanto en
el AE como en el VAE con KL. Esto cierra la preparación del dataset, pero no
selecciona ninguna arquitectura ni hiperparámetro de la Etapa 8.

### Etapa 8 - VAE final y experimentos acotados — EN CURSO

**Objetivo:** obtener reconstrucciones y muestras nuevas defendibles.

- Hacer primero que el AE determinista reconstruya el dataset; esto separa un
  problema de capacidad de un problema variacional.
- Incorporar el VAE ya validado.
- Registrar siempre loss total, reconstruccion y KL.
- Revisar reconstrucciones, distribucion de `mu`, escalas de varianza y muestras
  del prior. Una loss sola no detecta colapso latente.
- Estudiar pocos factores con orden: capacidad base, tasa de aprendizaje y peso
  de KL. No hacer producto cartesiano completo.
- Confirmar finalistas con varias semillas y generar una muestra fija para poder
  comparar modelos.

**PING-PONG 8 - diagnostico**

- Buena reconstruccion pero muestras incoherentes: posible espacio latente no
  regularizado.
- KL casi cero y decoder que ignora `z`: posible colapso.
- Ambas losses malas: revisar capacidad, escalas, gradientes u optimizacion antes
  de ampliar el barrido.

**Salida:** comparacion pequena, modelo elegido, interpolaciones y muestras del
prior juzgadas con el criterio definido en la etapa 7.

**Control AE completado:** se compararon secuencialmente tres anchuras con
latente 16, Adam `0.005`, una semilla y un máximo común de 3.000 épocas. Solo
`650-256-16-256-650` sostuvo los umbrales prefijados. Obtuvo 446/450 exactas en
validación y, tras la selección, 448/450 en test, con 100 % de compilación y
pertenencia a la gramática. El protocolo, la comparación y los fallos están en
`docs/vae/code-ae-capacity.md`. El siguiente paso es acordar un embudo VAE más
amplio; no se adopta automáticamente la arquitectura del AE como modelo final.

#### Hipótesis narrativa para el ejercicio 2

El AE y el VAE no se presentarán como dos competidores equivalentes. El AE es
el control que demuestra capacidad de reconstrucción. La hipótesis a evaluar es
que ese AE no impone una distribución conocida sobre sus códigos, mientras que
el VAE acepta un posible costo de reconstrucción para aproximar el latente a
`N(0,I)` y volver significativo el muestreo desde ese prior.

No se afirmará de antemano que el VAE “soluciona” el espacio latente. Se deberá
mostrar empíricamente si mejora la generación desde `N(0,I)`, cuánto cuesta en
reconstrucción y qué limitaciones conserva.

Esto no significa que el AE sea incapaz de reconstruir. Cuando recibe una
fuente, el recorrido `x -> encoder -> z -> decoder -> x_hat` usa un código
producido por su propio encoder y ya alcanzó reconstrucción casi perfecta. El
problema aparece al generar sin una entrada: el AE nunca fue obligado a ubicar
sus códigos cerca de una distribución conocida, por lo que un punto tomado de
`N(0,I)` puede caer en una región que su decoder nunca vio.

El VAE no busca “más representación”, sino una representación más organizada y
muestreable. Sus ramas `mu` y `log_var`, la reparametrización y la KL entrenan
al decoder con vecindades latentes y aproximan los códigos al prior normal. La
comparación deberá mostrar por separado:

- **reconstrucción:** qué ocurre cuando el código proviene del encoder;
- **generación:** qué ocurre cuando el código se toma directamente de
  `N(0,I)`, sin una fuente de entrada.

La hipótesis esperada es que el AE reconstruya igual o mejor y que el VAE genere
desde el prior con mayor validez, diversidad y novedad. Si las métricas no
muestran esa diferencia, no se concluirá que el VAE resolvió el problema.

La presentación debería separar dos preguntas:

1. **¿Por qué hace falta el VAE?** Comparación controlada AE-VAE sobre el mismo
   dataset, backbone, dimensión latente, semillas y puntos del prior.
2. **¿Por qué se eligió ese VAE?** Embudo de entrenamiento con curvas separadas,
   métricas de validación, diagnóstico del latente y confirmación multisemilla.

#### Evidencia común para comparar AE y VAE

- Reconstrucción determinista: entropía cruzada, exactitud por carácter y por
  fuente, compilación y pertenencia a la gramática por split.
- En el VAE, reconstrucción tanto con `z = mu` como con muestras repetidas del
  posterior; no esconder la variabilidad usando solo la media.
- Distribución latente: media y desvío por dimensión, correlaciones, KL por
  dimensión, cantidad de unidades activas y señales de colapso.
- Generación sobre los mismos puntos fijos `z ~ N(0,I)`: compilación, gramática,
  novedad sintáctica y semántica, fuentes únicas, duplicación, cobertura de los
  nueve pares de operadores y concentración en pocos resultados.
- Interpolaciones prefijadas: porcentaje de pasos válidos y ejemplos elegidos
  por regla, no mediante inspección posterior.
- Estabilidad: distribución de las métricas entre semillas, no solo la mejor
  corrida.

#### Recomendación de embudo — PENDIENTE DE APROBACIÓN DE LA USUARIA

Ningún bloque de esta propuesta autoriza corridas por sí solo. La usuaria da el
OK final sobre candidatos, presupuestos, umbrales y semillas antes de cada
tanda.

**Bloque A — completar el baseline AE**

- Congelar `650-256-16-256-650`; no volver a buscar anchuras.
- Confirmarlo con semillas 0-9 y el mismo presupuesto.
- Para cada semilla, evaluar una colección común de 5.000 puntos de `N(0,I)`.
- Propósito: obtener estabilidad y el baseline generativo que permita mostrar
  el problema del latente no regularizado.

**Bloque B — calibración técnica del VAE**

- Usar primero el mismo backbone y latente 16 para conservar una comparación
  causal con el AE.
- Recomendada: calibración breve de Adam con tasas `0.001`, `0.003` y `0.005`,
  una semilla y un peso KL intermedio provisional.
- Descartar solo divergencias o configuraciones claramente inviables; esta
  tanda no elige el modelo final.

**Bloque C — tamiz del peso KL**

- Incluir `KL = 0` únicamente como diagnóstico.
- Recomendado: escala logarítmica
  `{0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1}` con una semilla.
- Comparar reconstrucción, prior, posterior y unidades activas. No seleccionar
  por loss total, que no es comparable entre pesos distintos.
- Conservar la frontera de Pareto: configuraciones no dominadas entre
  reconstrucción, validez, novedad/diversidad y alineación con el prior.

**Bloque D — refinamientos con hipótesis**

- Probar valores KL intermedios solo si el tamiz muestra un cambio claro entre
  dos puntos.
- Recomendado: comparar KL constante contra warm-up lineal para el finalista si
  aparece colapso temprano o una pérdida brusca de reconstrucción.
- Dimensiones latentes `{8, 16, 32}` son una extensión posible, no automática.
  Se abre únicamente si latente 16 muestra saturación, colapso o diversidad
  insuficiente.
- No cambiar simultáneamente anchura, latente, tasa y KL.

**Bloque E — confirmación y comparación final**

- Confirmar hasta tres finalistas con semillas 0, 1 y 2.
- Congelar una configuración antes de test.
- Ejecutar el ganador con semillas 0-9 y compararlo contra las mismas diez
  semillas del AE usando los mismos puntos del prior.
- Publicar media, desvío y box-plots, además de una muestra fija y casos de
  fallo elegidos por reglas reproducibles.

#### Decisiones que conserva la usuaria

Antes de cada bloque se deben aprobar explícitamente las decisiones que le
correspondan:

1. si la comparación final tendrá diez semillas para ambos modelos;
2. si se aceptan las tres tasas recomendadas;
3. el rango de pesos KL y si se incluye warm-up;
4. los mínimos aceptables de reconstrucción y validez para filtrar candidatos;
5. si la dimensión latente queda fija en 16 o bajo qué diagnóstico se reabre;
6. cantidad de muestras del prior y presupuesto máximo de épocas.

**Piloto disponible, no cierre:** se ejecutó un AE control y se compararon tres
pesos KL sin que la usuaria hubiera aprobado previamente el embudo. Los datos
pueden informar el futuro diseño, pero no seleccionan arquitectura, peso KL ni
modelo final. Ninguna corrida del VAE final comenzará hasta aprobar juntos el
bloque correspondiente del protocolo experimental.

### Etapa 9 - Informe, presentacion y defensa

**Objetivo:** que cada figura responda una pregunta y que podamos explicar el
codigo sin depender de memoria reciente.

- Escribir el informe en paralelo, consolidarlo al final.
- Presentar el recorrido: problema, representacion, mecanismo, experimentos,
  decision y limites.
- Incluir arquitectura con dimensiones, loss, protocolo de semillas y metrica
  de pixeles.
- Mostrar pocos experimentos, elegidos por lo que ensenaron; dejar los barridos
  completos como respaldo.
- Preparar una explicacion oral simple de AE, DAE, `mu/log_var`,
  reparametrizacion y KL.
- Ensayar preguntas: por que esa arquitectura, por que esa loss, por que esos
  niveles de ruido, por que ese dataset, que fallo y como se diagnostico.

**PING-PONG 9 - ensayo de defensa**

Hacer una ronda donde yo pregunto como docente y otra donde la explicacion se
hace sin mirar formulas. Las partes que no se puedan explicar vuelven a una
prueba o figura concreta; no se memorizan de forma aislada.

## 5. Estrategia de hiperparametros y computo

No sabemos aun cuanto tarda una corrida; por eso el presupuesto se fija despues
de medir el primer piloto.

### Embudo de experimentos

1. **Depuracion:** una configuracion, una semilla y pocas epocas. No se usa para
   concluir.
2. **Calibracion:** pocas tasas sobre dos arquitecturas ancla para evitar una
   comparacion injusta entre optimizadores.
3. **Tamiz cruzado:** arquitecturas representativas por optimizadores viables y
   tres semillas, con un presupuesto comun y sin figuras por corrida.
4. **Descarte:** retirar opciones consistentemente dominadas o divergentes y
   guardar la evidencia; no completar combinaciones que ya no pueden cambiar la
   decision.
5. **Refinamiento:** probar valores intermedios o ampliar el rango solo donde la
   tendencia observada lo justifique.
6. **Confirmacion:** pocos finalistas con cinco semillas.
7. **Final:** configuracion congelada y artefactos reproducibles.

### Limites practicos

- No lanzar un grid de arquitectura x activacion x optimizador x tasa x batch.
  El unico cruce permitido es el tamiz pequeño arquitectura x optimizador,
  despues de calibrar y podar opciones inviables.
- Para los 32 sprites, full-batch es un baseline natural; comparar otra modalidad
  solo si hay una hipotesis.
- Usar una cantidad maxima de epocas como presupuesto comun. Se puede detener
  cuando se alcanza el requisito fijado de antemano, pero no elegir a posteriori
  el mejor checkpoint de una trayectoria inestable.
- Medir tiempo desde la primera corrida y estimar el costo total antes de cada
  tanda.
- En las tandas de seleccion guardar metricas e historias, pero generar figuras
  solo para controles y finalistas.
- Toda corrida larga debe poder dejarse ejecutando y reanudarse/analizarse sin
  repetir las anteriores.
- Si una tanda supera el presupuesto acordado, reducir candidatos o semillas,
  no la calidad de las metricas.

## 6. Calendario hasta el 17 de noviembre

El calendario deja el dia 16 como colchon. Si un dia no se puede trabajar, se
conserva el orden y se recorta primero una extension, nunca las validaciones.

| Fecha | Foco | Resultado minimo al cerrar |
|---|---|---|
| 4-6 nov | Plan inicial | Alcance y riesgos identificados, sin implementacion |
| 7 nov | Etapas 0-2 y entrada a 3 | Datos verificados, AE minimo probado, baseline y diseño experimental |
| 8 nov | Etapa 3, tamiz | Calibracion, descarte temprano y cruce grueso medido |
| 9 nov | Etapas 3-4 | Refinamiento, finalistas, mapa latente y generacion AE |
| 10 nov | Etapa 5 | DAE con curva inicial de ruido |
| 11 nov | Cierre parte 1 | Resultados repetibles, texto preliminar y margen para demoras |
| 12 nov | Etapa 6, pasos 1-4 | `mu`, `log_var`, reparametrizacion y gradientes entendidos |
| 13 nov | Etapas 6-7 | VAE de juguete y dataset final preparado |
| 14 nov | Etapa 8 | VAE final y experimentos acotados |
| 15 nov | Etapas 8-9 | Confirmacion, figuras e informe/presentacion |
| 16 nov | Colchon y ensayo | Corrida reproducible, defensa y revision final |
| 17 nov | Entrega | Solo empaquetado y verificacion, sin experimentos nuevos |

### Version minima si se complica la semana

Se preserva, en este orden:

1. AE que cumple el maximo de un pixel y su mapa latente.
2. DAE con bit-flip y tres niveles informativos de ruido.
3. VAE sobre un dataset visual simple con una comparacion de peso KL.
4. Tres semillas para configuraciones finales.
5. Informe y presentacion reproducibles.

Se recortan primero: ruido estructurado, dataset de codigo, arquitecturas extra,
grids densos y figuras redundantes.

## 7. Registro breve para retomar despues de una pausa

Al terminar cada sesion, completar estas cinco lineas en una bitacora:

```text
Estado actual:
Ultima evidencia confiable:
Decision tomada y por que:
Pregunta abierta:
Proximo paso exacto (un solo paso):
```

Esto evita volver a leer todo el proyecto y, sobre todo, evita confundir una
idea pendiente con una decision ya respaldada por resultados.

## 8. Próximo paso al retomar

**Estado actual:** Etapas 1A, 1B, 6 y 7 cerradas. Etapa 8 en curso y pausada
antes de ejecutar el Bloque A del embudo propuesto.

**Última evidencia confiable:** el control AE seleccionó
`650-256-16-256-650` con latente 16 mediante umbrales prefijados. Reconstruyó
446/450 fuentes de validación y 448/450 de test; todas las decodificaciones
compilaron y pertenecieron a la gramática.

**Decisiones tomadas:** dataset Python de plantilla fija, split
1.800/450/450, one-hot `26 x 25`, softmax y entropía cruzada por posición. El AE
es control de capacidad; no es todavía evidencia del comportamiento generativo
ni determina automáticamente la arquitectura final del VAE.

**Pregunta abierta:** aprobar o modificar el Bloque A recomendado: confirmar el
AE congelado con semillas 0-9, máximo de 3.000 épocas y evaluar los mismos 5.000
puntos de `N(0,I)` por semilla. Ese bloque establecerá el baseline generativo y
la estabilidad con los que luego se fijarán los umbrales del VAE.

**Próximo paso exacto:** no ejecutar nuevas corridas hasta recibir el OK de la
usuaria sobre el Bloque A. Después de completarlo, presentar sus resultados y
pedir aprobación separada para la calibración del VAE del Bloque B.
