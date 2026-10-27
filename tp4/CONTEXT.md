# TP4 - Contexto de trabajo

Este documento resume el estado y el alcance del TP4 para orientar futuras
consultas e implementaciones. No reemplaza al enunciado: ante una diferencia,
manda `docs/SIATP42026.pdf` y, para las técnicas, el material de cátedra de
`docs/`.

## Reglas generales del repositorio

- El trabajo, la documentación y los comentarios deben estar en español.
- No incorporar técnicas de IA que no se hayan visto en clase sin avisar y
  acordarlo antes. EDA y preprocesamiento tabular estándar sí son admisibles.
- Toda decisión de variables, preprocesamiento, arquitectura e
  hiperparámetros debe justificarse por escrito. Importa tanto la
  interpretación como que el algoritmo funcione.
- No agregar trailers de coautoría de Claude a los commits.

## Estado actual

- Existe un primer EDA reproducible en `scripts/analyze_creators.py`; genera el
  informe `docs/eda-creadores/README.md`, tablas CSV y visualizaciones SVG.
  PCA ya está implementado sobre la selección base. Kohonen está implementado
  de forma configurable, pasó una validación sintética reproducible y tiene un
  primer barrido de `k = 2, ..., 8` con cinco semillas. También se comparó la
  interacción completa entre esos tamaños y siete tasas constantes, y luego
  cada combinación contra una disminución lineal al 10 % de la tasa inicial.
  El barrido de radio inicial ya ejecutó otras 300 corridas sobre `k = 4–6`,
  cuatro perfiles de tasa por tamaño, cinco radios y cinco semillas.
  La inspección posterior de nueve mapas de hits y matrices U redujo el próximo
  barrido de épocas a cinco configuraciones: dos `k = 4`, dos `k = 5` y una
  `k = 6` conservada como control de frontera.
  El barrido de 25, 50, 100, 150, 250 y 400 épocas ya se ejecutó y confirmó 150
  como compromiso entre error, ocupación, estabilidad y tiempo.
  La comparación posterior de 50 corridas entre inicialización `sample` y
  `uniform` confirmó `sample`: `uniform` obtuvo algunas reducciones aisladas de
  error a costa de más unidades muertas y mayor variabilidad.
  El estudio de mapas por variable eligió finalmente `k = 5`, tasa constante
  `0.2`, radio inicial igual al diámetro con descenso hasta 1, 150 épocas e
  inicialización `sample`. La semilla 2 sólo representa el caso central en las
  figuras; las decisiones usan cinco semillas. La interpretación del caso de
  uso también está cerrada: se caracterizaron cuatro regiones temáticas, se
  compararon con PC1-PC2 y se cuantificaron las fronteras de la Matriz U.
  La configuración final se repitió después con diez semillas para validar la
  interpretación: 36 de 50 creadores conservaron su región temática en al menos
  ocho corridas, 11 fueron fronterizos y sólo 3 inestables. La mejor celda por
  campaña fue sensible a la semilla, aunque siempre mantuvo el tema esperado;
  por eso la síntesis comercial usará regiones completas y no una única celda.
  Oja también está implementado y validado. El barrido de tasa comparó ocho
  valores, dos schedules y diez semillas; el barrido posterior comparó ocho
  cantidades de épocas para tres perfiles finalistas. Se eligieron 100 épocas
  y descenso lineal de `0.001` a `0.0001`. En diez semillas, la similitud media
  con el autovector de PC1 es 0.999999847 y la correlación media entre el índice
  y los scores de PC1 es 0.999999962. El análisis final compara pesos,
  proyecciones y creadores e interpreta por qué el índice no es un ranking
  universal. La síntesis de las cuatro campañas también está implementada:
  asigna cada creador a su región temática modal en diez semillas, usa regiones
  completas y conserva estabilidad, datos originales, PC1-PC2 y Oja sin crear
  un ranking. Los conjuntos contienen 17 creadores para fútbol, 8 para belleza,
  13 para perros y 12 para arte; 47 de los 50 son estables o fronterizos.
  Esos candidatos primarios ya se compararon con un top global por followers de
  igual `K` para cada campaña. La salida incluye detalle individual, agregados,
  costo y gráficos. La selección regional aumenta la afinidad temática y reduce
  el costo, pero sacrifica volumen de alcance; no se infieren ventas ni
  adopciones porque no existe esa variable objetivo. Dentro de cada conjunto se
  proponen además tres candidatos reproducibles: alcance, equilibrio y
  especialización/eficiencia. Todos superan la afinidad temática mediana y no se
  usa un score compuesto. Los candidatos de fútbol son `INF021`, `INF041` e
  `INF027`; los de belleza `INF006`, `INF022` e `INF038`; los de perros
  `INF039`, `INF023` e `INF043`; y los de arte `INF024`, `INF012` e `INF016`.
  Esa síntesis regional se conserva como antecedente. La decisión revisada ya
  no convierte la región modal de Kohonen en un filtro: exige al menos 50 % de
  afinidad temática individual, prioriza alcance argentino nuevo y afín, usa
  visitas al perfil como señal secundaria de intención y compara contra dos
  baselines de igual tamaño, followers globales y followers entre afines. El
  informe revisado está en `docs/campanas-creadores-revisado/`.
  El mantenimiento también está documentado: una actualización mensual o la
  incorporación de creadores vuelve a entrenar el pipeline completo; campañas
  o variables nuevas requieren revisar explícitamente el diseño. Los informes
  canónicos se regeneran sin crear carpetas por mes o por semilla.
  El núcleo discreto de Hopfield ya está implementado y probado: construye la
  matriz de pesos de Hebb, actualiza el estado en forma sincrónica, conserva el
  valor previo ante activación cero, registra la energía y distingue estados
  fijos, ciclos y corte por máximo de iteraciones. El caso controlado de letras
  5×5 también está implementado: rankea exhaustivamente los grupos de cuatro
  letras y compara los extremos con once niveles de ruido y diez semillas.
  El análisis de los códigos QR también está completo: evalúa la recuperación
  conjunta de los cinco patrones, la capacidad exacta sobre todos sus
  subconjuntos y la robustez de los conjuntos individuales y las duplas
  estables frente a tres tipos de deterioro entre 0 % y 100 %.
- El EDA define 15 variables base para la primera ejecución de PCA. Las
  justificaciones y variables auxiliares están en `docs/decisiones.md`.
  La comparación controlada entre engagement total y sus cuatro componentes
  está en `docs/pca-engagement/`. Se confirmó el engagement total como entrada
  principal y las cuatro interacciones como auxiliares, por lo que las 15
  variables base quedan fijadas para Kohonen y Oja.
- El TP4 tiene configuración Python, pruebas y resultados de la primera
  ejecución de PCA en `docs/pca-creadores/`.
- El material disponible alcanza para resolver el ejercicio 1 completo:
  introducción al aprendizaje no supervisado, Kohonen, autovalores y
  autovectores, PCA, Oja y Sanger.
- La clase de Hopfield, su transcripción y el ejemplo de ortogonalidad ya están
  disponibles en `docs/` y habilitan el trabajo sobre el ejercicio 2.
- Datos disponibles:
  - `datos/creadores.csv`, para el ejercicio 1;
  - cinco códigos QR en `datos/hopfield/`, para el ejercicio 2.

## Mapa del TP

El TP estudia métodos de aprendizaje no supervisado y tiene dos ejercicios.

### Ejercicio 1 - Campañas de marketing

La agencia elige actualmente creadores principalmente por cantidad de
seguidores. Esa estrategia da visibilidad, pero no necesariamente llega a la
audiencia ni al tipo de contenido adecuados. Hay cuatro clases de clientes:
refugios de perros, ropa de entrenamiento de fútbol, productos de belleza y
materiales de arte. Para los refugios interesa generar consultas y adopciones;
para los otros clientes, ventas.

El ejercicio no pide convertirse en experto en marketing. Pide usar e
interpretar métodos no supervisados para fundamentar mejores decisiones. Se
pueden declarar supuestos razonables sobre las variables cuando haga falta.

El recorrido obligatorio es:

1. Hacer EDA, seleccionar variables y realizar PCA. Justificar si se reduce la
   dimensionalidad e interpretar las componentes.
2. Implementar una red de Kohonen para agrupar creadores, variar y justificar
   sus hiperparámetros, analizar la grilla, el mapa de ocupación y la matriz U,
   e interpretar regiones de la grilla en el caso de uso.
3. Implementar Oja sobre los mismos datos para obtener un índice, interpretarlo
   y compararlo con la proyección sobre PC1 de PCA.
4. Usar la evidencia durante todo el análisis para decidir campañas. Esta
   interpretación no debe quedar relegada a una sección aislada al final.

Las preguntas del punto 4 son disparadores, no un cuestionario que deba
responderse literalmente una por una. Entre otras cosas, hay que poder discutir:

- si una campaña debe priorizar alcance masivo, descubrimiento o profundidad
  educativa;
- qué mejora y qué pierde la selección propuesta frente a ordenar sólo por
  `followers`;
- qué pasa ante una nueva campaña, nuevos creadores, nuevas variables o la
  actualización mensual de los datos.

### Ejercicio 2 - Memoria asociativa

Debe implementarse una red de Hopfield para recuperar códigos QR deteriorados:

- representar cada QR como una matriz binaria de 37 x 37;
- almacenar los cinco patrones y construir consultas incompletas o ruidosas;
- probar distintos niveles de ruido;
- analizar iteraciones y energía en cada paso;
- explicar las limitaciones del caso de uso.

Antes de los QR, el enunciado recomienda validar la red con cuatro letras
bipolares (`1` y `-1`) de 5 x 5, estudiar ortogonalidad y capacidad, y provocar
o identificar un estado espurio con una consulta muy ruidosa.

La implementación debe mantenerse dentro de las reglas de actualización,
energía, capacidad y dinámica explicadas en la clase de Hopfield disponible en
`docs/`.

## Dataset de creadores

`datos/creadores.csv` contiene 50 filas, una por creador, y 33 columnas. No hay
campos vacíos en la versión inspeccionada. `creator_id` y `name` son
identificadores; las otras 31 columnas son numéricas.

### Familias conceptuales

- **Escala y exposición:** `followers`, `average_reach`,
  `average_impressions`, `average_video_views`, `monthly_posts`.
- **Audiencia y descubrimiento:** `argentina_audience_pct`,
  `estimated_argentina_reach`, `non_follower_reach_pct`.
- **Interacción por alcance:** `likes_per_1000`, `comments_per_1000`,
  `saves_per_1000`, `shares_per_1000`, `total_engagement_per_1000`,
  `profile_visits_per_1000`.
- **Costo y mezcla comercial:** `price_usd`, `price_ars`,
  `sponsored_posts_pct`, `organic_posts_pct`.
- **Tema:** `topic_football_pct`, `topic_makeup_pct`, `topic_dogs_pct`,
  `topic_art_pct`.
- **Tipo y formato de contenido:** `content_tutorial_pct`,
  `content_review_pct`, `content_entertainment_pct`,
  `content_showcase_pct`, `average_caption_words`,
  `average_watch_time_seconds`, `video_posts_pct`,
  `short_form_posts_pct`, `trend_posts_pct`.

### Redundancias verificadas en los datos

Estas son observaciones iniciales para el EDA, no una selección de variables ya
cerrada:

- `price_ars = 1500 * price_usd` exactamente salvo precisión numérica;
- `total_engagement_per_1000` es la suma exacta de likes, comentarios,
  guardados y compartidos por 1000;
- `estimated_argentina_reach` es `average_reach *
  argentina_audience_pct / 100`, redondeado a una cuenta;
- `sponsored_posts_pct + organic_posts_pct = 100`;
- los cuatro porcentajes de tema suman 100;
- los cuatro porcentajes de tipo de contenido suman 100.

Por lo tanto, conservar a la vez todas las columnas de cada identidad puede
sobrerrepresentar el mismo fenómeno e introduce dependencias lineales exactas.
Hay que decidir y justificar qué representación conservar. No deben usarse
`creator_id` ni `name` como entradas numéricas de PCA, Kohonen u Oja; sí deben
preservarse para etiquetar e interpretar resultados.

La inspección preliminar también muestra escalas muy diferentes y correlaciones
altas entre varias medidas de exposición (`reach`, `impressions`, vistas),
costo y formatos de video. Esto confirma la necesidad de EDA, análisis de
correlación y estandarización, pero no autoriza a eliminar columnas sin mostrar
la evidencia y considerar el significado comercial.

La explicación oral del enunciado orientó a reducir las variables originales a
aproximadamente 10-15 antes de modelar, manteniendo representados los aspectos
importantes. Es una guía práctica de clase, no un número obligatorio escrito en
el PDF.

## Teoría disponible y consecuencias para el ejercicio 1

### Estandarización

Para una variable, la estandarización o z-score es

```text
z = (x - media) / desvío_estándar
```

Deja media 0 y desvío 1; no restringe los valores a `[0, 1]`. Es necesaria
porque el dataset mezcla cuentas, porcentajes, palabras, segundos y precios.
Sin ella, las magnitudes mayores dominarían las distancias de Kohonen, la
varianza de PCA y el aprendizaje de Oja.

Hay que usar la misma matriz estandarizada -y documentar el mismo conjunto y
orden de variables- al comparar PCA, Kohonen y Oja. Si luego se evalúan nuevos
creadores, se deben reutilizar las medias y desvíos ajustados, no recalcularlos
por fila.

### PCA

PCA transforma variables posiblemente correlacionadas en combinaciones
lineales no correlacionadas y ordenadas por varianza. Con variables
estandarizadas, trabajar con la covarianza de `Z` equivale al análisis sobre la
matriz de correlaciones de las variables originales.

Procedimiento visto en clase:

1. construir la matriz con observaciones en filas y variables en columnas;
2. estandarizar;
3. calcular la matriz de correlaciones;
4. obtener autovalores y autovectores;
5. ordenar los autovalores de mayor a menor;
6. formar las componentes con los autovectores asociados;
7. proyectar cada creador y decidir cuántas componentes conservar según la
   varianza explicada y la interpretación.

Los autovalores expresan la varianza de cada componente. La proporción explicada
por la componente `i` es `lambda_i / sum(lambda)`. Una aclaración docente
posterior indicó superar el 80 % y acercarse al 90 %. Se adoptan 8 componentes,
que explican 90,00 %; 6 alcanzaban 81,48 % y 7, 86,24 %. La reducción se
justifica además con la curva, las cargas y lo que se pierde o conserva.

Las cargas permiten nombrar o describir cada componente. En un biplot PC1-PC2:

- la dirección de una flecha muestra cómo se relaciona una variable con las
  componentes;
- su magnitud en el plano refleja cuánto está representada/importa allí;
- direcciones parecidas sugieren correlación positiva, opuestas negativa y
  aproximadamente perpendiculares poca relación lineal;
- la ubicación de cada creador debe interpretarse respecto de esas direcciones.

El signo de un autovector es arbitrario: `v` y `-v` describen la misma
dirección. Por eso una ejecución puede invertir cargas y scores sin cambiar la
interpretación. Hay que fijar o alinear el signo sólo para presentar y comparar
resultados, nunca inferir que un score alto es intrínsecamente "mejor".

### Red de Kohonen / SOM

La SOM es una capa de salida organizada como una grilla `k x k`. Cada neurona
tiene un vector de pesos con la misma dimensión que una fila de entrada. Para
cada muestra `x`:

1. se busca la BMU o neurona ganadora cuyo peso es el más cercano a `x`;
2. se actualizan la ganadora y su vecindario;
3. para cada neurona actualizada se aplica
   `w <- w + eta * (x - w)`.

El material prioriza distancia euclídea sobre entradas estandarizadas. Los
pesos pueden inicializarse al azar, pero inicializarlos con muestras del dataset
reduce el riesgo de unidades muertas. La tasa debe ser menor que 1 y puede
decrecer; el radio puede mantenerse o decrecer desde una cobertura amplia hasta
un vecindario local.

Variables experimentales explícitas del material:

- tamaño `k` de la grilla;
- geometría rectangular o hexagonal (la rectangular es la opción más simple
  presentada);
- inicialización de pesos;
- tasa de aprendizaje inicial y su evolución;
- radio inicial y su evolución;
- cantidad de iteraciones/épocas;
- semilla y orden de presentación, necesarios para reproducibilidad aunque no
  sean el foco conceptual.

Un `k` demasiado grande para sólo 50 muestras deja muchas neuronas vacías y no
aprovecha el vecindario; uno demasiado chico mezcla perfiles que convendría
distinguir. La elección final debe apoyarse como mínimo en:

- mapa de hits u ocupación por neurona;
- cantidad y distribución de unidades muertas;
- matriz U;
- mapas por variable para caracterizar regiones;
- estabilidad e interpretación en más de una semilla/configuración;
- coherencia, no identidad exacta, con los perfiles visibles en PC1-PC2.

La matriz U asigna a cada neurona el promedio de la distancia entre su vector de
pesos y los de sus vecinas. Distancias bajas describen continuidad local;
distancias altas pueden marcar fronteras entre regiones o un mapa mal
organizado, según el patrón completo y la ocupación. No debe concluirse que
"todo valor alto es malo" sin mirar el contexto.

La fórmula de las diapositivas define el vecindario con distancia estrictamente
menor que `R`, pero la explicación dice que en una grilla rectangular `R=1`
incluye los cuatro vecinos a distancia 1. La implementación resolvió la
ambigüedad incluyendo el borde (`distancia <= R`): con radio 1 se actualizan la
ganadora y sus cuatro vecinas ortogonales. La convención está documentada en
`docs/kohonen.md` y cubierta por una prueba pequeña.

El enunciado exige evaluar la performance, pero el material disponible no fija
una única métrica numérica de selección para SOM. No introducir silenciosamente
métricas avanzadas externas a clase. Se acordó usar el error de cuantización
como apoyo explícito, porque deriva directamente de la distancia a la BMU, pero
nunca como criterio único. Primero sostener la comparación con las evidencias
enseñadas arriba y consultar/acordar cualquier métrica adicional.

### Regla de Oja

Oja es un perceptrón lineal no supervisado de una salida. Para una entrada
centrada `x`, calcula

```text
y = x · w
w <- w + eta * y * (x - y * w)
```

Los pesos pueden inicializarse con una distribución uniforme entre 0 y 1. La
clase propone comenzar explorando tasas menores que 0.5, una tasa decreciente o
valores pequeños como `1e-3`; son puntos de partida, no una configuración final
obligatoria.

Al converger, `w` aproxima el autovector asociado al mayor autovalor de la
matriz de correlaciones. Oja obtiene las cargas/dirección de PC1, no los scores
de PC1 directamente. El índice de cada creador es luego `y = Z @ w`.

La comparación con PCA debe contemplar la ambigüedad de signo. Se debe comparar
la dirección aprendida por Oja con el primer autovector de PCA y, después de
alinear el signo para visualizar, comparar los índices de Oja con los scores de
PC1. También conviene mostrar convergencia a través de épocas y sensibilidad a
`eta` y semilla.

Sanger es la extensión que aprende varias componentes, pero la clase aclaró que
no se implementa para este TP: el requisito es Oja y su comparación con PC1.

## Interpretación orientada a campañas

Ni PC1 ni el índice de Oja son un ranking universal de calidad. Representan la
dirección de mayor variabilidad del conjunto seleccionado, cuya semántica surge
de las cargas. Un valor extremo puede reflejar tamaño, costo, engagement,
especialización temática u otra combinación; sólo las cargas y los perfiles
permiten saberlo.

La decisión debería separar al menos:

- **compatibilidad temática:** fútbol, maquillaje, perros o arte;
- **tipo de comunicación:** tutorial, review, entretenimiento o showcase;
- **audiencia relevante:** porcentaje y alcance estimado en Argentina;
- **objetivo de embudo:** alcance, descubrimiento, interacción o profundidad;
- **costo y saturación comercial:** precio y proporción patrocinada;
- **escala frente a eficiencia:** volumen bruto frente a tasas por 1000.

Esto no prescribe una fórmula de scoring supervisada. Sirve para interpretar
regiones del SOM y candidatos dentro de ellas. La comparación con `followers`
debe usar ejemplos concretos del dataset y hacer visibles los trade-offs.

## Orden recomendado para comenzar

1. Crear una carga de datos validada y un diccionario de variables.
2. Hacer EDA reproducible: tipos, faltantes, rangos, distribuciones, outliers,
   identidades algebraicas y correlaciones.
3. Proponer y justificar un conjunto de variables sin redundancias exactas,
   dejando identificadores sólo como metadatos.
4. Ajustar una única estandarización reproducible.
5. Interpretar la primera ejecución de PCA, comparar la variante de engagement
   y decidir cuántas componentes conservar.
6. Implementar y validar Kohonen primero sobre un caso sintético pequeño
   (completado); después comparar configuraciones sobre creadores.
7. Implementar y validar Oja; comparar su dirección e índice con PCA
   (completado).
8. Caracterizar regiones/candidatos para las cuatro campañas y contrastar con
   el baseline de `followers` usando el mismo `K` (completado).
9. Documentar mantenimiento, llegada de datos mensuales y limitaciones
   (completado).
10. Recién después de la clase correspondiente, incorporar Hopfield.

## Decisiones todavía abiertas

- Implementación, CLI y formato de resultados de Hopfield cuando se habilite
  el ejercicio 2.

Esa decisión no debe cerrarse sólo por conveniencia de código: se debe registrar
la evidencia, la alternativa considerada y su efecto en la interpretación.

## Fuentes locales consultadas

- `docs/SIATP42026.pdf`: enunciado vigente.
- `docs/Enunciado TP4 pt1.VTT`: explicación oral del ejercicio 1.
- `docs/1 - Introducción Aprendizaje No Supervisado.pdf`.
- `docs/2 - Modelo de Kohonen.pdf`.
- `docs/Aprendizaje No Supervisado y Kohonen.VTT`.
- `docs/4 - Autovalores y Autovectores (1).pdf`.
- `docs/5 - PCA (2).pdf`.
- `docs/6 - Regla de Oja y Sanger (1).pdf`.
- `docs/autovalores, PCA y Oja.VTT`.
- `datos/creadores.csv`.
