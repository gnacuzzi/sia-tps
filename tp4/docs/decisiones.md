# Decisiones del TP4

Este documento registra las decisiones acordadas, su justificación y las
validaciones pendientes. El informe cuantitativo que las respalda está en
`docs/eda-creadores/README.md`.

## Estado de la selección

Se confirmó el conjunto de 15 variables para PCA, Kohonen y Oja. La comparación
de representaciones de engagement ya está ejecutada y se decidió conservar el
engagement total como entrada principal.

## Variables base

| Familia | Variable | Justificación |
| --- | --- | --- |
| Escala | `average_reach` | Mide cuentas únicas alcanzadas y representa mejor la exposición efectiva que la cantidad de seguidores. |
| Actividad | `monthly_posts` | Representa la frecuencia de publicación y la actividad disponible para campañas. |
| Audiencia | `argentina_audience_pct` | Mide la afinidad geográfica con el mercado objetivo. |
| Descubrimiento | `non_follower_reach_pct` | Mide cuánto alcance proviene de personas que todavía no siguen al creador. |
| Interacción | `total_engagement_per_1000` | Resume la intensidad total de interacción por alcance en la primera variante. |
| Interés | `profile_visits_per_1000` | Mide si el alcance genera una acción posterior hacia el perfil. |
| Costo | `price_usd` | Es la cotización original y no depende de actualizar el tipo de cambio fijo usado para ARS. |
| Mezcla comercial | `sponsored_posts_pct` | Expresa directamente el nivel de contenido patrocinado y la posible saturación comercial. |
| Tema | `topic_football_pct` | Representa afinidad con campañas de fútbol. |
| Tema | `topic_makeup_pct` | Representa afinidad con campañas de belleza. |
| Tema | `topic_dogs_pct` | Representa afinidad con campañas de adopción de perros. |
| Contenido | `content_tutorial_pct` | Representa profundidad educativa y explicación de procedimientos. |
| Contenido | `content_review_pct` | Representa evaluación de productos o experiencias. |
| Contenido | `content_entertainment_pct` | Representa contenido cuyo objetivo principal es entretener. |
| Consumo | `average_watch_time_seconds` | Aporta una medida de profundidad de consumo audiovisual. |

## Variables auxiliares

Las variables auxiliares no determinan inicialmente componentes, distancias ni
pesos. Se conservan vinculadas por `creator_id` para etiquetar gráficos,
caracterizar regiones y tomar decisiones de campaña.

- `creator_id` y `name`: identifican a cada creador.
- `followers`: baseline exigido por el enunciado. Los resultados de PCA y
  Kohonen se colorearán o etiquetarán con esta variable para comparar la nueva
  selección con la estrategia actual de la agencia.
- `estimated_argentina_reach`: traduce alcance y audiencia local a una cantidad
  estimada de cuentas argentinas.
- `price_ars`: comunica costos en moneda local, pero no entra junto con USD
  porque `price_ars = 1500 * price_usd`.
- `organic_posts_pct`: complemento exacto de `sponsored_posts_pct`.
- `topic_art_pct`: complemento de los otros tres porcentajes temáticos.
- `content_showcase_pct`: complemento de los otros tres tipos de contenido.
- `likes_per_1000`, `comments_per_1000`, `saves_per_1000` y
  `shares_per_1000`: explican la composición del engagement y se usaron en la
  variante comparativa.
- `video_posts_pct`, `short_form_posts_pct` y `trend_posts_pct`: permiten
  interpretar el formato de los perfiles sin darle tres veces ese peso al
  entrenamiento.
- `average_impressions`, `average_video_views` y
  `average_caption_words`: se conservan para describir casos, pero no entran al
  conjunto base por su redundancia con el bloque de escala y estilo.

## Decisiones y justificaciones

### Alcance en lugar de seguidores

`followers` queda fuera del conjunto base porque mide audiencia potencial, no
exposición efectiva. `average_reach` mide cuentas únicas que realmente reciben
una publicación. Conservar `followers` como auxiliar permite responder la
comparación que exige el enunciado.

`average_impressions` y `average_video_views` también quedan como auxiliares.
Tienen correlaciones de 0,987 y 0,997 con `average_reach`, respectivamente.
Se elige alcance porque cuenta personas únicas y evita que las visualizaciones
repetidas representen varias veces el mismo bloque de escala.

### Representación del engagement

La primera variante usa `total_engagement_per_1000`. Esto mantiene compacta y
equilibrada la cantidad de variables. Sin embargo, el total oculta la
composición: en estos datos los likes representan entre 45 % y 74 % del total,
los comentarios entre 1 % y 12 %, los guardados entre 7 % y 43 % y los
compartidos entre 0,5 % y 24 %.

Se ejecutó el siguiente experimento acordado:

1. ejecutar PCA con el conjunto base y engagement total;
2. ejecutar otra variante reemplazando el total por likes, comentarios,
   guardados y compartidos;
3. no incluir simultáneamente el total y sus cuatro componentes;
4. comparar varianza explicada, cargas, biplot y perfiles observados;
5. elegir la representación final según cuánto cambien PCA y los perfiles y
   según cuál resulte más interpretable para las campañas.

El informe está en `docs/pca-engagement/README.md`. Los scores de PC1 y PC2
entre variantes tienen correlaciones de 0,986 y 0,982, respectivamente, y la
correlación entre las distancias del plano PC1-PC2 es 0,965. Separar las
interacciones muestra una oposición entre likes y guardados que el total
oculta, pero no cambia sustancialmente la organización global de los
creadores.

Se decidió conservar el engagement total como entrada principal y las cuatro
interacciones como auxiliares. Esto evita
darle cuatro columnas al mismo bloque, mantiene el conjunto en 15 variables y
preserva la interpretación. Las interacciones individuales se usarán para
caracterizar regiones o para reabrir la selección si una campaña futura exige
una acción específica.

### Precio en USD

Se usa `price_usd` y `price_ars` queda como auxiliar. Como ARS es una
multiplicación exacta por 1500, ambas variables producen el mismo z-score y los
mismos resultados numéricos después de estandarizar. USD se conserva porque es
la cotización original y no requiere actualizar un tipo de cambio fijo.

### Porcentaje patrocinado

Se usa `sponsored_posts_pct` y no `organic_posts_pct`. Son complementos exactos
y contienen la misma información con signo opuesto después de estandarizar. El
porcentaje patrocinado se interpreta directamente como exposición comercial o
posible saturación publicitaria.

### Composiciones de tema y contenido

Los cuatro porcentajes temáticos suman 100. Se omite `topic_art_pct` para evitar
una dependencia lineal exacta y se reconstruye como complemento. Esto no
elimina la campaña de arte del análisis.

Los cuatro tipos de contenido también suman 100. Se omite
`content_showcase_pct` y se reconstruye de la misma manera. Arte y showcase se
mantienen visibles al interpretar los resultados.

### Video, formato breve y tendencias

`video_posts_pct`, `short_form_posts_pct` y `trend_posts_pct` quedan inicialmente
fuera del modelo porque están muy correlacionadas entre sí:

- video y formato breve: `r = 0,971`;
- formato breve y tendencias: `r = 0,961`;
- video y tendencias: `r = 0,952`.

En este dataset describen casi el mismo eje de estilo. Incluir las tres haría
que el contenido breve, audiovisual y de tendencia pesara tres veces en PCA y
en las distancias de Kohonen. Además, el conjunto base ya representa:

- descubrimiento con `non_follower_reach_pct`;
- entretenimiento con `content_entertainment_pct`;
- profundidad con `average_watch_time_seconds`.

Las tres quedan como auxiliares para caracterizar regiones. Si más adelante
falta representar explícitamente el formato, la primera candidata para entrar
es `short_form_posts_pct`, porque describe un formato concreto y no sólo la
presencia de cualquier video o la participación coyuntural en una tendencia.

`average_caption_words` tampoco entra al conjunto base. En estos datos está
fuertemente asociado en sentido inverso con alcance, impresiones y vistas
(`r` cercano a -0,86). La profundidad del contenido ya se representa de forma
más directa con tutoriales, reviews y tiempo de reproducción. Se conserva como
auxiliar para describir perfiles textuales si aparece una diferencia relevante.

### Alcance argentino estimado

`estimated_argentina_reach` queda como auxiliar porque deriva de
`average_reach * argentina_audience_pct / 100`. Se usará para comunicar
resultados en cuentas argentinas, pero incluirla junto con sus dos factores
repetiría esa información en las entradas.

### Preprocesamiento

La primera ejecución usará únicamente estandarización por z-score. No se aplica
una transformación logarítmica ahora. El logaritmo queda como comparación
opcional sólo si PCA resulta dominado por variables de volumen o costo, y debe
justificarse explícitamente antes de incorporarlo.

No se eliminan automáticamente observaciones marcadas por la regla de 1,5 IQR.
En las variables temáticas, esos casos representan especialización y pueden ser
precisamente los creadores relevantes para una campaña.

### Componentes conservadas en PCA

La aclaración docente fija como criterio superar el 80 % de varianza acumulada
y acercarse al 90 %. Con las 15 variables elegidas, 6 componentes explican
81,48 %, 7 explican 86,24 % y 8 explican 90,00 %. Se conservan por lo tanto
**8 componentes**.

Este corte pertenece a la representación reducida de PCA. Kohonen y Oja siguen
recibiendo las 15 variables estandarizadas: Kohonen agrupa considerando todas
ellas y Oja aprende una dirección que se compara con PC1.

## Recorrido experimental de Kohonen

PCA queda cerrado con 8 componentes y 90,00 % de varianza acumulada; la
comparación de engagement también está implementada y documentada. Kohonen fue
validado con cuatro grupos sintéticos y su primer barrido sobre los creadores
comparó `k = 2, ..., 8` con cinco semillas por tamaño. La lectura conjunta del
error de cuantización, la ocupación y la concentración de hits dejó `k = 4` y
`k = 5` como candidatos para los experimentos posteriores.

La verificación posterior cruzó los siete tamaños con siete tasas constantes y
cinco semillas. Mostró que las tasas altas empeoran el error y la estabilidad,
y que la región útil queda entre `0.05` y `0.2`. Para el experimento de tasa
decreciente se conservaron `k = 4, η = 0.05`, `k = 5, η = 0.2` y, como control
de la interacción, `k = 6, η = 0.1`. Esa fue una etapa intermedia, no la
configuración final.

Esos tres puntos no se usaron como único recorte: se ejecutó además la
comparación completa de las 245 configuraciones constantes contra las mismas
245 configuraciones con disminución lineal hasta el 10 % de la tasa inicial.
El descenso resultó útil con tasas iniciales altas para `k = 4` y `k = 5`, pero
no dominó las tasas constantes bajas en `k = 6`. Se conservan regiones de esos
tres tamaños para estudiar radio y épocas; `k = 2–3` siguen demasiado
concentrados y `k = 7–8`, demasiado fragmentados.

El barrido posterior de radio ejecutó las 300 combinaciones previstas. En
`k = 6`, los radios locales lograron menor error dejando hasta 37,8 % de
unidades muertas; los radios amplios mejoraron la ocupación, pero el tamaño
siguió menos aprovechado que `k = 4` y `k = 5`. En estos dos tamaños no aparece
un único radio ganador para todas las tasas. Se conservaron configuraciones
representativas con buena ocupación y estabilidad para comparar sus mapas de
hits y matrices U antes de decidir.

El tiempo de entrenamiento se registró como criterio auxiliar. Las diferencias
observadas son de milisegundos y no justifican elegir una configuración de
menor calidad. Sólo se usará como desempate entre resultados equivalentes.

La inspección de nueve mapas representativos cerró el recorte previo a épocas.
Avanzan cinco configuraciones: las dos alternativas `k = 4`; en `k = 5`, la
tasa constante `0.2` con radio completo y el descenso `0.7 → 0.07` con radio al
75 %; y, como control `k = 6`, el descenso `0.7 → 0.07` con radio completo. Los
mapas `k = 4` quedaron totalmente ocupados, las dos alternativas `k = 5`
mantuvieron pocos huecos y estructura continua, y `k = 6` conservó suficiente
fragmentación como para no tratarlo todavía como candidato principal.

El barrido de 25 a 400 épocas fijó **150 épocas**. En ese punto las variantes
`k = 4` promedian 1,2 % de unidades muertas, las `k = 5` 4,8 % y el control
`k = 6` 15,6 %. Entrenar hasta 400 épocas multiplica el tiempo por unas 2,6–2,7
veces sin mejorar conjuntamente error, ocupación y estabilidad. En particular,
`k = 6` reduce su error pero aumenta las unidades muertas a 26,1 %, por lo que
no se interpreta como una mejora del mapa.

La comparación de 50 corridas de inicialización conserva **`sample`** frente a
`uniform`. En ningún perfil `uniform` mejoró simultáneamente error, ocupación y
concentración. Sus reducciones de error en `k = 5` decreciente y `k = 6`
vinieron acompañadas por 10,4 y 5,6 puntos porcentuales adicionales de unidades
muertas, respectivamente, y mayor variabilidad. No hay evidencia suficiente
para reabrir tasa y radio específicamente para esa inicialización.

La comparación final de mapas elige **`k5-constant-global`**: `k = 5`, tasa
constante `0.2`, radio inicial igual al diámetro con descenso hasta 1, 150
épocas e inicialización `sample`. `k = 5` reduce cerca de 9 % el error de
`k = 4` y sus neuronas ocupadas tienen mayor especialización temática media
(48,7 % frente a 41–45 %), con regiones interpretables para las cuatro
campañas. La alternativa decreciente reduce el error menos de 1 % adicional,
pero es más variable y deja más huecos en la corrida representativa.

La semilla 2 queda únicamente como corrida central para visualizar y describir
la configuración elegida. Las conclusiones de hiperparámetros usan los
promedios y desvíos de cinco semillas.

## Protocolo acordado para Oja

Oja utilizará las mismas 15 variables, en el mismo orden, y el mismo z-score
muestral (`ddof=1`) que PCA. Reutilizar `Standardizer` evita que la comparación
se vea afectada por dos implementaciones distintas del preprocesamiento y
garantiza que los pesos de Oja y el autovector de PC1 pertenezcan al mismo
espacio.

La implementación sigue la actualización en línea vista en clase:
`w <- w + eta * y * (x - y * w)`, con `y = x . w`. Los pesos se inicializan
con una distribución uniforme entre 0 y 1. No se normalizan explícitamente
después de cada actualización, porque la normalización aproximada es justamente
el término que distingue la regla de Oja de la regla hebbiana sin cota; la norma
debe tender a 1 como parte de la evidencia de convergencia.

Las muestras se barajarán en cada época. El orden de una actualización en línea
puede cambiar la trayectoria y el resultado finito, por lo que mantener siempre
el orden del CSV introduciría una dependencia arbitraria. El generador queda
controlado por una semilla para que tanto la inicialización como cada orden sean
reproducibles.

Se comparará una tasa constante con un descenso lineal hasta el 10 % de la tasa
inicial. La clase propone experimentar tanto con tasa fija como decreciente,
pero no prescribe una fórmula de descenso. Se adopta la lineal porque es
simple, auditable y ya se utilizó para Kohonen; no se interpreta como una
ventaja teórica sobre otros schedules.

La convergencia se evaluará en conjunto mediante:

- similitud de dirección con el autovector de PC1, ignorando el signo;
- evolución de la norma de los pesos hacia 1;
- correlación entre el índice de Oja y los scores de PC1 después de alinear el
  signo;
- estabilidad de estas medidas entre semillas.

El experimento de Oja usará como mínimo 10 semillas por configuración. Las
semillas no forman una serie temporal: se resumirán con boxplots o con
promedios y barras de error, además de informar la dispersión y el peor caso.
Las curvas por época mostrarán primero la trayectoria completa; sólo se
agregará un zoom final si ayuda a distinguir configuraciones cercanas.

Antes del barrido sobre creadores se validará la implementación con datos
sintéticos de dirección principal conocida y con pruebas de reproducibilidad,
ambigüedad de signo y cálculo de las proyecciones. El barrido no comenzará
hasta acordar sus tasas y cantidad de épocas. Se versionarán configuraciones,
tablas agregadas, el informe y un conjunto pequeño de gráficos comparativos;
no se guardará una carpeta redundante por corrida.

### Resultado de la etapa 1: tasa y schedule

Se ejecutaron 160 corridas: ocho tasas iniciales entre `0.001` y `0.5`, tasa
constante o descenso lineal hasta el 10 %, 250 épocas y 10 semillas. La menor
tasa ya alcanzó el techo práctico de la comparación, por lo que no apareció un
pico interior: con descenso, `eta = 0.001` obtuvo una similitud media con PC1 de
0.99999994 y correlación media de índices de 0.99999998. Probar tasas menores
aportaría principalmente una convergencia más lenta, porque ambas métricas ya
están acotadas por 1.

Las tasas constantes se degradaron progresivamente entre `0.005` y `0.05` y
divergieron en las 10 semillas desde `0.1`. El descenso lineal mantuvo estables
las tasas de hasta `0.05`, pero con `0.1` sólo convergieron 2 de 10 semillas y
desde `0.2` divergieron todas. Esto muestra que disminuir la tasa reduce la
oscilación final, pero no compensa de forma confiable una tasa inicial excesiva.

Para estudiar la cantidad de épocas se conservan tres finalistas:

- descenso `0.001 -> 0.0001`, por su precisión final y norma más cercana a 1;
- descenso `0.005 -> 0.0005`, porque alcanza rápidamente una dirección muy
  cercana a PC1 y termina con precisión casi perfecta;
- tasa constante `0.001`, como control estable para aislar el efecto del
  schedule.

El informe reproducible, las corridas individuales, los agregados por época y
los gráficos comparativos quedan en `docs/oja-experimentos/01-rate/`.

### Resultado de la etapa 2: cantidad de épocas

Se compararon 5, 10, 25, 50, 100, 150, 250 y 500 épocas para los tres perfiles
finalistas de tasa, con 10 semillas por combinación y 240 corridas en total. No
hubo divergencias.

Se elige **100 épocas con descenso lineal de `0.001` a `0.0001`**. La similitud
media con PC1 fue 0.999999847 y el peor valor entre las 10 semillas fue
0.999999638. La correlación media entre las proyecciones de Oja y los scores de
PC1 fue 0.999999962, y el error medio de la norma respecto de 1 fue 0.000446.

A las 50 épocas el promedio ya era alto, pero la peor semilla todavía quedaba
en 0.997114. En cambio, desde 100 épocas todas las semillas forman una meseta
prácticamente perfecta. Extender hasta 500 épocas multiplica por cinco el
entrenamiento y mejora la similitud media sólo en aproximadamente `1.2e-7`; la
reducción adicional del error de norma es 0.000156. No se justifica ese costo
adicional para una diferencia sin efecto interpretativo.

El perfil decreciente desde `0.005` aprende más rápido, pero estabiliza la norma
más lejos de 1. La tasa constante `0.001` también converge, aunque mantiene un
error de norma cercano a 0.00231. Por eso la elección final considera
conjuntamente dirección, proyecciones, norma y estabilidad entre semillas.

### Interpretación final del índice

Las diez corridas de la configuración elegida confirman que Oja reproduce PC1:
la diferencia absoluta máxima entre el peso de la corrida representativa y el
autovector es 0.000341, y la correlación entre sus índices y los scores de PC1
es 0.999999962. Se usa una semilla central respecto de las medianas de las tres
métricas, no la mejor corrida aislada.

El índice ubica a cada creador sobre el principal eje de variabilidad de las 15
entradas estandarizadas. En la orientación elegida, valores altos describen
mayor escala, precio, descubrimiento y entretenimiento; valores bajos,
profundidad, visitas al perfil, tutoriales, reviews, tiempo de reproducción y
contenido de perros. Esta orientación se fija sólo para comunicar: invertir
todos los signos produce la misma solución.

No se usará el máximo como selección automática para cualquier campaña. PC1
explica 37.63 % de la varianza y no representa ventas, adopciones ni retorno de
inversión. Además, un índice alto puede implicar mayor precio y no garantiza
afinidad temática. La decisión debe combinar este eje con las regiones de
Kohonen, las componentes temáticas, la audiencia argentina, el objetivo de la
campaña, el costo y la saturación comercial.

## Protocolo de estabilidad para las campañas

Antes de convertir las regiones de Kohonen en recomendaciones se repetirá sólo
la configuración final con diez semillas (`0` a `9`). No se compararán
coordenadas de la grilla, porque una solución equivalente puede rotarse o
reflejarse: para cada creador se comparará el tema dominante del prototipo de su
BMU. También se registrará, para cada campaña, cuántas semillas incluyen al
creador en la neurona ocupada de mayor afinidad temática.

Los umbrales se fijan antes de observar el resultado: entre 8 y 10 apariciones
se considera estable, entre 6 y 7 fronterizo y entre 0 y 5 inestable. Las
neuronas cuya afinidad temática máxima sea menor que 35 % se mantienen
etiquetadas por su tema dominante, pero se señalan como transiciones. Este
conteo es un diagnóstico descriptivo adicional y no un criterio para elegir de
nuevo los hiperparámetros.

La configuración, semillas y umbrales quedan en un único JSON. Los resultados
se consolidan en tablas y gráficos agregados, sin guardar mapas o directorios
separados por semilla. Si sólo cambian creadores fronterizos, se informará la
incertidumbre. Si la inestabilidad es general o afecta a los candidatos
principales, primero se repetirán con diez semillas las configuraciones
finalistas; no se reabrirá automáticamente el barrido completo.

### Resultado de estabilidad

Con la configuración final y diez semillas, 36 de los 50 creadores resultaron
estables en su región temática, 11 fronterizos y sólo 3 inestables. Las celdas
de transición reunieron 60 de las 500 asignaciones creador-semilla. Además, las
40 celdas elegidas como máxima afinidad para las cuatro campañas conservaron el
tema dominante esperado.

La selección de una única celda sí fue sensible: fútbol y perros produjeron 10
shortlists distintas en 10 semillas, mientras que belleza y arte produjeron 9.
Fútbol conservó un candidato estable y otro fronterizo; belleza cuatro
fronterizos; perros ninguno estable o fronterizo; y arte uno estable y dos
fronterizos. Esto no muestra un colapso de las regiones, sino que los creadores
de un mismo tema se reparten entre neuronas internas diferentes.

Por lo tanto, no se usará la neurona de máxima afinidad como selector final. La
síntesis de campañas partirá de la región temática completa y comparará dentro
de ella las variables originales. No se repetirá todavía el barrido: primero se
evaluará esta selección regional, y sólo si también resulta inestable se
compararán con diez semillas las configuraciones finalistas.

## Síntesis reproducible para las campañas

Cada creador se asigna a la campaña correspondiente a su región temática modal
en las diez semillas de Kohonen. Si hubiera un empate modal, se conserva en las
dos regiones empatadas en vez de resolverlo arbitrariamente; no hubo empates en
estos datos. La región completa constituye el conjunto de análisis. Los casos
estables y fronterizos son candidatos primarios, mientras que los inestables se
mantienen visibles con una advertencia y no se eliminan silenciosamente.

La tabla maestra reúne las 31 variables numéricas en unidades originales, los
cuatro conteos de pertenencia regional, la clase y porcentaje de estabilidad,
PC1-PC2 y el índice representativo de Oja. No se construye un score compuesto
ni un ranking: Oja confirma PC1 con correlación 0.999999962 y, por lo tanto, no
constituye una segunda evidencia independiente que deba contarse dos veces.
`followers` se conserva como variable auxiliar para el contraste posterior con
la estrategia actual de la agencia.

Los conjuntos obtenidos son:

- fútbol: 17 creadores, 13 estables, 2 fronterizos y 2 inestables;
- belleza: 8 creadores, 6 estables y 2 fronterizos;
- perros: 13 creadores, 12 estables y 1 fronterizo;
- arte: 12 creadores, 5 estables, 6 fronterizos y 1 inestable.

Las afinidades temáticas originales medias son 53.1 % para fútbol, 69.1 % para
belleza, 64.9 % para perros y 48.9 % para arte. La región de arte se interpreta
también mediante `topic_art_pct`, conservada como variable auxiliar porque se
había excluido de las 15 entradas para evitar la dependencia lineal exacta de
los cuatro porcentajes temáticos.

Esta etapa no elige automáticamente un creador. Produce conjuntos auditables y
perfiles regionales para que el próximo contraste seleccione la misma cantidad
de candidatos que el baseline por `followers` y haga explícitos los trade-offs
de alcance, afinidad, respuesta, formato y costo. El protocolo y las salidas
quedan en `configs/campaigns/synthesis.json` y `docs/campanas-creadores/`.
