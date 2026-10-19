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

## Próximo paso acordado

PCA queda cerrado con 8 componentes y 90,00 % de varianza acumulada; la
comparación de engagement también está implementada y documentada. Kohonen fue
validado con cuatro grupos sintéticos y todavía no se ejecutó sobre los
creadores. El siguiente paso es definir y correr el protocolo de comparación de
hiperparámetros documentado en `docs/kohonen.md`.
