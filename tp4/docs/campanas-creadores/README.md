# Síntesis regional para las cuatro campañas

## Criterio

Cada creador se asigna a su región temática modal en las diez semillas de la
configuración final de Kohonen. Se usa la región completa, no la neurona de
máxima afinidad. Los casos estables y fronterizos forman el conjunto primario;
los inestables permanecen visibles, pero con una advertencia explícita.

La tabla maestra conserva las 31 variables numéricas originales en sus unidades,
los conteos de región de las diez semillas, los scores de PC1-PC2 y el índice de
Oja. No se calcula un puntaje compuesto ni se ordena a los creadores como
mejores o peores. `followers` se usa únicamente para construir el baseline
global de igual tamaño que cada conjunto regional.

## Integración de modelos

| Campaña | Creadores | Estables | Fronterizos | Inestables | PC1 medio | PC2 medio | Oja medio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fútbol | 17 | 13 | 2 | 2 | +1.07 | +0.83 | +1.07 |
| Belleza | 8 | 6 | 2 | 0 | +1.99 | -0.87 | +1.99 |
| Perros | 13 | 12 | 1 | 0 | -2.30 | -1.07 | -2.30 |
| Arte | 12 | 5 | 6 | 1 | -0.35 | +0.58 | -0.35 |

PC1 y PC2 explican juntas **49.90 %** de la varianza. El índice de
Oja de la corrida representativa tiene correlación **0.999999962**
con PC1. Por eso Oja confirma ese eje, pero no se cuenta como una segunda señal
independiente para seleccionar candidatos.

![Regiones sobre PCA](generated/regiones_pca_estabilidad.svg)

## Perfiles regionales

El siguiente gráfico usa las mismas 15 variables estandarizadas de PCA,
Kohonen y Oja. Cada valor es la media z-score de una región; no es una tasa ni
un score comercial.

![Perfiles de regiones](generated/perfiles_regiones.svg)

### Fútbol

- Afinidad temática media: **53.1 %**.
- Variables relativamente altas: `topic_football_pct:1.009;sponsored_posts_pct:0.523;non_follower_reach_pct:0.365`.
- Variables relativamente bajas: `topic_dogs_pct:-0.474;profile_visits_per_1000:-0.292;topic_makeup_pct:-0.274`.
- Candidatos regionalmente estables: INF005 (@tribunacreativa), INF011 (@vidaperruna), INF017 (@tacticavisual), INF021 (@golconmate), INF027 (@pasionalbirroja), INF030 (@lookdecancha), INF033 (@futbolentrepases), INF034 (@makeupconsejos), INF036 (@crearenreels), INF040 (@galeriacotidiana), INF041 (@jugadayestilo), INF045 (@futbolsinvueltas), INF050 (@reelconideas).
- Candidatos fronterizos: INF020 (@manosalaobra), INF025 (@reelargentino).
- Creadores inestables, conservados con advertencia: INF014 (@brochasycolor), INF037 (@pasiondepotrero).
### Belleza

- Afinidad temática media: **69.1 %**.
- Variables relativamente altas: `topic_makeup_pct:1.898;average_reach:0.890;price_usd:0.812`.
- Variables relativamente bajas: `profile_visits_per_1000:-0.786;content_tutorial_pct:-0.673;topic_dogs_pct:-0.623`.
- Candidatos regionalmente estables: INF001 (@mateconideas), INF002 (@brillourbano), INF006 (@glamencasa), INF010 (@maquillajereal), INF022 (@bellezasimple), INF042 (@brilloytendencia).
- Candidatos fronterizos: INF032 (@tallerurbano), INF038 (@cosmeticaconsciente).
- Creadores inestables, conservados con advertencia: ninguno.
### Perros

- Afinidad temática media: **64.9 %**.
- Variables relativamente altas: `topic_dogs_pct:1.480;profile_visits_per_1000:0.833;average_watch_time_seconds:0.747`.
- Variables relativamente bajas: `average_reach:-0.735;price_usd:-0.701;sponsored_posts_pct:-0.684`.
- Candidatos regionalmente estables: INF003 (@huellasyconsejos), INF015 (@guiadehuellas), INF018 (@estiloydetalle), INF019 (@perrosfelices.ar), INF023 (@clubdelasmascotas), INF026 (@colorenmovimiento), INF028 (@ideasconhuella), INF031 (@mundoanimal.ar), INF035 (@adoptafelicidad), INF039 (@huellasargentinas), INF043 (@consejoscaninos), INF047 (@patitascontentas).
- Candidatos fronterizos: INF007 (@patitasenred).
- Creadores inestables, conservados con advertencia: ninguno.
### Arte

- Afinidad temática media: **48.9 %**.
- Variables relativamente altas: `content_review_pct:0.461;monthly_posts:0.245;sponsored_posts_pct:0.233`.
- Variables relativamente bajas: `topic_dogs_pct:-0.518;content_entertainment_pct:-0.422;topic_football_pct:-0.287`.
- Candidatos regionalmente estables: INF008 (@artealpaso), INF024 (@artequeinspira), INF029 (@canvasycafe), INF044 (@arteentusmanos), INF048 (@creatividadaldia).
- Candidatos fronterizos: INF004 (@pincelcotidiano), INF009 (@futbolcononda), INF012 (@trazosyreels), INF013 (@canchaaldia), INF016 (@rinconcreativo), INF049 (@entrelienzos).
- Creadores inestables, conservados con advertencia: INF046 (@brochasymates).

## Comparación con el baseline por followers

Para cada campaña, `K` es la cantidad de candidatos regionales estables o
fronterizos. El baseline toma los `K` creadores con más followers de todo el
catálogo. De este modo ambas estrategias tienen el mismo tamaño y la diferencia
no se explica por comparar conjuntos de distinta cantidad.

| Campaña | K | Solapamiento | Afinidad regional / baseline (%) | Alcance argentino regional / baseline | Engagement regional / baseline (por 1000) | Costo regional / baseline (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fútbol | 15 | 7/15 (46.7 %) | 56.1 / 31.7 | 1,060,925 / 1,504,350 | 72.6 / 70.3 | 38,197 / 52,440 |
| Belleza | 8 | 4/8 (50.0 %) | 69.1 / 39.6 | 598,344 / 1,097,439 | 67.4 / 67.3 | 24,834 / 31,741 |
| Perros | 13 | 0/13 (0.0 %) | 64.9 / 15.6 | 156,762 / 1,416,823 | 65.0 / 70.3 | 6,371 / 44,743 |
| Arte | 11 | 0/11 (0.0 %) | 51.8 / 14.2 | 316,311 / 1,328,395 | 68.5 / 71.1 | 13,698 / 38,520 |

Las sumas de followers, alcance y alcance argentino son volúmenes descriptivos
no deduplicados: el dataset no permite conocer cuánta audiencia comparten dos
creadores. El costo es la suma de `price_usd` de los integrantes de cada
shortlist.

![Comparación de estrategias](generated/comparacion_estrategias.svg)

![Solapamiento](generated/solapamiento_shortlists.svg)

### Fútbol

- Afinidad temática: **56.1 %** regional frente a **31.7 %** por followers (+24.4 puntos porcentuales).
- Suma de alcance argentino estimado: **1,060,925** frente a **1,504,350** (-29.5 %).
- Engagement medio: **72.6** frente a **70.3** por 1000 (+3.4 %).
- Costo total: **USD 38,197** frente a **USD 52,440** (-27.2 %).
- Sólo en la shortlist regional: INF027 (@pasionalbirroja), INF030 (@lookdecancha), INF033 (@futbolentrepases), INF034 (@makeupconsejos), INF041 (@jugadayestilo), INF050 (@reelconideas), INF020 (@manosalaobra), INF025 (@reelargentino).
- Sólo en el baseline: INF001 (@mateconideas), INF002 (@brillourbano), INF010 (@maquillajereal), INF042 (@brilloytendencia), INF032 (@tallerurbano), INF038 (@cosmeticaconsciente), INF006 (@glamencasa), INF004 (@pincelcotidiano).
### Belleza

- Afinidad temática: **69.1 %** regional frente a **39.6 %** por followers (+29.5 puntos porcentuales).
- Suma de alcance argentino estimado: **598,344** frente a **1,097,439** (-45.5 %).
- Engagement medio: **67.4** frente a **67.3** por 1000 (+0.2 %).
- Costo total: **USD 24,834** frente a **USD 31,741** (-21.8 %).
- Sólo en la shortlist regional: INF006 (@glamencasa), INF022 (@bellezasimple), INF032 (@tallerurbano), INF038 (@cosmeticaconsciente).
- Sólo en el baseline: INF021 (@golconmate), INF036 (@crearenreels), INF045 (@futbolsinvueltas), INF005 (@tribunacreativa).
### Perros

- Afinidad temática: **64.9 %** regional frente a **15.6 %** por followers (+49.4 puntos porcentuales).
- Suma de alcance argentino estimado: **156,762** frente a **1,416,823** (-88.9 %).
- Engagement medio: **65.0** frente a **70.3** por 1000 (-7.5 %).
- Costo total: **USD 6,371** frente a **USD 44,743** (-85.8 %).
- Sólo en la shortlist regional: INF003 (@huellasyconsejos), INF015 (@guiadehuellas), INF018 (@estiloydetalle), INF019 (@perrosfelices.ar), INF023 (@clubdelasmascotas), INF026 (@colorenmovimiento), INF028 (@ideasconhuella), INF031 (@mundoanimal.ar), INF035 (@adoptafelicidad), INF039 (@huellasargentinas), INF043 (@consejoscaninos), INF047 (@patitascontentas), INF007 (@patitasenred).
- Sólo en el baseline: INF001 (@mateconideas), INF021 (@golconmate), INF002 (@brillourbano), INF010 (@maquillajereal), INF036 (@crearenreels), INF045 (@futbolsinvueltas), INF042 (@brilloytendencia), INF005 (@tribunacreativa), INF032 (@tallerurbano), INF038 (@cosmeticaconsciente), INF040 (@galeriacotidiana), INF011 (@vidaperruna), INF006 (@glamencasa).
### Arte

- Afinidad temática: **51.8 %** regional frente a **14.2 %** por followers (+37.6 puntos porcentuales).
- Suma de alcance argentino estimado: **316,311** frente a **1,328,395** (-76.2 %).
- Engagement medio: **68.5** frente a **71.1** por 1000 (-3.6 %).
- Costo total: **USD 13,698** frente a **USD 38,520** (-64.4 %).
- Sólo en la shortlist regional: INF008 (@artealpaso), INF024 (@artequeinspira), INF029 (@canvasycafe), INF044 (@arteentusmanos), INF048 (@creatividadaldia), INF004 (@pincelcotidiano), INF009 (@futbolcononda), INF012 (@trazosyreels), INF013 (@canchaaldia), INF016 (@rinconcreativo), INF049 (@entrelienzos).
- Sólo en el baseline: INF001 (@mateconideas), INF021 (@golconmate), INF002 (@brillourbano), INF010 (@maquillajereal), INF036 (@crearenreels), INF045 (@futbolsinvueltas), INF042 (@brilloytendencia), INF005 (@tribunacreativa), INF032 (@tallerurbano), INF038 (@cosmeticaconsciente), INF040 (@galeriacotidiana).

## Alcance de esta etapa

Esta síntesis define conjuntos de candidatos y hace visibles sus perfiles y su
estabilidad. La comparación muestra diferencias observadas de composición,
alcance, afinidad, interacción, formato y costo. No estima ventas, adopciones ni
retorno de inversión porque el dataset no contiene esas variables objetivo.
Tampoco elige automáticamente un ganador: la estrategia apropiada depende del
objetivo declarado para cada campaña.

## Archivos reproducibles

- `generated/creadores_por_campana.csv`: unión por creador de estabilidad,
  variables originales, PCA y Oja;
- `generated/resumen_regiones.csv`: tamaños, estabilidad, ejes y medias/medianas;
- `generated/perfiles_estandarizados.csv`: perfil z-score completo de cada región.
- `generated/shortlists_comparadas.csv`: comparación individual con igual `K`;
- `generated/comparacion_followers.csv`: resultados agregados por estrategia.
