# Selección de campañas guiada por los modelos

## Qué corrige esta versión

La selección anterior recorría todo el catálogo y ordenaba por un proxy comercial. Kohonen quedaba como una descripción posterior. Esta versión invierte el orden: los modelos delimitan primero el perfil buscado y las métricas originales sólo deciden entre los creadores que cumplen ese perfil.

En este informe, **afinidad temática** significa el porcentaje del contenido del creador dedicado al tema de la campaña. Es una variable original del dataset. No mide afinidad de la audiencia ni proviene de Kohonen.

## Regla de selección

1. **Kohonen define el conjunto temático.** Se consideran los creadores cuya región modal coincide con la campaña y cuya asignación es estable o fronteriza.
2. **La afinidad elimina los casos débiles dentro de la región.** Se conserva la mitad superior según el porcentaje del tema correspondiente. El umbral es la mediana de cada región, no un 50 % común inventado.
3. **PCA define el tipo de perfil dentro de esa región.** Para ventas se usa la mitad con PC1 más alto, asociada con escala y descubrimiento. Para perros se usa la mitad con PC1 más bajo, asociada con tutoriales, tiempo de reproducción y visitas al perfil.
4. **Oja valida PC1, pero no agrega otro voto.** Su correlación casi perfecta con PC1 indica que ambos describen el mismo eje.
5. **La métrica del objetivo decide dentro del conjunto resultante.** Para ventas se maximiza el alcance argentino temático proveniente de no seguidores. Para perros se incorpora además la tasa de visitas al perfil como señal de consideración.

Las cuentas son proxies, no predicciones de ventas ni adopciones:

- `alcance nuevo relevante = alcance argentino × afinidad temática × % no seguidores`
- `visitas nuevas relevantes = alcance nuevo relevante × visitas al perfil por 1000 / 1000`

Estas multiplicaciones permiten comparar alternativas con una unidad interpretable, pero suponen que los porcentajes se aplican proporcionalmente. El dataset no permite comprobar esa suposición ni observar ventas, consultas o adopciones.

## Resultado

| Campaña | Candidato | Aporte de los modelos | Métrica de decisión |
| --- | --- | --- | ---: |
| Fútbol | @golconmate | región fútbol, afinidad alta y lado de escala de PC1 | 102,045 cuentas estimadas |
| Belleza | @glamencasa | región belleza, afinidad alta y lado de escala de PC1 | 25,655 cuentas estimadas |
| Perros | @guiadehuellas | región perros, afinidad alta y lado profundo de PC1 | 51.4 visitas estimadas |
| Arte | @pincelcotidiano | región arte, afinidad alta y lado de escala de PC1 | 23,527 cuentas estimadas |

## Finalistas después de aplicar los modelos

| Campaña | Orden por objetivo | Creador | Afinidad | PC1 |
| --- | ---: | --- | ---: | ---: |
| Fútbol | 1 | @golconmate | 82.2 % | +3.34 |
| Fútbol | 2 | @lookdecancha | 69.3 % | +2.81 |
| Fútbol | 3 | @futbolsinvueltas | 63.4 % | +4.29 |
| Fútbol | 4 | @tacticavisual | 81.3 % | +3.26 |
| Belleza | 1 | @glamencasa | 74.1 % | +1.91 |
| Belleza | 2 | @bellezasimple | 77.8 % | +2.66 |
| Perros | 1 | @guiadehuellas | 68.8 % | -2.41 |
| Perros | 2 | @perrosfelices.ar | 70.7 % | -3.20 |
| Perros | 3 | @consejoscaninos | 82.0 % | -4.41 |
| Perros | 4 | @mundoanimal.ar | 74.2 % | -3.94 |
| Arte | 1 | @pincelcotidiano | 70.2 % | +0.69 |
| Arte | 2 | @artequeinspira | 70.2 % | +0.62 |
| Arte | 3 | @rinconcreativo | 64.1 % | +0.19 |

## Justificación por campaña

### Fútbol: @golconmate

Conserva la elección anterior, pero ahora llega a ella después de exigir coherencia con la región de fútbol, afinidad superior a la mediana y posición en el lado de escala de PC1.

- Afinidad temática: **82.2 %**.
- Estabilidad de región: **estable (100 %)**.
- PC1 / Oja: **3.34 / 3.34**.
- Alcance argentino estimado: **207,994**.
- Alcance de no seguidores: **59.7 %**.
- Precio: **USD 3,251**.

### Belleza: @glamencasa

Reemplaza a @maquillajereal. @glamencasa pertenece de forma estable a belleza y gana en alcance nuevo relevante dentro del perfil de escala definido por PCA.

- Afinidad temática: **74.1 %**.
- Estabilidad de región: **estable (100 %)**.
- PC1 / Oja: **1.91 / 1.91**.
- Alcance argentino estimado: **51,469**.
- Alcance de no seguidores: **67.2 %**.
- Precio: **USD 2,922**.

### Perros: @guiadehuellas

Reemplaza a @crearenreels. @guiadehuellas pertenece de forma estable a perros, queda en el lado de profundidad de PC1 y maximiza las visitas nuevas relevantes dentro de ese perfil. La cuenta sigue siendo una aproximación y no mide adopciones.

- Afinidad temática: **68.8 %**.
- Estabilidad de región: **estable (100 %)**.
- PC1 / Oja: **-2.41 / -2.41**.
- Alcance argentino estimado: **5,535**.
- Alcance de no seguidores: **66.8 %**.
- Precio: **USD 189**.

### Arte: @pincelcotidiano

Reemplaza a @galeriacotidiana. @pincelcotidiano es fronterizo, pero su región modal es arte y gana en alcance nuevo relevante dentro del perfil de escala. La condición fronteriza debe mostrarse como cautela.

- Afinidad temática: **70.2 %**.
- Estabilidad de región: **fronterizo (70 %)**.
- PC1 / Oja: **0.69 / 0.69**.
- Alcance argentino estimado: **41,603**.
- Alcance de no seguidores: **80.6 %**.
- Precio: **USD 2,362**.

## Qué queda para la presentación

La futura comparación con followers debe usar estos cuatro seleccionados. La comparación debe separar alcance, afinidad, señal de respuesta y costo; no debe reducirlos a una única barra sin unidad.

El archivo `generated/conjunto_finalista.csv` conserva todos los candidatos que superaron las tres etapas de los modelos, con sus umbrales y cuentas. Así se puede explicar por qué ganó cada creador y quién quedó segundo.
