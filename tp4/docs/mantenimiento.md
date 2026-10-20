# Mantenimiento del análisis de creadores

## Alcance

Esta guía describe cómo actualizar el ejercicio 1 sin cambiar silenciosamente
la definición del problema. El pipeline actual entrena PCA, Kohonen y Oja sobre
el catálogo completo. No persiste un modelo de referencia para puntuar una fila
nueva de forma aislada. Por lo tanto, toda actualización operativa incorpora el
nuevo catálogo y vuelve a generar los resultados canónicos.

Las configuraciones finales continúan siendo el punto de partida:

- 15 variables base y estandarización z-score;
- Kohonen 5×5, tasa constante 0.2, radio decreciente hasta 1, 150 épocas e
  inicialización `sample`, evaluado con diez semillas;
- Oja durante 100 épocas, tasa decreciente de 0.001 a 0.0001 y diez semillas;
- cuatro campañas y tres perfiles de candidato por campaña.

Mantener una configuración como punto de partida no significa asumir que será
válida para cualquier cambio. Las secciones siguientes fijan cuándo alcanza con
regenerar y cuándo debe revisarse el diseño.

## Capacidades y límites actuales

| Cambio | Estado actual | Acción necesaria |
| --- | --- | --- |
| Agregar creadores con el mismo esquema | Soportado | Incorporar las filas y regenerar todo el pipeline. |
| Actualizar valores mensuales | Soportado | Reemplazar los valores del catálogo y regenerar. |
| Agregar una variable sólo descriptiva | Parcialmente soportado | Actualizar el esquema del EDA y las tablas donde deba aparecer; no cambia los modelos. |
| Agregar una entrada a los modelos | Requiere revisión | Actualizar la selección base, repetir EDA, PCA, Kohonen y Oja. |
| Agregar una quinta campaña | No soportado deliberadamente | Extender configuración, regiones y reportes; hoy se rechaza para preservar el alcance del enunciado. |
| Agregar un tema nuevo | Cambio estructural | Incorporar una representación temática válida y volver a validar todos los modelos. |
| Puntuar un creador con modelos congelados | No soportado | Requeriría persistir el z-score y los modelos; no se agrega para este TP. |

El cargador de modelos valida que existan las columnas solicitadas, que
`creator_id` y `name` no estén vacíos, que los identificadores sean únicos, que
las entradas sean numéricas y finitas, que `followers` no sea negativo y que
ninguna entrada sea constante. El EDA es más estricto: verifica el esquema
completo y su orden. Una columna nueva exige actualizar explícitamente
`EXPECTED_COLUMNS` en `scripts/analyze_creators.py`.

## Actualización mensual

### 1. Preparar el catálogo

1. Actualizar `datos/creadores.csv` conservando una fila por creador.
2. Mantener estable `creator_id`; un cambio de nombre visible no debe crear una
   identidad nueva.
3. Registrar la fecha o corte de la fuente en el mensaje de commit o en la
   documentación de la entrega. El CSV no contiene actualmente una fecha de
   extracción.
4. No editar manualmente los CSV o SVG generados.
5. Revisar el diff del archivo fuente antes de entrenar para detectar columnas
   renombradas, unidades distintas, duplicados o cambios de escala.

### 2. Validar y regenerar

Desde `tp4/`, ejecutar:

```bash
python3 -m pytest
python3 scripts/analyze_creators.py
python3 scripts/analyze_pca.py
python3 scripts/analyze_kohonen_stability.py
python3 scripts/analyze_oja.py
python3 scripts/analyze_campaigns.py
```

`analyze_campaigns.py` vuelve a ajustar PCA, las diez redes de Kohonen y las
diez corridas de Oja necesarias para la síntesis. Los comandos anteriores se
mantienen porque también actualizan sus informes independientes. Si cambiaron
de forma visible las regiones o se necesitan nuevamente los mapas completos,
ejecutar además:

```bash
python3 scripts/analyze_kohonen_finalists.py
```

No es necesario volver a ejecutar todos los barridos en una actualización
rutinaria.

### 3. Revisar los resultados

Comparar contra la versión anterior:

- cantidad de filas y de creadores por región;
- creadores estables, fronterizos e inestables;
- asignaciones a celdas de transición;
- candidatos finales y motivos de cualquier cambio;
- cantidad de componentes necesaria para alcanzar 90 % de varianza;
- cargas e interpretación de PC1 y PC2;
- similitud de Oja con el autovector de PC1, norma de pesos y correlación de
  índices;
- solapamiento y trade-offs contra el baseline por `followers`;
- costo, alcance y afinidad de los tres perfiles propuestos.

Un cambio de candidato no es por sí solo un error. Debe explicarse mediante sus
variables originales y su estabilidad regional.

### 4. Versionar sin duplicar resultados

Se sobrescriben los informes canónicos de `docs/`; no se crea una carpeta por
mes ni por semilla. Git conserva el historial de datos, configuraciones,
tablas, gráficos y decisiones. Antes del commit:

```bash
git diff --check
git status --short
```

No se versionan cachés de Matplotlib, `__pycache__`, entornos virtuales ni
salidas temporales. Este procedimiento no requiere agregar archivos al
`.gitignore` actual.

## Nuevos creadores

Una fila nueva debe usar un `creator_id` nuevo y completar las 31 variables
numéricas originales en las mismas unidades. No se imputan faltantes ni se
convierten unidades automáticamente.

Después de agregarla se ejecuta la actualización mensual completa. Esto es
necesario porque:

- el z-score depende de la media y el desvío de todo el catálogo;
- PCA y Oja dependen de la nueva matriz estandarizada;
- las BMU, regiones y fronteras de Kohonen pueden cambiar;
- `K`, el baseline por followers y las medianas usadas para candidatos pueden
  cambiar.

Para evaluar un creador sin reentrenar sería necesario guardar la media y el
desvío originales, los autovectores, el vector de Oja y los pesos de Kohonen.
Ese modo de inferencia congelada no está implementado y no debe simularse
copiando resultados de una corrida anterior.

## Nuevas campañas

### Campaña dentro de un tema existente

Primero se debe declarar:

- región temática relevante;
- objetivo observable disponible, como alcance, engagement o visitas al
  perfil;
- formato de contenido pertinente;
- variable de afinidad;
- regla de candidatos y baseline.

Los modelos pueden reutilizarse si no cambian las entradas ni la semántica de
las regiones. Sin embargo, la síntesis actual exige exactamente las cuatro
campañas del enunciado y una correspondencia uno a uno con fútbol, maquillaje,
perros y arte. Una quinta campaña requiere generalizar `TOPICS`, `CAMPAIGNS`, la
validación de `synthesis.json`, las etiquetas y los reportes. No debe agregarse
duplicando una campaña existente o reemplazándola sólo para que la validación
pase.

### Campaña con un tema nuevo

Se necesita una variable que represente el tema y debe revisarse la composición
porcentual. Los porcentajes temáticos actuales suman 100, por lo que incluir
todas las categorías como entradas introduce una dependencia lineal exacta. La
nueva selección debe documentar qué categoría se omite o qué representación se
usa.

Luego corresponde repetir EDA, PCA, Oja y Kohonen, reinterpretar las regiones y
validar la estabilidad antes de proponer candidatos.

## Nuevas variables

### Variable auxiliar

Una variable auxiliar se usa sólo para explicar resultados o tomar decisiones
posteriores. Para incorporarla:

1. documentar definición, unidad y fuente;
2. agregarla al esquema esperado del EDA;
3. comprobar faltantes, rango, identidades y redundancias;
4. agregarla a `RAW_FEATURES` o a las tablas de resumen sólo si será utilizada;
5. mantener sin cambios `BASE_FEATURES`.

No hace falta reentrenar por la variable auxiliar en sí, aunque el pipeline se
regenera si al mismo tiempo se actualizan filas o valores.

### Nueva entrada del modelo

Antes de modificar `BASE_FEATURES` se debe justificar qué información nueva
aporta y qué variable redundante podría reemplazar. Después:

1. actualizar la selección en `data.py` y en el EDA;
2. conservar el mismo orden en PCA, Kohonen y Oja;
3. recalcular el z-score desde el catálogo completo;
4. volver a decidir la cantidad de componentes de PCA;
5. comparar Oja con el nuevo PC1;
6. repetir la configuración final de Kohonen con diez semillas;
7. reinterpretar regiones, baseline y candidatos.

Primero se comparan las configuraciones finalistas conocidas. Sólo se reabre el
barrido completo si esa comparación no recupera una solución estable e
interpretable.

Un cambio de definición o unidad en una columna existente se trata como una
variable nueva. No se comparan directamente métricas históricas que dejaron de
medir lo mismo.

## Criterios de alerta

### Detener la actualización

- faltan columnas, hay IDs duplicados o vacíos, aparecen valores no numéricos o
  no finitos, `followers` es negativo o una entrada queda constante;
- cambió una unidad, definición o identidad algebraica sin documentación;
- alguna campaña queda sin tres candidatos primarios que cumplan el protocolo.

### Revisar interpretación y candidatos

- un candidato final pasa a ser inestable, es decir, conserva su región en no
  más de 5 de las 10 semillas;
- aumentan los empates de región modal o las asignaciones a celdas de
  transición;
- una región desaparece, queda casi vacía o cambia de significado según sus
  variables originales;
- cambia la semántica de PC1 o la cantidad de componentes para alcanzar 90 %;
- Oja deja de converger consistentemente hacia PC1 en las diez semillas;
- las shortlists o candidatos cambian sin que las variables originales
  expliquen el movimiento.

### Reevaluar hiperparámetros

No se reabre automáticamente todo el barrido. El orden es:

1. repetir las configuraciones finalistas con diez semillas;
2. comparar ocupación, matriz U, regiones, estabilidad y coherencia con PCA;
3. conservar la configuración si los cambios son locales y explicables;
4. reabrir el barrido completo sólo si persiste una inestabilidad general, se
   modifica el conjunto de entradas, aparece un tema nuevo o cambia de forma
   sustancial el tamaño del catálogo.

No se elige una configuración únicamente por error de cuantización, ni se fija
un umbral nuevo después de observar un resultado desfavorable.

## Limitaciones que deben conservarse en la comunicación

- No hay etiquetas de ventas, adopciones, consultas ni retorno de inversión.
- Engagement y visitas al perfil son indicadores descriptivos, no conversiones.
- Las sumas de audiencia no descuentan seguidores compartidos entre creadores.
- PC1 y Oja no son rankings universales.
- Una región de Kohonen describe similitud, no causalidad ni éxito futuro.
- Los candidatos fronterizos requieren más cautela ante cada actualización.
- Los resultados dependen del catálogo y del período observado.
