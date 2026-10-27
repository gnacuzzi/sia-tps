# TP4 - Aprendizaje no supervisado

## Preparación

Desde este directorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

## Análisis disponibles

El EDA inicial no requiere dependencias externas:

```bash
python3 scripts/analyze_creators.py
```

El análisis PCA usa las 15 variables documentadas en `docs/decisiones.md`:

```bash
python3 scripts/analyze_pca.py
```

Genera el informe en `docs/pca-creadores/README.md`, junto con tablas de
varianza, cargas, autovectores, scores y los gráficos correspondientes. El
umbral acordado es 90 %, que en estos datos conserva 8 componentes. Puede
indicarse otro umbral de forma explícita:

```bash
python3 scripts/analyze_pca.py --variance-threshold 0.90
```

La comparación entre engagement total y sus cuatro componentes se reproduce
con:

```bash
python3 scripts/compare_engagement_pca.py
```

El informe comparativo queda en `docs/pca-engagement/README.md`.

La red de Kohonen está implementada en `src/sia_tp4/kohonen.py`. Estandariza
las 15 entradas, permite configurar la grilla y el entrenamiento, y expone el
mapa de hits, la matriz U y los mapas por variable. Cada JSON representa una
sola corrida; la configuración base está en `configs/kohonen/base.json`. Su
informe individual se genera con:

```bash
python3 scripts/analyze_kohonen.py
```

Antes de usar los creadores, la implementación se valida con cuatro grupos
sintéticos conocidos:

```bash
python3 scripts/validate_kohonen.py
```

El resultado reproducible está en `docs/kohonen-sintetico/README.md`.

Los experimentos usan otros JSON indicados mediante `--config`.
Cada nombre de corrida produce por defecto una carpeta diferente dentro de
`docs/kohonen-creadores/`; `--output` permite sobrescribir ese destino. También
puede cambiarse el dataset con `--data`. Su diseño y las decisiones adoptadas
están documentados en `docs/kohonen.md`.

El primer experimento compara `k = 2, ..., 8` con cinco semillas por tamaño:

```bash
python3 scripts/experiment_kohonen_grid.py
```

Cada una de las 35 corridas conserva su propio JSON dentro de
`configs/kohonen/01-grid/`. El informe reproducible queda en
`docs/kohonen-experimentos/01-grid/README.md`. Esta primera etapa dejó `k = 4`
y `k = 5` como candidatos, pero todavía no fijó los hiperparámetros finales.

Para comprobar la interacción entre el tamaño y la tasa se ejecuta:

```bash
python3 scripts/generate_kohonen_rate_configs.py
python3 scripts/experiment_kohonen_rate.py
```

El generador crea un JSON por corrida y el segundo comando compara los 245
casos de `k = 2, ..., 8`, siete tasas constantes y cinco semillas. El informe
queda en `docs/kohonen-experimentos/02-rate/README.md`.

La comparación completa entre tasa constante y disminución lineal al 10 % se
reproduce con:

```bash
python3 scripts/generate_kohonen_decay_configs.py
python3 scripts/experiment_kohonen_rate_schedule.py
```

El informe de las 490 redes queda en
`docs/kohonen-experimentos/03-rate-schedule/README.md`.

El siguiente experimento compara el radio inicial en las regiones útiles de
`k = 4`, `k = 5` y `k = 6`:

```bash
python3 scripts/generate_kohonen_radius_configs.py
python3 scripts/experiment_kohonen_radius.py
```

Son 300 corridas: cuatro perfiles de tasa por tamaño, cinco radios y cinco
semillas. Se registran métricas de calidad y tiempo de entrenamiento, pero se
generan únicamente curvas agregadas para no almacenar cientos de mapas
redundantes. El informe queda en
`docs/kohonen-experimentos/04-radius/README.md`.

El barrido posterior compara seis cantidades de épocas sobre las cinco
configuraciones finalistas:

```bash
python3 scripts/generate_kohonen_epoch_configs.py
python3 scripts/experiment_kohonen_epochs.py
```

Son 150 corridas independientes: `25`, `50`, `100`, `150`, `250` y `400`
épocas, con cinco semillas. El informe queda en
`docs/kohonen-experimentos/05-epochs/README.md`.

La comparación controlada de inicialización se reproduce con:

```bash
python3 scripts/generate_kohonen_initialization_configs.py
python3 scripts/experiment_kohonen_initialization.py
```

Compara `sample` y `uniform` sobre las cinco finalistas, con 150 épocas y cinco
semillas: 50 corridas. El informe queda en
`docs/kohonen-experimentos/06-initialization/README.md`.

El estudio interpretativo de las cuatro candidatas `k = 4` y `k = 5` se
genera con:

```bash
python3 scripts/analyze_kohonen_finalists.py
```

Produce mapas comparables para las 15 variables, variables reconstruidas y
auxiliares, además de asignaciones y perfiles neuronales. También etiqueta las
regiones por tema dominante, colorea con ellas el biplot PC1-PC2 y compara las
distancias entre neuronas vecinas dentro y fuera de esas regiones. El informe
queda en `docs/kohonen-experimentos/07-interpretation/README.md`.

La robustez de la interpretación final se evalúa sin volver a buscar
hiperparámetros. Se entrena únicamente la configuración elegida con diez
semillas y se agregan la estabilidad temática por creador y la frecuencia de
inclusión en las shortlists de las cuatro campañas:

```bash
python3 scripts/analyze_kohonen_stability.py
```

El diseño completo está en `configs/kohonen/stability.json`. El comando genera
un único informe en `docs/kohonen-experimentos/08-stability/`, con tablas y
gráficos agregados y sin una carpeta por semilla.

## Índice mediante la regla de Oja

La implementación de Oja usa las mismas 15 variables y el mismo z-score
muestral que PCA. La selección de hiperparámetros se reproduce en dos etapas.
La primera compara ocho tasas de aprendizaje, dos schedules y diez semillas:

```bash
python3 scripts/experiment_oja_rate.py
```

Su diseño está en `configs/oja/01-rate.json` y el informe se genera en
`docs/oja-experimentos/01-rate/README.md`. La segunda etapa compara ocho
cantidades de épocas para los tres perfiles finalistas:

```bash
python3 scripts/experiment_oja_epochs.py
```

La configuración está en `configs/oja/02-epochs.json` y el informe queda en
`docs/oja-experimentos/02-epochs/README.md`. A partir de esos barridos se elige
un descenso lineal de `0.001` a `0.0001` durante 100 épocas.

El análisis final entrena esa configuración con diez semillas, compara los
pesos aprendidos con el autovector de PC1 y contrasta el índice de cada creador
con su score de PCA:

```bash
python3 scripts/analyze_oja.py
```

El diseño final está en `configs/oja/final.json`. Las métricas, tablas,
gráficos e interpretación del índice quedan en `docs/oja-creadores/README.md`.
El índice describe la dirección de máxima variabilidad de los datos; no es una
función de utilidad ni un ranking automático para cualquier campaña.

## Decisión para las campañas

La síntesis para las cuatro campañas integra los datos originales, las regiones
temáticas completas de las diez redes de Kohonen, PC1-PC2 y el índice de Oja:

```bash
python3 scripts/analyze_campaigns.py
```

El protocolo está en `configs/campaigns/synthesis.json`. Genera un único informe
en `docs/campanas-creadores/` con tablas agregadas y dos gráficos. La región
modal de cada creador define su campaña; no se usa una única neurona ni se crea
un ranking automático. Los casos estables y fronterizos quedan como candidatos
primarios y los inestables permanecen visibles con esa advertencia. Para cada
campaña también compara esos candidatos contra el top global por `followers`
con el mismo `K`, tanto por creador como en forma agregada, e incluye el costo y
dos gráficos comparativos. Finalmente propone tres perfiles complementarios por
campaña —alcance, equilibrio y especialización/eficiencia— mediante umbrales y
ordenamientos explícitos, sin combinar las variables en un score.

La revisión de la decisión comercial conserva esa síntesis como antecedente,
pero deja de usar la región de Kohonen como filtro obligatorio. Exige mayoría
temática individual, estima alcance argentino nuevo y afín, y compara la
selección con followers globales y con followers entre creadores afines:

```bash
python3 scripts/analyze_campaigns_revised.py
```

El informe queda en `docs/campanas-creadores-revisado/`. Kohonen se mantiene
como evidencia descriptiva y de estabilidad, no como una etiqueta comercial.

El procedimiento para incorporar creadores, actualizar el catálogo cada mes y
decidir cuándo revisar campañas, variables o hiperparámetros está documentado
en `docs/mantenimiento.md`.

## Memoria asociativa de Hopfield

El núcleo discreto está implementado en `src/sia_tp4/hopfield.py`. Almacena
patrones bipolares por fila, construye una matriz de pesos simétrica con
diagonal cero y recupera una consulta mediante actualizaciones sincrónicas.
Cada resultado conserva la trayectoria de estados y energías, e informa si
terminó en un estado fijo, en un ciclo o por alcanzar el máximo de iteraciones.

```python
import numpy as np

from sia_tp4.hopfield import fit_hopfield

patterns = np.asarray([[1, 1, -1, -1], [-1, -1, 1, 1]])
model = fit_hopfield(patterns)
result = model.recall(np.asarray([1, -1, -1, -1]))
```

La carga bipolar y el análisis de ortogonalidad de los códigos QR se reproducen
mediante:

```bash
python3 scripts/analyze_hopfield_qr.py
```

El comando valida los cinco PNG, calcula las matrices de Gram cruda y
normalizada, registra el balance de píxeles y prueba la recuperación de cada
QR original después de almacenar los cinco juntos. Luego compara flip uniforme,
borrado uniforme y borrado sesgado hacia los bordes entre 0 % y 100 %, con diez
semillas por nivel aleatorio. El extremo de 100 % se ejecuta una sola vez porque
es determinista. El diseño está en `configs/hopfield/qr.json`; el informe, las
tablas y los gráficos quedan en `docs/hopfield-qr/`.

El barrido de capacidad sin ruido enumera los 31 subconjuntos no vacíos de los
cinco QR y consulta todos los patrones almacenados mediante:

```bash
python3 scripts/analyze_hopfield_qr_capacity.py
```

Su único resguardo numérico está en
`configs/hopfield/qr_capacity.json`. Las 80 consultas, los agregados y los
gráficos quedan en `docs/hopfield-qr-capacidad/`.

La robustez de los cinco conjuntos individuales y las diez duplas que resultan
estables se evalúa con el mismo diseño de ruido mediante:

```bash
python3 scripts/analyze_hopfield_qr_robustness.py
```

El script usa `configs/hopfield/qr_robustness.json` para mantener los tres
ruidos, las diez semillas y el barrido extendido entre 0 % y 100 %. El extremo
de 100 % se ejecuta una sola vez porque allí la consulta es determinista. Sus
tablas y gráficos quedan en `docs/hopfield-qr-robustez/`.

El experimento controlado con letras 5×5 se reproduce mediante:

```bash
python3 scripts/analyze_hopfield_letters.py
```

La configuración está en `configs/hopfield/letters.json` y exige al menos diez
semillas. El script evalúa las 14.950 combinaciones de cuatro letras, conserva
únicamente los extremos del ranking y compara el grupo más favorable con el
más desfavorable bajo los mismos niveles y realizaciones de ruido. El informe,
las tablas y tres gráficos agregados quedan en `docs/hopfield-letras/`.

Para ejecutar las pruebas:

```bash
python3 -m pytest
```
