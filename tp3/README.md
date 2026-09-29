# TP3 — Perceptrón simple y multicapa

Sistemas de Inteligencia Artificial, ITBA, segundo cuatrimestre de 2026.

## Material

- [Guía visual interactiva](validacion-visual.html): abrir en el navegador para leer las explicaciones y explorar los gráficos.
- [Enunciado](docs/Enunciado%20TP3%20-%202Q%202026.pdf).
- [Validación: fórmulas y cálculos paso a paso](docs/validacion.md).
- [Decisiones: qué elegimos, por qué y cómo se aplica](docs/decisiones.md).
- [Datos originales y correspondencia con la consigna](data/README.md).
- [Documentación del dataset de fraude](docs/fraud_dataset_documentation.pdf).
- [Clases: índice, referencias y criterios para comenzar](docs/material-clases.md).
- [Implementación y API reutilizable](docs/implementacion.md).
- [Resultados de validación y gráficos de entrenamiento](docs/resultados-validacion.md).
- [Análisis exploratorio de training: hallazgos y decisiones pendientes](docs/eda-training.md).

Seguimos la organización de los TPs anteriores: una carpeta independiente
por TP, documentación en español y material fuente dentro de `docs/`.

## Alcance actual

Implementados los cuatro modelos y el ejercicio de validación de la página 2:

| Modelo | Caso de validación |
|---|---|
| Perceptrón simple escalón | AND con entradas y salidas bipolares |
| Perceptrón simple lineal | Ajustar muestras de `y = x` |
| Perceptrón simple no lineal | Ajustar muestras de `y = tanh(x)` |
| Perceptrón multicapa | XOR, arquitecturas `[2, 2, 1]` y `[2, 3, 2, 1]` |

Estos ejercicios **no se presentan**, según la consigna: sirven para verificar
las herramientas antes de trabajar con los datos de los ejercicios obligatorios.
Los ejemplos del apunte y de la guía interactiva son cálculos didácticos.
Las corridas reales están en el informe de resultados: **15/15 aprobadas**
con la configuración final y tres semillas. También hay pruebas automatizadas
para el motor, la carga de datos y el EDA, incluido el control de solapamiento
exacto entre las entradas de training y test (con el extra `plot`).
También se conservan los fallos de la configuración inicial de XOR.

## Instalar y ejecutar

Desde `tp3/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,plot]'
pytest -q
sia-tp3 --config configs/validation.json --output output/mi-validacion
python scripts/plot_validation.py output/mi-validacion --output output/mi-validacion/plots
```

La carpeta de salida del entrenamiento debe ser nueva o vacía. El comando
retorna 0 si todos los casos cumplen los criterios, 1 si alguno no converge
y 2 ante errores de configuración o ejecución. No descarta semillas fallidas.

También puede ejecutarse sin instalar el paquete, teniendo NumPy disponible:

```bash
PYTHONPATH=src python3 -m sia_tp3 --config configs/validation.json --output output/otra-validacion
```

`src/sia_tp3/models.py` y `training.py` son el motor reutilizable; no conocen
AND, XOR, fraude ni dígitos. `validation.py` crea los casos sintéticos y usa
esa misma API. Las redes admiten múltiples entradas y salidas, y permiten
guardar y cargar parámetros. `data.py` carga los datasets reales y conserva
una separación estricta entre training y test, sin crear validation. Para
fraude, `preprocessing.py` ajusta la estandarización sólo con training y el
loader aplica esos mismos parámetros a training y test.

## Continuación del TP

1. Usar la estandarización de fraude ya implementada y mantener dígitos en `[0,1]`.
2. Usar las métricas de clasificación implementadas y completar las variantes
   de entrenamiento obligatorias.
3. Resolver fraude y dígitos sin utilizar los conjuntos de test para elegir
   parámetros o hiperparámetros.

Los datasets recibidos están en `data/`, conservando sus nombres originales.
Hay diferencias de nombres respecto del enunciado, documentadas en el README
de datos. Los ocho PDFs de clase están en `docs/`. Se versionan las
transcripciones de clases 12.2 y 13; las anteriores permanecen locales.

## Análisis de training (punto 3)

```bash
python scripts/analyze_training.py --output output/eda-nueva
```

Requiere el extra `plot` y una carpeta de salida nueva o vacía. Genera tablas
CSV, seis gráficos, un resumen Markdown y configuración/versiones/hashes en
JSON. Usa el split vigente de fraude (test 20 %, semilla 0) y `digits.csv` para
los estadísticos. Sólo lee las entradas de test para comprobar duplicados
exactos entre particiones; no usa sus etiquetas, distribuciones ni métricas, y
no abre los datos adicionales. Mantiene training/test sin validation y no
aplica preprocesamiento. La evidencia versionada está en
[`docs/eda-training/`](docs/eda-training/resumen.md).

## Preprocesamiento (punto 4)

`load_fraud_train_test` separa primero los datos, calcula una media y un desvío
por cada una de las nueve entradas usando sólo `X_train`, y aplica esos mismos
parámetros a `X_train` y `X_test`. El objetivo BigModel y `flagged_fraud` no se
transforman. El `Standardizer` devuelto en `data.standardizer` puede guardarse
como NPZ junto con el futuro modelo para reproducir exactamente las predicciones.

Los dígitos conservan sus píxeles en el rango original `[0,1]`; no se dividen
nuevamente por 255 ni se estandarizan en esta etapa.

## Métricas de clasificación (punto 5)

`classification_metrics(y_true, y_pred, labels=...)` recibe clases enteras ya
decididas y devuelve matriz de confusión, accuracy, precision, recall, F1, TPR
y FPR por clase, además de promedios macro. Las filas de la matriz son clases
reales y las columnas son predicciones; `labels` es explícito para que una clase
ausente, como el 8 en training, no desaparezca del informe.

La conversión a clase queda fuera del módulo: fraude aplicará el umbral que se
decida posteriormente y dígitos usará `argmax`. Las divisiones sin denominador
se informan como `NaN`, porque esa métrica no puede evaluarse con esas muestras.
