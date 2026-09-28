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
con la configuración final y tres semillas, además de **38 tests aprobados**
(31 del motor/validación y 7 de carga de datos; revalidado el 28 de septiembre).
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
una separación estricta entre training y test, sin crear validation.

## Continuación del TP

1. Analizar los datos de training y decidir si necesitan transformaciones.
2. Implementar las métricas y variantes de entrenamiento obligatorias.
3. Resolver fraude y dígitos sin utilizar los conjuntos de test para elegir
   parámetros o hiperparámetros.

Los datasets recibidos están en `data/`, conservando sus nombres originales.
Hay diferencias de nombres respecto del enunciado, documentadas en el README
de datos. Los cinco PDFs de clase están en `docs/`. Las transcripciones son
material local excluido del repositorio por decisión del equipo.
