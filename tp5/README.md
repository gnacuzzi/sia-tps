# TP5 - Autoencoders

Este trabajo implementa autoencoders desde cero con NumPy. Incluye un
autoencoder básico para reconstruir caracteres de una fuente de `7x5`, un
denoising autoencoder (DAE) que recupera patrones afectados por ruido *bit
flip* y un autoencoder variacional (VAE) que genera funciones simples de
Python.

## Preparación

Desde este directorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

La implementación requiere Python 3.9 o posterior. Los modelos se entrenan con
NumPy; Matplotlib se utiliza para generar las figuras.

## Ejercicio 1A - Autoencoder básico

La corrida final usa una arquitectura `35-45-2-45-35`, pérdida BCE y descenso
de gradiente. Reconstruye los 32 caracteres de la fuente y representa cada uno
en un espacio latente bidimensional:

```bash
python3 -m sia_tp5.experiment configs/ae_final_bce_width45_seed0.json
```

La interpolación que recorre el espacio latente entre dos caracteres se
reproduce con:

```bash
python3 -m sia_tp5.latent_interpolation configs/latent_interpolation_ax.json
```

El recorrido experimental, la comparación de arquitecturas, optimizadores y
funciones de pérdida se documentan en
[`docs/ejercicio_1a.md`](docs/ejercicio_1a.md).

## Ejercicio 1B - Denoising autoencoder

Cada configuración del DAE puede ejecutarse con:

```bash
python3 -m sia_tp5.denoising_experiment <configuracion.json>
```

Las curvas para distintos niveles de ruido se generan con:

```bash
python3 -m sia_tp5.denoising_curve <configuracion.json>
```

El análisis agregado de la configuración final se reproduce con:

```bash
python3 -m sia_tp5.denoising_final_analysis configs/dae_final_analysis.json
```

El estudio completo y el índice de artefactos publicados están en
[`docs/ejercicio_1b.md`](docs/ejercicio_1b.md) y
[`docs/dae/README.md`](docs/dae/README.md), respectivamente.

## Ejercicio 2 - Autoencoder variacional

El ejercicio compara primero un autoencoder determinista con distintas
capacidades y luego entrena un VAE sobre funciones simples de Python:

```bash
python3 -m sia_tp5.code_ae_study configs/code_ae_capacity.json
python3 -m sia_tp5.vae_code_study configs/vae_code_study.json
```

El dataset se encuentra en `data/python_functions.csv`. Su construcción, la
selección del peso de regularización KL y el análisis de las muestras generadas
se explican en [`docs/ejercicio_2.md`](docs/ejercicio_2.md).

## Pruebas

La suite valida, entre otros componentes, los gradientes, optimizadores,
carga de datos, ruido, experimentos y el VAE:

```bash
python3 -m pytest
```

## Documentación y entrega

- [`docs/ejs.md`](docs/ejs.md): síntesis completa de los tres ejercicios y sus
  resultados.
- [`docs/guia_conceptual_tp5.md`](docs/guia_conceptual_tp5.md): guía de los
  conceptos y del flujo de entrenamiento.
- [`docs/casos_borde_y_decisiones_tp5.md`](docs/casos_borde_y_decisiones_tp5.md):
  decisiones de implementación y casos borde.
- [`docs/enunciadoTP5.pdf`](docs/enunciadoTP5.pdf): consigna del trabajo.
- [`docs/TP5_SIA_GRUPO_10.pptx`](docs/TP5_SIA_GRUPO_10.pptx): presentación
  final.

Los JSON de `configs/` fijan las arquitecturas, hiperparámetros y semillas de
cada corrida. Los documentos de cada ejercicio indican cuáles corresponden a
los barridos intermedios y cuáles a los modelos finales.
