# Datos originales del TP3

Origen: ZIP `data and documentation.zip` compartido por el usuario.
Los cuatro CSV y `digit_dataset_loader.py` se copiaron sin modificar su contenido.
Se verificó la igualdad SHA-256 entre cada archivo extraído y su entrada del ZIP.
La documentación de fraude se conserva en `../docs/fraud_dataset_documentation.pdf`.

| Archivo recibido | Nombre en la consigna | Uso previsto |
|---|---|---|
| `fraud_dataset.csv` | `transactions.csv` | Ejercicio 1, fraude |
| `digits.csv` | `digits.csv` | Ejercicio 2, ajuste de parámetros e hiperparámetros |
| `digits_test.csv` | `digits_test.csv` | Evaluación final de generalización, ejercicios 2 y 3 |
| `more_digits.csv` | `more_data_digits.csv` | Ejercicio 3, datos adicionales |

Las correspondencias de nombres distintos son inferencias por contenido y por
el conjunto de archivos recibido, no equivalencias declaradas en el PDF.
Se conservan los nombres originales para no ocultar esa diferencia.

## Documentación relevante

En fraude, `big_model_fraud_probability` contiene la probabilidad estimada por
BigModel y será el objetivo de la destilación. `flagged_fraud` contiene la
etiqueta real obtenida de reportes: la documentación prohíbe usarla durante el
entrenamiento. No debe incluirse como entrada ni como objetivo de entrenamiento.
El loader nuevo la usa para preservar aproximadamente la proporción de clases
al separar los datos y solo devuelve sus valores de test para la futura
evaluación final.
El script de EDA, por acuerdo posterior del equipo, accede a las etiquetas
de las filas de training únicamente para contar el balance de clases. No
cambia la API del loader ni las usa en correlaciones o selección de variables.

Los CSV de dígitos tienen columnas `label` e `image`. El loader suministrado
interpreta `image` con `ast.literal_eval`, la convierte a `float32` y propone
reconstruir imágenes de 28 × 28. Se conserva como material provisto. El loader
integrado en `src/sia_tp3/data.py` hace la misma deserialización sin depender de
pandas, comprueba que cada imagen tenga 784 píxeles y mantiene
`digits_test.csv` separado.

Se implementó únicamente una separación training/test, sin validation:

- fraude: 6000 muestras de training y 1500 de test con la configuración por
  defecto (`test_fraction=0.2`, `seed=0`);
- ejercicio 2: 12449 imágenes de `digits.csv` para training y las 2497 de
  `digits_test.csv` para test;
- ejercicio 3: 28190 imágenes de `digits.csv` más `more_digits.csv` para
  training y el mismo test externo de 2497 imágenes.

El [análisis exploratorio de training](../docs/eda-training.md) está realizado
para fraude y `digits.csv`. Todavía no se aplicaron normalización,
estandarización ni codificación de las etiquetas. El test queda reservado:
no debe emplearse para escoger parámetros o hiperparámetros. Sus entradas se
usan únicamente en un control de integridad que comprueba si existen muestras
exactamente repetidas entre training y test. Ese control no utiliza etiquetas,
distribuciones ni métricas de test; se conserva la estratificación de fraude
ya definida.
