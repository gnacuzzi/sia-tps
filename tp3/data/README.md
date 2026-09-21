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
Su uso posterior para evaluación y elección de umbral requiere definir primero
el protocolo de generalización.

Los CSV de dígitos tienen columnas `label` e `image`. El loader suministrado
interpreta `image` con `ast.literal_eval`, la convierte a `float32` y propone
reconstruir imágenes de 28 × 28. Se conserva como material provisto; todavía
no fue integrado ni ejecutado. Su ejemplo usa una ruta relativa a la carpeta
de trabajo y requiere NumPy, pandas y Matplotlib.

Hasta ahora se revisaron encabezados, documentación y loader, y se contaron
registros. No se hizo aún el análisis exploratorio de valores faltantes,
rangos, duplicados, balance de clases o dimensiones de todas las imágenes.
Los datos no se transformaron ni se dividieron en subconjuntos.
