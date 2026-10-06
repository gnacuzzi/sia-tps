# Datos originales del TP3

Origen: ZIP `data and documentation.zip` compartido por el usuario.
Los cuatro CSV y `digit_dataset_loader.py` se copiaron sin modificar su contenido.
Se verificó la igualdad SHA-256 entre cada archivo extraído y su entrada del ZIP.
La documentación de fraude se conserva en `../docs/fraud_dataset_documentation.pdf`.

| Archivo recibido | Nombre en la consigna | Uso previsto |
|---|---|---|
| `fraud_dataset.csv` | `fraud_dataset.csv` | Ejercicio 1, fraude |
| `digits.csv` | `digits.csv` | Ejercicio 2, ajuste de parámetros e hiperparámetros |
| `digits_test.csv` | `digits_test.csv` | Evaluación final de generalización, ejercicios 2 y 3 |
| `more_digits.csv` | `more_digits.csv` | Ejercicio 3, datos adicionales |

Los nombres coinciden con la versión actualizada de la consigna.

## Documentación relevante

En fraude, `big_model_fraud_probability` contiene la probabilidad estimada por
BigModel y será el objetivo de la destilación. `flagged_fraud` contiene la
etiqueta real obtenida de reportes: la documentación prohíbe usarla durante el
entrenamiento. No debe incluirse como entrada ni como objetivo de entrenamiento.
El loader de generalización la usa para preservar aproximadamente la proporción
de clases al separar los datos y solo devuelve sus valores de test para la
evaluación final. El loader de aprendizaje inicial usa las 7500 muestras sin
separarlas.
El script de EDA accede a las etiquetas de todas las filas únicamente para
contar el balance de clases. No
cambia la API del loader ni las usa en correlaciones o selección de variables.

Los CSV de dígitos tienen columnas `label` e `image`. El loader suministrado
interpreta `image` con `ast.literal_eval`, la convierte a `float32` y propone
reconstruir imágenes de 28 × 28. Se conserva como material provisto. El loader
integrado en `src/sia_tp3/data.py` hace la misma deserialización sin depender de
pandas, comprueba que cada imagen tenga 784 píxeles y mantiene
`digits_test.csv` separado.

Se implementaron dos protocolos para fraude y uno para dígitos:

- aprendizaje inicial de fraude: las 7500 muestras juntas, sin validation/test;
- generalización de fraude: split configurable y validation temporal, después
  de seleccionar el tipo de perceptrón;
- ejercicio 2: 12449 imágenes de `digits.csv` para training y las 2497 de
  `digits_test.csv` para test;
- ejercicio 3: 28190 filas originales de `digits.csv` más `more_digits.csv`;
  después de eliminar 3689 duplicados exactos quedan 24501 imágenes únicas de
  development, y se conserva el mismo test externo de 2497 imágenes.

El [análisis exploratorio](../docs/eda-training.md) usa las 7500 transacciones
de fraude y `digits.csv`. En aprendizaje, las nueve entradas de fraude se
estandarizan con parámetros de las 7500 muestras; en generalización se ajustan
sólo con training y luego se aplican a validation/test. El objetivo y las
etiquetas no se transforman. Los
píxeles de dígitos se conservan en `[0,1]`. El test queda reservado:
no debe emplearse para escoger parámetros o hiperparámetros. Sus entradas se
usan únicamente en un control de integridad que comprueba si existen muestras
exactamente repetidas entre training y test. Ese control no utiliza etiquetas,
distribuciones ni métricas de test. En el EDA de fraude no hay test porque esa
etapa describe el CSV completo.
