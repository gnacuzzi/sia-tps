# EDA de training: resumen generado

Generado por `scripts/analyze_training.py`. No se entrenan modelos ni se transforman datos.

- Fraude: 6000 filas de training; test_fraction=0.2, seed=0.
- Dígitos: 12449 imágenes de 784 píxeles.
- No se abren `digits_test.csv` ni `more_digits.csv`.
- En fraude se leen las etiquetas para reproducir el split, pero sólo se analizan filas de training.
- Estadísticos sobre valores finitos; faltantes/NaN e infinitos se cuentan por separado.
- Desvío descriptivo con ddof=0; cuartiles/percentiles con interpolación lineal de NumPy.
- Correlaciones por pares finitos; duplicados de fraude sólo entre filas numéricas completas.
- El loader de dígitos comprueba columnas, etiquetas 0..9 e imágenes finitas de 784 píxeles;
  interrumpe el análisis ante datos inválidos, sin corregirlos ni omitirlos.
- Histogramas: 30 intervalos para fraude, 40 para píxeles. Son opciones gráficas, no transformaciones.
- Regla IQR: candidatos fuera de Q1−1,5×IQR y Q3+1,5×IQR; no implica datos erróneos.
- Chequeos semánticos de fraude: no negativos; cantidades/resolución positivas; enteros según documentación;
  BigModel en [0,1]. Son alertas para revisar, no filtros aplicados.

## Fraude: distribución y escala

| Variable | Mínimo | Mediana | Máximo | Desvío | Faltantes/NaN | Infinitos | Candidatos IQR |
|---|---:|---:|---:|---:|---:|---:|---:|
| timestamp | 1.7e+09 | 1.7158e+09 | 1.73153e+09 | 9.19748e+06 | 0 | 0 | 0 |
| amount_usd | 1 | 63.82 | 2000 | 151.786 | 0 | 0 | 440 |
| quantity_purchased | 1 | 5 | 24 | 4.16607 | 0 | 0 | 272 |
| session_duration_seconds | 5 | 286.9 | 726.8 | 135.589 | 0 | 0 | 5 |
| days_since_last_purchase | 0 | 8.51 | 142.32 | 14.9382 | 0 | 0 | 291 |
| account_age_days | 1 | 1640.5 | 3649 | 1124.51 | 0 | 0 | 0 |
| device_screen_resolution | 1.00673e+06 | 2.07326e+06 | 8.31094e+06 | 2.66782e+06 | 0 | 0 | 1140 |
| time_since_last_login_s | 10 | 2484.5 | 40160.8 | 3560.8 | 0 | 0 | 282 |
| items_viewed_before_purchase | 1 | 8 | 29 | 5.40398 | 0 | 0 | 140 |
| big_model_fraud_probability | 0.000897 | 0.359399 | 1 | 0.302384 | 0 | 0 | 0 |

## Fraude: balance y duplicados

| Clase | Cantidad | Porcentaje |
|---|---:|---:|
| 0 | 5305 | 88.42% |
| 1 | 695 | 11.58% |

Entradas duplicadas: 0 grupos, 0 filas involucradas,
0 copias adicionales y 0 grupos con objetivos distintos.

## Dígitos: balance y duplicados

| Clase | Cantidad | Porcentaje |
|---|---:|---:|
| 0 | 1480 | 11.89% |
| 1 | 1685 | 13.54% |
| 2 | 1489 | 11.96% |
| 3 | 1532 | 12.31% |
| 4 | 1460 | 11.73% |
| 5 | 271 | 2.18% |
| 6 | 1479 | 11.88% |
| 7 | 1566 | 12.58% |
| 8 | 0 | 0.00% |
| 9 | 1487 | 11.94% |

Entradas duplicadas: 0 grupos, 0 filas involucradas,
0 copias adicionales y 0 grupos con objetivos distintos.

Píxeles constantes: 97; constantes en cero: 97.
Clases ausentes en training: [8]. Píxeles en cero: 81.27%.
Imágenes con todos los píxeles iguales: 0; completamente en cero: 0.
Rango observado de píxeles: [0.0, 1.0].

## Gráficos

![fraud-histograms](fraud-histograms.png)

![fraud-boxplots](fraud-boxplots.png)

![fraud-correlations](fraud-correlations.png)

![digits-distributions](digits-distributions.png)

![digits-pixel-maps](digits-pixel-maps.png)

![digits-examples](digits-examples.png)

Los CSV conservan el detalle completo; `summary.json` registra configuración, versiones y hashes.
Las decisiones de tratamiento requieren revisar estos resultados; no se aplican automáticamente.
