# TP3 — Perceptrón simple y multicapa

Aplican las reglas generales del `CLAUDE.md` de la raíz.

- Consigna: `docs/Enunciado TP3 - 2Q 2026.pdf`.
- Desarrollo actual: `docs/validacion.md`.
- El alcance incluye el motor Python reutilizable y su validación ejecutable.
  Código en `src/sia_tp3/`, config en `configs/validation.json`, tests en `tests/`.
  No confundir los pesos manuales de la guía con los modelos aprendidos.
- Ejecución desde `tp3/`: `pytest -q` y
  `sia-tp3 --config configs/validation.json --output output/<corrida-nueva>`.
  El registro real está en `docs/resultados-validacion.md`.
- Conservar la codificación bipolar y el orden de muestras del enunciado.
- Las teóricas están en `docs/`. Las transcripciones, si están disponibles
  localmente, viven en `docs/transcripciones/`. Por pedido del equipo se
  versionan las de clases 12.2 y 13; las anteriores siguen excluidas de Git.
  Consultar `docs/material-clases.md` para ubicar cada tema y las diferencias
  de notación. Se contrastaron las fórmulas de validación con las clases.
- Documentar decisiones y separar cálculos ilustrativos de resultados medidos,
  siguiendo el criterio de los TPs anteriores.
- Por pedido explícito del usuario, registrar **todas** las decisiones en
  `docs/decisiones.md`: qué se decidió, por qué, cómo/dónde se aplicó y estado.
  Explicarlas también al presentar avances. Distinguir requisitos de la
  consigna, elecciones propias y propuestas pendientes.
- Datos originales en `data/`; documentación de fraude en `docs/`.
  `flagged_fraud` no puede usarse para entrenar, según esa documentación.
- No comenzar opcionales antes de completar los ejercicios obligatorios.
- Punto 3: EDA en `scripts/analyze_training.py`, informe en `docs/eda-training.md`.
  El EDA de fraude analiza las 7500 muestras, igual que la comparación inicial,
  sin validation/test. Para los futuros experimentos
  Después de seleccionar el perceptrón, la generalización deriva validation
  temporalmente desde training. Ambos protocolos están implementados en
  `experiments.py`; falta integrar validation en los runners de dígitos. En
  dígitos el EDA usa entradas de test únicamente para detectar solapamientos
  exactos. Ese control no usa etiquetas ni métricas de test.
  `flagged_fraud` se usa únicamente para contar clases en el EDA y para
  estratificar el protocolo posterior de generalización. Se decidió
  conservar variables, outliers y píxeles constantes en esta etapa.
- Punto 4: `load_fraud_train_test` estandariza las nueve entradas después del
  split. `Standardizer` se ajusta sólo con `X_train` y transforma también
  `X_test` con esos parámetros; objetivos y etiquetas quedan intactos. Dígitos
  conserva sus píxeles en `[0,1]`.
- Punto 5: `metrics.py` recibe clases enteras reales/predichas y no elige umbral
  ni aplica `argmax`. Matriz con filas reales/columnas predichas; métricas
  one-vs-rest por clase y macro; divisiones indefinidas se expresan como `NaN`.
- Punto 6: `optimizers.py` implementa descenso básico, Momentum, eta adaptativo,
  RMSProp y Adam según clase 12.1. `fit` mantiene independiente `batch_size`:
  1 online, intermedio mini-batch y `None`/N batch. Rosenblatt escalón permanece
  online con descenso básico. La implementación no selecciona todavía la mejor
  combinación para fraude o dígitos.
- Ejercicio de fraude: `configs/fraud-learning.json` declara una comparación
  lineal/logística con todas las muestras. `fraud_experiment.py` separa los
  protocolos `learning` y `generalization`; este último exige un único modelo
  seleccionado y mantiene test reservado. La corrida histórica
  `output/fraud-baseline-01` fue una sanidad con split y no responde por sí sola
  la comparación inicial exigida.
