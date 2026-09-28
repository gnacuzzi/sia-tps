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
  Se mantiene training/test sin validation por decisión del equipo. El EDA
  sólo analiza training; `flagged_fraud` se usa únicamente para estratificar
  y contar clases. Los tratamientos del punto 4 aún no están acordados.
