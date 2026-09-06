# TP2 — Algoritmos Genéticos

Motor de algoritmos genéticos que aproxima una imagen con `T` triángulos RGBA sobre un canvas
de color fijo. Enunciado completo en `docs/SIA - TP2 - 2026 2Q.pdf`.

El ejercicio 1 (representación ASCII, solo teórico) está resuelto en `docs/ej1.md`.
El ejercicio 2 (triángulos) es todo el código de `src/sia_tp2/`.

## Reglas del enunciado que condicionan el código

- **No se pueden usar librerías de algoritmos genéticos.** Selección, cruza, mutación,
  supervivencia y criterios de corte se implementan a mano. Sí se permiten librerías de imagen
  (Pillow) y numérica (numpy) para renderizar y medir error.
- Los únicos *parámetros* del problema son la imagen y la cantidad de triángulos. Todo lo demás
  son *hiperparámetros* y viven en el JSON de configuración.
- Requisitos mínimos de implementación: los 7 métodos de selección (elite, ruleta, universal,
  Boltzmann, torneo determinístico, torneo probabilístico, ranking), las 2 estrategias de
  supervivencia (aditiva y exclusiva), al menos 2 métodos de cruza y al menos 2 de mutación.
- Salidas obligatorias de una corrida: imagen generada, enumeración de triángulos y métricas.

## Comandos

Desde `tp2/`. Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
pytest                                            # 31 tests
sia-tp2 --config configs/default.json inspect-config   # valida e imprime config efectiva
sia-tp2 --config configs/smoke.json render-random      # baseline: solo población inicial
sia-tp2 --config configs/smoke.json evolve             # corrida evolutiva completa
```

Sin instalar: `PYTHONPATH=src python3 -m sia_tp2 --config configs/smoke.json evolve`.

`configs/smoke.json` y `configs/phase3-smoke.json` son corridas chicas (segundos) para validar
cambios. `configs/default.json` es la configuración de referencia.

Cada corrida escribe en `output/runs/<modo>-seed-<n>/` (gitignoreado): `config.effective.json`,
`metadata.json`, `metrics.csv`, `triangles.json`, `best.png` y `checkpoints/`.

## Arquitectura

```
src/sia_tp2/
  config.py            validación estricta del JSON → AppConfig (dataclasses)
  cli.py               argparse: inspect-config | render-random | evolve
  workflow.py          arma los closures del dominio y los inyecta en el motor
  domain/              todo lo específico de triángulos e imágenes
    model.py           TriangleGene, Individual (frozen, se validan solos)
    renderer.py        genotipo → fenotipo (Pillow, alpha compositing en orden de locus)
    fitness.py         NMSE y fitness = max(epsilon, 1 - NMSE)
    initialization.py  población inicial aleatoria reproducible
    operators.py       cruza y mutación sobre cromosomas de triángulos
    diversity.py       diversidad genotípica en [0, 1]
  ga/                  motor genérico, sin conocimiento del dominio
    engine.py          loop evolutivo, métricas por generación, criterios de corte
    selection.py       los 7 selectores + dispatcher select_population
    survival.py        supervivencia aditiva y exclusiva
  reporting/
    serialization.py   artefactos de la corrida (JSON, CSV, PNG)
```

**Invariante central: `ga/` no importa `domain/`.** El motor es genérico sobre `T` y recibe
`evaluate`, `error`, `fitness`, `select_parents`, `crossover`, `mutate`, `survive` y `diversity`
como callables. `workflow.py` es el único lugar donde se unen motor y dominio. Si un cambio
necesita que `ga/` sepa qué es un triángulo, el cambio está mal planteado.

Otros invariantes:

- `TriangleGene` e `Individual` son `frozen` y validan en `__post_init__`: vértices en `[0, 1]`,
  área estrictamente positiva, RGBA en `[0, 255]`, `A > 0`. Un individuo inválido no se puede
  construir; no hay estados intermedios inválidos.
- Todo el azar pasa por un único `random.Random(seed)` creado en el motor y propagado como
  argumento. Nada usa `random` global ni `numpy.random`. Misma seed + misma config = misma corrida.
- El error está en `[0, 1]` y el fitness en `(0, 1]`. Los selectores maximizan fitness; el motor
  guarda el mejor por menor error.
- El `best` reportado es el *best-so-far* global, no el mejor de la última población.
- El orden de los genes es parte de la solución: con alfa < 255, permutar dos triángulos cambia
  el fenotipo. Los operadores respetan loci.

## Convenciones

- **Código y docstrings en inglés; documentación y commits del contenido en español.** Un
  docstring de una línea por función pública, en imperativo. Sin comentarios inline salvo que
  expliquen algo no obvio (ver el clamp de `diversity.py`).
- Argumentos keyword-only (`*`) en casi todas las funciones públicas. Errores de dominio como
  `ValueError` con mensaje en minúscula.
- Tipos con `typing` compatible con Python 3.9 (`Tuple`, `Optional`, no `list[...]` ni `X | Y`).
- Los tests viven en `tests/test_<modulo>.py` y usan seeds fijas. Todo operador nuevo necesita
  test de determinismo y test de validez del resultado.

## Agregar un operador o hiperparámetro

Hay que tocar cinco lugares, y falta uno rompe la corrida o la documentación:

1. Implementación en `ga/selection.py` (si es genérica) o `domain/operators.py` (si toca genes).
2. Alta en el dispatcher correspondiente (`select_population`, `crossover_individuals`,
   `mutate_individual`).
3. Validación en `config.py`: agregar el valor al whitelist y validar sus `params` exactos
   (claves desconocidas = error).
4. Documentar el campo en `docs/config-contract.md` con tipo, rango y validaciones cruzadas.
5. Tests en `tests/`.

Si la decisión tiene un porqué de diseño, va también a `docs/decisions.md`.

## Configuración

`docs/config-contract.md` es la fuente de verdad del formato. El JSON es estricto: todos los
campos son obligatorios (incluso con valor `null`), las claves desconocidas fallan, las rutas
relativas se resuelven desde `tp2/` y la validación acumula todos los errores antes de abortar.
No inventar defaults escondidos: si un campo hace falta, se agrega al contrato.

## Estado actual

Implementado: los 7 métodos de selección, supervivencia aditiva y exclusiva, cruza de un punto y
uniforme, mutación de un gen y multigen uniforme (con modos de alelo `local_delta` y
`global_resample`), NMSE + fitness, diversidad, los cuatro criterios de corte
(`max_generations`, `target_nmse`, estancamiento, `max_seconds`), artefactos de corrida y CLI.

Pendiente: scripts de experimentación y gráficos, README de ejecución completo, presentación, y
las decisiones abiertas listadas al final de `docs/decisions.md`.
