# Estudio comparativo de operadores

Este directorio define un estudio secuencial: en cada etapa se cambia una
familia de operadores y se conservan las decisiones de la etapa anterior. Las
cinco semillas fijas (`101`, `202`, `303`, `404`, `505`) permiten repetir cada
comparación exactamente.

Los resultados crudos se escriben fuera de Git, bajo `.context/`. El analizador
copia automáticamente los CSV resumidos, las decisiones y las figuras finales a
`experiments/results/`, que sí se versiona. Ejecutar desde `tp2` con el entorno
del proyecto activo:

## Ejecución automática completa

Para ejecutar todas las fases, tomar las decisiones numéricas y trasladarlas a
la etapa siguiente sin intervención manual:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_full_study.py
```

El estudio se guarda aisladamente en
`../.context/tp2-colombia-final-study/`. Si se interrumpe, el mismo comando
reanuda las corridas completas. No reutiliza resultados del estudio de Bélgica
ni publica nada en el repositorio: deja una copia preparada para revisar en
`published/`. Para omitir las tres corridas visuales extendidas se puede agregar
`--no-showcase`.

`study-manifest.json` conserva el punto de partida necesario para repetir todas
las campañas. `final-study-manifest.json` congela por separado la configuración
ganadora; no se usa para recalcular las decisiones previas.

## Ejecución manual por fases

Los resultados que ya estaban versionados fueron generados con el protocolo
anterior. Su configuración exacta queda preservada en
`legacy-study-manifest.json`; no deben mezclarse con las nuevas corridas.

```bash
PYTHONPATH=src python scripts/run_study.py profile
PYTHONPATH=src python scripts/analyze_study.py profile

PYTHONPATH=src python scripts/run_study.py resolution
PYTHONPATH=src python scripts/analyze_study.py resolution --figures

PYTHONPATH=src python scripts/run_study.py capacity
PYTHONPATH=src python scripts/analyze_study.py capacity --figures

PYTHONPATH=src python scripts/run_study.py selection
PYTHONPATH=src python scripts/analyze_study.py selection --figures

PYTHONPATH=src python scripts/run_study.py crossover \
  --selected ../.context/tp2-colombia-final-study/decisions/selection.json
PYTHONPATH=src python scripts/analyze_study.py crossover --figures

PYTHONPATH=src python scripts/run_study.py mutation \
  --selected ../.context/tp2-colombia-final-study/decisions/selection.json \
  --selected ../.context/tp2-colombia-final-study/decisions/crossover.json
PYTHONPATH=src python scripts/analyze_study.py mutation --figures

PYTHONPATH=src python scripts/run_study.py survival \
  --selected ../.context/tp2-colombia-final-study/decisions/selection.json \
  --selected ../.context/tp2-colombia-final-study/decisions/crossover.json \
  --selected ../.context/tp2-colombia-final-study/decisions/mutation.json
PYTHONPATH=src python scripts/analyze_study.py survival --figures

PYTHONPATH=src python scripts/run_study.py validation \
  --selected ../.context/tp2-colombia-final-study/decisions/selection.json \
  --selected ../.context/tp2-colombia-final-study/decisions/crossover.json \
  --selected ../.context/tp2-colombia-final-study/decisions/mutation.json \
  --selected ../.context/tp2-colombia-final-study/decisions/survival.json
PYTHONPATH=src python scripts/analyze_study.py validation --figures

PYTHONPATH=src python scripts/run_study.py showcase \
  --selected ../.context/tp2-colombia-final-study/decisions/selection.json \
  --selected ../.context/tp2-colombia-final-study/decisions/crossover.json \
  --selected ../.context/tp2-colombia-final-study/decisions/mutation.json \
  --selected ../.context/tp2-colombia-final-study/decisions/survival.json \
  --validation-records ../.context/tp2-colombia-final-study/records/validation.csv
PYTHONPATH=src python scripts/analyze_study.py showcase --figures
PYTHONPATH=src python scripts/generate_comparative_report.py
```

`showcase` elige, para cada imagen, la semilla con NMSE final mediano entre las
cinco validaciones. Guarda checkpoints cada 500 generaciones y usa 3.000
generaciones para la bandera; 5.000 para señal e ícono. No se usa la mejor
semilla, porque la presentación debe mostrar un resultado representativo.

Los CSV de `summaries/` contienen mediana, cuartiles, tiempo y diversidad. Las
curvas están normalizadas por el NMSE de la generación cero y siempre se basan
en el mejor histórico, no sólo en la última población.

El último comando genera `experiments/results/COMPARATIVE-REPORT.md`. Copia las
imágenes de la semilla mediana de cada condición y deja visibles las secciones
que aún no tienen cinco réplicas completas.

En una ejecución manual, las decisiones de `resolution` y `capacity` deben
aplicarse sobre una copia de trabajo del manifiesto, no sobre
`study-manifest.json`. El runner automático administra esa copia en
`working-manifest.json`. El orden de las corridas de cada fase se mezcla con
`execution_seed`, por lo que es aleatorio pero repetible.
