# Estudio comparativo de operadores

Este directorio define un estudio secuencial: en cada etapa se cambia una
familia de operadores y se conservan las decisiones de la etapa anterior. Las
cinco semillas fijas (`101`, `202`, `303`, `404`, `505`) permiten repetir cada
comparación exactamente.

Los resultados crudos se escriben fuera de Git, bajo `.context/`. El analizador
copia automáticamente los CSV resumidos, las decisiones y las figuras finales a
`experiments/results/`, que sí se versiona. Ejecutar desde `tp2` con el entorno
del proyecto activo:

```bash
PYTHONPATH=src python scripts/run_study.py profile
PYTHONPATH=src python scripts/analyze_study.py profile

PYTHONPATH=src python scripts/run_study.py selection
PYTHONPATH=src python scripts/analyze_study.py selection --figures

PYTHONPATH=src python scripts/run_study.py crossover \
  --selected ../.context/tp2-comparative-study/decisions/selection.json
PYTHONPATH=src python scripts/analyze_study.py crossover --figures

PYTHONPATH=src python scripts/run_study.py mutation \
  --selected ../.context/tp2-comparative-study/decisions/selection.json \
  --selected ../.context/tp2-comparative-study/decisions/crossover.json
PYTHONPATH=src python scripts/analyze_study.py mutation --figures

PYTHONPATH=src python scripts/run_study.py validation \
  --selected ../.context/tp2-comparative-study/decisions/selection.json \
  --selected ../.context/tp2-comparative-study/decisions/crossover.json \
  --selected ../.context/tp2-comparative-study/decisions/mutation.json
PYTHONPATH=src python scripts/analyze_study.py validation --figures

PYTHONPATH=src python scripts/run_study.py showcase \
  --selected ../.context/tp2-comparative-study/decisions/selection.json \
  --selected ../.context/tp2-comparative-study/decisions/crossover.json \
  --selected ../.context/tp2-comparative-study/decisions/mutation.json \
  --validation-records ../.context/tp2-comparative-study/records/validation.csv
PYTHONPATH=src python scripts/analyze_study.py showcase --figures
```

`showcase` elige, para cada imagen, la semilla con NMSE final mediano entre las
cinco validaciones. Guarda checkpoints cada 500 generaciones y usa 3.000
generaciones para la bandera; 2.000 para señal e ícono. No se usa la mejor
semilla, porque la presentación debe mostrar un resultado representativo.

Los CSV de `summaries/` contienen mediana, cuartiles, tiempo y diversidad. Las
curvas están normalizadas por el NMSE de la generación cero y siempre se basan
en el mejor histórico, no sólo en la última población.
