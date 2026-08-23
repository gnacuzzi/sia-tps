# Niveles seleccionados para los experimentos

Esta carpeta contiene copias de los tres niveles elegidos para la comparación
principal. Los archivos originales se mantienen en `levels/` para no romper
las configuraciones y los comandos existentes.

| Nivel | Rol en la comparación | Referencia |
| --- | --- | --- |
| `level_03.txt` | sencillo/intermedio | `levels/level_03.txt` |
| `level_02.txt` | intermedio | `levels/level_02.txt` |
| `level_04.txt` | desafiante pero resoluble | `levels/level_04.txt` |

La opción adicional de tres cajas se conserva como `levels/level_05.txt`, pero
no integra la selección principal porque se superpone con los niveles medios.
Los tableros Aenigma quedan reservados como pruebas de estrés, ya que suelen
terminar por cutoff y no permiten comparar todos los métodos en igualdad de
condiciones.

Para ejecutar los seis casos obligatorios sobre los tres niveles desde la
carpeta `tp1`:

```bash
python3 scripts/run_experiments.py levels/experiments/*.txt \
  --suite core \
  --repetitions 10 \
  --output results/comparacion_principal.csv
```
