# Contrato de configuración

## Principios

- El archivo se interpreta como JSON estricto y usa `config_version: 1`.
- Todos los campos del ejemplo son obligatorios, incluso cuando su valor sea `null`. Esto hace visible la configuración efectiva y evita defaults importantes escondidos.
- Las claves desconocidas producen un error de validación.
- Las rutas relativas se resuelven desde la raíz de `tp2`.
- Antes de ejecutar se valida todo el archivo y se informan juntos los errores encontrados.
- Cada corrida guarda una copia exacta de la configuración efectiva junto con sus resultados.
- El código no modifica el archivo de configuración recibido.

## Configuración completa

```json
{
  "config_version": 1,
  "input": {
    "image": "assets/targets/01_simple.png",
    "working_max_side": 64
  },
  "representation": {
    "triangle_count": 20,
    "canvas_rgb": [255, 255, 255],
    "alpha_range": [1, 255],
    "initialization": "uniform_random"
  },
  "genetic": {
    "population_size": 100,
    "offspring_count": 100,
    "parent_selection": {
      "method": "tournament_deterministic",
      "params": {
        "tournament_size": 3
      }
    },
    "crossover": {
      "method": "uniform",
      "probability": 0.9,
      "params": {
        "swap_probability": 0.5
      }
    },
    "mutation": {
      "method": "multigene_uniform",
      "probability": 0.05,
      "allele_change": {
        "mode": "local_delta",
        "position_delta": 0.08,
        "color_delta": 20,
        "alpha_delta": 20
      }
    },
    "survival": {
      "strategy": "additive",
      "selection": {
        "method": "elite",
        "params": {}
      }
    }
  },
  "fitness": {
    "metric": "normalized_mse",
    "epsilon": 1e-12
  },
  "termination": {
    "max_generations": 1000,
    "target_nmse": null,
    "stagnation": {
      "patience": 100,
      "min_improvement": 1e-6
    },
    "max_seconds": null
  },
  "run": {
    "seed": 0
  },
  "output": {
    "directory": "output/runs",
    "metrics_every": 1,
    "checkpoint_every": 100,
    "render_original_size": true
  }
}
```

## Campos y validaciones

### `config_version`

- Tipo: entero.
- Único valor admitido actualmente: `1`.

### `input`

#### `image`

- Tipo: string no vacío.
- Debe resolver a un archivo existente y legible.
- Formatos iniciales permitidos: PNG, JPEG y WebP aceptados por Pillow.
- La imagen se convierte explícitamente a RGB.

#### `working_max_side`

- Tipo: entero mayor que `0`.
- El lado mayor de la imagen de trabajo toma este valor.
- El otro lado se calcula preservando la relación de aspecto y debe resultar al menos `1` píxel.
- La imagen nunca se deforma para forzarla a un cuadrado.

### `representation`

#### `triangle_count`

- Tipo: entero mayor que `0`.
- Es la cantidad exacta de genes de todo cromosoma.

#### `canvas_rgb`

- Tipo: arreglo de exactamente tres enteros.
- Cada canal debe estar entre `0` y `255` inclusive.
- El canvas es opaco; el alfa pertenece a los triángulos, no al fondo.

#### `alpha_range`

- Tipo: arreglo `[min, max]` de dos enteros.
- Debe cumplir `1 <= min <= max <= 255`.
- No se permite inicialmente un triángulo completamente transparente.

#### `initialization`

- Tipo: string.
- Único valor inicial: `"uniform_random"`.
- Los vértices y canales se muestrean dentro de sus dominios. Un triángulo degenerado se vuelve a generar.

### `genetic`

#### `population_size`

- Tipo: entero mayor o igual que `2`.
- Define `P`, la cantidad de individuos de toda generación.

#### `offspring_count`

- Tipo: entero positivo y par.
- Define `K`, la cantidad de hijos generados por generación.
- Se exige par porque los operadores iniciales toman dos padres y generan dos hijos.

#### Objetos de selección

`parent_selection` y `survival.selection` usan el mismo contrato:

```json
{
  "method": "nombre",
  "params": {}
}
```

Métodos y parámetros admitidos:

| Método | `params` |
| --- | --- |
| `elite` | `{}` |
| `roulette` | `{}` |
| `universal` | `{}` |
| `ranking` | `{}` |
| `boltzmann` | `initial_temperature`, `final_temperature`, `decay_rate` |
| `tournament_deterministic` | `tournament_size` |
| `tournament_probabilistic` | `threshold` |

Validaciones específicas:

- `initial_temperature` y `final_temperature`: números finitos mayores que `0`.
- `decay_rate`: número finito mayor o igual que `0`.
- La temperatura por generación es `T(t) = final + (initial - final) x exp(-decay_rate x t)`.
- `tournament_size`: entero mayor o igual que `2` y no mayor que el conjunto del que se selecciona.
- `threshold`: número finito entre `0.5` y `1` inclusive.
- Los métodos sin parámetros exigen exactamente `params: {}`.
- Los métodos de selección deben devolver exactamente la cantidad solicitada y pueden seleccionar un individuo más de una vez cuando el método lo permita.

#### `crossover`

- `method`: `"one_point"` o `"uniform"`.
- `probability`: número finito entre `0` y `1` inclusive. Si no ocurre la cruza, los hijos comienzan como copias de los padres y luego pasan por mutación.
- Para `one_point`, `params` debe ser `{}` y `triangle_count >= 2`.
- Para `uniform`, `params` contiene únicamente `swap_probability`, número finito entre `0` y `1` inclusive.
- La cruza opera inicialmente entre genes completos y conserva `triangle_count`.

#### `mutation`

- `method`: `"single_gene"` o `"multigene_uniform"`.
- `probability`: número finito entre `0` y `1` inclusive.
- En `single_gene`, es la probabilidad por hijo de mutar exactamente un gen.
- En `multigene_uniform`, es la probabilidad independiente de mutar cada gen.

`allele_change` es una unión discriminada por `mode`:

```json
{
  "mode": "local_delta",
  "position_delta": 0.08,
  "color_delta": 20,
  "alpha_delta": 20
}
```

o:

```json
{
  "mode": "global_resample"
}
```

Reglas:

- Se elige uniformemente una de siete propiedades: uno de los tres vértices, uno de los tres canales RGB o alfa.
- `position_delta`: número finito en `(0, 1]`, aplicado sobre coordenadas normalizadas.
- `color_delta` y `alpha_delta`: enteros entre `1` y `255` inclusive.
- En `local_delta`, se aplica un cambio aleatorio acotado por el delta correspondiente.
- En `global_resample`, se toma un nuevo valor de todo el dominio permitido de la propiedad.
- El valor debe cambiar efectivamente y el triángulo resultante debe seguir siendo válido; si no, se vuelve a intentar.
- Los campos delta son obligatorios sólo para `local_delta` y están prohibidos para `global_resample`.

#### `survival`

- `strategy`: `"additive"` o `"exclusive"`.
- `selection`: objeto de selección válido.
- En aditiva se seleccionan `P` individuos del conjunto de `P` individuos actuales más `K` hijos.
- En exclusiva, si `K > P`, se seleccionan `P` entre los hijos. Si `K <= P`, entran los `K` hijos y se seleccionan `P-K` individuos de la población actual.
- El resultado debe contener exactamente `P` individuos.

### `fitness`

#### `metric`

- Tipo: string.
- Único valor inicial: `"normalized_mse"`.

#### `epsilon`

- Tipo: número finito.
- Debe cumplir `0 < epsilon < 1`.
- El cálculo inicial es `fitness = max(epsilon, 1 - NMSE)`.

### `termination`

#### `max_generations`

- Tipo: entero mayor que `0`.
- Siempre está activo y garantiza una cota finita.

#### `target_nmse`

- Tipo: `null` o número finito entre `0` y `1` inclusive.
- `null` deshabilita este criterio.

#### `stagnation`

- Tipo: `null` o un objeto con `patience` y `min_improvement`.
- `patience`: entero mayor que `0`.
- `min_improvement`: número finito mayor o igual que `0`.
- El corte ocurre cuando el mejor NMSE no disminuye al menos `min_improvement` durante `patience` generaciones consecutivas.

#### `max_seconds`

- Tipo: `null` o número finito mayor que `0`.
- `null` deshabilita el límite temporal.

Los criterios habilitados se combinan con OR: la ejecución termina cuando se cumple el primero. Siempre se devuelve el mejor individuo histórico.

### `run`

#### `seed`

- Tipo: entero mayor o igual que `0`.
- Toda aleatoriedad del motor deriva de esta semilla.

### `output`

#### `directory`

- Tipo: string no vacío.
- Cada corrida crea un subdirectorio propio y nunca reemplaza otra corrida silenciosamente.

#### `metrics_every`

- Tipo: entero mayor que `0`.
- Define cada cuántas generaciones se registra el historial de métricas.

#### `checkpoint_every`

- Tipo: `null` o entero mayor que `0`.
- `null` deshabilita checkpoints intermedios.

#### `render_original_size`

- Tipo: booleano.
- Si es `true`, el mejor cromosoma se vuelve a renderizar con las dimensiones originales usando sus coordenadas normalizadas.

## Validaciones cruzadas

Además de validar cada campo:

1. `one_point` requiere `triangle_count >= 2`.
2. El `tournament_size` no puede superar el tamaño del conjunto usado por ese selector.
3. Todos los resultados de selección, cruza, mutación y supervivencia deben conservar individuos válidos.
4. Todo cromosoma debe contener exactamente `triangle_count` genes antes y después de cada operador.
5. `alpha_range` y las mutaciones deben respetar siempre `A > 0`.
6. La configuración del método elegido no puede contener parámetros pertenecientes a otro método.
7. `output.directory` no puede ser el archivo de imagen de entrada ni apuntar a una ruta de archivo existente.
8. Una configuración inválida impide comenzar la corrida; no se corrige silenciosamente.

## Archivos mínimos de una corrida

Cada subdirectorio de corrida contiene como mínimo:

```text
config.effective.json
metadata.json
metrics.csv
triangles.json
best.png
```

`metadata.json` registra semilla, motivo de corte, generación final, mejor generación, NMSE, fitness, dimensiones de trabajo y dimensiones originales.

