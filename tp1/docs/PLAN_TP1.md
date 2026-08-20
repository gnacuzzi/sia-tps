# TP1 - Plan de trabajo y contrato de colaboración

Estado: Fases 0, 1, 2, 3, 4 y 5 completadas. Próxima etapa: Fase 6, reproducción visual y GIF.
Objetivo: completar el TP entendiendo las decisiones conceptuales, usando asistencia para recuperar fluidez de programación sin delegar el razonamiento central.

## 1. Acuerdo de trabajo

### Lo que hago yo

- Explicar conceptos y hacer preguntas para comprobar que se entienden.
- Ayudar a transformar decisiones propias en interfaces, pseudocódigo y luego código.
- Proponer estructura, tests, instrumentación, experimentos y revisión.
- Escribir código mecánico o de infraestructura cuando se solicite.
- Detectar errores y explicar su causa antes de corregirlos.

### Lo que queda reservado al estudiante

- Diseñar las heurísticas del ejercicio 1 y del ejercicio 2.
- Elegir qué información usa cada heurística y justificar por qué no sobreestima.
- Escribir primero la definición de cada heurística en palabras o matemática.
- Aprobar la traducción de esa definición a código.
- Poder explicar las decisiones y resultados de la presentación.

**Regla explícita:** Codex no debe inventar, completar ni escribir la implementación de una heurística. Puede formular preguntas, revisar una propuesta del estudiante, buscar contraejemplos y explicar admisibilidad o consistencia.

## 2. Qué pide exactamente el enunciado

### Ejercicio 1 - 8-puzzle (conceptual, sin implementación)

- Proponer una estructura de estado.
- Proponer al menos dos heurísticas admisibles no triviales.
- Elegir métodos de búsqueda, asociarlos con heurísticas y justificar la elección.

### Ejercicio 2 - implementación

- Elegir un dominio: Sokoban o Grid World multiagente.
- Implementar una estructura de estado.
- Implementar BFS, DFS, Greedy y A*; IDDFS es opcional.
- Usar al menos dos heurísticas admisibles; heurísticas no admisibles son opcionales.
- Informar éxito o fracaso, costo, nodos expandidos, nodos en frontera, camino solución y tiempo.
- Entregar código, presentación y README de ejecución.

## 3. Decisión de alcance

**Decisión confirmada: Sokoban pequeño.**

Motivos:

- El enunciado define mejor el juego y enlaza sus reglas.
- Grid World deja abiertas decisiones importantes: movimientos simultáneos o secuenciales, colisiones, asignación de objetivos y costo conjunto.
- Sokoban permite comenzar con tableros mínimos e incrementar cajas/objetivos gradualmente.
- Tiene dificultades reales que sirven para el análisis, pero pueden controlarse limitando los niveles.

Riesgo a tener presente: el espacio de estados de Sokoban crece rápido. El alcance inicial será un tablero pequeño, una caja y un objetivo; recién después se agregará complejidad.

### Puerta de decisión

Antes de programar, responder:

1. [x] El grupo eligió Sokoban.
2. ¿Hay reglas adicionales de la cátedra que no estén en el PDF?
3. ¿La unidad de costo es cada movimiento del jugador? En Sokoban, el enunciado dice optimizar movimientos, por lo que esta será la hipótesis inicial.

Si no aparece información nueva de la cátedra, continuar con Sokoban y costo unitario por movimiento como experimento principal.

### Decisión de diseño sobre el costo

El costo no quedará escrito dentro de BFS, DFS, Greedy o A*. El dominio expondrá conceptualmente una operación como:

```text
costo_transicion(estado, accion, estado_siguiente) -> numero
```

Esto permite cambiar el modelo sin reescribir los algoritmos, por ejemplo para realizar un experimento secundario donde distintos tipos de acción tengan distinto costo. Sin embargo:

- el resultado principal usará costo 1 por movimiento porque el enunciado pide optimizar cantidad de movimientos;
- cada ejecución tendrá un único modelo de costo, declarado en su configuración;
- si los costos dejan de ser uniformes, BFS ya no garantiza minimizar el costo ponderado;
- cambiar el modelo obliga a revisar qué significa “óptimo” y volver a justificar la admisibilidad de las heurísticas bajo ese modelo;
- nunca se mezclarán resultados de modelos de costo distintos en una misma comparación sin identificarlos.

La extensibilidad queda prevista ahora, pero el segundo modelo de costo es opcional y no debe retrasar la entrega obligatoria.

### Qué significa “formato de niveles”

Sí: son los tableros del juego guardados como archivos. El formato debe indicar de manera no ambigua paredes, piso, objetivos, cajas y jugador. También hay que decidir:

- qué carácter representa cada elemento;
- cómo se representa una caja o el jugador cuando está sobre un objetivo;
- si todas las filas deben tener el mismo ancho;
- qué validaciones hace el parser;
- si cada archivo contiene un nivel o varios;
- cómo se selecciona el nivel desde `config.json`.

La codificación concreta se decidirá mediante preguntas antes de crear el parser.

Decisión confirmada sobre almacenamiento:

- cada archivo de texto contiene exactamente un nivel;
- los niveles viven en el directorio `levels/`;
- `config.json` selecciona el archivo mediante una ruta, por ejemplo `levels/level_01.txt`;
- agregar una instancia nueva no requiere modificar el código del motor.

Alfabeto confirmado para los archivos de niveles:

```text
#  pared
_  piso transitable
.  objetivo vacío
$  caja sobre piso
@  persona sobre piso
*  caja sobre objetivo
+  persona sobre objetivo
   espacio en blanco: posición exterior, inexistente o fuera del tablero
```

Los símbolos `*` y `+` conservan la información de que debajo de la caja o la persona hay un objetivo.

### Forma irregular de los niveles

La zona jugable no tiene que ser rectangular. Se distingue entre:

- la **grilla de coordenadas** usada para guardar y consultar posiciones; y
- la **forma jugable**, que puede ser irregular dentro de esa grilla.

Los espacios en blanco y las posiciones que no existen porque una fila es más corta se interpretarán como exterior (`void`), no como piso. Por lo tanto:

- las filas del archivo podrán tener longitudes diferentes;
- internamente el parser podrá normalizarlas hasta el ancho de la fila más larga;
- esa normalización no crea piso ni cambia la forma del nivel;
- la persona y las cajas no pueden entrar en una posición exterior;
- no se exigirá que todo el borde del rectángulo contenedor sea una pared;
- sí se validará que ninguna posición inicial dinámica esté sobre el exterior.

El parser rechazará un nivel con una posición transitable directamente abierta al exterior. El exterior sirve para representar la forma irregular, pero el área jugable debe estar delimitada por paredes reales. Esta validación se aplica a vecinos ortogonales; una pared sí puede limitar directamente con el exterior.

## 4. Modelo conceptual que debe existir antes del código

Según la terminología de la materia, el problema debe fijar:

- estado inicial;
- acciones aplicables;
- modelo de transición;
- costo de cada transición;
- condición de solución.

Para Sokoban, separar conceptualmente:

- información estática del nivel: dimensiones, paredes y objetivos;
- información dinámica del estado: aquello que cambia al ejecutar una acción;
- nodo de búsqueda: estado, referencia al padre, acción aplicada, profundidad y costo acumulado.

No decidir todavía la representación concreta. Primero completar esta ficha:

```text
Estado inicial: el nivel fijo cargado desde el archivo, junto con la única posición inicial de la persona y las posiciones iniciales de una o más cajas.
Información estática: paredes, posiciones transitables (piso) y objetivos de las cajas.
Información dinámica: posición de la persona y posiciones de las cajas.
Acciones posibles: intentar moverse arriba, abajo, a la izquierda o a la derecha.
Precondición de cada acción: el movimiento es inválido si la casilla inmediata en la dirección elegida contiene una pared o el exterior. Si contiene una caja, no se puede empujar cuando la casilla siguiente en esa misma dirección contiene una pared, el exterior u otra caja. Los objetivos son transitables.
Efecto de cada acción: si la casilla inmediata es transitable y está libre, solamente se desplaza la persona. Si contiene una caja y la casilla siguiente es transitable y está libre, la caja avanza una posición y la persona ocupa la posición anterior de la caja. Una caja puede entrar, atravesar y abandonar un objetivo.
Costo de una acción: para el experimento principal, 1 por cada movimiento válido de la persona, empuje o no; el diseño permitirá modelos alternativos opcionales.
Condición de objetivo: verdadera cuando todas las cajas están ubicadas en posiciones objetivo.
Cuándo dos estados son iguales: cuando, dentro del mismo nivel fijo, la persona y todas las cajas ocupan las mismas posiciones.
Qué parte del estado se usa para detectar repetidos: las posiciones dinámicas de la persona y de las cajas; el camino, el padre, la profundidad y el costo acumulado pertenecen al nodo y no a la identidad del estado.
```

### Ficha conceptual en construcción

Primera decisión aportada por el estudiante:

- **Nivel o información estática:** las paredes, las posiciones que se pueden ocupar y los objetivos no cambian durante una partida.
- **Estado dinámico:** la persona cambia de posición al moverse; una caja cambia de posición únicamente cuando el movimiento efectivamente la empuja.
- **Igualdad de estados:** dos tableros del mismo nivel representan el mismo estado cuando coinciden la posición de la persona y las posiciones de todas las cajas, aunque se haya llegado mediante caminos diferentes.
- **Estado versus nodo:** dos nodos pueden contener el mismo estado y, sin embargo, tener distinto padre, camino, profundidad o costo acumulado.
- **Acciones:** la persona puede intentar moverse arriba, abajo, a la izquierda o a la derecha.
- **Empuje:** no se modela por ahora como una quinta acción independiente. Es un posible efecto de un movimiento direccional cuando hay una caja involucrada y el movimiento resulta válido.
- **Movimiento inválido, definición parcial:** no se puede entrar en una pared. Para empujar una caja, hay que observar también la casilla que está inmediatamente detrás de ella en la dirección del movimiento; una pared u otra caja allí impide el empuje.
- **Cajas consecutivas:** una caja no empuja a otra. En una secuencia persona-caja-caja, el intento de movimiento hacia ellas es inválido, aunque detrás de la segunda caja haya piso libre.
- **Objetivos:** son posiciones transitables, no obstáculos. La persona puede ocupar un objetivo vacío. Una caja puede entrar en un objetivo y posteriormente salir de él; alcanzar un objetivo no la bloquea ni la elimina del tablero.
- **Condición de solución:** el nivel está resuelto cuando la posición de cada caja coincide con alguna posición objetivo.
- **Cantidad de cajas y objetivos:** por decisión de alcance, un nivel válido debe tener exactamente la misma cantidad de cajas que de objetivos. El parser rechazará niveles que no cumplan esta condición.
- **Cantidad de personas:** un nivel válido debe contener exactamente una persona. El parser rechazará tanto la ausencia de una persona como la presencia de más de una.
- **Cantidad mínima:** un nivel válido debe contener al menos una caja y al menos un objetivo. No se aceptan niveles vacíos que comiencen resueltos por ausencia de cajas.
- **Intentos inválidos:** no producen un estado sucesor y no suman costo; el motor de búsqueda solamente recibe acciones aplicables.

Esta separación todavía es conceptual. La representación concreta en Python se elegirá después de completar la ficha.

Consecuencia para la búsqueda: detectar que un estado ya apareció no siempre significa ignorar automáticamente el nuevo nodo. En una búsqueda que compara costos, si el nuevo camino es más barato habrá que conservar o actualizar el mejor costo conocido para ese estado. La política concreta se definirá al diseñar cada algoritmo.

### Comprobación manual aprobada

Nivel mínimo usado para validar el modelo:

```text
#######
#@_$_.#
#######
```

Secuencia `DERECHA, DERECHA, DERECHA`:

1. Primer movimiento válido: se mueve solamente la persona; `pushed = false`.
2. Segundo movimiento válido: la persona empuja la caja; `pushed = true`.
3. Tercer movimiento válido: vuelve a empujar la caja hasta el objetivo; `pushed = true` y el nivel queda resuelto.

Resultado esperado:

- movimientos: 3;
- empujes: 2;
- costo unitario total: 3;
- la persona termina inmediatamente a la izquierda de la caja;
- la caja no puede seguir avanzando porque detrás del objetivo hay una pared;
- intentar `IZQUIERDA` desde el estado inicial es inválido, no genera sucesor, no cambia el estado y no suma costo.

Este ejemplo se convertirá en un test de aceptación del dominio durante la Fase 1.

## 5. Qué es una heurística (sin resolver las del TP)

Una heurística es una función numérica:

```text
h(estado) -> estimación del costo que falta hasta una solución
```

Primero es una idea matemática; después se traduce a una función de código. Para una meta, debe valer cero. Es admisible si nunca devuelve más que el costo óptimo real restante. Una forma de buscar candidatas, indicada en la clase, es estudiar subproblemas o relajar reglas del problema.

### Taller reservado para el estudiante

Para cada candidata, completar sin ayuda generativa de soluciones:

```text
Nombre provisional:
Qué observa del estado:
Qué cantidad calcula:
Qué reglas del problema ignora o relaja:
Por qué debería ser una cota inferior:
Caso objetivo (¿da 0?):
Ejemplo manual sencillo:
Posible contraejemplo que podría hacerla sobreestimar:
Costo computacional aproximado:
```

Luego Codex puede actuar como revisor:

- pedir una demostración breve de admisibilidad;
- intentar encontrar un contraejemplo;
- comprobar el valor a mano en tableros pequeños;
- revisar que el código implementa exactamente la definición aprobada por el estudiante.

## 6. Arquitectura propuesta

El motor de búsqueda debe depender de una interfaz de problema y no de detalles de Sokoban. Así los cuatro algoritmos comparten nodos, métricas y reconstrucción del camino.

```text
CLI/configuracion -> problema Sokoban -> interfaz de problema
                                      -> motor de busqueda
                                         |- BFS
                                         |- DFS
                                         |- Greedy
                                         `- A*
                                      -> resultado + metricas
```

Responsabilidades:

- **Dominio:** reglas, estados sucesores, objetivo y costo.
- **Búsqueda:** frontera, repetidos, prioridades y reconstrucción del camino.
- **Heurísticas:** módulo separado y propiedad intelectual conceptual del estudiante.
- **Experimentos:** ejecutar la misma instancia con distintas combinaciones y guardar resultados.
- **Presentación:** consumir resultados reproducibles; no producir números manualmente.

## 7. Estructura futura del monorepo

No hace falta reorganizar ahora. Cuando se cree el repositorio de la materia, una estructura simple sería:

```text
sia/
├── README.md
├── .gitignore
├── tp1/
│   ├── README.md
│   ├── pyproject.toml
│   ├── config.json
│   ├── src/
│   │   └── sia_tp1/
│   │       ├── cli.py
│   │       ├── domain/
│   │       │   └── sokoban.py
│   │       ├── search/
│   │       │   ├── common.py
│   │       │   ├── uninformed.py
│   │       │   └── informed.py
│   │       ├── heuristics.py
│   │       └── metrics.py
│   ├── levels/
│   ├── experiments/
│   ├── tests/
│   ├── docs/
│   │   ├── decisions.md
│   │   └── exercise_1.md
│   └── presentation/
└── tp2/
```

Mantener un `pyproject.toml` por TP evita configurar un workspace complejo antes de necesitarlo. Los PDFs de la cátedra pueden quedar fuera de Git si tienen restricciones de distribución; en ese caso se documentan como material local.

### Configuración confirmada

La presentación de la cátedra pide explícitamente incluir `config.json` en la entrega y menciona `config.yml` como alternativa similar. Se usará `config.json` para respetar literalmente el lineamiento.

Su responsabilidad será centralizar parámetros de una ejecución, por ejemplo la instancia, el método, el identificador de heurística, el modelo de costo, límites defensivos y semilla. El archivo no contendrá resultados ni lógica Python. La CLI lo cargará, validará y pasará los valores a los componentes; las clases del dominio y los algoritmos no leerán variables globales del archivo por su cuenta.

La semilla se guardará para reproducibilidad aunque el recorrido normal sea determinista. No se introducirá aleatoriedad sólo para justificar su existencia.

### Dependencias de Python

Objetivo: mantener el motor con pocas dependencias.

- Biblioteca estándar para estructuras de datos, prioridades, JSON, CLI, rutas y medición de tiempo.
- `pytest` como dependencia de desarrollo para tests.
- La dependencia para generar GIF se elegirá en la Fase 6 y las herramientas de
  análisis y gráficos en la Fase 7, siempre separadas del motor de búsqueda.

`pyproject.toml` documentará la versión de Python y las dependencias. Se usará el flujo tradicional `venv + pip + pyproject.toml`, sin incorporar Poetry ni uv por ahora. La versión concreta de Python se confirmará al crear el repositorio; el `python3` disponible actualmente es 3.9.6.

## 8. Definiciones de métricas antes de implementar

Estas definiciones deben permanecer idénticas para todos los métodos:

- **Resultado:** `SUCCESS` si se encuentra una meta; `FAILURE` únicamente si la frontera queda vacía y no existe solución alcanzable; `CUTOFF` si se interrumpe por un límite de tiempo, expansiones u otro recurso, sin afirmar si existe solución.
- **Costo de solución:** suma de costos de las acciones del camino devuelto.
- **Nodos expandidos:** cantidad de nodos a los que efectivamente se les generan sucesores. Agregar o descubrir un nodo no cuenta como expansión. Un nodo retirado de la frontera que satisface el objetivo no se cuenta como expandido, porque la búsqueda termina sin generarle sucesores.
- **Frontera al finalizar (`frontier_size_at_end`):** cantidad de nodos pendientes cuando termina la búsqueda.
- **Máximo de frontera (`max_frontier_size`):** máximo tamaño observado durante la ejecución; representa mejor el pico de memoria atribuible a la frontera.
- **Tiempo:** tiempo monotónico exclusivamente del algoritmo de búsqueda. Excluye lectura y validación de `config.json` y del nivel, impresión o animación del camino, escritura de resultados y generación de gráficos. Cada ejecución guarda su tiempo individual; los promedios se calculan posteriormente en los experimentos.
- **Camino durante la búsqueda:** cada nodo guarda solamente su propio estado, una referencia a su nodo padre y el movimiento que lo produjo. No guarda una copia de todo el camino, porque eso duplicaría la misma información en muchos nodos.
- **Reconstrucción del camino:** al encontrar la meta se recorren las referencias a padres desde el nodo objetivo hasta la raíz y luego se invierte el orden.
- **Camino en el resultado:** después de reconstruirlo, el resultado puede exponer la secuencia ordenada de movimientos y la secuencia de estados desde el inicial hasta la meta. Esto no implica que cada nodo haya almacenado ambas secuencias completas.
- **Detalle de movimiento:** cada transición registra la dirección elegida y un booleano `pushed` que indica si el movimiento empujó una caja. Esta marca pertenece a la transición, no a la identidad del estado.
- **Movimientos y empujes:** un resultado exitoso informa por separado la cantidad total de movimientos y la cantidad de movimientos que empujaron una caja. Con costo unitario, el costo coincide con la cantidad de movimientos, pero se conservan ambas métricas para admitir modelos de costo opcionales.

Pregunta para la cátedra si hay oportunidad: “cantidad de nodos frontera” ¿significa tamaño al finalizar o máximo de frontera? Mientras tanto se registrarán ambas con nombres inequívocos.

## 9. Plan por fases

### Fase 0 - alcance y especificación (45-60 min)

- [x] Confirmar dominio: Sokoban.
- [x] Completar la ficha conceptual básica de la sección 4.
- [x] Definir formato mínimo de niveles y sus validaciones.
- [x] Definir métricas y criterio de repetición.
- [x] Elegir lenguaje: Python.
- [x] Elegir formato de configuración: `config.json`, requerido por el lineamiento de la cátedra.
- [x] Elegir herramienta mínima de dependencias: `venv + pip + pyproject.toml`.

**Terminado cuando:** se puede simular a mano una acción válida, una inválida y una condición de victoria sin hablar de BFS o A*.

**Estado:** completada mediante la comprobación manual documentada en la sección 4.

### Fase 1 - esqueleto vertical (1.5-2 h)

Decisiones de representación confirmadas:

- una posición se identifica mediante coordenadas `(fila, columna)` con índices internos desde cero;
- la forma jugable puede ser irregular aunque las posiciones usen una grilla de coordenadas;
- el nivel se representa de forma dispersa: se almacenan explícitamente las posiciones que existen;
- paredes, pisos transitables y objetivos pertenecen a la información estática;
- una coordenada ausente del mapa representa exterior (`void`) y no puede ser ocupada;
- los objetivos son posiciones transitables con una propiedad adicional.
- los estados son inmutables: una acción válida crea un estado nuevo y conserva intacto el estado padre, permitiendo generar varias ramas desde él sin interferencias.
- la posición de la persona se representa con una tupla `(fila, columna)`;
- las posiciones de las cajas se representan con un `frozenset` de posiciones: las cajas son indistinguibles, su orden no forma parte del estado y la colección no puede mutarse accidentalmente.
- las colecciones estáticas de paredes, pisos y objetivos también se representan como `frozenset`; el nivel no cambia durante una búsqueda y puede compartirse de forma segura entre todos los estados y nodos.

Separación de objetos prevista:

```text
Problema / nivel (compartido por toda la búsqueda)
├── paredes
├── pisos transitables
├── objetivos
└── forma o límites necesarios para validar y mostrar

Estado dinámico (una configuración dentro de ese nivel)
├── posición de la persona
└── posiciones de las cajas

Nodo de búsqueda (cómo se llegó a un estado)
├── estado dinámico
├── nodo padre
├── transición aplicada: dirección + pushed
├── profundidad
└── costo acumulado
```

El tablero completo en un instante se obtiene combinando el nivel fijo con el estado dinámico. Las paredes y los objetivos no se copian dentro de cada estado porque son idénticos para todos los estados de una misma búsqueda. La igualdad de estados se evalúa dentro del mismo problema o nivel.

Contrato conceptual del parser:

```text
parse_level(archivo) -> (nivel_fijo, estado_inicial)
```

- `nivel_fijo` contiene paredes, pisos transitables, objetivos y la información necesaria para validar o mostrar la forma irregular;
- `estado_inicial` contiene la posición inicial de la persona y el `frozenset` de posiciones iniciales de cajas;
- el parser valida el archivo, pero no crea nodos de búsqueda;
- el motor crea luego el nodo raíz con el estado inicial, `parent = None`, profundidad 0, costo acumulado 0 y sin transición previa.

Descomposición confirmada de símbolos durante el parseo:

| Símbolo | Nivel fijo | Estado inicial |
| --- | --- | --- |
| `#` | pared | - |
| `_` | piso transitable | - |
| `.` | piso transitable y objetivo | - |
| `$` | piso transitable | caja en la posición |
| `@` | piso transitable | persona en la posición |
| `*` | piso transitable y objetivo | caja en la posición |
| `+` | piso transitable y objetivo | persona en la posición |
| espacio | exterior (`void`) | - |

La información superpuesta se conserva por capas: si una caja o la persona abandona un objetivo, el objetivo permanece en el nivel fijo.

Desplazamientos confirmados para las acciones direccionales:

```text
ARRIBA    -> (-1,  0)
ABAJO     -> (+1,  0)
IZQUIERDA -> ( 0, -1)
DERECHA   -> ( 0, +1)
```

La posición inmediata se calcula sumando el desplazamiento a la posición actual. La misma suma aplicada nuevamente permite obtener la posición detrás de una caja.

Algoritmo conceptual de transición:

1. Calcular `next = player + delta`.
2. Si `next` es pared o exterior, el intento es inválido y no se genera sucesor. En una interfaz visual el tablero parece quedar igual, pero el motor no devuelve el estado actual como un nuevo sucesor.
3. Si `next` es piso transitable y no contiene una caja, crear un estado nuevo con la persona en `next`, las mismas cajas y `pushed = false`.
4. Si `next` contiene una caja, calcular `box_destination = next + delta`.
5. Si `box_destination` es pared, exterior u otra caja, el intento es inválido y no se genera sucesor.
6. Si `box_destination` es piso transitable libre, crear un estado nuevo donde:
   - la persona ocupa `next`, es decir, la posición anterior de la caja;
   - la caja deja `next` y ocupa `box_destination`;
   - todas las demás cajas conservan su posición;
   - `pushed = true`.

Un intento inválido no cambia el estado, no suma costo y no aparece en la lista de sucesores del motor de búsqueda.

Interfaz confirmada:

```text
apply_move(level, state, direction)
    -> Transition, si el movimiento es válido
    -> None, si el movimiento es inválido
```

`Transition` contendrá como mínimo el estado sucesor, la dirección aplicada y el booleano `pushed`. La función no modifica `level` ni `state`.

- [x] Crear el paquete Python y un nivel diminuto.
- [x] Crear la CLI.
- [x] Parsear y validar el nivel.
- [x] Mostrar un estado.
- [x] Aplicar una acción y obtener un nuevo estado inmutable.
- [x] Detectar objetivo.
- [x] Agregar tests iniciales del modelo y el parser.
- [x] Agregar tests de movimientos y condición de objetivo.

Implementación inicial creada:

- `src/sia_tp1/model.py`: posiciones, direcciones, nivel, estado y transición inmutables;
- `src/sia_tp1/parser.py`: lectura por capas y validaciones del formato acordado;
- `src/sia_tp1/domain.py`: aplicación inmutable de movimientos y condición de objetivo;
- `src/sia_tp1/render.py`: reconstrucción del tablero combinando nivel y estado;
- `src/sia_tp1/config.py`: carga y validación de `config.json`;
- `src/sia_tp1/cli.py` y `src/sia_tp1/__main__.py`: reproducción manual desde línea de comandos;
- `levels/level_01.txt`: caso mínimo razonado manualmente;
- `config.json`: configuración inicial centralizada;
- tests de modelo, parser, dominio, renderizado, configuración y CLI: 28 pruebas automáticas;
- `pyproject.toml`: proyecto compatible con Python 3.9 o superior, sin dependencias de ejecución y con `pytest` como extra de desarrollo.

Verificación actual: los 28 tests pasan con `unittest`, incluidos el caso de aceptación de tres movimientos, bloqueos, empujes, inmutabilidad, forma irregular, renderizado, configuración y CLI. Los módulos compilan y `config.json` es JSON válido. `pytest` todavía no está instalado localmente; los tests usan `unittest` y también podrán ser descubiertos por `pytest` cuando se instale el extra de desarrollo.

**Terminado cuando:** una secuencia escrita manualmente puede reproducirse y validarse de punta a punta.

**Estado:** completada. La CLI reproduce `RIGHT RIGHT RIGHT` sobre `level_01.txt`, informa `pushed = false, true, true`, costo 3 y estado resuelto.

### Fase 2 - motor común y BFS (1.5-2 h)

Diseño conceptual de nodo en curso:

```text
Node
├── state: estado dinámico representado
├── parent: nodo anterior o None para la raíz
├── transition: movimiento desde el padre o None para la raíz
├── depth: cantidad de movimientos desde la raíz
└── path_cost: costo acumulado g(n)
```

El nodo raíz se construye con el estado inicial, `parent = None`, `transition = None`, `depth = 0` y `path_cost = 0`.

Un nodo hijo creado desde una transición válida cumple:

```text
state = transition.state
parent = nodo expandido
transition = dirección aplicada + indicador pushed
depth = parent.depth + 1
path_cost = parent.path_cost + costo de esa transición
```

Para un primer movimiento `RIGHT` sin empuje y de costo unitario: el hijo contiene el nuevo objeto `State`, apunta a la raíz, registra `RIGHT` y `pushed = false`, tiene profundidad 1 y costo acumulado 1.

Política conceptual de repetidos para BFS:

- los nodos con padres o costos distintos siguen siendo nodos diferentes;
- la detección de configuraciones repetidas usa `State`, no `Node`;
- `visited` o `discovered` tendrá tipo conceptual `set[State]`;
- esto evita ciclos como `RIGHT` seguido de `LEFT`, que crean un nodo nuevo pero regresan al mismo tablero;
- con costo unitario, la primera vez que BFS descubre un estado lo alcanza a profundidad mínima;
- A* y modelos con costos variables se diseñarán después con un registro del mejor costo conocido por estado, no suponiendo que este conjunto simple alcanza.

Momento de marcado confirmado para BFS:

- el estado inicial se agrega a `discovered_states` al colocar la raíz en la frontera;
- un sucesor se conoce únicamente después de ejecutar `apply_move` durante la expansión de su padre;
- si el sucesor no fue descubierto, se agrega a `discovered_states` inmediatamente antes de incorporarlo a la cola;
- marcar al encolar evita que dos padres agreguen copias del mismo estado antes de que alguna sea expandida;
- estar descubierto no significa estar expandido: la frontera contiene estados conocidos que todavía esperan expansión.

Diseño conceptual de resultado en curso:

```text
SearchResult
├── status: SUCCESS | FAILURE | CUTOFF
├── goal_node: Node si SUCCESS; None si FAILURE o CUTOFF
├── expanded_nodes: disponible para cualquier estado final
├── frontier_size_at_end: disponible para cualquier estado final
├── max_frontier_size: disponible para cualquier estado final
└── elapsed_seconds: disponible para cualquier estado final
```

Para reconstruir una solución exitosa se comienza en `goal_node`, se siguen referencias `parent` hasta la raíz y se invierte la secuencia obtenida. Los estados salen de todos los nodos del camino; las transiciones salen de todos excepto la raíz, cuya transición es `None`.

El costo, la cantidad de movimientos y los empujes solamente existen para `SUCCESS`. Se derivarán del nodo objetivo y del camino reconstruido, en lugar de guardar copias independientes que podrían contradecirse. Para `FAILURE` y `CUTOFF`, el costo de solución es `None`.

`SearchResult` también incluye `cutoff_reason`: identifica si el corte ocurrió por `timeout` o `max_expanded_nodes`. Su valor es `None` para `SUCCESS` y `FAILURE`.

Frontera de BFS:

- se implementa como cola FIFO con `collections.deque`;
- la raíz se incorpora al inicializar la búsqueda;
- `max_frontier_size` comienza en 1 porque la raíz ya está esperando;
- los nodos se agregan con `append` y se retiran con `popleft`;
- `frontier_size_at_end` se obtiene con `len(frontier)` al construir el resultado;
- el máximo se actualiza después de incorporar sucesores.

Caso inicial ya resuelto: `SUCCESS`, 0 nodos expandidos, frontera final 0, máximo de frontera 1 y costo de solución 0.

Límites operativos confirmados:

- `timeout_seconds` y `max_expanded_nodes` son opcionales y viven en `config.json`;
- con valor `null` quedan desactivados;
- no son necesarios para la terminación teórica de BFS sobre el espacio finito y con repetidos controlados, pero protegen tiempo y memoria frente a explosión combinatoria;
- un límite de N expansiones permite exactamente N y produce `CUTOFF` antes de intentar la expansión N+1;
- un corte no se informa como ausencia de solución;
- las comparaciones experimentales deben usar los mismos límites para todos los métodos.

El nivel no se copia dentro de los nodos porque es compartido por toda la búsqueda. Tampoco se almacena el camino completo ni los hijos: el camino se reconstruye mediante `parent` y los sucesores se generan al expandir.

- [x] Definir nodo, resultado y métricas.
- [x] Implementar frontera FIFO y detección de repetidos.
- [x] Reconstruir el camino mediante padres.
- [x] Verificar una solución en el nivel mínimo.
- [x] Reproducir el camino devuelto y comprobar que termina en meta.

**Terminado cuando:** BFS produce un camino válido y métricas consistentes.

**Estado:** completada. La implementación determinista usa el orden `UP, DOWN, LEFT, RIGHT`, costo unitario, marcado de estados al encolar y límites opcionales.

Implementación:

- `src/sia_tp1/search/model.py`: `Node`, `SearchLimits`, estados de resultado, motivos de corte, `SearchResult` y reconstrucción;
- `src/sia_tp1/search/bfs.py`: BFS con `deque`, repetidos por `State`, métricas, timeout y máximo de expansiones;
- la CLI admite `--search` y ejecuta el algoritmo configurado;
- 38 tests automáticos pasan, incluyendo éxito inmediato, solución mínima, fracaso, ambos cortes, reconstrucción y métricas.

Resultado reproducible sobre `level_01.txt`: `SUCCESS`, costo 3, 3 movimientos, 2 empujes, 4 nodos expandidos, frontera final 1 y máximo de frontera 2.

### Fase 3 - DFS y robustez (1-1.5 h)

Diseño conceptual de DFS confirmado:

- la frontera es una pila LIFO: los nodos se agregan con `append` y se retiran con `pop`;
- la prioridad observable de expansión se conserva como `UP, DOWN, LEFT, RIGHT`;
- como la pila retira primero el último elemento agregado, los hijos se incorporan en el orden inverso `RIGHT, LEFT, DOWN, UP`;
- los repetidos se detectan mediante `set[State]`, no mediante nodos, y cada estado se marca como descubierto al agregarlo a la pila;
- esta política evita ciclos y duplicados pendientes en la frontera;
- DFS termina al encontrar la primera solución y no continúa buscando una alternativa más corta;
- por lo anterior, DFS no se presenta como óptimo aunque el costo de cada movimiento sea uniforme;
- usa los mismos límites, estados de resultado, definición de expansión y métricas que BFS.

- [x] Reutilizar el motor o la interfaz común con una frontera LIFO.
- [x] Definir un límite defensivo configurable para experimentos difíciles.
- [x] Probar éxito, fracaso y estado inicial ya resuelto.
- [x] Confirmar que DFS no se presenta como óptimo.

**Estado:** fase completada. `run_search` selecciona BFS o DFS según
`algorithm`; ambos reutilizan la misma presentación de `SearchResult`. Los
algoritmos informados continúan rechazándose hasta que sus diseños estén
aprobados. La suite completa contiene 45 tests aprobados.

**Terminado cuando:** BFS y DFS resuelven los mismos casos básicos y exponen el mismo formato de resultado.

### Fase 4 - taller de heurísticas (60-90 min, estudiante)

#### Primera heurística provisional propuesta por el estudiante

Definición aprobada en palabras:

```text
h(estado) =
    mínimo, entre todas las asignaciones uno-a-uno de cajas a objetivos,
    de la suma de las distancias Manhattan entre cada caja y su objetivo
    asignado
```

Para posiciones `(fila_1, columna_1)` y `(fila_2, columna_2)`, la distancia
Manhattan usada en la definición es:

```text
|fila_1 - fila_2| + |columna_1 - columna_2|
```

La propuesta ignora paredes, exterior, ubicación y reposicionamiento de la
persona, y bloqueos entre cajas. Cada caja se asigna a exactamente un objetivo
y cada objetivo a exactamente una caja. Si varias asignaciones tienen el mismo
total mínimo, el valor numérico de la heurística no depende de cuál se elija.

Traducción aprobada a código:

- `src/sia_tp1/heuristics.py` implementa
  `minimum_matching_manhattan_distance` mediante todas las permutaciones de
  objetivos y conserva la suma mínima;
- la función valida defensivamente que coincidan las cantidades de cajas y
  objetivos;
- `tests/test_heuristics.py` reproduce los valores manuales `2, 2, 1, 0` de
  `level_01.txt`, comprueba el mínimo global 7 frente a la elección local que
  daba 13 y prueba la validación de cantidades;
- después de este bloque, la suite completa contiene 48 tests aprobados.

La implementación inicial podrá enumerar las `n!` asignaciones porque el
alcance previsto usa niveles pequeños. Antes de escribir código todavía falta:

- calcular la heurística manualmente en varios estados;
- buscar contraejemplos a esa justificación;
- confirmar más casos manuales además del estado objetivo.

Justificación de admisibilidad construida por el estudiante:

1. Toda solución real termina con cada caja sobre un objetivo distinto y, por
   lo tanto, induce una asignación uno-a-uno de cajas a objetivos.
2. La heurística toma el mínimo entre todas las asignaciones, incluida la que
   induce esa solución real. Su valor no puede superar la suma correspondiente
   a esa asignación concreta.
3. Para una caja, cada empuje cambia sólo una fila o una columna en una unidad.
   Por eso alcanzar su objetivo requiere al menos tantos empujes como indique
   su distancia Manhattan.
4. La suma de distancias de la asignación final no supera la cantidad total de
   empujes de la solución.
5. Cada empuje también es un movimiento válido de la persona. Como el costo
   unitario cuenta todos sus movimientos válidos, el costo total no puede ser
   menor que la cantidad de empujes.

En consecuencia:

```text
h(estado)
    <= suma de distancias de la asignación de una solución óptima
    <= empujes de esa solución óptima
    <= movimientos de esa solución óptima
    = costo óptimo real restante
```

Además, en un estado objetivo existe una asignación de cada caja al objetivo
que ya ocupa, con distancia total cero. Como las distancias no son negativas,
`h(objetivo) = 0`.

Comprobación manual sobre `level_01.txt`, cuya solución conocida es
`RIGHT, RIGHT, RIGHT`:

| Estado | Posición de la caja | `h` | Costo óptimo real restante |
| --- | --- | ---: | ---: |
| Inicial | `(1, 3)` | 2 | 3 |
| Después del primer `RIGHT`, sin empuje | `(1, 3)` | 2 | 2 |
| Después del segundo `RIGHT`, con empuje | `(1, 4)` | 1 | 1 |
| Después del tercer `RIGHT`, resuelto | `(1, 5)` | 0 | 0 |

La heurística permanece en 2 durante el movimiento sin empuje porque no usa la
posición de la persona. En todos los estados comprobados su valor es menor o
igual que el costo óptimo real restante.

Revisión de posibles contraejemplos y limitaciones:

- si una pared obliga a realizar un rodeo, el costo real aumenta mientras la
  estimación puede quedar baja; eso es subestimar y no rompe la admisibilidad;
- ignorar el reposicionamiento de la persona y los bloqueos entre cajas también
  puede reducir la estimación, no aumentarla por encima del costo real;
- en un estado sin solución, por ejemplo una caja trabada en una esquina que no
  es objetivo, la heurística puede devolver un valor finito. El costo real se
  considera infinito, por lo que esto no rompe la admisibilidad, pero muestra
  que la heurística no detecta todos los estados insolubles;
- un contraejemplo real necesitaría cumplir estrictamente
  `h(estado) > costo óptimo real restante`. No se encontró uno bajo el modelo
  actual de costo unitario y las validaciones acordadas.

#### Segunda heurística propuesta por el estudiante, revisión aprobada

La segunda propuesta observa la posición de la persona, información que la
primera heurística ignora. Su definición aprobada en palabras es:

```text
si el estado está resuelto:
    h(estado) = 0

en otro caso:
    h(estado) = longitud del camino más corto de la persona hasta cualquier
                posición desde la que pueda realizar un empuje físicamente
                posible, manteniendo inmóviles todas las cajas

si ninguna de esas posiciones es alcanzable:
    h(estado) = infinito
```

Para una caja y una dirección, una posición de empuje es candidata cuando:

- la posición ortogonal inmediatamente detrás de la caja es piso y no contiene
  otra caja;
- la posición ortogonal inmediatamente delante de la caja, es decir, su destino
  si se la empuja, es piso y no contiene otra caja.

La distancia se obtiene mediante el camino transitable más corto de la persona:
las paredes y todas las cajas son obstáculos, y durante este recorrido no se
permite empujar. El movimiento que efectúa el primer empuje no se suma. Si la
persona ya está en una posición candidata, la heurística devuelve cero aunque
el estado no esté resuelto.

Si un estado no objetivo no tiene ninguna posición candidata alcanzable sin
empujar, no puede existir un primer empuje y el estado no tiene solución. El
costo real restante es infinito y la heurística usa también infinito. La
comprobación de objetivo debe tener prioridad, porque un estado resuelto no
necesita ningún empuje y debe devolver cero.

Traducción aprobada a código:

- `shortest_push_access_distance` obtiene primero las posiciones candidatas y
  ejecuta una BFS interna que mueve solamente a la persona;
- durante esa BFS, las paredes, el exterior y todas las cajas son obstáculos;
- la función devuelve la distancia al retirar la primera posición candidata,
  cero para un estado objetivo e infinito si agota las posiciones alcanzables;
- los tests reproducen `1, 0, 0, 0` sobre `level_01.txt`, comprueban un rodeo
  cuyo camino real cuesta 4 frente a una distancia Manhattan de 2 y verifican
  el resultado infinito cuando ninguna posición candidata es alcanzable;
- después de implementar ambas heurísticas, la suite completa contiene 51
  tests aprobados.

Justificación de admisibilidad construida por el estudiante:

1. Toda solución de un estado no objetivo tiene un primer empuje.
2. La posición desde la que la persona realiza ese primer empuje cumple las
   condiciones de una posición candidata y es alcanzable sin empujar.
3. La heurística toma el mínimo de los caminos más cortos hasta todas las
   candidatas alcanzables, por lo que no supera el camino más corto hasta la
   posición usada por la solución real.
4. Los movimientos de la solución previos al primer empuje forman un camino
   válido hasta esa posición, manteniendo todavía inmóviles todas las cajas. El
   camino más corto no puede ser más largo que ese prefijo.
5. Los movimientos previos al primer empuje son parte de la solución completa y
   no pueden superar su costo total.

Por lo tanto, para cualquier solución real:

```text
h(estado)
    <= camino más corto hasta la posición de su primer empuje
    <= movimientos reales previos a ese empuje
    <= costo total de esa solución
```

La desigualdad vale en particular para la solución óptima. En un estado
objetivo la definición devuelve cero. Si un estado no objetivo no tiene ninguna
posición de empuje alcanzable, ninguna solución puede realizar su primer empuje
y el costo real es infinito; devolver infinito tampoco lo sobreestima.

Comprobación manual sobre la solución conocida de `level_01.txt`:

| Estado | Persona | Caja | `h` |
| --- | --- | --- | ---: |
| Inicial | `(1, 1)` | `(1, 3)` | 1 |
| Después del primer `RIGHT` | `(1, 2)` | `(1, 3)` | 0 |
| Después del segundo `RIGHT` | `(1, 3)` | `(1, 4)` | 0 |
| Resuelto | `(1, 4)` | `(1, 5)` | 0 |

Los valores confirman que la propuesta no es idénticamente cero y, por lo
tanto, no es la heurística trivial en sentido formal. Usar el camino transitable
más corto permite distinguir rodeos causados por paredes y cajas que Manhattan
ignoraba. Sin embargo, sigue devolviendo cero en estados no objetivo cuando la
persona ya ocupa alguna posición candidata, aunque todavía falten cajas por
llevar a sus objetivos. Esta debilidad deberá observarse en los experimentos.

Revisión de posibles contraejemplos y decisiones descartadas:

- la posición candidata más cercana puede permitir un empuje que luego conduce
  a un mal camino. Esto sólo reduce la estimación: si cuesta 3 alcanzarla pero
  una solución óptima necesita 6 movimientos antes de su primer empuje, se
  conserva `h(estado) = 3 <= 6`;
- no se restringen las candidatas a empujes considerados "útiles" o dirigidos
  hacia un objetivo. Una solución puede necesitar primero un empuje que parezca
  desfavorable; excluirlo podría eliminar la posición del primer empuje real y
  romper la demostración de admisibilidad;
- las candidatas inalcanzables con las cajas inmóviles no participan del mínimo.
  Si no queda ninguna alcanzable, no puede existir el primer empuje de una
  solución y corresponde el valor infinito ya definido;
- un estado objetivo se comprueba antes que la ausencia de empujes, de modo que
  siempre recibe valor cero.

No se encontró un estado solucionable donde esta definición supere el costo
óptimo real restante.

#### Comparación conceptual de las dos propuestas

- La primera estima desplazamientos de cajas durante toda la solución. Ignora
  el recorrido de la persona, las paredes y los bloqueos, pero en
  `level_01.txt` conserva información hasta alcanzar la meta.
- La segunda estima solamente el recorrido real de la persona anterior al
  primer empuje. Considera paredes y cajas inmóviles durante ese prefijo, pero
  puede devolver cero mucho antes de que el nivel esté resuelto.
- Una heurística no orienta mejor por bajar más rápido. Greedy prioriza valores
  bajos de `h` y A* usa `g + h`, pero una cota admisible más cercana al costo
  real suele discriminar mejor los estados. Devolver cero prematuramente genera
  empates y aporta poca información.
- Después del primer `RIGHT` de `level_01.txt`, el costo óptimo restante es 2:
  la primera heurística devuelve 2 y la segunda devuelve 0. El estudiante
  identificó que la primera es más precisa en ese estado.
- Las propuestas observan dificultades diferentes. La segunda puede aportar
  información en otro nivel cuando paredes o cajas obliguen a la persona a dar
  un rodeo antes de poder efectuar el primer empuje.

- [x] Completar dos fichas de heurísticas candidatas.
- [x] Calcularlas a mano en varios estados.
- [x] Escribir una justificación de admisibilidad para la primera propuesta.
- [x] Escribir una justificación de admisibilidad para la segunda propuesta.
- [x] Buscar contraejemplos para la primera propuesta.
- [x] Buscar contraejemplos para la segunda propuesta.
- [x] Traducir las definiciones aprobadas a funciones.

**Estado:** fase completada. El estudiante diseñó, justificó, calculó y revisó
ambas heurísticas, y comprendió que la BFS local de la segunda función usa una
frontera independiente de la frontera del algoritmo principal.

**Terminado cuando:** el estudiante puede explicar cada función sin mostrar código y defender por qué no sobreestima.

### Fase 5 - Greedy y A* (1.5-2 h, sin delegar heurísticas)

Decisiones conceptuales confirmadas:

- Greedy prioriza `h(n)` y A* prioriza `g(n) + h(n)`;
- la frontera de prioridad desempata mediante un contador creciente de
  inserción, por lo que un empate conserva el orden observable de generación
  `UP, DOWN, LEFT, RIGHT` y nunca intenta comparar objetos `Node`;
- Greedy usa `set[State]`, marca al incorporar a la frontera y descarta toda
  aparición posterior del mismo estado; no garantiza costo mínimo;
- A* guarda el menor `g` conocido por estado y vuelve a incorporar un estado
  solamente si aparece con un costo estrictamente menor;
- las entradas anteriores de A* que queden obsoletas en el heap se descartan al
  retirarlas y no se expanden;
- `bfs` y `dfs` exigirán `heuristic: null`; `greedy` y `astar` exigirán una
  heurística reconocida, sin ignorar silenciosamente configuraciones inválidas;
- los identificadores de configuración serán `minimum_matching_manhattan` y
  `shortest_push_access`;
- ambos algoritmos comprobarán objetivo al retirar de la frontera y reutilizarán
  sin cambios los límites, métricas, estados de resultado, costo unitario y
  orden de sucesores de BFS y DFS.

Implementación actual:

- `src/sia_tp1/search/greedy.py` implementa Greedy con `heapq` y entradas
  `(h, orden_de_inserción, nodo)`;
- marca estados al incorporarlos y descarta repetidos mediante `set[State]`;
- reutiliza límites, métricas, reconstrucción y estados de resultado comunes;
- los tests cubren solución, meta inicial, fracaso, ambos cortes y desempate
  determinista que conserva `UP` como primera prioridad observable;
- `src/sia_tp1/search/astar.py` implementa A* con prioridad `g + h`, registro
  del menor `g` conocido, reapertura y descarte perezoso de entradas obsoletas;
- en A*, `frontier_size_at_end` cuenta estados activos y excluye entradas
  obsoletas, mientras `max_frontier_size` usa el tamaño físico máximo del heap
  porque esas entradas sí ocuparon memoria durante la ejecución;
- el registro de heurísticas asocia `minimum_matching_manhattan` y
  `shortest_push_access` con las funciones diseñadas por el estudiante;
- `config.json` rechaza una heurística en BFS/DFS, exige una heurística conocida
  en Greedy/A* y nunca ignora silenciosamente combinaciones inválidas;
- la CLI despacha los cuatro algoritmos y reutiliza la misma salida de métricas
  y camino solución;
- A* tiene una prueba específica de reapertura y coincide con el costo óptimo de
  BFS sobre `level_01.txt` bajo costo unitario;
- la suite completa contiene 69 tests aprobados al cerrar la fase.

- [x] Implementar una frontera de prioridad con desempate determinista para Greedy.
- [x] Greedy ordena por `h(n)`.
- [x] A* ordena por `g(n) + h(n)`.
- [x] Hacer intercambiable la función heurística.
- [x] Comparar el costo de A* y BFS en niveles pequeños de costo uniforme.

**Estado:** fase completada. Greedy y A* usan las heurísticas aprobadas, exponen
el mismo `SearchResult` que los métodos desinformados y pueden seleccionarse
desde `config.json`.

**Terminado cuando:** ambos algoritmos usan las funciones escritas por el estudiante y el camino se valida por reproducción.

### Fase 6 - reproducción visual y GIF de soluciones (1.5-2 h)

El requisito visual consiste en mostrar el Sokoban ejecutando el camino solución
de cada algoritmo: la persona camina y las cajas se desplazan. No se animará el
árbol de búsqueda, la frontera ni el orden de expansión.

La animación no necesita otro parser ni instrumentación adicional en los
algoritmos. Consumirá el resultado ya existente:

```text
SearchResult exitoso
-> solution_nodes
-> estado inicial + un estado por movimiento
-> una imagen por estado
-> GIF animado
```

- [ ] Elegir una dependencia liviana para dibujar imágenes y codificar GIF.
- [ ] Dibujar paredes, pisos, objetivos, persona y cajas con un estilo legible.
- [ ] Mantener dimensiones y escala constantes durante toda la animación.
- [ ] Incluir el estado inicial y luego un cuadro por cada transición.
- [ ] Mostrar opcionalmente algoritmo, número de movimiento, dirección y empuje.
- [ ] Exportar un GIF reproducible para BFS, DFS, Greedy y A* sobre el nivel
  elegido para la presentación.
- [ ] Verificar que la cantidad de cuadros sea `solution_moves + 1` y que el
  último estado satisfaga la condición de objetivo.
- [ ] Informar claramente que `FAILURE` y `CUTOFF` no tienen camino para animar.

**Terminado cuando:** una ejecución exitosa puede convertirse en un GIF que
reproduce exactamente su camino desde el estado inicial hasta la meta.

### Fase 7 - experimentos (1.5-2 h)

- [ ] Elegir pocas instancias: mínima, intermedia y una que exponga diferencias.
- [ ] Ejecutar repeticiones para tiempo; mantener fijos nivel, equipo y configuración.
- [ ] Exportar CSV/JSON con método, heurística, éxito, costo, expandidos, frontera final, máximo de frontera y tiempo.
- [ ] Marcar timeout o límite de recursos como resultado, no ocultarlo.
- [ ] Interpretar resultados antes de graficar.

**Terminado cuando:** una sola orden reproduce la tabla base de la presentación.

### Fase 8 - ejercicio 1, README y entrega (2-3 h)

- [ ] Resolver el 8-puzzle conceptualmente con el mismo taller de heurísticas.
- [ ] Escribir instrucciones exactas de instalación y ejecución.
- [ ] Documentar formato de niveles y opciones de CLI.
- [ ] Agregar decisiones, supuestos y limitaciones.
- [ ] Ejecutar tests desde un entorno limpio.
- [ ] Preparar la presentación usando el lineamiento de análisis de datos.

**Terminado cuando:** otra persona puede clonar, ejecutar un caso y entender una fila de resultados sin explicación oral.

## 10. Tests mínimos

- Igualdad y hash de estados equivalentes.
- Rechazo de movimientos ilegales.
- Transición correcta de un movimiento legal.
- Condición de objetivo verdadera y falsa.
- Los sucesores no modifican el estado padre.
- Detección de repetidos en un ciclo pequeño.
- Reconstrucción exacta de un camino conocido.
- BFS retorna costo mínimo en un caso pequeño de costo uniforme.
- DFS retorna un camino válido sin afirmar optimalidad.
- La solución de cada método se reproduce desde el estado inicial.
- Para cada heurística creada por el estudiante: valor cero en meta y casos manuales documentados.
- Los cuadros de cada GIF coinciden con los estados del camino solución y el
  último cuadro muestra un estado objetivo.

## 11. Estrategia para trabajar con Codex

Usar sesiones cortas con un resultado verificable. Ejemplos de pedidos adecuados:

```text
Estoy en la Fase 0. Haceme preguntas de a una para que yo defina el estado de Sokoban. No propongas heurísticas ni escribas código.
```

```text
Revisá mi ficha de estado y señalá ambigüedades. No implementes todavía.
```

```text
Ya definí el modelo. Ayudame a escribir solamente los tests del parser y explicame cada test.
```

```text
Esta es mi definición de heurística. No propongas otra. Intentá encontrar un contraejemplo a su admisibilidad y haceme preguntas.
```

```text
Traducí mi pseudocódigo aprobado a Python sin cambiar la definición de la heurística. Después explicá la correspondencia línea por línea.
```

El modo Plan es útil para acordar alcance, decisiones y criterios de aceptación antes de editar archivos. Después se pasa a ejecución para una fase pequeña. No conviene pedir “hacé todo el TP”; conviene pedir una fase y verificarla antes de continuar.

## 12. ¿Usar GSD Core?

GSD Core propone un ciclo `Discuss -> Plan -> Execute -> Verify -> Ship`, con artefactos persistentes de planificación. La idea es buena y este documento adopta ese ciclo de forma liviana.

Para este TP y con poco tiempo, **no se recomienda instalarlo todavía**:

- hay que aprender sus comandos y estructura además del contenido de la materia;
- su flujo completo está pensado para proyectos y fases con más automatización;
- la ejecución con agentes podría cruzar el límite pedagógico de no delegar heurísticas;
- todavía no existe el repositorio definitivo.

Reevaluar después de completar Fase 2. Si el flujo manual ya resulta claro, no hace falta agregarlo. Si se decide instalarlo, usar instalación local y configurar por escrito la prohibición sobre heurísticas antes de ejecutar agentes.

## 13. Recorte si el tiempo aprieta

Prioridad obligatoria:

1. Un dominio correctamente definido.
2. BFS, DFS, Greedy y A* correctos.
3. Dos heurísticas admisibles diseñadas y defendidas por el estudiante.
4. GIF de los caminos solución requerido para la presentación.
5. Métricas comparables y una batería pequeña reproducible.
6. README y presentación.

Recortar primero:

- IDDFS;
- heurísticas no admisibles;
- niveles grandes;
- framework GSD completo;
- automatización sofisticada del monorepo.

## 14. Próxima sesión propuesta

Empezar exclusivamente por Fase 0:

1. [x] decidir Sokoban o Grid World: Sokoban;
2. completar juntos la ficha conceptual;
3. definir el formato de los archivos de niveles;
4. fijar las métricas;
5. cerrar una especificación de una página;
6. recién entonces crear el repositorio y el esqueleto.
