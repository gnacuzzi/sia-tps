# Control de capacidad del autoencoder de código

## Pregunta experimental

Antes de introducir muestreo y KL se comprobó si un autoencoder determinista
puede reconstruir la representación one-hot aprobada. El objetivo no fue elegir
el VAE final, sino separar un posible problema de capacidad de las dificultades
propias del entrenamiento variacional.

## Protocolo prefijado

Se mantuvieron constantes el latente de dimensión 16, activaciones
`tanh-lineal-tanh-softmax agrupado`, entropía cruzada categórica, Adam `0.005`,
semilla de modelo 0 y un máximo de 3.000 épocas. Las arquitecturas se probaron
en orden creciente de anchura.

Una arquitectura pasaba el control únicamente si sostenía durante tres
evaluaciones consecutivas:

- 100 % de reconstrucción exacta en las 1.800 fuentes de entrenamiento;
- al menos 99 % de reconstrucción exacta en las 450 de validación;
- 100 % de pertenencia a la gramática en validación.

El test permaneció sin consultar hasta seleccionar una arquitectura. La tabla
versionada está en `docs/vae/code-ae-capacity-comparison.csv`.

## Resultados

| Arquitectura | Parámetros | Train exacto | Validación exacta | Caracteres correctos | Pasa |
|---|---:|---:|---:|---:|---:|
| `650-64-16-64-650` | 86.042 | 100 % | 97,33 % | 99,897 % | no |
| `650-128-16-128-650` | 171.418 | 100 % | 95,78 % | 99,769 % | no |
| `650-256-16-256-650` | 342.170 | 100 % | 99,11 % | 99,949 % | sí |

La anchura intermedia memorizó entrenamiento más rápido pero generalizó peor;
por lo tanto, aumentar parámetros no produjo una mejora monótona. La última
arquitectura fue la única que superó la regla prefijada y quedó seleccionada
como control de capacidad, no como VAE final.

Al abrir test una sola vez, reconstruyó exactamente 448 de 450 fuentes
(`99,56 %`) y acertó `99,974 %` de los caracteres. Las 450 decodificaciones
compilaron y pertenecieron a la gramática. Los dos errores conservaron sintaxis
y estructura válidas y afectaron únicamente dígitos de las constantes:

```text
esperada: def f(x): return (x-64)-31
obtenida: def f(x): return (x-66)-31

esperada: def f(x): return (x* 4)-50
obtenida: def f(x): return (x* 1)-10
```

Por par de operadores, test obtuvo 100 % exacto en siete de los nueve grupos y
98 % en los dos restantes. No hubo errores en caracteres fijos ni operadores;
los únicos errores posicionales estuvieron en dígitos de constantes.

## Conclusión acotada

La representación one-hot y el decoder categórico poseen capacidad suficiente
para reconstruir datos no vistos con una red densa. Esto habilita comenzar el
embudo del VAE sin atribuir anticipadamente sus posibles fallos a la salida de
650 valores. Todavía no justifica reutilizar automáticamente la misma anchura,
dimensión latente u optimizador en el modelo variacional: esas decisiones deben
formar parte del protocolo siguiente.
