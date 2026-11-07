# Ejercicio 1.a.2 - Arquitectura y optimización del autoencoder

## Objetivo y criterio de éxito

El objetivo de esta parte es estudiar arquitecturas y técnicas de optimización
que permitan representar los 32 patrones binarios de `font.h`, de 7 x 5
píxeles, en un espacio latente de dimensión 2.

Una reconstrucción se convierte a valores binarios con umbral 0.5 únicamente
para evaluarla. El modelo cumple el requisito si **cada uno de los 32 patrones**
tiene como máximo un píxel incorrecto. No alcanza con que el promedio sea menor
que uno, porque podría ocultar un patrón reconstruido con muchos errores.

Las salidas continuas y la loss se conservan durante el entrenamiento. Aplicar
el umbral dentro de backpropagation eliminaría la información sobre cuánto se
acerca una salida al objetivo: la función escalón tiene derivada cero fuera del
salto y no está definida en el salto.

## Implementación y controles previos

El autoencoder fue implementado con NumPy a partir del MLP propio del TP3. La
implementación permite acceder por separado a:

```text
encode: 35 píxeles -> (z1, z2)
decode: (z1, z2) -> 35 salidas continuas
```

Antes de entrenar los 32 patrones se realizaron dos controles:

1. Backpropagation se comparó contra gradientes numéricos en una red diminuta.
2. La arquitectura provisional `35 -> 15 -> 2 -> 15 -> 35` memorizó una única
   letra `a`: pasó de 16 píxeles incorrectos a cero en 12 épocas con Adam y tasa
   0.01.

Estos controles verifican que el flujo básico funciona, pero no demuestran que
la arquitectura tenga capacidad para aprender los 32 patrones.

## Primera corrida sobre los 32 patrones

Se mantuvo la configuración de la prueba mínima para cambiar una sola cosa: la
cantidad de muestras.

```text
Arquitectura: 35 -> 15 -> 2 -> 15 -> 35
Activaciones: tanh -> lineal -> tanh -> logística
Loss: MSE
Optimizador: Adam
Tasa: 0.01
Semilla: 0
Presupuesto: 20.000 épocas
```

La corrida no cumplió el requisito:

| Métrica final | Resultado |
|---|---:|
| Máximo de píxeles incorrectos | 7 |
| Promedio de píxeles incorrectos | 2,0625 |
| Patrones exactos | 7 de 32 |
| Patrones con como máximo un error | 17 de 32 |

La trayectoria mostró que el modelo había sido mejor antes del final:

| Época | Loss | Máximo de errores |
|---:|---:|---:|
| 0 | 4,497 | 21 |
| 100 | 2,011 | 11 |
| 1.000 | 0,506 | 3 |
| 3.493 | 0,446 | 3 |
| 10.000 | 0,484 | 4 |
| 20.000 | 1,032 | 7 |

Este resultado motivó dos decisiones metodológicas:

- no asumir que más épocas siempre mejoran el modelo;
- guardar la trayectoria completa y distinguir convergencia lenta de
  inestabilidad posterior.

No se seleccionará retrospectivamente el mejor checkpoint de una única
semilla. En los experimentos de selección se evalúa el estado final después de
un presupuesto común. El mínimo observado se registra solo para diagnosticar si
una tasa nunca aprendió, todavía seguía mejorando o aprendió y luego empeoró.

## Por qué estudiar conjuntamente arquitectura y optimizador

La capacidad de una arquitectura y la facilidad para optimizarla no son
independientes. Una tasa puede ser estable en una red y excesiva en otra; una
red profunda puede ser más sensible al método de optimización. Por eso no se
hará un estudio exhaustivo de arquitecturas con un único optimizador ni un
producto cartesiano de todos los hiperparámetros.

Se usa un embudo:

1. calibrar pocas tasas sobre dos arquitecturas ancla;
2. descartar tasas divergentes o claramente ineficientes;
3. cruzar arquitecturas representativas con perfiles viables;
4. refinar solo donde los resultados muestran una frontera o tendencia;
5. confirmar pocos finalistas con más semillas.

## Optimizadores implementados

- **Descenso de gradiente:** utiliza el gradiente actual.
- **Momentum:** agrega una fracción del cambio anterior para acumular dirección
  y reducir zigzag.
- **RMSProp:** divide cada paso por una escala basada en gradientes cuadrados
  recientes.
- **Adam:** combina promedio del gradiente, adaptación por segundo momento y
  corrección del sesgo inicial.

Los parámetros internos se mantuvieron fijos para no multiplicar el espacio de
búsqueda: `alpha=0.9` para Momentum, `gamma=0.9` para RMSProp y
`beta1=0.9`, `beta2=0.999` para Adam. En esta etapa solo se calibró la tasa.

## Calibración inicial de tasas

Se usaron dos arquitecturas ancla:

```text
anchor-15: 35 -> 15 -> 2 -> 15 -> 35
anchor-35: 35 -> 35 -> 2 -> 35 -> 35
```

Cada combinación se entrenó durante exactamente 2.000 épocas con semilla 0.
Esta única semilla sirve para eliminar valores claramente inviables; no permite
elegir un optimizador final.

Las tasas iniciales fueron:

| Optimizador | Tasas |
|---|---|
| Descenso | 0,01; 0,1; 1,0 |
| Momentum | 0,01; 0,05; 0,2 |
| RMSProp | 0,0001; 0,001; 0,01 |
| Adam | 0,0001; 0,001; 0,01 |

Las 24 corridas terminaron en 11,3 segundos y ninguna divergió. Ninguna alcanzó
un máximo de un píxel; el mejor máximo observado fue 3.

### Tasas demasiado lentas para el presupuesto

`GD 0.01`, `RMSProp 0.0001` y `Adam 0.0001` terminaron con máximos de entre 12 y
17 errores en ambas arquitecturas. Sus curvas continuaban descendiendo: no se
concluye que sean incapaces, sino que quedaron dominadas en eficiencia por tasas
mayores del mismo optimizador para un presupuesto común.

### Tasas inestables y evidencia de interacción

Momentum 0.2 con `anchor-15` alcanzó una loss mínima de 0,521 en la época 1.032,
pero terminó con loss 1,769 y 22 errores máximos. GD 1.0 mostró una inestabilidad
menor: mínimo 0,500 en la época 683 y final 0,625 con 8 errores máximos.

Sin embargo, con `anchor-35` esas mismas tasas continuaron mejorando hasta el
final: GD 1.0 terminó con máximo 3 y Momentum 0.2 con máximo 4. Por lo tanto, no
pueden clasificarse como universalmente malas; su estabilidad depende de la
arquitectura.

RMSProp y Adam con tasas 0.001 y 0.01 fueron estables en ambas anclas. La tasa
0.01 produjo menor loss y más patrones dentro del límite en este presupuesto,
por lo que se conserva para el tamiz. La tasa 0.001 queda como alternativa si
los entrenamientos más largos revelan inestabilidad.

## Refinamiento de GD y Momentum

La calibración dejó intervalos ambiguos. Se agregaron GD 0.3 y 0.5 entre la tasa
estable 0.1 y la agresiva 1.0, y Momentum 0.1 entre 0.05 y 0.2. Se mantuvieron
las dos anclas, la semilla 0 y las 2.000 épocas.

| Arquitectura | Optimizador | Tasa | Loss final | Loss mínima (época) | Máximo final | Patrones <= 1 error |
|---|---|---:|---:|---:|---:|---:|
| anchor-15 | GD | 0,3 | 0,521 | 0,519 (1.999) | 3 | 25 |
| anchor-15 | GD | 0,5 | 0,405 | 0,395 (1.944) | 4 | 26 |
| anchor-15 | Momentum | 0,1 | 0,532 | 0,501 (1.330) | 5 | 24 |
| anchor-35 | GD | 0,3 | 0,167 | 0,167 (2.000) | 2 | 29 |
| anchor-35 | GD | 0,5 | 0,253 | 0,250 (1.958) | 2 | 28 |
| anchor-35 | Momentum | 0,1 | 0,411 | 0,411 (2.000) | 6 | 24 |

GD 0.3 ofrece el compromiso más estable entre ambas anclas: termina cerca de su
mínimo y obtiene máximos de 3 y 2. GD 0.5 no mejora consistentemente el error
binario y presenta un pequeño deterioro final. Momentum 0.1 tampoco resuelve la
interacción observada.

Momentum todavía no se descarta como optimizador: la calibración usa una sola
semilla y el protocolo exige más de una para retirar una opción. Se conserva
Momentum 0.01, que fue estable en ambas anclas y obtuvo máximo 3 en la red ancha.

## Perfiles que pasan al tamiz de arquitecturas

Los perfiles provisionales son:

```text
GD 0.3
Momentum 0.01
RMSProp 0.01
Adam 0.01
```

Se cruzarán primero con semilla 0 contra arquitecturas representativas:

```text
A0: 35 -> 2 -> 35
A1: 35 -> 10 -> 2 -> 10 -> 35
A2: 35 -> 20 -> 2 -> 20 -> 35
A3: 35 -> 35 -> 2 -> 35 -> 35
A4: 35 -> 20 -> 10 -> 2 -> 10 -> 20 -> 35
```

La semilla 0 funciona como tamiz barato. Solo las combinaciones prometedoras se
repetirán con semillas 1 y 2. Si el ancho continúa mejorando, se ampliará el
rango por encima de 35; si aparece una frontera entre éxito y fracaso, se
probarán valores intermedios. Si la profundidad ayuda, se compararán perfiles
internos adicionales y, cuando sea posible, arquitecturas con cantidades de
parámetros parecidas.

## Estado de las conclusiones

Hasta aquí las conclusiones son provisionales:

- la implementación aprende y los gradientes fueron validados;
- la arquitectura de 15 con Adam 0.01 no cumple por sí sola;
- el resultado depende de la interacción arquitectura-optimizador-tasa;
- tasas muy pequeñas son ineficientes bajo el presupuesto, no necesariamente
  incapaces;
- GD 0.3, Momentum 0.01, RMSProp 0.01 y Adam 0.01 justifican continuar al tamiz;
- todavía no se eligió arquitectura, optimizador ni cantidad final de épocas.

Las configuraciones reproducibles son `configs/optimizer_calibration.json` y
`configs/optimizer_calibration_refinement.json`. Los resultados crudos quedan
en `results/optimizer-calibration-v1/` y
`results/optimizer-calibration-refinement-v1/`.
