# Trabajos Prácticos de Sistemas de Inteligencia Artificial

Monorepo de trabajos prácticos de la materia Sistemas de Inteligencia
Artificial del ITBA.

## Integrantes

- Facundo Lasserre
- Gianna Lucía Nacuzzi
- Javier Bautista Peral Belmont
- Manuel José Santamarina Balbín
- Santiago Mesa Rubio

## TP1 — Métodos de búsqueda

El primer trabajo estudia métodos de búsqueda informados y no informados. Para
el 8-puzzle se proponen una representación de estado, heurísticas admisibles y
estrategias de búsqueda. Para Sokoban se implementa un motor con BFS, DFS,
Greedy y A*, distintas heurísticas, detección de estados repetidos, métricas,
experimentos reproducibles y animaciones de las soluciones.

La implementación, las instrucciones de ejecución y el detalle de los
resultados se encuentran en el [README del TP1](tp1/README.md).

## TP2 — Algoritmos genéticos

El segundo trabajo estudia algoritmos genéticos. Se implementa un motor que
aproxima una imagen mediante triángulos translúcidos sobre un canvas, con los
métodos de selección vistos en clase (elite, ruleta, universal, Boltzmann,
torneos y ranking), ambas estrategias de supervivencia, distintos métodos de
cruza y mutación, criterios de corte configurables, métricas de convergencia y
diversidad, y experimentos reproducibles comparando operadores e
hiperparámetros.

La implementación, las instrucciones de ejecución y el detalle de los
resultados se encuentran en el [README del TP2](tp2/README.md).

## TP3 — Perceptrón simple y multicapa

El tercer trabajo implementa perceptrones simples y redes multicapa con NumPy.
Además de validar la base con AND, ajuste de funciones y XOR, resuelve los tres
ejercicios obligatorios: destilación de un modelo de fraude, clasificación de
dígitos y reentrenamiento con datos adicionales. Incluye comparación de
arquitecturas, learning rates, optimizadores y tamaños de lote; validación
cruzada; métricas de clasificación; data augmentation y evaluaciones finales
reproducibles sobre conjuntos de test reservados.

La consigna, las fórmulas, los comandos y resultados están en el [README del TP3](tp3/README.md).
