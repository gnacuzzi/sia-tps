Comentario general
- Busquen siempre incluir los hiperparámetros elegidos para cada experimento
- Recuerden incluir barras de error cuando hacen promedios de corridas y/o trabajan con variables aleatorias
- Expresen sus datos de manera gráfica y comparativa
- Recuerden siempre justificar las afirmaciones con experimentos
- Excelentes conclusiones en todos los ejercicios



Ejercicio 1.
- Está excelente analizar el conjunto que se va a usar antes de realizar un ajuste de modelo pero siempre busquen contrastar la información que muestran. Busquen contar una historia y qué significa el dato: un gráfico de barras comparando total de transacciones cuantas son fraudulentas y cuántas no lo son aportan mucha más información que números aislados
- Cuidado al escalar los datos. Uno suele escalar la salida cuando la función de activación demanda que la salida esté entre dos valores (como sucede en la tangente (-1 y 1) o la función logística (0 y 1)). En ese caso, hay que tener cuidado con qué se puede dejar fuera de rango. Si en el conjunto de prueba queda un valor que equivale a más de 1, es esperable que la red sature pero de todas maneras prediga 1 (lo cuál no es un problema). En el caso de las entradas, se suele realizar una estandarización suponiendo una distribución normal y llevando la variable a media 0 y desvío 1. Una idea interesante es comparar diferentes formas de normalizar las características (la entrada): estandarización vs escalamiento vs no normalizar
- Bien mostrar la saturación de los perceptrones
- Recuerden mostrar la distribución de datos obtenida en cada k-fold para mostrar que los mismos no tienen bias y realmente capturan la riqueza del dataset. También es deseable comparar diferentes ks para justificar la elección de k=5
- Antes de mostrar gráficos sobre la variación de los errores de validación y entrenamiento en la época de convergencia, muestren el gráfico de error de entrenamiento y validación a lo largo de las épocas. A priori no deberían guiarse únicamente por el valor de error de training, el mínimo error de validación puede darse mucho antes de cuando el entrenamiento no muestra mejoras (momento de saturación de la capacidad del perceptrón)
- Cuando comparan los diferentes learning rates, vean de incluir un gráfico de zoom en las últimas épocas. Estos gráficos son muy útiles cuando hay una gran diferencia en el rango de valores al inicio y al final del experimento
- Excelente la decisión del umbral. Muy interesante las propuestas alternativas dependiendo de qué se quiere maximizar
- Muy buena la comparación entre las predicciones de TinyModel y BigModel. Recuerden expresar los resultados de manera gráfica. Para el caso de las métricas de precisión, recall, VP, FP, VN y FN, es útil presentarlo con una matriz de confusión
- Muy bien las conclusiones



Ejercicio 2.
- Muy bien la detección del desbalance entre clases
- Muy interesante el análisis de la evolución del F1-score con diferentes learning rates a lo largo de las épocas. Recuerden siempre incluir los hiperparámetros con los que corrieron el experimento (esto es muy importante)
- Recuerden siempre al hacer promedios de corridas incluir barras de error
- Observen que en la diapositiva 26 afirman que el mejor valor en términos de compromiso tarda 23% menos tiempo. Para realizar esa afirmación deben incluir un gráfico que compare los tiempos entre las configuraciones
- Cuidado al comparar arquitecturas. Mientras más parámetros (pesos) contenga una red, más tarda en entrenarse. Por ello, para comparar arquitecturas primero se debe hacer el gráfico de error de train/validation a lo largo de las épocas y validar cuando se consiguen los mejores valores para dicha arquitectura. No se debe asumir que 200 épocas es el valor idílico para cualquier arquitectura
- Muy bueno el análisis sobre optimizadores (recuerden lo del gráfico train/val). En general, el experimento se debe hacer con un número elevado de corridas (al menos 100) pero dado el tiempo acotado es razonable que sea un número bajo. La próxima busquen usar al menos 10 semillas para el box-plot
- Excelente la matriz de confusión



Ejercicio 3.
- Para F1 en función de la semilla: Cuidado con unir puntos que no pertenecen a una serie temporal, en este caso el valor de la semilla es arbitrario y no guarda correlación con la numeración. En estos casos un buen gráfico es un box plot para cada variable (control y translation, por ejemplo)
- Excelentes los experimentos realizados. Se destacan los analísis realizados a partir de la evolución del accuracy de validación a lo largo de las épocas