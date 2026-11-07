# TP5 - Plan de trabajo y aprendizaje

Fecha de entrega y presentacion: **17 de noviembre de 2026**.

Este plan no supone que hay que entender toda la matematica antes de empezar.
La idea es avanzar en capas: entender una pieza pequena, implementarla, verificarla
y recien entonces agregar la siguiente. Las pausas marcadas como **PING-PONG** son
parte del trabajo, no demoras: no se avanza hasta poder explicar con palabras
propias que entra, que sale y que error se esta minimizando.

## Estado actual - 7 de noviembre

La **Etapa 0** quedo cerrada conceptualmente. Se tomaron estas decisiones:

- Se pueden usar NumPy, Matplotlib y herramientas generales, pero no una
  libreria que implemente o entrene el autoencoder por nosotros. Si aparece un
  enunciado 2026 actualizado, se vuelve a verificar esta interpretacion.
- TP5 tendra su propio paquete. Se reutilizaran y adaptaran las partes necesarias
  del MLP del TP3 (capas densas, activaciones, backpropagation, optimizadores y
  persistencia), sin importar directamente el paquete `sia_tp3` ni modificarlo.
- El autoencoder se entrena completo: para una entrada `x`, el objetivo es esa
  misma `x`; el error de reconstruccion actualiza encoder y decoder.
- El encoder comprime los 35 pixeles a `(z1, z2)` y el decoder reconstruye los
  35 pixeles a partir de esos dos valores.
- El criterio de cumplimiento sera el peor caso: cada uno de los 32 patrones
  debe tener como maximo un pixel incorrecto despues de umbralizar la salida. Un
  promedio menor que uno no alcanza si algun patron supera ese limite.
- Los hiperparametros viviran en archivos JSON y un runner generico guardara
  configuracion, semilla, historia, metricas, pesos y figuras por corrida. No se
  editaran constantes dentro del script para distinguir experimentos.

La **Etapa 1** tambien quedo cerrada el 7 de noviembre:

- El cargador lee directamente `data/font.h` y convierte cada fila hexadecimal
  en cinco bits, del bit mas significativo (izquierda) al menos significativo.
- Se obtienen imagenes de forma `(32, 7, 5)` y patrones aplanados de forma
  `(32, 35)`, con valores exclusivamente binarios.
- Diez pruebas verifican conversiones conocidas, cantidad y orden de caracteres,
  forma, valores y orientacion izquierda-derecha.
- La grilla `docs/font-original.png` fue inspeccionada visualmente y las letras
  se ven correctamente orientadas.

Proximo paso exacto: comenzar la Etapa 2 adaptando el MLP y demostrar que puede
sobreajustar un solo sprite; todavia no se elige la arquitectura final.

La **Etapa 2** quedo cerrada en su version minima el 7 de noviembre:

- Se adapto el nucleo del MLP del TP3 a un autoencoder con `encode`, `decode` y
  acceso explicito a la capa latente.
- La arquitectura didactica provisional es `35 -> 15 -> 2 -> 15 -> 35`, con
  `tanh` en las capas ocultas, codigo latente lineal y salida logistica.
- Se usa MSE inicialmente por ser una loss ya implementada y facil de verificar;
  esto no la convierte todavia en la eleccion final frente a BCE.
- Backpropagation coincidio con gradientes numericos en una red diminuta.
- Con Adam y tasa `0.01`, una unica letra `a` paso de 16 pixeles incorrectos a
  cero en 12 epocas. Su codigo latente aprendido fue aproximadamente
  `(1.742, -1.581)` para esa semilla.
- Este resultado solo demuestra que la implementacion supera una prueba basica
  de funcionamiento. No demuestra que la arquitectura pueda aprender los 32
  patrones ni que esos hiperparametros sean finales.

Proximo paso exacto: crear una corrida base reproducible sobre los 32 patrones
manteniendo inicialmente la misma arquitectura, loss y optimizador, para que el
unico cambio conceptual sea la cantidad de muestras.

## 1. Que pide el TP

### Parte 1A - Autoencoder basico

- Leer los 32 sprites binarios de `data/font.h`, de 7 filas por 5 columnas.
- Implementar encoder y decoder sin usar una libreria que construya o entrene la
  red neuronal.
- Comprimir los 35 pixeles a un espacio latente de dimension 2.
- Reconstruir los 32 patrones con **como maximo un pixel incorrecto por patron**.
- Comparar y justificar arquitecturas y tecnicas de optimizacion.
- Graficar los 32 codigos latentes en dos dimensiones.
- Elegir un punto nuevo del espacio latente y mostrar la letra/sprite que genera
  el decoder.

### Parte 1B - Denoising autoencoder (DAE)

- Entrenar una arquitectura conveniente con una entrada distorsionada y el
  patron limpio como salida esperada.
- Evaluar distintos niveles de ruido y medir cuanto del original recupera.

### Parte 2 - Variational autoencoder (VAE)

- Elegir o construir un dataset nuevo.
- Implementar el encoder variacional, el muestreo mediante reparametrizacion y
  la perdida de reconstruccion mas regularizacion KL.
- Generar muestras nuevas y justificar que pertenecen a la familia aprendida.

La explicacion oral de 2026 agrega dos precisiones que no aparecen en el PDF de
2023: **no usar emojis** y construir el VAE incrementalmente. Proponer codigo
simple fue un desafio opcional del docente, no un requisito. Tambien aclara que
se pueden usar herramientas auxiliares, pero no una libreria que implemente la
red por nosotros. Conviene confirmar estas reglas si se publica una version
actualizada del enunciado.

## 2. Criterios que vamos a conservar durante todo el TP

1. **Una sola novedad por paso.** No cambiar arquitectura, optimizador, loss y
   tasa de aprendizaje simultaneamente.
2. **Prueba pequena antes de corrida larga.** Primero una muestra, luego las 32,
   luego varias semillas.
3. **Cada corrida debe responder una pregunta.** Si no sabemos que decision
   permitiria tomar, no se ejecuta.
4. **Separar probabilidad de pixel de pixel decidido.** La salida sigmoidea es
   continua; para contar pixeles incorrectos se aplica un umbral (inicialmente
   0.5) y se compara con el patron binario.
5. **Reproducibilidad.** Toda corrida guarda configuracion, semilla, historia de
   loss, metricas, tiempo y artefactos relevantes.
6. **Seleccionar con varias semillas.** Una semilla sirve para depurar; una
   conclusion necesita repeticiones.
7. **Documentar al decidir, no al final.** Cada etapa deja una nota breve con
   hipotesis, resultado, interpretacion y proximo paso.

## 3. La matematica minima, en el orden en que la necesitaremos

No hace falta estudiar toda la teoria de informacion de una vez.

1. **Autoencoder:** `x -> encoder -> z -> decoder -> x_hat`. El encoder aprende
   una representacion de 35 valores en solo 2; el decoder intenta recuperar
   `x`. Se necesita entender propagacion hacia adelante, funcion de perdida y
   regla de la cadena/backpropagation.
2. **DAE:** el mecanismo no cambia. Solo cambia el par de entrenamiento:
   `entrada = x_con_ruido`, `objetivo = x_limpio`. Si el objetivo tambien fuera
   ruidoso, la red aprenderia a copiar ruido.
3. **VAE, paso 1:** en vez de producir directamente `z`, el encoder produce dos
   vectores: `mu` y `log_var`.
4. **VAE, paso 2:** se toma `epsilon ~ N(0, I)` y se calcula
   `z = mu + exp(0.5 * log_var) * epsilon`. Esto separa el azar de los parametros
   entrenables y permite retropropagar.
5. **VAE, paso 3:** la perdida suma reconstruccion y
   `KL(q(z|x) || N(0,I))`. Reconstruccion pide conservar informacion; KL pide un
   espacio latente continuo y sampleable. El peso de KL controla la tension
   entre ambos objetivos.

Antes de programar el VAE se escribiran en papel las dimensiones de cada tensor
y los gradientes de `mu` y `log_var`. Luego se validaran numericamente en un caso
minusculo mediante diferencias finitas.

## 4. Etapas de implementacion

### Etapa 0 - Alinear reglas y preparar el terreno

**Objetivo:** que sepamos exactamente que esta permitido y como mediremos exito.

- Revisar el enunciado, el cierre de la transcripcion y las diapositivas
  relevantes.
- Confirmar si la version vigente mantiene: NumPy y Matplotlib permitidos, red
  neuronal propia, emojis excluidos.
- Definir estructura del proyecto, configuraciones y formato de resultados.
- Decidir si reutilizamos conceptos/codigo general del MLP del TP3. Reutilizar
  activaciones u optimizadores propios puede ahorrar trabajo, pero el TP5 debe
  conservar una interfaz clara para encoder, decoder y luego las dos ramas del
  VAE.

**PING-PONG 0 - alcance y reglas**

Preguntas a poder contestar antes de seguir:

- ¿Por que la salida esperada del autoencoder es igual a su entrada?
- ¿Que significa que el cuello de botella tenga dimension 2?
- ¿La catedra exige implementar tambien los optimizadores o solo la red?
- ¿Que criterio exacto usaremos para afirmar que se aprendieron los 32 patrones?

**Salida:** reglas anotadas y esqueleto reproducible, todavia sin entrenar.

### Etapa 1 - Entender y probar el dataset

**Objetivo:** eliminar cualquier ambiguedad de bits, filas y orientacion.

- Implementar un parser pequeno para `font.h` o trasladar sus constantes de
  forma verificable.
- Convertir cada entero de fila a 5 bits y aplanar cada sprite a 35 valores.
- Mostrar una grilla con los 32 patrones y sus caracteres.
- Agregar pruebas sobre forma `(32, 35)`, valores binarios y algunos patrones
  conocidos. Verificar visualmente izquierda/derecha para no entrenar letras
  espejadas.

**PING-PONG 1 - representacion**

- Convertir juntos una fila hexadecimal a cinco bits.
- Explicar por que aplanar 7x5 no destruye informacion para una MLP, aunque si
  pierde la nocion explicita de vecindad.
- Diferenciar cantidad de muestras (32) de cantidad de features (35).

**Salida:** cargador probado y figura de los datos originales.

### Etapa 2 - Autoencoder didactico minimo

**Objetivo:** verificar forward y backward sin que el dataset o la arquitectura
oculten errores.

- Implementar una red simetrica muy pequena con bottleneck 2.
- Hacer una pasada forward e imprimir dimensiones y rangos.
- Intentar sobreajustar primero **un solo sprite**. Si no puede copiar una
  muestra, no tiene sentido entrenar 32.
- Validar gradientes de una red diminuta con diferencias finitas.
- Graficar loss y reconstruccion durante el entrenamiento.

La primera loss candidata sera coherente con pixeles binarios (sigmoide de
salida y binary cross-entropy), pero antes de fijarla compararemos su derivada
con MSE, que es mas intuitiva. No se elige por costumbre: se deja escrito que
cambia en la señal de error.

**PING-PONG 2 - forward, loss y backprop**

- Seguir un ejemplo desde `x` hasta `x_hat` indicando dimensiones.
- Calcular a mano la contribucion de un pixel a la loss.
- Explicar por que el umbral para contar errores no se usa dentro del
  entrenamiento (no es derivable).

**Salida:** un sprite reconstruido correctamente y gradientes validados.

### Etapa 3 - Autoencoder sobre los 32 patrones

**Objetivo:** alcanzar el criterio obligatorio sin hacer un barrido ciego.

Orden de experimentacion propuesto:

1. Fijar una arquitectura base razonable y comprobar que aprende.
2. Ajustar la tasa del optimizador en un rango corto y logaritmico.
3. Comparar dos o tres capacidades simetricas, por ejemplo una red con una capa
   oculta por lado y otra con dos. Estos tamaños son candidatos iniciales, no
   decisiones tomadas.
4. Solo si sigue fallando, comparar activacion oculta o un segundo optimizador
   ya implementado en el TP3.
5. Reentrenar los dos finalistas con varias semillas.

Metricas obligatorias por corrida:

- maximo de pixeles incorrectos en un patron;
- cantidad de patrones con 0, 1 y mas de 1 errores;
- promedio de pixeles incorrectos;
- loss continua final y mejor epoca;
- tiempo y cantidad de actualizaciones.

No alcanza con que el error promedio sea menor que uno: el requisito dice que
el **peor patron** debe tener como maximo un pixel incorrecto.

**PING-PONG 3 - decidir experimentos**

- Mirar la corrida base y formular una hipotesis antes de tocar parametros.
- Elegir juntos que variable cambiar y que resultado apoyaria o refutaria la
  hipotesis.
- Separar “no tiene capacidad” de “tiene capacidad pero no optimiza bien”.

**Salida:** configuracion final justificada y tabla compacta de alternativas.

### Etapa 4 - Interpretar y generar con el autoencoder

**Objetivo:** no limitarse a que la loss baje.

- Graficar los 32 puntos `(z1, z2)` etiquetados por caracter.
- Observar vecinos, zonas vacias y posibles valores latentes extremos.
- Decodificar una grilla regular del plano para ver como cambia la salida.
- Elegir puntos entre dos letras cercanas y fuera de los codigos exactos.
- Definir “nueva letra” como una salida binarizada que no coincide con ninguno
  de los 32 patrones, mostrando tambien sus probabilidades sin umbral.

**PING-PONG 4 - generacion**

- ¿Por que interpolar dos codigos no garantiza una letra valida en un AE comun?
- ¿Que evidencia permite decir que la salida es nueva y no una copia?
- ¿Que diferencia hay entre una salida novedosa y una salida simplemente mala?

**Salida:** mapa latente, grilla generativa y al menos un caso discutido.

### Etapa 5 - Denoising autoencoder

**Objetivo:** medir recuperacion, no mostrar solo ejemplos favorables.

- Empezar con ruido salt-and-pepper/bit-flip, natural para datos binarios.
- En cada epoca generar corrupciones nuevas en vez de memorizar una unica copia
  ruidosa.
- Entrenar con `(x_con_ruido, x_limpio)`.
- Evaluar en corrupciones no usadas durante la actualizacion de pesos.
- Comparar varios niveles de probabilidad de flip, incluyendo 0 como control.
- Reportar error de entrada y error de salida; la mejora es la diferencia entre
  ambos. Incluir promedio, dispersion y ejemplos faciles/dificiles.
- Si el tiempo alcanza, probar ruido estructurado (bloques o filas) como
  extension, sin mezclarlo con la conclusion principal.

**PING-PONG 5 - protocolo de ruido**

- Distinguir ruido de entrenamiento y de evaluacion.
- Decidir que niveles son informativos despues de un piloto corto.
- Explicar por que a ruido extremo puede ser imposible identificar el original.

**Salida:** curva capacidad de recuperacion vs. ruido y ejemplos representativos.

### Etapa 6 - VAE de juguete, todavia sin dataset final

**Objetivo:** aprender el mecanismo variacional por partes, tal como indico el
docente.

Subpasos que deben funcionar por separado:

1. Autoencoder determinista sobre un conjunto diminuto.
2. Encoder con dos salidas, `mu` y `log_var`, aun usando `z = mu`.
3. Agregar `epsilon` y reparametrizacion, con KL temporalmente en cero.
4. Verificar los gradientes de ambas ramas.
5. Agregar KL con un peso pequeno y registrar reconstruccion y KL por separado.
6. Muestrear `z ~ N(0,I)` y pasar por el decoder.

**PING-PONG 6A - probabilidad sin formulas largas**

- Diferenciar “un punto z” de “una distribucion de posibles z”.
- Explicar que representan `mu`, varianza y `epsilon`.

**PING-PONG 6B - loss del VAE**

- Explicar la tension reconstruccion vs. KL con un ejemplo extremo para cada
  termino.
- Derivar juntos la reparametrizacion y revisar los gradientes numericos.

**Salida:** VAE minimo probado, con curvas separadas de reconstruccion y KL.

### Etapa 7 - Elegir y preparar el dataset nuevo

**Objetivo:** elegir algo evaluable y terminable, no solo llamativo.

Criterios de eleccion:

- representacion fija y pequena;
- suficientes muestras y variacion para aprender una distribucion;
- criterio objetivo o al menos claro de validez;
- generacion visible/evaluable;
- preparacion que no consuma mas tiempo que el modelo;
- sin emojis.

Opciones para discutir:

- **sprites geometricos parametrizados:** riesgo bajo, generacion visual y
  pertenencia facil de justificar;
- **glifos o simbolos sinteticos distintos de `font.h`:** riesgo medio y buena
  continuidad visual, cuidando que sea realmente un dataset nuevo;
- **codigo de un lenguaje minimo:** interesante pero de riesgo alto; exige una
  codificacion fija y validar sintaxis/semantica. Solo elegirlo si ya esta
  estable el VAE y queda margen.

**PING-PONG 7 - decision del dataset**

Armar una tabla corta con valor pedagogico, costo, riesgo y forma de evaluar.
No implementar el dataset definitivo hasta tomar esta decision conscientemente.

**Salida:** dataset versionado, visualizacion/ejemplos, reglas de pertenencia y
division reproducible si corresponde.

### Etapa 8 - VAE final y experimentos acotados

**Objetivo:** obtener reconstrucciones y muestras nuevas defendibles.

- Hacer primero que el AE determinista reconstruya el dataset; esto separa un
  problema de capacidad de un problema variacional.
- Incorporar el VAE ya validado.
- Registrar siempre loss total, reconstruccion y KL.
- Revisar reconstrucciones, distribucion de `mu`, escalas de varianza y muestras
  del prior. Una loss sola no detecta colapso latente.
- Estudiar pocos factores con orden: capacidad base, tasa de aprendizaje y peso
  de KL. No hacer producto cartesiano completo.
- Confirmar finalistas con varias semillas y generar una muestra fija para poder
  comparar modelos.

**PING-PONG 8 - diagnostico**

- Buena reconstruccion pero muestras incoherentes: posible espacio latente no
  regularizado.
- KL casi cero y decoder que ignora `z`: posible colapso.
- Ambas losses malas: revisar capacidad, escalas, gradientes u optimizacion antes
  de ampliar el barrido.

**Salida:** comparacion pequena, modelo elegido, interpolaciones y muestras del
prior juzgadas con el criterio definido en la etapa 7.

### Etapa 9 - Informe, presentacion y defensa

**Objetivo:** que cada figura responda una pregunta y que podamos explicar el
codigo sin depender de memoria reciente.

- Escribir el informe en paralelo, consolidarlo al final.
- Presentar el recorrido: problema, representacion, mecanismo, experimentos,
  decision y limites.
- Incluir arquitectura con dimensiones, loss, protocolo de semillas y metrica
  de pixeles.
- Mostrar pocos experimentos, elegidos por lo que ensenaron; dejar los barridos
  completos como respaldo.
- Preparar una explicacion oral simple de AE, DAE, `mu/log_var`,
  reparametrizacion y KL.
- Ensayar preguntas: por que esa arquitectura, por que esa loss, por que esos
  niveles de ruido, por que ese dataset, que fallo y como se diagnostico.

**PING-PONG 9 - ensayo de defensa**

Hacer una ronda donde yo pregunto como docente y otra donde la explicacion se
hace sin mirar formulas. Las partes que no se puedan explicar vuelven a una
prueba o figura concreta; no se memorizan de forma aislada.

## 5. Estrategia de hiperparametros y computo

No sabemos aun cuanto tarda una corrida; por eso el presupuesto se fija despues
de medir el primer piloto.

### Embudo de experimentos

1. **Depuracion:** una configuracion, una semilla, pocas epocas. No se usa para
   concluir.
2. **Piloto:** pocos valores bien separados, una semilla y corte temprano si es
   claramente inviable.
3. **Refinamiento:** uno o dos valores alrededor de la mejor region.
4. **Confirmacion:** dos finalistas con al menos 3 semillas; usar 5 solo si el
   costo medido es razonable o los resultados son muy variables.
5. **Final:** configuracion congelada y artefactos reproducibles.

### Limites practicos

- No lanzar un grid de arquitectura x activacion x optimizador x tasa x batch.
- Para los 32 sprites, full-batch es un baseline natural; comparar otra modalidad
  solo si hay una hipotesis.
- Usar una cantidad maxima de epocas como seguro, pero guardar el mejor estado
  segun el criterio definido.
- Medir tiempo desde la primera corrida y estimar el costo total antes de cada
  tanda.
- Toda corrida larga debe poder dejarse ejecutando y reanudarse/analizarse sin
  repetir las anteriores.
- Si una tanda supera el presupuesto acordado, reducir candidatos o semillas,
  no la calidad de las metricas.

## 6. Calendario hasta el 17 de noviembre

El calendario deja el dia 16 como colchon. Si un dia no se puede trabajar, se
conserva el orden y se recorta primero una extension, nunca las validaciones.

| Fecha | Foco | Resultado minimo al cerrar |
|---|---|---|
| 4 nov | Plan y lectura | Alcance, riesgos y preguntas identificados |
| 5 nov | Etapas 0-1 | Reglas, parser probado y grilla de sprites |
| 6 nov | Etapa 2 | Forward/backward y un patron sobreajustado |
| 7 nov | Etapa 3, pilotos | Baseline de 32 patrones y primer diagnostico |
| 8 nov | Etapas 3-4 | Finalistas, mapa latente y generacion AE |
| 9 nov | Etapa 5 | DAE con curva inicial de ruido |
| 10 nov | Cierre parte 1 | Resultados repetibles y texto preliminar |
| 11 nov | Etapa 6, pasos 1-3 | `mu`, `log_var` y reparametrizacion entendidos |
| 12 nov | Etapa 6, pasos 4-6 | Gradientes verificados y VAE de juguete |
| 13 nov | Etapa 7 | Dataset final elegido, preparado y visualizado |
| 14 nov | Etapa 8 | VAE final y pilotos medidos |
| 15 nov | Etapas 8-9 | Confirmacion, figuras e informe/presentacion |
| 16 nov | Colchon y ensayo | Corrida reproducible, defensa y revision final |
| 17 nov | Entrega | Solo empaquetado y verificacion, sin experimentos nuevos |

### Version minima si se complica la semana

Se preserva, en este orden:

1. AE que cumple el maximo de un pixel y su mapa latente.
2. DAE con bit-flip y tres niveles informativos de ruido.
3. VAE sobre un dataset visual simple con una comparacion de peso KL.
4. Tres semillas para configuraciones finales.
5. Informe y presentacion reproducibles.

Se recortan primero: ruido estructurado, dataset de codigo, arquitecturas extra,
grids densos y figuras redundantes.

## 7. Registro breve para retomar despues de una pausa

Al terminar cada sesion, completar estas cinco lineas en una bitacora:

```text
Estado actual:
Ultima evidencia confiable:
Decision tomada y por que:
Pregunta abierta:
Proximo paso exacto (un solo paso):
```

Esto evita volver a leer todo el proyecto y, sobre todo, evita confundir una
idea pendiente con una decision ya respaldada por resultados.

## 8. Primer paso cuando empecemos a implementar

Empezaremos por **PING-PONG 0** y la Etapa 1. No elegiremos todavia la
arquitectura final ni el dataset del VAE. La primera meta concreta sera poder
mirar los 32 sprites, explicar su representacion y confiar en que el modelo
recibira exactamente los bits que creemos.
