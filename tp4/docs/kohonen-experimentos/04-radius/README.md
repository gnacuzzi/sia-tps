# Experimento 4: radio inicial

Se compararon `k = 4`, `k = 5` y `k = 6`, cuatro perfiles de tasa por tamaño,
cinco radios iniciales y cinco semillas: 300 corridas. Los radios recorren 0 %,
25 %, 50 %, 75 % y 100 % del trayecto entre el vecindario local `R = 1` y el
diámetro de cada grilla. Todos terminan en `R = 1` mediante descenso lineal;
el primer nivel permanece constante porque comienza y termina en 1.

![Curvas de calidad por radio](generated/curvas_radio.png)

## Mínimo error dentro de cada perfil de tasa

| k | Schedule | Tasa | Nivel de radio | Radio | Error | Muertas (%) | Máximo de hits (%) | Tiempo (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | constant | 0.05 → 0.05 | 50% | 2.621 → 1 | 2.5468 | 3.8 | 13.2 | 0.0777 |
| 4 | constant | 0.1 → 0.1 | 25% | 1.811 → 1 | 2.5435 | 7.5 | 12.8 | 0.0778 |
| 4 | decay | 0.3 → 0.03 | 25% | 1.811 → 1 | 2.5376 | 3.8 | 12.0 | 0.0781 |
| 4 | decay | 0.7 → 0.07 | 100% | 4.243 → 1 | 2.5448 | 3.8 | 12.4 | 0.0799 |
| 5 | constant | 0.1 → 0.1 | 25% | 2.164 → 1 | 2.2728 | 16.0 | 10.4 | 0.0793 |
| 5 | constant | 0.2 → 0.2 | 25% | 2.164 → 1 | 2.3165 | 10.4 | 10.8 | 0.0798 |
| 5 | decay | 0.7 → 0.07 | 50% | 3.328 → 1 | 2.2453 | 12.8 | 9.6 | 0.0815 |
| 5 | decay | 0.9 → 0.09 | 25% | 2.164 → 1 | 2.2524 | 16.0 | 10.0 | 0.0813 |
| 6 | constant | 0.1 → 0.1 | 25% | 2.518 → 1 | 2.0242 | 21.1 | 7.6 | 0.0847 |
| 6 | constant | 0.2 → 0.2 | 0% | 1.000 → 1 | 1.9297 | 37.8 | 8.0 | 0.0825 |
| 6 | decay | 0.7 → 0.07 | 25% | 2.518 → 1 | 1.9563 | 31.1 | 8.4 | 0.0839 |
| 6 | decay | 0.9 → 0.09 | 0% | 1.000 → 1 | 1.9652 | 28.9 | 7.6 | 0.0833 |

La tabla no selecciona automáticamente esos mínimos: el radio organiza la
topología del mapa y debe juzgarse también mediante matrices U y mapas de hits.
Esas figuras se generarán sólo para las configuraciones representativas, no
para las 300 corridas.

## Lectura de los resultados

- En `k = 4` las diferencias de error son pequeñas. Dos compromisos especialmente
  estables son la tasa constante `η = 0.1` con `R = 1` (1,25 % de unidades
  muertas) y el descenso `0.3 → 0.03` con radio inicial a mitad del recorrido
  (también 1,25 %). Se conservan ambos para la inspección visual.
- En `k = 5` el radio interactúa con la tasa. Con `η = 0.1`, comenzar a mitad
  del recorrido mantiene prácticamente el mínimo error y reduce las unidades
  muertas de 17,6 % a 10,4 %. Con `η = 0.2`, comenzar en el diámetro completo
  las reduce a 4,8 %. Las alternativas decrecientes también dejan candidatos
  y deben decidirse mirando la estructura del mapa.
- En `k = 6`, los radios locales producen errores bajos a costa de entre 28,9 %
  y 37,8 % de unidades muertas. Los radios amplios bajan esa fracción hasta
  aproximadamente 15,6–22,2 %, aunque el tamaño sigue peor ocupado que `k = 4`
  y `k = 5`. Se conserva sólo como caso de frontera para la comparación visual.

Por lo tanto, el barrido no justifica elegir el mínimo error aislado. Primero
se filtran soluciones con ocupación razonable y estabilidad entre semillas;
luego se comparan sus mapas de hits y matrices U.

## Mapas de las configuraciones finalistas

Se conservaron nueve compromisos: dos de `k = 4`, cuatro de `k = 5` y tres de
`k = 6`. Para cada uno se muestra la semilla más central respecto de error,
unidades muertas y concentración máxima, luego de normalizar la variabilidad
de cada métrica. No se eligió la mejor semilla.

| Corrida representativa | Semilla | Error | Ocupadas | Máximo de hits |
| --- | ---: | ---: | ---: | ---: |
| `k04-constant-eta-10-radius-000-seed-04` | 4 | 2.5494 | 16/16 | 5 |
| `k04-decay-eta-30-radius-050-seed-03` | 3 | 2.5515 | 16/16 | 5 |
| `k05-constant-eta-20-radius-100-seed-02` | 2 | 2.3249 | 24/25 | 4 |
| `k05-decay-eta-70-radius-050-seed-04` | 4 | 2.2563 | 22/25 | 5 |
| `k05-decay-eta-70-radius-075-seed-01` | 1 | 2.3110 | 23/25 | 4 |
| `k05-decay-eta-90-radius-000-seed-04` | 4 | 2.2757 | 23/25 | 4 |
| `k06-constant-eta-10-radius-100-seed-00` | 0 | 2.0643 | 31/36 | 4 |
| `k06-decay-eta-70-radius-100-seed-03` | 3 | 2.0658 | 32/36 | 4 |
| `k06-decay-eta-90-radius-075-seed-02` | 2 | 2.0437 | 29/36 | 4 |

![Mapas de hits finalistas](generated/mapas_hits_finalistas.png)

![Matrices U finalistas](generated/matrices_u_finalistas.png)

Los valores dibujados también están disponibles en
`generated/matrices_finalistas.csv`.

### Validación visual y paso a épocas

- Los dos mapas `k = 4` representativos ocupan sus 16 neuronas y distribuyen
  los hits sin una celda dominante. El radio local constante produce contrastes
  más marcados en la matriz U; el descenso `0.3 → 0.03` con radio intermedio
  genera una transición más suave. Ambos pasan al barrido de épocas para no
  decidir anticipadamente entre separación y continuidad topológica.
- En `k = 5`, `η = 0.2` con radio completo y `0.7 → 0.07` con radio al 75 %
  presentan sólo una y dos celdas vacías, respectivamente, y matrices U con
  regiones continuas. El radio al 50 % deja más huecos y la tasa `0.9 → 0.09`
  con radio local fue menos estable entre semillas; no avanzan.
- En `k = 6` siguen apareciendo varias unidades vacías. El descenso
  `0.7 → 0.07` con radio completo es el representante mejor ocupado y se
  conserva únicamente como control de frontera. Las otras dos alternativas no
  aportan una mejora visual que compense la fragmentación.

Por lo tanto, el barrido de épocas continúa con cinco configuraciones: las dos
de `k = 4`, las dos seleccionadas de `k = 5` y una de `k = 6`. Esta reducción
se apoya conjuntamente en las cinco semillas y en los mapas representativos;
no convierte una sola figura en criterio de selección.

## Performance computacional

La consigna usa “performance” principalmente para pedir una justificación de
la calidad del modelo. Además se registró el tiempo de entrenamiento de cada
corrida como dato auxiliar. Es una medición dependiente de la máquina y no se
usa como criterio principal; permite comparar el costo cuando dos alternativas
producen resultados equivalentes.

Los tiempos medios por configuración estuvieron entre
`77.2` y `91.4` ms. La diferencia
es demasiado pequeña para compensar una peor ocupación o una estructura menos
interpretable. Sí se tendrá en cuenta si dos configuraciones resultan
equivalentes en calidad. Para esta implementación, el trabajo por época crece
principalmente con la cantidad de muestras, variables y neuronas, es decir,
aproximadamente con `N × d × k²`; el radio no cambia ese orden de complejidad.

![Tiempos por radio](generated/tiempos_radio.png)

Los resultados individuales están en `generated/corridas.csv` y los promedios
por configuración en `generated/resumen_radio.csv`.
