# Resultados finales de experimentación

## Protocolo

Las campañas se ejecutaron de forma secuencial: cada etapa mantuvo fijos los
factores ya elegidos y modificó solamente el factor estudiado. Cada condición
comparativa recibió cinco semillas (`101`, `202`, `303`, `404`, `505`) y el
mismo presupuesto de evaluaciones dentro de su campaña.

La medida primaria fue el NMSE final mediano. La AUC normalizada permitió
comparar velocidad de convergencia y la diversidad ayudó a interpretar posibles
casos de convergencia prematura. Cuando participaron varias imágenes, las
condiciones se ordenaron primero dentro de cada objetivo para no promediar NMSE
absolutos de problemas con distinta dificultad.

La auditoría final verificó 163 corridas completas, sin identificadores
repetidos, con el presupuesto esperado y con `best.png`, `metadata.json` y
`metrics.csv` presentes en cada ejecución.

## Configuración resultante

| Componente | Decisión |
| --- | --- |
| Resolución de trabajo | 64 píxeles de lado máximo |
| Triángulos para Colombia | 25 |
| Triángulos para señal e ícono | 100 |
| Población / descendencia | 50 / 50 |
| Selección de padres | Torneo determinístico de tamaño 5 |
| Cruza | Uniforme por genes completos, probabilidad 0,9 |
| Mutación | Un gen con cambio local, probabilidad 1,0 |
| Supervivencia | Aditiva con selección Elite |
| Fitness | `1 - NMSE`, con el NMSE registrado por separado |

## 1. Resolución de trabajo

Con Colombia y 25 triángulos, 64 píxeles obtuvo un NMSE mediano de `0,002182`,
frente a `0,002388` para 32 píxeles: una reducción aproximada del 8,6 %. El
tiempo mediano aumentó de 11,90 a 14,48 segundos y los intervalos intercuartiles
se superpusieron.

La diferencia no fue amplia, pero se eligió 64 porque conserva más detalle para
las imágenes posteriores y su costo adicional fue aceptable. El descubrimiento
fue que reducir la resolución acelera la evaluación, pero también puede ocultar
errores que interesan visualmente.

## 2. Cantidad de triángulos

Los NMSE medianos fueron `0,003965`, `0,002182` y `0,002360` para 10, 25 y 50
triángulos. Diez triángulos convergieron rápido, pero no tuvieron capacidad
suficiente. Cincuenta ofrecieron más capacidad teórica, aunque duplicaron casi
el tiempo de 25 —25,81 frente a 14,42 segundos— y no consiguieron aprovecharla
con el mismo presupuesto.

Se eligieron 25 triángulos para Colombia como equilibrio entre expresividad y
dificultad de búsqueda. Esta decisión no se trasladó a las imágenes complejas:
la señal y el ícono conservaron 100 triángulos.

## 3. Selección de padres

Tournament 5 y Ranking obtuvieron NMSE medianos de `0,001718` y `0,001722`.
La diferencia de 0,23 % es menor que la variación entre semillas: Ranking ganó
tres comparaciones emparejadas y Tournament 5 ganó dos. Por eso se interpretó
como un empate práctico y ambos avanzaron a la campaña de cruza.

Tournament 5 mostró mejor AUC (`0,097548` frente a `0,101129`), pero terminó con
menos diversidad (`0,000830` frente a `0,001593`). La presión alta aceleró la
explotación, aunque introdujo riesgo de convergencia prematura.

## 4. Cruza e interacción con selección

La cruza uniforme superó a un punto con ambos selectores y en las dos imágenes.
Con Tournament 5, sus NMSE medianos fueron:

| Objetivo | Uniforme | Un punto |
| --- | ---: | ---: |
| Colombia | `0,001718` | `0,002314` |
| Señal | `0,021563` | `0,028750` |

Además, Tournament 5 con uniforme superó a Ranking con uniforme de manera clara
en la señal (`0,021563` frente a `0,025112`). Esto permitió resolver el empate
de la etapa anterior.

Los loci consecutivos representan capas de renderizado, pero no forman bloques
semánticos o espaciales garantizados. La cruza uniforme puede combinar
triángulos útiles distribuidos por todo el cromosoma, mientras que un punto
preserva prefijos y sufijos cuya agrupación puede ser arbitraria. En ambos casos
se cruzan genes completos, sin mezclar propiedades incompatibles.

## 5. Mutación

Se compararon dos métodos con el mismo delta local y un gen afectado en
esperanza. `single_local` modifica exactamente un triángulo; la multigénica usa
probabilidad `1/T` para cada locus y puede modificar cero, uno o varios.

| Objetivo | Single local | Multigénica balanceada |
| --- | ---: | ---: |
| Colombia | `0,001481` | `0,001718` |
| Señal | `0,019957` | `0,021563` |

Single ganó siete de las diez comparaciones emparejadas. Su cambio obligatorio
parece compensar la baja diversidad producida por Tournament 5: evita que una
proporción de la descendencia atraviese selección, cruza y mutación sin recibir
ninguna variación nueva.

## 6. Supervivencia

La estrategia aditiva obtuvo un error final ligeramente menor en ambos
objetivos:

| Objetivo | Aditiva | Exclusiva |
| --- | ---: | ---: |
| Colombia | `0,001481` | `0,001595` |
| Señal | `0,019957` | `0,020106` |

La exclusiva mantuvo más diversidad y tuvo mejor AUC en la señal, pero esa
exploración no produjo mejor calidad final. Se eligió la aditiva porque permite
que los padres buenos compitan con sus hijos y protege los refinamientos ya
alcanzados. En la señal la diferencia final fue pequeña y debe presentarse como
tal, no como una superioridad universal.

## 7. Validación y ejecuciones visuales extendidas

La configuración final se validó con cinco semillas sobre las tres imágenes.
Después se tomó la semilla mediana de cada objetivo y se extendió sin elegir el
caso más favorable.

| Objetivo | NMSE mediano a 1000 | NMSE extendido | Generaciones extendidas |
| --- | ---: | ---: | ---: |
| Colombia | `0,001481` | `0,000462` | 3000 |
| Señal | `0,019957` | `0,004501` | 5000 |
| Ícono | `0,015952` | `0,006481` | 5000 |

La señal y el ícono no alcanzaron 90 % de reducción dentro de las 1000
generaciones comparativas, pero lo hicieron aproximadamente en las generaciones
1350 y 1400. Los mejores resultados extendidos aparecieron en las generaciones
2999, 5000 y 4999. Por lo tanto, el cutoff corto servía para comparar métodos,
pero no era suficiente para producir las mejores imágenes finales.

## Descubrimientos principales

1. Más resolución o más triángulos no garantizan una mejora proporcional: cada
   incremento también encarece y amplía el espacio de búsqueda.
2. La presión selectiva útil depende del objetivo y del resto de los operadores.
   Tournament 5 fue riesgoso por su baja diversidad, pero la cruza uniforme y la
   mutación obligatoria permitieron aprovechar su velocidad.
3. La cruza uniforme fue la conclusión más estable: los bloques consecutivos de
   triángulos no mostraron una estructura útil para la cruza de un punto.
4. La interacción entre operadores importa. La mutación multigénica había sido
   competitiva con menor presión, mientras que Single funcionó mejor junto con
   Tournament 5.
5. El presupuesto apropiado para comparar configuraciones no necesariamente es
   suficiente para generar la mejor demostración visual.

## Limitaciones

- Se utilizaron cinco semillas: permiten observar consistencia, pero no realizar
  afirmaciones estadísticas fuertes.
- El diseño fue secuencial y no evaluó el producto cartesiano de todos los
  operadores; encuentra una configuración razonable, no garantiza el óptimo
  global.
- El NMSE compara píxeles y puede no coincidir completamente con la percepción
  humana de bordes, formas o estructura.
- La selección inicial se estudió sobre la imagen simple; las etapas siguientes
  incorporaron la señal para comprobar interacción y transferencia.
- Cada showcase usa una sola semilla mediana y funciona como evidencia visual,
  no como una nueva comparación estadística.

## Reproducibilidad

El estudio completo puede repetirse con:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_full_study.py
```

Las corridas crudas permanecen fuera del repositorio. Los CSV resumidos, las
figuras, las decisiones y las imágenes representativas se publican en
`experiments/results/`.
