# TP2 — Informe comparativo de operadores

## Estado

Fases con resultados completos: **validation**. Las secciones pendientes se mantienen visibles para no presentar conclusiones antes de tener las cinco semillas por condición.

## Protocolo controlado

Todas las condiciones usan las semillas `101, 202, 303, 404, 505`. Se compara dentro del mismo objetivo y presupuesto; los NMSE absolutos de objetivos distintos no se comparan entre sí. Las curvas usan el mejor histórico.

## Resolución de trabajo

**Pendiente:** esta fase aún no terminó sus cinco semillas por condición.

## Cantidad de triángulos

**Pendiente:** esta fase aún no terminó sus cinco semillas por condición.

## Selección de padres

**Pendiente:** esta fase aún no terminó sus cinco semillas por condición.

## Cruza

**Pendiente:** esta fase aún no terminó sus cinco semillas por condición.

## Mutación

**Pendiente:** esta fase aún no terminó sus cinco semillas por condición.

## Supervivencia

**Pendiente:** esta fase aún no terminó sus cinco semillas por condición.

## Validación final

**Resultado:** la configuración elegida se evaluó sobre los tres niveles de dificultad. Sus NMSE se interpretan dentro de cada objetivo, no como un ranking directo entre imágenes.

| Objetivo | Condición | NMSE mediano | IQR NMSE | AUC normalizada | Diversidad final | Éxitos 90 % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| flag | winner | 0.001481 | 0.000572 | 0.093298 | 0.002438 | 5/5 |
| icon | winner | 0.018887 | 0.000711 | 0.300575 | 0.001429 | 0/5 |
| sign | winner | 0.013635 | 0.004686 | 0.239499 | 0.001863 | 1/5 |

### Imágenes representativas

- `flag` — `winner`, semilla mediana `404`: ![resultado](images/validation/flag-winner-seed-404.png)
- `icon` — `winner`, semilla mediana `303`: ![resultado](images/validation/icon-winner-seed-303.png)
- `sign` — `winner`, semilla mediana `404`: ![resultado](images/validation/sign-winner-seed-404.png)

## Demostraciones visuales extendidas

**Pendiente:** esta fase aún no terminó sus cinco semillas por condición.

## Limitaciones y lectura para la exposición

El NMSE se mide en la resolución de trabajo, por lo que una mejora numérica no demuestra fidelidad perfecta a tamaño original. Las imágenes representativas son la semilla mediana, no la mejor: muestran un caso típico. Las conclusiones finales deben distinguir velocidad (AUC), calidad (NMSE) y riesgo de convergencia prematura (diversidad).
