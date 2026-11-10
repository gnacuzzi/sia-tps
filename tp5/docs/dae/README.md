# Denoising autoencoder — índice de documentación

Esta carpeta reúne la documentación y los artefactos publicados del ejercicio
1B. Los modelos y las evaluaciones crudas permanecen en `results/`, que no se
versiona.

## Documento principal

- [`ejercicio_1b.md`](../ejercicio_1b.md): recorrido experimental, decisiones,
  configuración congelada, resultados, conclusión y extensiones opcionales.

## Carpetas

- [`pilot/`](pilot/): curva y datos del primer piloto con distintos niveles de
  bit-flip.
- [`experiments/`](experiments/): tablas de las comparaciones intermedias de
  tasa, arquitectura, activación, latente, entrenamiento y optimizador.
- [`final/`](final/): confirmación con diez semillas, curva agregada, dificultad
  por carácter y ejemplos representativos.

La configuración que regenera el análisis final es
`configs/dae_final_analysis.json` y la utilidad correspondiente es
`sia_tp5.denoising_final_analysis`.
