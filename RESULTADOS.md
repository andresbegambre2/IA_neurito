# Resultados reproducibles

3000 ejemplos de entrenamiento (semilla 42) y los mismos 1000 ejemplos de prueba (default_rng, semilla 123) en todas las configuraciones. Una ejecución por configuración. La pérdida se mide después de la última actualización, sobre entrenamiento.

| Neuronas | Épocas | Tasa | Pérdida | Precisión de prueba |
|---:|---:|---:|---:|---:|
| 3 | 1200 | 0.1 | 0.1618 | 96.3% |
| 6 | 1200 | 0.1 | 0.1224 | 99.7% |
| 10 | 1200 | 0.1 | 0.1261 | 98.5% |
| 20 | 1200 | 0.1 | 0.1220 | 99.2% |
| 6 | 200 | 0.1 | 0.4961 | 91.7% |
| 6 | 600 | 0.1 | 0.1942 | 98.4% |
| 6 | 2400 | 0.1 | 0.0807 | 99.5% |
| 6 | 1200 | 0.001 | 1.0333 | 42.1% |
| 6 | 1200 | 0.01 | 0.7372 | 81.4% |
| 6 | 1200 | 1 | 0.0447 | 98.5% |
| 6 | 1200 | 10 | 3.0699 | 34.3% |

## Variar hambre con energía 0.4 y aburrimiento 0.3

| Hambre | Comer | Dormir | Jugar | Predicción |
|---:|---:|---:|---:|---|
| 0.4 | 2.33% | 97.38% | 0.29% | Dormir |
| 0.5 | 13.40% | 86.36% | 0.24% | Dormir |
| 0.59 | 45.40% | 54.45% | 0.14% | Dormir |
| 0.6 | 50.06% | 49.81% | 0.13% | Comer |
| 0.61 | 54.72% | 45.16% | 0.12% | Comer |
| 0.7 | 86.66% | 13.31% | 0.03% | Comer |
| 0.8 | 97.68% | 2.32% | 0.01% | Comer |

## Prioridades casi iguales

| Hambre | Energía | Aburrimiento | Comer | Dormir | Jugar |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.5 | 0.5 | 35.21% | 33.45% | 31.34% |
| 0.51 | 0.5 | 0.49 | 41.78% | 32.98% | 25.24% |
| 0.49 | 0.5 | 0.51 | 28.95% | 33.09% | 37.96% |

Interpretación y respuestas: [EXPERIMENTOS.md](EXPERIMENTOS.md).
