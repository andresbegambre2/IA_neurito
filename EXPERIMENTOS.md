# Experimentos y explicación de Neurito

## Modelo y metodología

Neurito recibe tres valores entre 0 y 1: hambre, energía y aburrimiento. Su arquitectura original es 3 entradas, 6 neuronas ocultas con ReLU y 3 salidas con Softmax: comer, dormir y jugar.

Las etiquetas sintéticas se calculan eligiendo la mayor prioridad entre hambre, `1 - energía` y aburrimiento. Se entrenó con 3000 ejemplos (semilla 42). Para comparar configuraciones se utilizaron los mismos 1000 ejemplos independientes de prueba, generados con `np.random.default_rng(123)`. Se modificó un parámetro por vez. La pérdida corresponde a entropía cruzada sobre entrenamiento después de la última actualización; la precisión corresponde a prueba.

Ejecutar `python experimentos.py` regenera las tablas en [RESULTADOS.md](RESULTADOS.md). Son resultados de una ejecución por configuración; no son promedios de varias semillas. La precisión que muestra la aplicación puede variar respecto de estas tablas porque utiliza otro conjunto de prueba.

## 1. Mantener dos necesidades fijas y variar la tercera

Se fijaron energía en 0.4 y aburrimiento en 0.3. Por tanto, el cansancio fue 0.6.

| Hambre | Comer | Dormir | Jugar | Acción predicha |
|---:|---:|---:|---:|---|
| 0.50 | 13.40 % | 86.36 % | 0.24 % | Dormir |
| 0.59 | 45.40 % | 54.45 % | 0.14 % | Dormir |
| 0.60 | 50.06 % | 49.81 % | 0.13 % | Comer |
| 0.70 | 86.66 % | 13.31 % | 0.03 % | Comer |

La acción cambió entre las muestras 0.59 y 0.60, cerca del empate entre hambre y cansancio. La regla ideal elegiría comer cuando el hambre supera 0.6; en el empate exacto, `argmax` selecciona la primera clase, comer. La red aproxima esa frontera y puede equivocarse alrededor de ella.

## 2. Comparar prioridades casi iguales

| Hambre | Energía | Aburrimiento | Comer | Dormir | Jugar |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.50 | 0.50 | 35.21 % | 33.45 % | 31.34 % |
| 0.51 | 0.50 | 0.49 | 41.78 % | 32.98 % | 25.24 % |
| 0.49 | 0.50 | 0.51 | 28.95 % | 33.09 % | 37.96 % |

Las probabilidades quedan más repartidas cerca del empate y pequeños cambios pueden cambiar la acción ganadora. No resultan exactamente iguales porque la red aprendida no es perfectamente simétrica. Softmax transforma los puntajes de la última capa, no directamente las necesidades. Sus valores no garantizan una confianza bien calibrada.

## 3. Cambiar el tamaño de la capa oculta

Se mantuvieron 1200 épocas y tasa de aprendizaje 0.1.

| Neuronas ocultas | Pérdida de entrenamiento | Precisión de prueba |
|---:|---:|---:|
| 3 | 0.1618 | 96.3 % |
| 6 | 0.1224 | 99.7 % |
| 10 | 0.1261 | 98.5 % |
| 20 | 0.1220 | 99.2 % |

En esta ejecución, 6 neuronas obtuvieron la mayor precisión. Aumentar la capacidad no garantizó mejorar. Al cambiar el tamaño también cambia la inicialización de los parámetros; se requieren varias semillas para establecer una ventaja consistente. Estos datos por sí solos no demuestran sobreajuste.

Ejemplos: `python main.py --hidden 3`, `python main.py --hidden 10` y `python main.py --hidden 20`.

## 4. Cambiar épocas o tasa de aprendizaje

Con 6 neuronas ocultas:

| Épocas | Tasa | Pérdida de entrenamiento | Precisión de prueba |
|---:|---:|---:|---:|
| 200 | 0.1 | 0.4961 | 91.7 % |
| 600 | 0.1 | 0.1942 | 98.4 % |
| 1200 | 0.1 | 0.1224 | 99.7 % |
| 2400 | 0.1 | 0.0807 | 99.5 % |
| 1200 | 0.001 | 1.0333 | 42.1 % |
| 1200 | 0.01 | 0.7372 | 81.4 % |
| 1200 | 1.0 | 0.0447 | 98.5 % |
| 1200 | 10.0 | 3.0699 | 34.3 % |

Más épocas redujeron la pérdida, pero la precisión dejó de mejorar entre 1200 y 2400. Esto no implica que toda la optimización se haya estancado: la pérdida siguió bajando. Las tasas pequeñas aprendieron lentamente dentro de las 1200 épocas disponibles. La tasa 10 dejó una pérdida alta y precisión cercana al azar para tres clases; el resultado es compatible con actualizaciones demasiado grandes. Para describir oscilaciones o divergencia habría que revisar la curva completa de pérdida. Una pérdida menor no garantiza mayor precisión porque ambas métricas miden cosas diferentes.

Ejemplos: `python main.py --epochs 2400` y `python main.py --learning-rate 0.01`.

## 5. Datos y etiquetas para una cuarta acción

Una posible cuarta acción es **estar tranquilo**. Se pueden conservar las tres entradas y agregar ejemplos etiquetados como tranquilo cuando `hambre < 0.4`, `energía > 0.6` y `aburrimiento < 0.4`. Los demás ejemplos conservarían la acción con mayor prioridad.

Se necesitan ejemplos de las cuatro clases, suficientes casos de tranquilidad y casos próximos a los límites para evaluar las transiciones. Las etiquetas podrían ser enteros 0=comer, 1=dormir, 2=jugar y 3=tranquilo. Habría que ampliar la salida a cuatro neuronas (`W2` de tamaño oculta×4 y `b2` de tamaño 1×4), ajustar la generación de etiquetas, las acciones mostradas y la evaluación, y reentrenar. Se debe comprobar la precisión por clase para evitar ocultar fallos de la clase menos frecuente.

Actualmente, el estado neutral se activa con una regla de la interfaz. **No es una cuarta clase aprendida por la red.** Este experimento propone cómo incorporarla; no se implementó una cuarta salida.

## Preguntas de explicación

### 1. ¿En qué se diferencian pesos y sesgos?

Una neurona calcula `z = suma(w_i * x_i) + b`. Los pesos `w_i` determinan cuánto y en qué dirección influye cada entrada. El sesgo `b` desplaza el valor de activación independientemente de las entradas y permite ajustar el umbral de respuesta. Ambos se aprenden mediante descenso de gradiente. En Neurito, `W1` y `W2` son los pesos; `b1` y `b2` son los sesgos.

### 2. ¿Por qué el cansancio es 1 - energía?

La energía está normalizada entre 0 y 1. Su complemento permite representar que poca energía implica mucho cansancio: energía 0.2 corresponde a cansancio 0.8. Así, hambre, cansancio y aburrimiento tienen el mismo sentido: cuanto mayor es el valor, mayor es la prioridad. Es una simplificación definida para este modelo, no una ley biológica. Se utiliza para generar etiquetas; la entrada de la red sigue siendo energía.

### 3. ¿Qué hace ReLU y qué hace Softmax?

**ReLU** calcula `max(0, z)`: convierte valores negativos en cero y conserva los positivos. Introduce no linealidad en la capa oculta para que la red pueda aprender relaciones más complejas que una transformación lineal.

**Softmax** calcula `p_i = exp(z_i) / suma(exp(z_j))`: transforma los puntajes finales en valores positivos que suman 1 y se interpretan como probabilidades del modelo. La acción se elige mediante `argmax`. El código resta el máximo antes de exponenciar para mejorar la estabilidad numérica sin alterar el resultado matemático.

### 4. ¿Por qué una precisión alta puede ser engañosa?

Los ejemplos de entrenamiento y prueba siguen la misma política sintética. Una precisión alta demuestra que la red reproduce esa regla en datos similares, pero no que comprenda necesidades reales ni que funcione igual en otras condiciones.

Además, la precisión global puede ocultar errores cerca de los empates, fallos en clases poco frecuentes y probabilidades mal calibradas. Si se evaluara sobre entrenamiento, también podría ocultar memorización. Por ello conviene usar datos independientes, revisar resultados por clase y una matriz de confusión, examinar casos difíciles y repetir con varias semillas. Si se seleccionan hiperparámetros usando estas tablas, se necesitaría un conjunto adicional sin utilizar para informar una evaluación final imparcial.
