# Neurito 3.0: aprende a encontrar comida en un laberinto

Neurito busca comida dentro de tres laberintos. Su cerebro debe elegir entre arriba, abajo, izquierda y derecha. La red se construye con NumPy, y Pygame dibuja el escenario, la mascota y una gráfica de la pérdida.

## Cómo aprende

1. BFS calcula rutas cortas y genera ejemplos etiquetados: mapa, posición, meta y movimiento recomendado.
2. Los ejemplos se dividen en entrenamiento (80 %) y prueba (20 %). Los pares posición-meta de prueba no aparecen en entrenamiento.
3. Cada entrada incluye un vector one-hot de tres mapas y dos vectores one-hot de 81 casillas: 165 características.
4. La arquitectura es 165 → 96 → 4. Usa ReLU en la capa oculta y Softmax en las cuatro salidas.
5. El entrenamiento minimiza entropía cruzada multiclase mediante retropropagación y descenso de gradiente con mini-lotes.
6. Durante el juego, el modelo decide sin consultar BFS. BFS es el experto que proporcionó las etiquetas.

Esto es aprendizaje supervisado por imitación, NO aprendizaje por refuerzo.

## Dos modos

- Red pura: toma la salida de mayor probabilidad. Puede chocar con paredes o quedarse en ciclos.
- Asistido: descarta movimientos contra paredes y penaliza casillas visitadas. Son reglas adicionales, no conocimientos aprendidos por la red.

La precisión de clasificación no equivale al porcentaje de laberintos completados.

## Estructura

- laberinto_mundo.py: mapas, movimiento, BFS y navegación asistida.
- laberinto_red.py: red neuronal y entrenamiento con NumPy.
- laberinto.py: juego Pygame, sprites y pérdida.
- test_laberinto.py: pruebas automáticas.
- GUIA_LABERINTO.md: esta guía.

Reutiliza assets/neutral_1.png, neutral_2.png y neutral_3.png. No modifica las versiones anteriores.

## Ejecutar

    uv sync
    uv run laberinto.py

## Controles

| Tecla | Acción |
|---|---|
| 1 | Entrenar una época |
| 2 | Entrenar cien épocas |
| 3 | Entrenamiento continuo |
| A | Alternar red pura y navegación asistida |
| P | Alternar navegación automática y paso a paso |
| Espacio | Avanzar un paso en modo manual |
| N | Cambiar mapa |
| G | Nuevas posiciones y meta |
| C | Reiniciar contadores |
| R | Reiniciar pesos y entrenamiento |
| Esc | Salir |

## Probar automáticamente

    uv run python -m unittest -v test_laberinto.py

## Experimentos

1. Observa Neurito sin entrenar. ¿Por qué se atasca?
2. Entrena 100, 200 y 400 épocas. Compara pérdida, precisión y movimientos.
3. Compara red pura frente a navegación asistida. ¿Qué parte corresponde al modelo y cuál a las reglas?
4. Cambia las 96 neuronas ocultas por 16, 32 o 128.
5. Cambia la tasa de aprendizaje 0.45 y observa si converge.
6. Cambia de mapa. Los mapas forman parte del entrenamiento, pero las combinaciones de inicio y meta usadas para probar no se entrenaron.
7. Haz vibecoding para verificar como mejora o empeora Neurito al cambiar DFS, por BFS y A*

## Alcance y limitaciones

El modelo no interpreta capturas de pantalla: trabaja con características estructuradas. Conoce el identificador de cada mapa. Los tres mapas pertenecen al mismo universo de entrenamiento. La evaluación demuestra generalización a pares de posición-meta no entrenados, no a laberintos totalmente desconocidos. Para eso habría que representar también las paredes de cualquier mapa nuevo y entrenar y evaluar en mapas separados.


## Respuestas de los experimentos

### Condiciones de las pruebas

Se usaron 2860 ejemplos de entrenamiento y 716 de prueba, con semilla 42 para datos y pesos iniciales. Los tres mapas aparecen en ambos conjuntos, pero cada combinación mapa, posición y meta pertenece a uno solo. Cada configuración se ejecutó una vez. Para repetir: `uv run experimentos_laberinto.py`. Los valores completos están en [resultados_laberinto.csv](resultados_laberinto.csv).

La pérdida de las tablas se calcula sobre todo el entrenamiento después de actualizar los pesos. La gráfica del juego muestra el promedio de las pérdidas calculadas durante los mini-lotes, por eso puede diferir ligeramente. La precisión mide el siguiente movimiento frente al experto; el éxito mide llegar a la comida en un recorrido completo.

Para comparar navegación se probaron los mismos 716 pares con límite `min(65, 2 * distancia_BFS + 8)` por reto. Un choque termina el reto en esta evaluación. Es un protocolo independiente del juego, cuya interfaz reinicia tras 45 intentos. Los intentos promedio incluyen retos fallidos; un valor menor no implica mejor desempeño si aumenta el número de fallos.

### 1. Observa Neurito sin entrenar. ¿Por qué se atasca?

Los pesos iniciales son aleatorios y todavía no relacionan la posición, la meta y el mapa con el movimiento correcto. El modo puro puede elegir una pared o repetir movimientos. Sin entrenar acertó el 27,65 % de las etiquetas y completó el 3,07 % de los retos. Hubo 589 retos terminados por pared. La asistencia completó el 57,54 % incluso sin entrenamiento, mostrando que sus reglas aportan parte del resultado.

### 2. Entrena 100, 200 y 400 épocas. Compara pérdida, precisión y movimientos.

| Épocas | Pérdida | Precisión de prueba | Éxito red pura | Éxito asistido |
|---|---:|---:|---:|---:|
| 0 | 1.3950 | 27.65% | 3.07% | 57.54% |
| 100 | 0.1650 | 84.92% | 51.26% | 97.91% |
| 200 | 0.0414 | 90.08% | 81.28% | 98.88% |
| 400 | 0.0108 | 91.48% | 82.68% | 99.58% |

Al entrenar más, la pérdida disminuyó y mejoraron los recorridos. A las 100 épocas la red pura completaba el 51,26 %; a las 400, el 82,68 %. Los retos que terminaron por pared bajaron de 15 a 3. La mejora de precisión entre 200 y 400 épocas fue pequeña: de 90,08 % a 91,48 %. Un error al inicio de una ruta puede llevar a un ciclo, por eso la precisión no coincide con la tasa de éxito.

### 3. Compara red pura frente a navegación asistida. ¿Qué parte corresponde al modelo y cuál a las reglas?

La red calcula cuatro probabilidades y el modo puro elige la mayor. La asistencia descarta acciones contra paredes y resta `0.35 * visitas` a la puntuación de cada casilla visitada. Estos controles están programados; no los aprendió la red. Después de esa resta las puntuaciones ya no son probabilidades.

Con 400 épocas, el mismo modelo completó el 82,68 % sin asistencia y el 99,58 % con ella. La asistencia evita choques, pero no garantiza encontrar siempre la comida ni seguir la ruta más corta. Los tres fallos asistidos de esta prueba agotaron el límite de intentos.

### 4. Cambia las 96 neuronas ocultas por 16, 32 o 128.

Se mantuvieron 400 épocas, experto BFS y tasa 0.45.

| Neuronas ocultas | Pérdida | Precisión de prueba | Éxito red pura | Éxito asistido |
|---|---:|---:|---:|---:|
| 16 | 0.0420 | 87.15% | 74.30% | 99.02% |
| 32 | 0.0162 | 89.25% | 78.35% | 98.32% |
| 96 | 0.0108 | 91.48% | 82.68% | 99.58% |
| 128 | 0.0102 | 90.78% | 85.34% | 98.74% |

Con 16 neuronas hubo menor precisión y menos recorridos completados. Las 96 lograron la mayor precisión de clasificación de este grupo, pero las 128 completaron más retos sin asistencia. Aumentar neuronas no mejoró todas las métricas al mismo tiempo. Esta comparación tiene una sola semilla; no demuestra que una arquitectura sea siempre superior.

### 5. Cambia la tasa de aprendizaje 0.45 y observa si converge.

Se mantuvieron 96 neuronas, 400 épocas y experto BFS.

| Tasa | Pérdida | Precisión de prueba | Éxito red pura | Éxito asistido |
|---|---:|---:|---:|---:|
| 0.01 | 0.6810 | 73.88% | 20.39% | 87.57% |
| 0.1 | 0.1481 | 87.15% | 60.20% | 98.88% |
| 0.45 | 0.0108 | 91.48% | 82.68% | 99.58% |
| 1.2 | 0.0021 | 93.85% | 86.17% | 99.30% |

La tasa 0.01 aprendió más lentamente y terminó con pérdida 0.6810. Con 0.45 bajó a 0.0108; con 1.2, a 0.0021. En esta ejecución 1.2 convergió y obtuvo la mayor precisión. Eso no significa que cualquier tasa más alta funcione: la estabilidad depende de los datos, la arquitectura y la inicialización.

### 6. Cambia de mapa. ¿Qué sucede con combinaciones de inicio y meta no entrenadas?

Con BFS, 96 neuronas, tasa 0.45 y 400 épocas:

| Mapa | Retos | Éxitos red pura | Éxitos asistidos |
|---|---:|---:|---:|
| 1 | 210 | 180/210 (85.71%) | 210/210 (100.00%) |
| 2 | 251 | 197/251 (78.49%) | 248/251 (98.80%) |
| 3 | 255 | 215/255 (84.31%) | 255/255 (100.00%) |

La red pudo resolver muchas combinaciones nuevas dentro de los mapas conocidos. No se evaluaron mapas completamente nuevos. La entrada contiene el identificador del mapa, pero no una representación general de sus paredes; para generalizar a otro laberinto habría que cambiar esa representación y separar mapas completos entre entrenamiento y prueba.

### 7. Verifica cómo mejora o empeora Neurito al cambiar DFS por BFS y A*.

Se implementaron DFS y A* en `experimentos_laberinto.py`. Se conservaron los pares de entrenamiento y prueba y se reemplazaron las etiquetas por la primera acción de la ruta del experto elegido. Cada red se entrenó desde cero con 96 neuronas, tasa 0.45 y 400 épocas. El juego sigue usando BFS como configuración original; las variantes se comparan con el script.

| Experto | Pérdida | Precisión de prueba | Éxito red pura | Éxito asistido |
|---|---:|---:|---:|---:|
| bfs | 0.0108 | 91.48% | 82.68% | 99.58% |
| dfs | 0.0101 | 94.83% | 11.45% | 35.75% |
| astar | 0.0116 | 92.60% | 86.31% | 99.44% |

La precisión de cada fila se mide contra su propio experto, así que no equivale a comparar la calidad de las rutas. DFS logró 94,83 % de precisión frente a sus etiquetas, pero la red pura solo completó el 11,45 % de los retos. Las rutas completas del experto DFS midieron, en promedio, 2,06 veces la longitud óptima de BFS. Además, imitar solo su primera acción sin conservar la pila ni las visitas del buscador puede producir decisiones que se repiten. El límite de pasos, basado en distancia BFS, también penaliza rutas largas.

BFS encuentra rutas mínimas en estos mapas con coste uniforme. A* empleó distancia Manhattan, también produjo rutas mínimas y su red completó el 86,31 % sin asistencia, frente al 82,68 % con BFS. Con asistencia quedaron en 99,44 % y 99,58 %, respectivamente. Los empates entre rutas óptimas pueden generar etiquetas distintas. La diferencia observada en una sola ejecución no demuestra superioridad general ni compara tiempos de búsqueda.

La comparación ejecuta las rutas DFS y A* para verificar que no atraviesen paredes y lleguen a la meta. También comprueba que las rutas de A* tengan la misma longitud óptima que BFS en los 716 retos.

### ¿Por qué una red puede clasificar bien y aun así no encontrar la comida?

La clasificación evalúa decisiones individuales; navegar exige encadenarlas hasta la meta. Un solo error puede llevar a una pared o a un ciclo. Además, reproducir las etiquetas de un experto no asegura que sus primeras acciones formen una buena política sin memoria. Por eso se reportan pérdida, precisión y recorridos completos por separado.

### Validación

Las cinco pruebas de `test_laberinto.py` pasaron. Se verificaron mapas conectados, separación de datos, aprendizaje, probabilidades válidas, asistencia y retos no entrenados. También se compiló el código y se comprobó el arranque de Pygame, la carga de sprites, el dibujo de un fotograma y el cierre sin ventana visible. Esto no sustituye una revisión manual de todos los controles.


---

# Versiones anteriores

# Neurito 2.0: aprende a saltar obstáculos

Neurito aprende **cuándo saltar** ante obstáculos que se aproximan con diferentes velocidades. El cerebro se implementa manualmente con NumPy y el aprendizaje puede observarse en Pygame.

## ¿Qué aprende?

La red recibe tres características normalizadas: distancia al obstáculo, velocidad y tiempo estimado hasta alcanzarlo (distancia / velocidad). La última es una característica derivada, no una observación independiente.

La arquitectura es **3 → 12 → 1**: doce neuronas ocultas con ReLU y una salida con sigmoide. La última devuelve una puntuación entre cero y uno que, con el umbral 0.5, determina si Neurito debe saltar cuando está en el suelo.

## ¿Cómo aprende?

Aplicamos **aprendizaje supervisado**, no aprendizaje por refuerzo. Creamos datos sintéticos que incluyen ejemplos donde es conveniente saltar y otros donde debe esperar. Para generar las etiquetas, definimos una ventana de referencia: saltar cuando el tiempo estimado está entre **7 y 13 fotogramas**. No es una regla aprendida por recompensa ni garantiza evitar choques en todas las situaciones.

La red ajusta sus pesos a partir de la **entropía cruzada binaria**, la **retropropagación** y el **descenso de gradiente**:

`W_nuevo = W_anterior - tasa_de_aprendizaje * gradiente`

El conjunto de entrenamiento equilibra aproximadamente las dos decisiones. De otra manera, una red podría aparentar buena precisión diciendo casi siempre «esperar».

La precisión se mide sobre ejemplos distintos de los utilizados en el entrenamiento. **Una precisión alta no garantiza un buen resultado en el juego**, que también depende de la física del salto.

## Archivos

```text
Neurito/
├── main.py                 # Mascota original: comer, dormir, jugar
├── salto_modelo.py         # Red neuronal manual y datos de entrenamiento
├── salto_juego.py          # Física y detección de colisiones
├── saltar.py               # Juego y visualización del entrenamiento
├── GUIA_SALTO.md
└── assets/
    └── jugar_1.png        # Reutilizado en el nuevo juego
```

## Ejecución

```bash
uv sync
uv run saltar.py
```

No se necesitan dependencias nuevas: se reutilizan NumPy y Pygame.

## Controles

| Tecla | Función |
|---|---|
| `1` | Entrenar una época |
| `2` | Entrenar cien épocas |
| `3` | Activar o pausar entrenamiento continuo |
| `A` | Alternar control automático y manual |
| `Espacio` | Saltar manualmente |
| `C` | Reiniciar pista y contadores |
| `R` | Reiniciar pesos y contadores |
| `Esc` | Salir |

## Experimentos

1. Prueba el juego sin entrenar. Anota colisiones y obstáculos esquivados.
2. Entrena de cien en cien épocas; compara la pérdida, la precisión y el comportamiento.
3. Cambia `ocultas=12` por `ocultas=3` o `ocultas=24`; compara generalización.
4. Modifica la tasa de aprendizaje `0.8` a `0.01` o `1.2` y estudia la convergencia.
5. Cambia la ventana de entrenamiento en `etiqueta()` y en la generación de positivos. Comprueba si la política resultante es compatible con la física.

**Pregunta central:** ¿por qué una red puede clasificar ejemplos con buena precisión y aun así chocar?

---

# Experimentos de la mascota original

## Experimentos

### 1. Mantén dos necesidades fijas y varía la tercera. ¿Cuándo cambia la acción predicha?

Se fijaron energía en 0.4 y aburrimiento en 0.3. Por tanto, el cansancio fue 0.6.

| Hambre | Comer | Dormir | Jugar | Acción predicha |
|---:|---:|---:|---:|---|
| 0.50 | 13.40 % | 86.36 % | 0.24 % | Dormir |
| 0.59 | 45.40 % | 54.45 % | 0.14 % | Dormir |
| 0.60 | 50.06 % | 49.81 % | 0.13 % | Comer |
| 0.70 | 86.66 % | 13.31 % | 0.03 % | Comer |

La acción pasó de dormir a comer entre 0.59 y 0.60 de hambre. Esto ocurre porque el hambre alcanza al cansancio, que estaba fijo en 0.6. Cerca del empate, las probabilidades son muy parecidas.

### 2. Compara prioridades casi iguales. ¿Cómo cambian las salidas de Softmax?

| Hambre | Energía | Aburrimiento | Comer | Dormir | Jugar |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.50 | 0.50 | 35.21 % | 33.45 % | 31.34 % |
| 0.51 | 0.50 | 0.49 | 41.78 % | 32.98 % | 25.24 % |
| 0.49 | 0.50 | 0.51 | 28.95 % | 33.09 % | 37.96 % |

Cuando las necesidades son parecidas, las probabilidades quedan más repartidas. Un cambio pequeño puede hacer que gane otra acción. No dan exactamente un tercio para cada acción porque los pesos aprendidos son distintos.

### 3. Cambia la capa oculta de 6 a 3, 10 o 20 neuronas; compara pérdida y precisión.

Se mantuvieron 1200 épocas y tasa de aprendizaje 0.1. En todas las configuraciones se usaron los mismos 3000 ejemplos de entrenamiento (semilla 42) y 1000 de prueba (semilla 123). La pérdida es de entrenamiento y la precisión es de prueba. Los resultados corresponden a una ejecución por configuración.

| Neuronas ocultas | Pérdida de entrenamiento | Precisión de prueba |
|---:|---:|---:|
| 3 | 0.1618 | 96.3 % |
| 6 | 0.1224 | 99.7 % |
| 10 | 0.1261 | 98.5 % |
| 20 | 0.1220 | 99.2 % |

La mejor precisión en esta prueba fue con 6 neuronas. Con 3 el resultado bajó, y con 10 o 20 no hubo una mejora frente a 6. Tener más neuronas no aseguró un mejor resultado. Habría que repetir con otras semillas para comprobar si se mantiene la diferencia.

### 4. Cambia las épocas o la tasa de aprendizaje. ¿El entrenamiento mejora, se estanca o se vuelve inestable?

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

Al aumentar las épocas, la pérdida bajó. Sin embargo, pasar de 1200 a 2400 no mejoró la precisión de prueba. Con tasas de 0.001 y 0.01 el aprendizaje fue más lento. Con una tasa de 10 el resultado empeoró mucho: la precisión quedó en 34.3 %. Esto sugiere que los ajustes fueron demasiado grandes; para confirmar oscilaciones habría que revisar la pérdida durante todo el entrenamiento.

### 5. ¿Qué datos y etiquetas requeriría una cuarta acción?

Una cuarta acción podría ser estar tranquilo. Se usarían las mismas entradas y se etiquetarían como tranquilo los ejemplos con `hambre < 0.4`, `energía > 0.6` y `aburrimiento < 0.4`. En los demás casos se seguiría eligiendo la necesidad más alta.

Las etiquetas serían 0=comer, 1=dormir, 2=jugar y 3=tranquilo. Se necesitarían suficientes ejemplos de cada acción, incluyendo casos cerca de los límites. También habría que agregar una cuarta neurona de salida, actualizar las etiquetas y volver a entrenar.

Actualmente, el estado neutral se activa con una condición en la interfaz. La red todavía tiene tres salidas; la cuarta acción queda como propuesta.

## Explica tus resultados

### 1. ¿En qué se diferencian pesos y sesgos?

Los pesos indican cuánto influye cada entrada en una neurona. Los sesgos son valores que se suman al resultado y permiten ajustar cuándo se activa. Ambos cambian durante el entrenamiento. En el código, los pesos son `W1` y `W2`, y los sesgos son `b1` y `b2`.

### 2. ¿Por qué el cansancio es 1 - energía?

Porque la energía va de 0 a 1 y el cansancio representa lo contrario: a menor energía, mayor cansancio. Por ejemplo, si la energía es 0.2, el cansancio es 0.8. Esto permite compararlo con el hambre y el aburrimiento, donde un valor alto también representa una necesidad alta. Esta conversión se usa para generar las etiquetas.

### 3. ¿Qué hace ReLU y qué hace Softmax?

ReLU convierte los valores negativos en cero y deja pasar los positivos. Se usa en la capa oculta y permite aprender relaciones que no son solamente lineales.

Softmax convierte los valores de salida en probabilidades que suman 1. Después, `argmax` elige la acción con la probabilidad más alta. Estas probabilidades son estimaciones del modelo y no garantizan que la predicción sea correcta.

### 4. ¿Por qué una precisión alta puede ser engañosa?

Porque acertar muchos ejemplos no significa que el modelo funcione bien en cualquier situación. En este proyecto, los datos se generan con una regla sencilla, así que una precisión alta indica que la red aprendió a copiar esa regla.

Además, el promedio puede ocultar errores cuando dos necesidades están casi empatadas o cuando una acción aparece pocas veces. Por eso conviene probar con datos separados del entrenamiento y revisar los errores de cada acción. Si se usan estas pruebas para escoger los parámetros, hace falta otro conjunto de datos para la evaluación final.
