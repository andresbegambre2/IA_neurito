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
├── README_2.0.md
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

## Respuestas de los experimentos

### Condiciones de las pruebas

Se utilizaron 5000 ejemplos de entrenamiento (semilla 42), 1500 de prueba (semilla 123) y semilla 42 para inicializar cada red. Los datos se generan por separado de los pesos para conservar los mismos ejemplos al cambiar la cantidad de neuronas. Por eso estos resultados pueden diferir ligeramente de una ejecución desde la interfaz.

Cada configuración se ejecutó una vez. Se evaluaron 100 obstáculos independientes, empezando cada uno desde el suelo y desde la posición inicial. El primero usa velocidad 5.5 y los demás velocidades entre 4.5 y 8.5 con semilla 19. Todas las configuraciones enfrentan esa misma secuencia. Esta evaluación no reproduce una partida continua ni garantiza los resultados de otras semillas.

La pérdida de las tablas se mide sobre todo el entrenamiento después de actualizar los pesos. La interfaz muestra la pérdida calculada antes de la última actualización. La precisión compara las decisiones con las etiquetas de la ventana usada en cada configuración.

Para repetir los experimentos sin abrir el juego:

```bash
uv run experimentos_salto.py
```

### 1. Prueba el juego sin entrenar. Anota colisiones y obstáculos esquivados.

Sin entrenamiento, con 12 neuronas y los pesos iniciales, hubo **100 choques y 0 obstáculos esquivados**. La precisión de clasificación fue 44,60 %. Los pesos aleatorios todavía no relacionan correctamente la distancia y la velocidad con el momento de saltar. El personaje puede saltar en momentos inadecuados aunque la salida tenga un valor alto.

### 2. Entrena de cien en cien épocas; compara pérdida, precisión y comportamiento.

Se mantuvieron 12 neuronas, tasa 0.8 y ventana de 7 a 13 fotogramas.

| Épocas | Pérdida | Precisión de prueba | Choques | Esquivados |
|---|---:|---:|---:|---:|
| 0 | 0.7095 | 44.60% | 100 | 0 |
| 100 | 0.2618 | 91.13% | 10 | 90 |
| 200 | 0.2318 | 91.80% | 1 | 99 |
| 300 | 0.2059 | 92.33% | 0 | 100 |
| 400 | 0.1714 | 92.53% | 0 | 100 |

La pérdida disminuyó y la precisión aumentó. A las 100 épocas todavía hubo 10 choques; a las 200 quedó uno, y a las 300 y 400 no hubo choques en esta secuencia. Entre 300 y 400 épocas mejoró la clasificación, aunque el número de obstáculos esquivados ya no podía aumentar porque había llegado a 100. Esto no garantiza que nunca choque con otras velocidades o condiciones.

### 3. Cambia `ocultas=12` por `ocultas=3` o `ocultas=24`; compara generalización.

Se mantuvieron 400 épocas, tasa 0.8 y ventana de 7 a 13.

| Neuronas ocultas | Pérdida | Precisión de prueba | Choques | Esquivados |
|---|---:|---:|---:|---:|
| 3 | 0.2188 | 92.60% | 0 | 100 |
| 12 | 0.1714 | 92.53% | 0 | 100 |
| 24 | 0.1146 | 95.40% | 0 | 100 |

Con 24 neuronas se obtuvo menor pérdida y mayor precisión sobre ejemplos no utilizados para entrenar. Con 3 neuronas la precisión fue ligeramente mayor que con 12, aunque la pérdida fue peor. Las tres redes esquivaron los 100 obstáculos, así que esta prueba del juego no distingue su rendimiento. Una sola semilla no permite afirmar que aumentar neuronas siempre mejore la generalización.

### 4. Modifica la tasa de aprendizaje `0.8` a `0.01` o `1.2` y estudia la convergencia.

Se mantuvieron 12 neuronas, 400 épocas y ventana de 7 a 13.

| Tasa | Pérdida | Precisión de prueba | Choques | Esquivados |
|---|---:|---:|---:|---:|
| 0.01 | 0.5386 | 86.67% | 71 | 29 |
| 0.8 | 0.1714 | 92.53% | 0 | 100 |
| 1.2 | 0.1269 | 93.27% | 1 | 99 |

La tasa 0.01 aprendió más lentamente: terminó con pérdida 0.5386 y 71 choques. Con 0.8 la pérdida fue 0.1714 y no hubo choques. La tasa 1.2 terminó con menor pérdida y mayor precisión que 0.8, pero tuvo un choque. Por tanto, mejorar la clasificación no asegura mejorar cada resultado del juego. No se observó divergencia en estas configuraciones; tampoco se demuestra que tasas superiores sean estables.

### 5. Cambia la ventana de entrenamiento en `etiqueta()` y en la generación de positivos. Comprueba si la política resultante es compatible con la física.

El script experimental cambia conjuntamente la regla de las etiquetas y el intervalo de los ejemplos positivos, conservando el juego original. Se compararon ventanas de 3 a 6, de 7 a 13 y de 20 a 26 fotogramas, con 12 neuronas, tasa 0.8 y 400 épocas. Los positivos se generan entre el límite inferior más 0.5 y el superior menos 0.5; la otra mitad de ejemplos se distribuye entre 0 y 60. Cada precisión se mide frente a las etiquetas de su propia ventana.

| Ventana en fotogramas | Pérdida | Precisión de prueba | Choques | Esquivados |
|---|---:|---:|---:|---:|
| 3 a 6 | 0.1486 | 97.07% | 0 | 100 |
| 7 a 13 | 0.1714 | 92.53% | 0 | 100 |
| 20 a 26 | 0.0859 | 96.27% | 100 | 0 |

Las redes entrenadas con ventanas de 3 a 6 y de 7 a 13 esquivaron los 100 obstáculos evaluados. La ventana de 20 a 26 produjo una precisión alta, 96,27 %, pero terminó en 100 choques: saltar demasiado pronto hizo que el personaje descendiera antes de superar el obstáculo. Esa política resultó incompatible con esta física y esta secuencia de velocidades.

La compatibilidad también depende del umbral 0.5, de los errores cerca de los límites y del tiempo que ocupa el obstáculo bajo el personaje. Esquivar todos los obstáculos de una prueba no demuestra que la red reproduzca exactamente la ventana ni que funcione en cualquier situación.

### Pregunta central: ¿por qué una red puede clasificar ejemplos con buena precisión y aun así chocar?

Porque la precisión mide si la red imita las etiquetas, mientras que esquivar depende de que esas etiquetas indiquen un momento compatible con la gravedad, el impulso, la velocidad y el tamaño del obstáculo. La red no recibe recompensas por evitar choques: aprende una regla supervisada.

Además, los errores cerca del instante de salto pueden ser pocos dentro del conjunto de prueba y tener un efecto grande durante el juego. La ventana de 20 a 26 lo mostró: 96,27 % de precisión y ningún obstáculo esquivado. Por eso se deben comparar la pérdida, la precisión y el comportamiento físico por separado.
