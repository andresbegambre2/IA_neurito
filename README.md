# Neurito

Neurito elige entre comer, dormir y jugar según el hambre, la energía y el aburrimiento. Usa una red de 3 entradas, una capa oculta con 6 neuronas y 3 salidas.

## Ejecución

Requiere Python 3.12 o superior.

```powershell
python -m pip install numpy pygame-ce
python main.py
```

Para cambiar los parámetros o repetir las pruebas:

```powershell
python main.py --hidden 10 --epochs 1200 --learning-rate 0.1
python experimentos.py
```

`experimentos.py` muestra las tablas en la terminal.

## Controles

| Teclas | Función |
|---|---|
| H / J | Aumentar / disminuir hambre |
| E / D | Aumentar / disminuir energía |
| A / S | Aumentar / disminuir aburrimiento |
| R | Valores aleatorios |
| N | Estado tranquilo |
| Esc | Salir |

## Experimentos

Se usaron 3000 ejemplos de entrenamiento con semilla 42 y los mismos 1000 ejemplos de prueba con semilla 123 para comparar las configuraciones. Se cambió un parámetro a la vez. La pérdida se calculó sobre entrenamiento y la precisión sobre prueba. Los resultados corresponden a una ejecución por configuración. La aplicación usa otro conjunto de prueba, por lo que su precisión puede ser diferente.

## 1. Mantener dos necesidades fijas y variar la tercera

Se fijaron energía en 0.4 y aburrimiento en 0.3. Por tanto, el cansancio fue 0.6.

| Hambre | Comer | Dormir | Jugar | Acción predicha |
|---:|---:|---:|---:|---|
| 0.50 | 13.40 % | 86.36 % | 0.24 % | Dormir |
| 0.59 | 45.40 % | 54.45 % | 0.14 % | Dormir |
| 0.60 | 50.06 % | 49.81 % | 0.13 % | Comer |
| 0.70 | 86.66 % | 13.31 % | 0.03 % | Comer |

La acción pasó de dormir a comer entre 0.59 y 0.60 de hambre. Esto ocurre porque el hambre alcanza al cansancio, que estaba fijo en 0.6. Cerca del empate, las probabilidades son muy parecidas.

## 2. Comparar prioridades casi iguales

| Hambre | Energía | Aburrimiento | Comer | Dormir | Jugar |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.50 | 0.50 | 35.21 % | 33.45 % | 31.34 % |
| 0.51 | 0.50 | 0.49 | 41.78 % | 32.98 % | 25.24 % |
| 0.49 | 0.50 | 0.51 | 28.95 % | 33.09 % | 37.96 % |

Cuando las necesidades son parecidas, las probabilidades quedan más repartidas. Un cambio pequeño puede hacer que gane otra acción. No dan exactamente un tercio para cada acción porque los pesos aprendidos son distintos.

## 3. Cambiar el tamaño de la capa oculta

Se mantuvieron 1200 épocas y tasa de aprendizaje 0.1.

| Neuronas ocultas | Pérdida de entrenamiento | Precisión de prueba |
|---:|---:|---:|
| 3 | 0.1618 | 96.3 % |
| 6 | 0.1224 | 99.7 % |
| 10 | 0.1261 | 98.5 % |
| 20 | 0.1220 | 99.2 % |

La mejor precisión en esta prueba fue con 6 neuronas. Con 3 el resultado bajó, y con 10 o 20 no hubo una mejora frente a 6. Tener más neuronas no aseguró un mejor resultado. Habría que repetir con otras semillas para comprobar si se mantiene la diferencia.

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

Al aumentar las épocas, la pérdida bajó. Sin embargo, pasar de 1200 a 2400 no mejoró la precisión de prueba. Con tasas de 0.001 y 0.01 el aprendizaje fue más lento. Con una tasa de 10 el resultado empeoró mucho: la precisión quedó en 34.3 %. Esto sugiere que los ajustes fueron demasiado grandes; para confirmar oscilaciones habría que revisar la pérdida durante todo el entrenamiento.

Ejemplos: `python main.py --epochs 2400` y `python main.py --learning-rate 0.01`.

## 5. Datos y etiquetas para una cuarta acción

Una cuarta acción podría ser estar tranquilo. Se usarían las mismas entradas y se etiquetarían como tranquilo los ejemplos con `hambre < 0.4`, `energía > 0.6` y `aburrimiento < 0.4`. En los demás casos se seguiría eligiendo la necesidad más alta.

Las etiquetas serían 0=comer, 1=dormir, 2=jugar y 3=tranquilo. Se necesitarían suficientes ejemplos de cada acción, incluyendo casos cerca de los límites. También habría que agregar una cuarta neurona de salida, actualizar las etiquetas y volver a entrenar.

Actualmente, el estado neutral se activa con una condición en la interfaz. La red todavía tiene tres salidas; la cuarta acción queda como propuesta.

## Preguntas de explicación

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
