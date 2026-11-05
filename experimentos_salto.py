"""Experimentos reproducibles del salto, sin abrir Pygame."""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
import random
import numpy as np
from salto_modelo import RedSalto
from salto_juego import JuegoSalto


def generar_datos(semilla, cantidad, ventana):
    rng = np.random.default_rng(semilla)
    inferior, superior = ventana
    velocidad = rng.uniform(4.5, 8.5, cantidad)
    tiempos = np.concatenate((rng.uniform(inferior + .5, superior - .5, cantidad // 2),
                              rng.uniform(0, 60, cantidad - cantidad // 2)))
    rng.shuffle(tiempos)
    distancia = velocidad * tiempos
    X = RedSalto.preparar_entradas(distancia, velocidad)
    y = ((tiempos >= inferior) & (tiempos <= superior)).astype(float)[:, None]
    return X, y


def evaluar_juego(red):
    azar = random.Random(19)
    velocidades = [5.5] + [azar.uniform(4.5, 8.5) for _ in range(99)]
    choques = esquivados = 0
    for velocidad in velocidades:
        mundo = JuegoSalto()
        mundo.velocidad = velocidad
        for _ in range(500):
            if mundo.en_suelo and red.probabilidad(mundo.distancia, mundo.velocidad) >= .5:
                mundo.saltar()
            mundo.actualizar()
            if mundo.terminado:
                choques += 1
                break
            if mundo.esquivados:
                esquivados += 1
                break
        else:
            raise AssertionError('Obstáculo sin resolver dentro del límite')
    return choques, esquivados


def ejecutar():
    configuraciones = [(12, .8, (7, 13)), (3, .8, (7, 13)), (24, .8, (7, 13)),
                       (12, .01, (7, 13)), (12, 1.2, (7, 13)),
                       (12, .8, (3, 6)), (12, .8, (20, 26))]
    for ocultas, tasa, ventana in configuraciones:
        X, y = generar_datos(42, 5000, ventana)
        Xt, yt = generar_datos(123, 1500, ventana)
        red = RedSalto(ocultas=ocultas, tasa=tasa)
        checkpoints = {0, 100, 200, 300, 400} if (ocultas, tasa, ventana) == configuraciones[0] else {400}
        for epoca in range(401):
            if epoca:
                red.entrenar_epoca(X, y)
            if epoca not in checkpoints:
                continue
            p = np.clip(red.adelante(X)[2], 1e-8, 1 - 1e-8)
            perdida = float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p)))
            choques, esquivados = evaluar_juego(red)
            assert np.isfinite(perdida) and choques + esquivados == 100
            print(json.dumps(dict(ocultas=ocultas, tasa=tasa, ventana=ventana,
                                  epocas=epoca, perdida=perdida,
                                  precision=red.precision(Xt, yt),
                                  choques=choques, esquivados=esquivados)), flush=True)


if __name__ == '__main__':
    ejecutar()
