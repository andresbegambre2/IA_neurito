"""Compara entrenamiento y navegación con los mismos datos y semillas."""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import heapq
import json
import numpy as np

from laberinto_mundo import MAPAS, Laberinto, distancias_bfs, elegir_accion, vecino, valido
from laberinto_red import RedLaberinto, generar_datos


def ruta_experta(mapa, inicio, meta, metodo):
    """DFS usa pila; A* usa distancia Manhattan y coste acumulado."""
    padres = {inicio: None}
    acciones = {}
    costes = {inicio: 0}
    pila = [inicio]
    cola = [(0, inicio)]
    while pila if metodo == 'dfs' else cola:
        actual = pila.pop() if metodo == 'dfs' else heapq.heappop(cola)[1]
        if actual == meta:
            ruta = []
            while padres[actual] is not None:
                ruta.append(acciones[actual])
                actual = padres[actual]
            return ruta[::-1]
        # Orden fijo: arriba, abajo, izquierda, derecha.
        for a in (reversed(range(4)) if metodo == 'dfs' else range(4)):
            nueva = vecino(actual, a)
            if not valido(mapa, nueva):
                continue
            coste = costes[actual] + 1
            if metodo == 'dfs':
                if nueva in padres:
                    continue
            elif coste >= costes.get(nueva, float('inf')):
                continue
            padres[nueva], acciones[nueva], costes[nueva] = actual, a, coste
            if metodo == 'dfs':
                pila.append(nueva)
            else:
                h = abs(nueva[0] - meta[0]) + abs(nueva[1] - meta[1])
                heapq.heappush(cola, (coste + h, nueva))
    raise ValueError('Meta inaccesible')


def casos_desde(X):
    for x in X:
        yield (int(x[:3].argmax()), divmod(int(x[3:84].argmax()), 9),
               divmod(int(x[84:].argmax()), 9))


def navegar(red, casos, asistido):
    exitos = paredes = intentos = 0
    por_mapa = [[0, 0] for _ in MAPAS]
    for indice, inicio, meta, _ in casos:
        mundo = Laberinto(indice)
        mundo.posicion, mundo.meta = inicio, meta
        mundo.visitadas = {inicio: 1}
        limite = min(65, 2 * distancias_bfs(mundo.mapa, meta)[inicio] + 8)
        por_mapa[indice][1] += 1
        for _ in range(limite):
            accion, _ = elegir_accion(red, mundo, mundo.visitadas, asistido)
            resultado = mundo.mover(accion)
            intentos += 1
            if resultado == 'pared':
                paredes += 1
                break
            if resultado == 'comida':
                exitos += 1
                por_mapa[indice][0] += 1
                break
    return {'exito': exitos / len(casos), 'paredes': paredes,
            'intentos_promedio': intentos / len(casos), 'por_mapa': por_mapa}


def ejecutar():
    (X, y), (Xt, yt), casos = generar_datos()
    print(json.dumps({'entrenamiento': len(X), 'prueba': len(Xt)}), flush=True)
    configuraciones = [('bfs', 96, .45), ('bfs', 16, .45), ('bfs', 32, .45),
                       ('bfs', 128, .45), ('bfs', 96, .01), ('bfs', 96, .1),
                       ('bfs', 96, 1.2), ('dfs', 96, .45), ('astar', 96, .45)]
    for experto, ocultas, tasa in configuraciones:
        etiquetas, etiquetas_test = y, yt
        if experto != 'bfs':
            etiquetas = np.array([ruta_experta(MAPAS[i], p, m, experto)[0]
                                   for i, p, m in casos_desde(X)])
            etiquetas_test = np.array([ruta_experta(MAPAS[i], p, m, experto)[0]
                                        for i, p, m in casos_desde(Xt)])
        red = RedLaberinto(ocultas=ocultas, tasa=tasa)
        checkpoints = {0, 100, 200, 400} if (experto, ocultas, tasa) == ('bfs', 96, .45) else {400}
        for epoca in range(401):
            if epoca:
                red.entrenar_epoca(X, etiquetas)
            if epoca not in checkpoints:
                continue
            p = red.adelante(X)[2]
            perdida = float(-np.log(np.clip(p[np.arange(len(X)), etiquetas], 1e-10, 1)).mean())
            registro = dict(experto=experto, ocultas=ocultas, tasa=tasa,
                            epocas=epoca, perdida=perdida,
                            precision=red.exactitud(Xt, etiquetas_test),
                            coincidencia_bfs=red.exactitud(Xt, yt),
                            pura=navegar(red, casos, False),
                            asistida=navegar(red, casos, True))
            print(json.dumps(registro), flush=True)
    for metodo in ('dfs', 'astar'):
        razones = []
        for i, p, m, _ in casos:
            ruta = ruta_experta(MAPAS[i], p, m, metodo)
            actual = p
            for accion in ruta:
                actual = vecino(actual, accion)
                assert valido(MAPAS[i], actual)
            assert actual == m
            optima = distancias_bfs(MAPAS[i], m)[p]
            if metodo == 'astar':
                assert len(ruta) == optima
            razones.append(len(ruta) / optima)
        print(json.dumps({'rutas_expertas': metodo, 'longitud_relativa_bfs': float(np.mean(razones))}), flush=True)


if __name__ == '__main__':
    ejecutar()
