"""Muestra los resultados de los experimentos en la terminal."""
import os

# Evita sobrecostos de BLAS para matrices pequeñas.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from brain import Brain, policy


def main():
    test = np.random.default_rng(123).random((1000, 3))
    targets = policy(test)
    configs = [(3, 1200, .1), (6, 1200, .1), (10, 1200, .1), (20, 1200, .1),
               (6, 200, .1), (6, 600, .1), (6, 2400, .1),
               (6, 1200, .001), (6, 1200, .01), (6, 1200, 1.), (6, 1200, 10.)]
    lines = ["# Resultados reproducibles", "",
             "3000 ejemplos de entrenamiento (semilla 42) y los mismos 1000 ejemplos de prueba "
             "(default_rng, semilla 123) en todas las configuraciones. Una ejecución por configuración. "
             "La pérdida se mide después de la última actualización, sobre entrenamiento.", "",
             "| Neuronas | Épocas | Tasa | Pérdida | Precisión de prueba |",
             "|---:|---:|---:|---:|---:|"]
    for hidden, epochs, lr in configs:
        model = Brain(hidden).train(epochs, lr)
        accuracy = np.mean(model.predict(test).argmax(axis=1) == targets)
        row = f"| {hidden} | {epochs} | {lr:g} | {model.loss():.4f} | {accuracy:.1%} |"
        lines.append(row)
        if (hidden, epochs, lr) == (6, 1200, .1):
            baseline = model
    lines += ["", "## Variar hambre con energía 0.4 y aburrimiento 0.3", "",
              "| Hambre | Comer | Dormir | Jugar | Predicción |", "|---:|---:|---:|---:|---|"]
    actions = ["Comer", "Dormir", "Jugar"]
    for hunger in [.4, .5, .59, .6, .61, .7, .8]:
        p = baseline.predict(np.array([[hunger, .4, .3]]))[0]
        lines.append(f"| {hunger:g} | {p[0]:.2%} | {p[1]:.2%} | {p[2]:.2%} | {actions[p.argmax()]} |")
    lines += ["", "## Prioridades casi iguales", "",
              "| Hambre | Energía | Aburrimiento | Comer | Dormir | Jugar |",
              "|---:|---:|---:|---:|---:|---:|"]
    for x in [[.5, .5, .5], [.51, .5, .49], [.49, .5, .51]]:
        p = baseline.predict(np.array([x]))[0]
        lines.append(f"| {x[0]:g} | {x[1]:g} | {x[2]:g} | {p[0]:.2%} | {p[1]:.2%} | {p[2]:.2%} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
