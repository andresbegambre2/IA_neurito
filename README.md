# Neurito: mi cerebro artificial

Mascota educativa en Python y Pygame que utiliza una red neuronal implementada con NumPy para elegir entre comer, dormir y jugar según hambre, energía y aburrimiento.

## Ejecutar

Requiere Python 3.12 o superior. Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pygame-ce
.\.venv\Scripts\python.exe main.py
```

También se puede usar `uv sync` y `uv run python main.py`.

## Configurar experimentos

```powershell
python main.py --hidden 3
python main.py --hidden 10 --epochs 2400 --learning-rate 0.1
python main.py --help
python experimentos.py
```

Los valores por defecto son 6 neuronas ocultas, 1200 épocas y tasa 0.1. La interfaz está en español. `experimentos.py` funciona sin abrir una ventana y regenera [RESULTADOS.md](RESULTADOS.md).

Las respuestas a los cinco experimentos y a las cuatro preguntas están en [EXPERIMENTOS.md](EXPERIMENTOS.md).

## Controles

| Teclas | Función |
|---|---|
| H / J | Aumentar / disminuir hambre |
| E / D | Aumentar / disminuir energía |
| A / S | Aumentar / disminuir aburrimiento |
| R | Necesidades aleatorias |
| N | Restablecer estado tranquilo |
| Esc | Salir |

## Archivos

- `main.py`: interfaz y animaciones de Pygame.
- `brain.py`: red neuronal, entrenamiento y política sintética.
- `experimentos.py`: comparación reproducible de configuraciones.
- `assets/`: imágenes de las animaciones.

El estado neutral se activa mediante una regla visual; la red tiene tres salidas. La precisión mide coincidencia con una política sintética y no constituye una evaluación de comportamiento real.

Se conserva la estructura y la autoría declarada del proyecto original; esta versión añade parámetros de experimentación y documentación.
