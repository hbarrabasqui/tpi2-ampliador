"""Verificaciones mínimas que NO necesitan descargar modelos.

Ejecutar con:  uv run python tests/verificar.py
Prueban la parte clásica (bicúbica, mapa de diferencias, coherencia, exportación)
y que la app arme el informe de punta a punta, reemplazando los dos modelos
por versiones falsas y rápidas.
"""

import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis, diffusion, honesty  # noqa: E402

EJEMPLO = os.path.join(os.path.dirname(__file__), "..", "assets", "ejemplos", "edificio_perm_1910.png")


def probar_parte_clasica():
    chica = Image.open(EJEMPLO).convert("RGB")
    grande = honesty.bicubica(chica)
    assert grande.size == (chica.width * 4, chica.height * 4)

    # Comparar una imagen consigo misma: el mapa tiene que dar 0 % de cambio
    _, fraccion = honesty.mapa_de_diferencias(grande, grande)
    assert fraccion == 0.0

    # La bicúbica, achicada de nuevo, tiene que parecerse mucho a la foto de entrada
    assert honesty.coherencia(chica, grande) < 6

    # Una versión con "detalle inventado" (ruido) tiene que marcar cambios
    ruido = np.random.default_rng(0).normal(0, 40, (grande.height, grande.width, 3))
    inventada = Image.fromarray(np.clip(np.asarray(grande) + ruido, 0, 255).astype(np.uint8))
    _, fraccion = honesty.mapa_de_diferencias(inventada, grande)
    assert fraccion > 0.3

    # El archivo exportado lleva el aviso en los metadatos y una franja visible
    ruta = os.path.join(tempfile.mkdtemp(), "salida.png")
    honesty.exportar_con_aviso(grande, ruta)
    exportada = Image.open(ruta)
    assert "IA" in exportada.info["Description"]
    assert exportada.height > grande.height
    print("OK parte clásica: bicúbica, mapa, coherencia y exportación")


def probar_entrada():
    transparente = Image.new("RGBA", (300, 203), (255, 0, 0, 0))
    preparada, achicada = diffusion.preparar_entrada(transparente)
    assert preparada.mode == "RGB" and achicada
    assert preparada.width % 8 == 0 and preparada.height % 8 == 0
    assert max(preparada.size) <= diffusion.lado_maximo()
    assert preparada.width * preparada.height <= diffusion.area_maxima()
    print("OK entrada: PNG con transparencia, achicado y múltiplos de 8")


def probar_app_con_modelos_falsos():
    import app

    # Modelos falsos: "ampliar" = bicúbica + realce; "detectar" = una persona fija
    diffusion.ampliar = lambda imagen, tipo, libertad, semilla=0, pasos=20: honesty.bicubica(imagen).filter(ImageFilter.UnsharpMask(3, 200))
    analysis.detectar = lambda imagen: [{"label": "person", "score": 0.9,
                                         "box": {"xmin": 10, "ymin": 10, "xmax": 100, "ymax": 200}}]

    comparacion, mapa, cajas_sin, cajas_con, chequeo, resumen, ruta = app.procesar(EJEMPLO, "Edificios o paisajes", 20, 0, progreso=lambda *a, **k: None)
    assert comparacion[0].size == comparacion[1].size == cajas_sin.size == cajas_con.size
    assert "sin IA" in chequeo and "con IA" in chequeo and "persona" in chequeo
    assert "RT-DETR" not in resumen and "bicúbica" not in resumen  # sin jerga para la persona usuaria
    assert os.path.exists(ruta)
    app.crear_interfaz()  # que la interfaz se arme sin errores
    print("OK app de punta a punta con modelos falsos")


if __name__ == "__main__":
    probar_parte_clasica()
    probar_entrada()
    probar_app_con_modelos_falsos()
    print("Todas las verificaciones pasaron.")
