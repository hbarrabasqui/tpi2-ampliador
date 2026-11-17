"""Genera las evidencias de "qué es real y qué es inventado" con los ejemplos.

Ejecutar (con GPU, por ejemplo en Colab):  uv run python scripts/generar_evidencias.py

Para cada ejemplo tenemos el original grande (la VERDAD), así que podemos ver
qué inventó el modelo comparando contra lo que realmente había. Guarda en resultados/:
- <ejemplo>_comparacion.png : entrada | bicúbica | IA | verdad | mapa de lo inventado
- <ejemplo>_semillas.png    : misma foto, 3 semillas distintas -> 3 invenciones distintas
- <ejemplo>_libertad.png    : el mismo detalle con libertad 0, 20, 60 y 100
- resumen.md                : números (PSNR contra la verdad, coherencia, % cambiado, detecciones)
"""

import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from scripts.preparar_ejemplos import CASOS, EJEMPLOS, verdad  # noqa: E402
from src import analysis, diffusion, honesty  # noqa: E402

SALIDA = os.path.join(RAIZ, "resultados")

TIPOS = {
    "caras_kirillov_1909": "Personas o retratos",
    "retrato_1860": "Personas o retratos",
    "edificio_perm_1910": "Edificios o paisajes",
    "puente_1888": "Edificios o paisajes",
}

# Zona para mirar de cerca, en coordenadas de la imagen ampliada (caras, letras, ventanas)
ZOOM = {
    "caras_kirillov_1909": (100, 20, 880, 260),
    "retrato_1860": (150, 60, 360, 400),
    "edificio_perm_1910": (100, 120, 228, 248),
    "puente_1888": (200, 150, 500, 350),
}

LIBERTAD_BASE = 20


def fila(imagenes: list[tuple[str, Image.Image]], alto: int = 360) -> Image.Image:
    """Pone imágenes una al lado de la otra, con un título arriba de cada una."""
    escaladas = [(t, im.resize((max(1, round(im.width * alto / im.height)), alto), Image.NEAREST)) for t, im in imagenes]
    ancho = sum(im.width for _, im in escaladas) + 10 * (len(escaladas) - 1)
    lienzo = Image.new("RGB", (ancho, alto + 28), "white")
    dibujo = ImageDraw.Draw(lienzo)
    x = 0
    for titulo, im in escaladas:
        lienzo.paste(im, (x, 28))
        letra = honesty.fuente(16)
        dibujo.text((x + 4, 5), honesty.texto_seguro(titulo, letra), fill="black", font=letra)
        x += im.width + 10
    return lienzo


def psnr(a: Image.Image, b: Image.Image) -> float:
    """Parecido píxel a píxel con la verdad, en decibeles (más alto = más parecido)."""
    error = np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2)
    return float("inf") if error == 0 else 10 * np.log10(255 ** 2 / error)


def main():
    os.makedirs(SALIDA, exist_ok=True)
    print("Cargando modelos...")
    diffusion.cargar_modelo()
    analysis.cargar_modelo()
    filas_resumen = []

    for nombre in CASOS:
        inicio = time.time()
        tipo = TIPOS[nombre]
        entrada, _ = diffusion.preparar_entrada(Image.open(os.path.join(EJEMPLOS, f"{nombre}.png")))
        real = verdad(nombre).resize((entrada.width * 4, entrada.height * 4), Image.LANCZOS)
        sin_ia = honesty.bicubica(entrada)
        con_ia = diffusion.ampliar(entrada, tipo, LIBERTAD_BASE, semilla=0)
        mapa, fraccion = honesty.mapa_de_diferencias(con_ia, sin_ia)
        vecino = entrada.resize(sin_ia.size, Image.NEAREST)  # cómo se ve la foto chica "pixelada"

        sin_ia.save(os.path.join(SALIDA, f"{nombre}_bicubica.png"))
        con_ia.save(os.path.join(SALIDA, f"{nombre}_ia.png"))
        mapa.save(os.path.join(SALIDA, f"{nombre}_mapa.png"))
        fila([("Foto chica (pixelada)", vecino), ("Sin IA (bicúbica)", sin_ia), (f"Con IA (libertad {LIBERTAD_BASE})", con_ia),
              ("VERDAD (original grande)", real), ("Mapa de lo inventado", mapa)]).save(
            os.path.join(SALIDA, f"{nombre}_comparacion.png"))

        # Tres semillas: si el detalle fuera "recuperado", saldría igual las tres veces
        zona = ZOOM[nombre]
        semillas = [(f"IA semilla {s}", diffusion.ampliar(entrada, tipo, LIBERTAD_BASE, semilla=s).crop(zona)) for s in range(3)]
        fila([("VERDAD", real.crop(zona)), ("Sin IA", sin_ia.crop(zona))] + semillas).save(
            os.path.join(SALIDA, f"{nombre}_semillas.png"))

        # Cuatro niveles de libertad
        niveles = [(f"libertad {n}", diffusion.ampliar(entrada, tipo, n, semilla=0).crop(zona)) for n in (0, 20, 60, 100)]
        fila([("VERDAD", real.crop(zona))] + niveles).save(os.path.join(SALIDA, f"{nombre}_libertad.png"))

        deteccion = analysis.comparar(sin_ia, con_ia)
        deteccion["imagen_con_cajas"].save(os.path.join(SALIDA, f"{nombre}_detecciones.png"))
        filas_resumen.append(
            f"| {nombre} | {psnr(sin_ia, real):.1f} | {psnr(con_ia, real):.1f} | {honesty.coherencia(entrada, con_ia):.1f} "
            f"| {fraccion:.0%} | {deteccion['texto_sin_ia']} | {deteccion['texto_con_ia']} | {time.time() - inicio:.0f} s |")
        print(filas_resumen[-1], flush=True)

    with open(os.path.join(SALIDA, "resumen.md"), "w", encoding="utf-8") as archivo:
        archivo.write("| Ejemplo | PSNR bicúbica vs verdad (dB) | PSNR IA vs verdad (dB) | Coherencia IA (0-255) "
                      "| % cambiado vs bicúbica | Detecciones sin IA | Detecciones con IA | Tiempo |\n")
        archivo.write("|---|---|---|---|---|---|---|---|\n")
        archivo.write("\n".join(filas_resumen) + "\n")
    print("Listo. Resultados en", SALIDA)


if __name__ == "__main__":
    main()
