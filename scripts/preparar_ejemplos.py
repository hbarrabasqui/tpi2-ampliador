"""Arma las fotos de ejemplo chicas a partir de fotos de dominio público.

Ejecutar con:  uv run python scripts/preparar_ejemplos.py

Las fotos originales (assets/fuentes/) son de dominio público, de Wikimedia Commons
(ver assets/FUENTES.md). Se recorta una zona y se achica, para simular una foto de
baja resolución. Como tenemos el original grande, podemos comparar lo que inventa
el modelo con la VERDAD: eso es lo que hace scripts/generar_evidencias.py.
"""

import os

from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUENTES = os.path.join(RAIZ, "assets", "fuentes")
EJEMPLOS = os.path.join(RAIZ, "assets", "ejemplos")

# nombre: (archivo fuente, recorte (izq, arriba, der, abajo) o None, tamaño chico final)
CASOS = {
    "caras_kirillov_1909": ("original_kirillov_1909.jpg", (440, 420, 1400, 900), (256, 128)),
    "edificio_perm_1910": ("original_perm_1910.jpg", (920, 380, 1432, 892), (128, 128)),
    "retrato_1860": ("danado_retrato_1860.jpg", (440, 120, 1040, 720), (128, 128)),
    "puente_1888": ("danado_puente_1888.jpg", None, (176, 128)),
}


def recorte_original(nombre: str) -> Image.Image:
    """La zona del original grande que corresponde al ejemplo (la 'verdad')."""
    archivo, caja, _ = CASOS[nombre]
    imagen = Image.open(os.path.join(FUENTES, archivo)).convert("RGB")
    return imagen.crop(caja) if caja else imagen


def verdad(nombre: str) -> Image.Image:
    """La verdad al mismo tamaño que la salida del modelo (4x el ejemplo)."""
    ancho, alto = CASOS[nombre][2]
    return recorte_original(nombre).resize((ancho * 4, alto * 4), Image.LANCZOS)


def main():
    os.makedirs(EJEMPLOS, exist_ok=True)
    for nombre, (_, _, tamano) in CASOS.items():
        chica = recorte_original(nombre).resize(tamano, Image.LANCZOS)  # LANCZOS: achica sin aliasing fuerte
        chica.save(os.path.join(EJEMPLOS, f"{nombre}.png"))
        print(f"{nombre}: {tamano[0]} x {tamano[1]} px")


if __name__ == "__main__":
    main()
