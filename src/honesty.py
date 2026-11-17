"""Herramientas de honestidad: qué se conservó y qué inventó el modelo.

Idea central: la foto chica es el único DATO. Todo píxel nuevo es una suposición.
- La bicúbica amplía sin inventar: promedia vecinos (no agrega información).
- La resta difusión - bicúbica muestra dónde el modelo agregó detalle propio.
- Al achicar el resultado de nuevo al tamaño original se puede comprobar si el
  modelo respetó lo que sí estaba en la foto (coherencia).
"""

import os
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

# A partir de esta diferencia (escala 0-255) consideramos que un píxel "cambió".
UMBRAL_CAMBIO = 20

AVISO_ARCHIVO = "Ampliada con IA (difusión). Los detalles finos fueron generados por el modelo: no son datos de la foto original."


_FUENTES_TTF = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux / Colab
    "C:/Windows/Fonts/arial.ttf",  # Windows
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
]


def fuente(tamano: int = 16):
    """Una fuente con tildes si existe en el sistema; si no, la básica de Pillow."""
    for ruta in _FUENTES_TTF:
        if os.path.exists(ruta):
            return ImageFont.truetype(ruta, tamano)
    try:
        return ImageFont.load_default(size=tamano)
    except TypeError:  # versiones viejas de Pillow
        return ImageFont.load_default()


def texto_seguro(texto: str, letra) -> str:
    """Si la fuente no es TrueType del sistema, saca las tildes para que no salgan cuadraditos."""
    if isinstance(letra, ImageFont.FreeTypeFont) and any(os.path.exists(r) for r in _FUENTES_TTF):
        return texto
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def bicubica(imagen: Image.Image, factor: int = 4) -> Image.Image:
    """Línea base clásica: interpolación bicúbica (16 vecinos), sin IA."""
    return imagen.resize((imagen.width * factor, imagen.height * factor), Image.BICUBIC)


def _a_numeros(imagen: Image.Image) -> np.ndarray:
    return np.asarray(imagen.convert("RGB"), dtype=np.float32)


def _paleta_calor(valores: np.ndarray) -> np.ndarray:
    """Convierte valores 0-1 en colores: negro -> violeta -> naranja -> amarillo claro."""
    anclas = np.array([0.0, 0.33, 0.66, 1.0])
    colores = np.array([[0, 0, 4], [120, 28, 109], [237, 105, 37], [252, 255, 164]], dtype=np.float32)
    salida = np.zeros(valores.shape + (3,), dtype=np.float32)
    for canal in range(3):
        salida[..., canal] = np.interp(valores, anclas, colores[:, canal])
    return salida.astype(np.uint8)


def mapa_de_diferencias(con_ia: Image.Image, sin_ia: Image.Image) -> tuple[Image.Image, float]:
    """Dónde la versión con IA se aparta de la ampliación clásica.

    Devuelve el mapa (oscuro = casi igual, claro = el modelo agregó o cambió algo)
    y la fracción de píxeles con cambio notable.
    """
    if con_ia.size != sin_ia.size:
        sin_ia = sin_ia.resize(con_ia.size, Image.BICUBIC)
    diferencia = np.abs(_a_numeros(con_ia) - _a_numeros(sin_ia)).mean(axis=2)  # promedio de R, G y B
    fraccion = float((diferencia > UMBRAL_CAMBIO).mean())
    normalizada = np.clip(diferencia / 80.0, 0, 1)  # 80 o más se ve como el máximo
    return Image.fromarray(_paleta_calor(normalizada)), fraccion


def coherencia(entrada: Image.Image, con_ia: Image.Image) -> float:
    """Achica la versión con IA al tamaño de la foto original y mide cuánto difiere.

    Devuelve la diferencia promedio por píxel (0-255). Un valor bajo dice que el
    modelo respetó lo que había en la foto; NO dice que el detalle nuevo sea real.
    """
    reducida = con_ia.resize(entrada.size, Image.LANCZOS)
    return float(np.abs(_a_numeros(reducida) - _a_numeros(entrada)).mean())


def exportar_con_aviso(con_ia: Image.Image, ruta: str) -> str:
    """Guarda el resultado con una franja visible y metadatos que dicen que es reconstrucción."""
    alto_franja = max(18, con_ia.height // 20)
    salida = Image.new("RGB", (con_ia.width, con_ia.height + alto_franja), "black")
    salida.paste(con_ia, (0, 0))
    dibujo = ImageDraw.Draw(salida)
    letra = fuente(max(10, alto_franja - 8))
    dibujo.text((6, con_ia.height + 3), "Ampliada con IA - detalles reconstruidos, no originales", fill="white", font=letra)
    metadatos = PngInfo()
    metadatos.add_text("Description", AVISO_ARCHIVO)
    metadatos.add_text("Software", "tpi2-ampliador (stabilityai/stable-diffusion-x4-upscaler)")
    salida.save(ruta, pnginfo=metadatos)
    return ruta
