"""Ampliación 4x con un modelo de difusión (stabilityai/stable-diffusion-x4-upscaler).

El modelo NO recupera información perdida: genera detalle plausible que,
al achicarlo, se parece a la foto de entrada. Por eso todo lo que devuelve
este módulo se trata en la app como una reconstrucción, no como un dato.
"""

import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

MODELO = "stabilityai/stable-diffusion-x4-upscaler"

# Límites de la foto de entrada. La salida es 4x en cada lado (16x en área), así
# que una entrada grande llena enseguida la memoria. Con GPU (T4 de Colab, 15 GB)
# se admite hasta 256 px de lado y unos 40.000 píxeles (por ejemplo 200 x 200 o
# 256 x 128); en CPU se achica más para que no tarde demasiado.
LADO_MAXIMO_GPU = 256
AREA_MAXIMA_GPU = 40_000
LADO_MAXIMO_CPU = 128
AREA_MAXIMA_CPU = 128 * 128

# Frases que recibe el modelo según el tipo de foto. Van en inglés porque el
# codificador de texto del modelo (CLIP) se entrenó con textos en inglés.
PROMPTS = {
    "Personas o retratos": "a photograph of people, natural faces",
    "Edificios o paisajes": "a photograph of a building or landscape, sharp details",
    "Objetos u otra cosa": "a sharp photograph",
}

_pipe = None  # el modelo se carga una sola vez, la primera vez que se usa


def hay_gpu() -> bool:
    return torch.cuda.is_available()


def lado_maximo() -> int:
    return LADO_MAXIMO_GPU if hay_gpu() else LADO_MAXIMO_CPU


def cargar_modelo():
    """Carga el upscaler (unos 3 GB la primera vez que se descarga)."""
    global _pipe
    if _pipe is None:
        if hay_gpu():
            # float16: usa la mitad de memoria de la GPU
            _pipe = StableDiffusionUpscalePipeline.from_pretrained(MODELO, dtype=torch.float16).to("cuda")
        else:
            # en CPU float16 es lento o falla; se usa float32
            _pipe = StableDiffusionUpscalePipeline.from_pretrained(MODELO, dtype=torch.float32)
        # calcula la "atención" por partes: un poco más lento, mucha menos memoria
        _pipe.enable_attention_slicing()
    return _pipe


def area_maxima() -> int:
    return AREA_MAXIMA_GPU if hay_gpu() else AREA_MAXIMA_CPU


def preparar_entrada(imagen: Image.Image) -> tuple[Image.Image, bool]:
    """Pasa a RGB, achica si supera el lado o el área máxima y deja lados múltiplos de 8.

    Devuelve la imagen preparada y si hubo que achicarla (para avisarle a la persona).
    """
    imagen = imagen.convert("RGB")  # quita transparencia (PNG) y modos raros
    escala = min(1.0, lado_maximo() / max(imagen.size), (area_maxima() / (imagen.width * imagen.height)) ** 0.5)
    achicada = escala < 1.0
    if achicada:
        nuevo = (max(8, int(imagen.width * escala)), max(8, int(imagen.height * escala)))
        imagen = imagen.resize(nuevo, Image.LANCZOS)  # LANCZOS: achica sin aliasing fuerte
    ancho = max(8, imagen.width - imagen.width % 8)
    alto = max(8, imagen.height - imagen.height % 8)
    if (ancho, alto) != imagen.size:
        imagen = imagen.resize((ancho, alto), Image.LANCZOS)
    return imagen, achicada


def ampliar(imagen: Image.Image, tipo_de_foto: str, libertad: int, semilla: int = 0, pasos: int = 20) -> Image.Image:
    """Amplía la imagen 4x con difusión.

    libertad -> noise_level del modelo: cuánto ruido se le suma a la foto chica
    antes de ampliarla. Más ruido = el modelo se apoya menos en la foto y
    inventa más. 0 a 100 es un rango razonable (el modelo admite hasta 350).
    """
    pipe = cargar_modelo()
    dispositivo = "cuda" if hay_gpu() else "cpu"
    generador = torch.Generator(dispositivo).manual_seed(int(semilla))  # misma semilla = mismo resultado
    resultado = pipe(
        prompt=PROMPTS.get(tipo_de_foto, PROMPTS["Objetos u otra cosa"]),
        image=imagen,
        num_inference_steps=pasos,
        noise_level=int(libertad),
        generator=generador,
    ).images[0]
    if max(resultado.getextrema(), key=lambda canal: canal[1])[1] < 5:
        # problema numérico conocido de float16: la imagen sale negra
        raise RuntimeError("imagen negra")
    return resultado
