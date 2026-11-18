"""Análisis visual con RT-DETR (PekingU/rtdetr_r50vd): ¿apareció o desapareció algo?

Se compara la ampliación clásica (bicúbica) con la ampliación con IA. Las dos
tienen el mismo tamaño y salen de la misma foto, así que si el detector
encuentra cosas distintas, la diferencia la introdujo el modelo de difusión.

Límite importante: el detector confirma que "hay una persona", no que la cara
sea la de esa persona. Sirve para detectar cambios gruesos, no para validar detalles.
"""

from collections import Counter

import torch
from PIL import Image, ImageDraw
from transformers import pipeline

MODELO = "PekingU/rtdetr_r50vd"
CONFIANZA_MINIMA = 0.5  # descarta detecciones dudosas

# Traducción de las etiquetas más comunes del conjunto COCO
ETIQUETAS = {
    "person": "persona", "car": "auto", "bicycle": "bicicleta", "motorcycle": "moto",
    "bus": "colectivo", "truck": "camión", "boat": "bote", "dog": "perro", "cat": "gato",
    "horse": "caballo", "bird": "pájaro", "chair": "silla", "bench": "banco", "tie": "corbata",
    "handbag": "cartera", "umbrella": "paraguas", "bowl": "bol", "cup": "taza", "clock": "reloj",
    "potted plant": "planta", "dining table": "mesa", "book": "libro", "vase": "florero",
    "bottle": "botella", "cell phone": "celular", "tv": "televisor", "laptop": "notebook",
    "train": "tren", "traffic light": "semáforo", "cow": "vaca", "sheep": "oveja", "teddy bear": "peluche",
    "cake": "torta", "pizza": "pizza", "donut": "dona", "sandwich": "sándwich", "apple": "manzana",
    "orange": "naranja", "banana": "banana", "backpack": "mochila", "suitcase": "valija", "kite": "barrilete",
    "couch": "sillón", "bed": "cama", "airplane": "avión", "elephant": "elefante", "bear": "oso",
}

_detector = None


def cargar_modelo():
    global _detector
    if _detector is None:
        dispositivo = 0 if torch.cuda.is_available() else -1  # 0 = primera GPU, -1 = CPU
        _detector = pipeline("object-detection", model=MODELO, device=dispositivo)
    return _detector


def detectar(imagen: Image.Image) -> list[dict]:
    detecciones = cargar_modelo()(imagen)
    return [d for d in detecciones if d["score"] >= CONFIANZA_MINIMA]


def _nombre(etiqueta: str) -> str:
    return ETIQUETAS.get(etiqueta, etiqueta)


def dibujar_cajas(imagen: Image.Image, detecciones: list[dict]) -> Image.Image:
    copia = imagen.convert("RGB").copy()
    dibujo = ImageDraw.Draw(copia)
    for d in detecciones:
        caja = d["box"]
        dibujo.rectangle([caja["xmin"], caja["ymin"], caja["xmax"], caja["ymax"]], outline=(0, 255, 120), width=2)
        dibujo.text((caja["xmin"] + 3, caja["ymin"] + 2), f"{_nombre(d['label'])} {d['score']:.2f}", fill=(0, 255, 120))
    return copia


def _contar(detecciones: list[dict]) -> Counter:
    return Counter(_nombre(d["label"]) for d in detecciones)


def _describir(conteo: Counter) -> str:
    if not conteo:
        return "nada que pudiera reconocer con seguridad"
    return ", ".join(f"{cantidad} {nombre}" for nombre, cantidad in conteo.most_common())


def comparar(sin_ia: Image.Image, con_ia: Image.Image) -> dict:
    """Detecta objetos en ambas versiones y resume las diferencias en lenguaje simple."""
    antes, despues = detectar(sin_ia), detectar(con_ia)
    conteo_antes, conteo_despues = _contar(antes), _contar(despues)
    cambios = []
    for nombre in sorted(set(conteo_antes) | set(conteo_despues)):
        a, b = conteo_antes[nombre], conteo_despues[nombre]
        if b > a:
            cambios.append(f"aparecieron {b - a} {nombre}")
        elif a > b:
            cambios.append(f"desaparecieron {a - b} {nombre}")
    return {
        "personas_sin_ia": conteo_antes["persona"],
        "personas_con_ia": conteo_despues["persona"],
        "texto_sin_ia": _describir(conteo_antes),
        "texto_con_ia": _describir(conteo_despues),
        "cambios": cambios,  # texto simple: "aparecieron 1 perro"
        "imagen_con_cajas": dibujar_cajas(con_ia, despues),
    }
