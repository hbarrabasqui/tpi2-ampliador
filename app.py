"""Ampliador honesto de fotos chicas.

Ejecutar con:  uv run app.py
En Google Colab se abre además un link público (share=True) automáticamente.
"""

import os
import sys
import tempfile
import time

import gradio as gr
from PIL import Image, UnidentifiedImageError

from src import analysis, diffusion, honesty

TAMANO_MAXIMO_MB = 10
CARPETA_EJEMPLOS = os.path.join(os.path.dirname(__file__), "assets", "ejemplos")

AVISO_PRINCIPAL = (
    "**Importante:** esta herramienta *no recupera* lo que tu foto perdió. "
    "La IA **inventa** detalles creíbles para que la foto agrandada se vea nítida. "
    "Usá el resultado para mirar o compartir, **no como prueba** de cómo era algo o alguien."
)

AVISOS_POR_TIPO = {
    "Personas o retratos": (
        "Las **caras** ampliadas son una reconstrucción: pueden verse más jóvenes, más simétricas "
        "o directamente parecerse a otra persona. Compará con otras fotos antes de compartirla."
    ),
    "Edificios o paisajes": (
        "Ventanas, ladrillos, hojas y texturas finas fueron **dibujados por la IA**: "
        "la forma general viene de tu foto, el detalle no."
    ),
    "Objetos u otra cosa": "El detalle fino (texturas, bordes nítidos) fue **inventado por la IA**.",
}


def _abrir_imagen(ruta: str) -> Image.Image:
    """Abre la imagen con mensajes de error comprensibles."""
    if ruta is None:
        raise gr.Error("Primero subí una foto o elegí un ejemplo.")
    tamano_mb = os.path.getsize(ruta) / (1024 * 1024)
    if tamano_mb > TAMANO_MAXIMO_MB:
        raise gr.Error(f"La foto pesa {tamano_mb:.0f} MB. Probá con una de menos de {TAMANO_MAXIMO_MB} MB.")
    try:
        return Image.open(ruta).convert("RGB")
    except UnidentifiedImageError as error:
        raise gr.Error("No pude leer ese archivo como imagen. Probá con un JPG o PNG.") from error


def _interpretar_coherencia(diferencia: float) -> str:
    if diferencia < 6:
        return "la IA **respetó** las formas y colores de tu foto"
    if diferencia < 15:
        return "la IA **cambió un poco** tu foto (algunos colores o formas)"
    return "la IA **se alejó bastante** de tu foto: revisá el resultado con cuidado"


def procesar(ruta, tipo_de_foto, libertad, semilla, progreso=gr.Progress()):
    original = _abrir_imagen(ruta)

    progreso(0.05, desc="Preparando la foto...")
    entrada, fue_achicada = diffusion.preparar_entrada(original)
    sin_ia = honesty.bicubica(entrada)

    progreso(0.15, desc="La IA está agrandando tu foto e inventando detalles...")
    inicio = time.time()
    try:
        con_ia = diffusion.ampliar(entrada, tipo_de_foto, libertad, semilla)
    except RuntimeError as error:
        if diffusion.hay_gpu():
            import torch
            torch.cuda.empty_cache()  # libera la memoria de la GPU para el próximo intento
        if "imagen negra" in str(error):
            raise gr.Error("La IA devolvió una imagen negra (es un error conocido). "
                           "Probá con otra variante en Opciones avanzadas.") from error
        raise gr.Error("No pude ampliar la foto (probablemente falta memoria). Probá con una foto más chica.") from error
    segundos = time.time() - inicio

    progreso(0.75, desc="Comparando con la foto agrandada sin IA...")
    mapa, fraccion_cambiada = honesty.mapa_de_diferencias(con_ia, sin_ia)
    diferencia_coherencia = honesty.coherencia(entrada, con_ia)

    progreso(0.85, desc="Otra IA está buscando personas y objetos para chequear...")
    deteccion = analysis.comparar(sin_ia, con_ia)

    ruta_salida = os.path.join(tempfile.mkdtemp(), "foto_ampliada_con_IA.png")
    honesty.exportar_con_aviso(con_ia, ruta_salida)

    informe = _armar_informe(entrada, con_ia, fue_achicada, fraccion_cambiada,
                             diferencia_coherencia, deteccion, tipo_de_foto, segundos)
    return (sin_ia, con_ia), mapa, deteccion["imagen_con_cajas"], informe, ruta_salida


def _armar_informe(entrada, con_ia, fue_achicada, fraccion, coherencia, deteccion, tipo, segundos):
    # Lenguaje simple: la persona usuaria no tiene conocimientos técnicos.
    # Los nombres técnicos (bicúbica, RT-DETR, difusión) están en el README.
    lineas = ["### Qué pasó con tu foto"]
    if fue_achicada:
        lineas.append(f"- Tu foto era grande: primero la achiqué a {entrada.width} × {entrada.height} puntos, "
                      "porque la IA trabaja con fotos chicas.")
    lineas.append(f"- Ahora es **4 veces más grande** de cada lado ({con_ia.width} × {con_ia.height} puntos). "
                  f"**Todo ese detalle nuevo no estaba en tu foto.** (Tardó {segundos:.0f} segundos.)")
    lineas.append(f"- La IA agregó o cambió detalles en un **{fraccion:.0%}** de la foto. "
                  "En la imagen *\"Dónde inventó la IA\"*, **las zonas claras son las inventadas**.")
    lineas.append(f"- Comparando el resultado achicado con tu foto: {_interpretar_coherencia(coherencia)}. "
                  "Ojo: eso no quiere decir que los detalles nuevos sean verdaderos.")

    lineas.append("\n### Chequeo con otra IA")
    lineas.append("Otra IA, que reconoce personas y objetos, miró las dos versiones:")
    lineas.append(f"- En la agrandada **sin IA** vio: {deteccion['texto_sin_ia']}.")
    lineas.append(f"- En la agrandada **con IA** vio: {deteccion['texto_con_ia']}.")
    if deteccion["cambios"]:
        lineas.append(f"- **Atención: {'; '.join(deteccion['cambios'])}.** La IA que agrandó la foto pudo haber "
                      "cambiado algo importante. Miralo con cuidado. (También puede ser un error de la IA que chequea.)")
    else:
        lineas.append("- No apareció ni desapareció nada. Igual, eso no garantiza que los detalles "
                      "(una cara, una letra) sean los verdaderos.")

    lineas.append("\n### Antes de usar la foto")
    lineas.append(f"- {AVISOS_POR_TIPO.get(tipo, AVISOS_POR_TIPO['Objetos u otra cosa'])}")
    lineas.append("- Si hay **letras o números**, la IA puede inventarlos o deformarlos.")
    lineas.append("- La foto que descargás lleva la leyenda *\"Ampliada con IA\"* para que nadie la confunda con la original.")
    return "\n".join(lineas)


def _ejemplos():
    if not os.path.isdir(CARPETA_EJEMPLOS):
        return []
    tipos = {"caras": "Personas o retratos", "retrato": "Personas o retratos",
             "edificio": "Edificios o paisajes", "puente": "Edificios o paisajes"}
    lista = []
    for archivo in sorted(os.listdir(CARPETA_EJEMPLOS)):
        if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
            tipo = next((t for clave, t in tipos.items() if archivo.startswith(clave)), "Objetos u otra cosa")
            lista.append([os.path.join(CARPETA_EJEMPLOS, archivo), tipo])
    return lista


def crear_interfaz() -> gr.Blocks:
    with gr.Blocks(title="Ampliador honesto de fotos") as demo:
        gr.Markdown("# Ampliador honesto de fotos chicas\n"
                    "Subí una foto chica o de baja calidad y la agrando **4 veces** con inteligencia artificial (IA). "
                    "Te muestro qué detalles agregó la IA por su cuenta.")
        gr.Markdown(AVISO_PRINCIPAL)

        with gr.Row():
            with gr.Column(scale=1):
                entrada = gr.Image(type="filepath", label="Tu foto (chica: hasta unos 200 × 200 px)")
                tipo = gr.Radio(list(diffusion.PROMPTS), value="Personas o retratos",
                                label="¿Qué muestra la foto?",
                                info="Le da una pista a la IA y ajusta los consejos.")
                libertad = gr.Slider(0, 100, value=20, step=5,
                                     label="Cuánto puede inventar la IA",
                                     info="Poco: se parece más a tu foto. Mucho: más detalle, pero inventa más y se equivoca más.")
                with gr.Accordion("Opciones avanzadas", open=False):
                    semilla = gr.Number(value=0, precision=0, label="Variante",
                                        info="Cambiá el número para ver otra versión inventada de la misma foto. Con el mismo número, sale siempre igual.")
                boton = gr.Button("Ampliar", variant="primary")
                ejemplos = _ejemplos()
                if ejemplos:
                    gr.Examples(ejemplos, inputs=[entrada, tipo], label="Fotos de ejemplo (antiguas, de uso libre)")

            with gr.Column(scale=2):
                comparador = gr.ImageSlider(label="Mové la línea · Izquierda: agrandada de forma común, sin IA · Derecha: agrandada con IA",
                                            type="pil")
                with gr.Row():
                    mapa = gr.Image(label="Dónde inventó la IA (más claro = más inventado)", type="pil")
                    cajas = gr.Image(label="Lo que reconoce otra IA en la foto agrandada con IA", type="pil")
                informe = gr.Markdown()
                descarga = gr.File(label="Descargar resultado (lleva la leyenda \"Ampliada con IA\")")

        boton.click(procesar, [entrada, tipo, libertad, semilla], [comparador, mapa, cajas, informe, descarga])
    return demo


if __name__ == "__main__":
    print("Cargando modelos (la primera vez se descargan, unos 3 GB)...")
    diffusion.cargar_modelo()
    analysis.cargar_modelo()
    if diffusion.hay_gpu():
        import torch
        print("GPU disponible:", torch.cuda.get_device_name(0))
    else:
        print("Sin GPU: va a funcionar, pero lento (varios minutos por foto).")
    en_colab = "COLAB_RELEASE_TAG" in os.environ or "--share" in sys.argv
    crear_interfaz().launch(share=en_colab)
