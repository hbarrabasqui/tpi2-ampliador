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
    "Un modelo de IA **inventa** detalles creíbles para que la foto ampliada se vea nítida. "
    "Usá el resultado para mirar o compartir, **no como prueba** de cómo era algo o alguien."
)

AVISOS_POR_TIPO = {
    "Personas o retratos": (
        "Las **caras** ampliadas son una reconstrucción: pueden verse más jóvenes, más simétricas "
        "o directamente parecerse a otra persona. Compará con otras fotos antes de compartirla."
    ),
    "Edificios o paisajes": (
        "Ventanas, ladrillos, hojas y texturas finas fueron **dibujados por el modelo**: "
        "la forma general viene de tu foto, el detalle no."
    ),
    "Objetos u otra cosa": "El detalle fino (texturas, bordes nítidos) fue **generado por el modelo**.",
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
        return "el modelo **respetó** lo que había en tu foto"
    if diferencia < 15:
        return "el modelo **cambió un poco** tu foto (colores o formas)"
    return "el modelo **se alejó bastante** de tu foto: revisá el resultado con cuidado"


def procesar(ruta, tipo_de_foto, libertad, semilla, progreso=gr.Progress()):
    original = _abrir_imagen(ruta)

    progreso(0.05, desc="Preparando la foto...")
    entrada, fue_achicada = diffusion.preparar_entrada(original)
    sin_ia = honesty.bicubica(entrada)

    progreso(0.15, desc="Ampliando con el modelo de difusión (inventa detalle plausible)...")
    inicio = time.time()
    try:
        con_ia = diffusion.ampliar(entrada, tipo_de_foto, libertad, semilla)
    except RuntimeError as error:
        if diffusion.hay_gpu():
            import torch
            torch.cuda.empty_cache()  # libera la memoria de la GPU para el próximo intento
        if "imagen negra" in str(error):
            raise gr.Error("El modelo devolvió una imagen negra (un problema numérico conocido). "
                           "Probá con otra semilla en Opciones avanzadas.") from error
        raise gr.Error("No pude ampliar la foto (probablemente falta memoria). Probá con una foto más chica.") from error
    segundos = time.time() - inicio

    progreso(0.75, desc="Comparando con la ampliación clásica...")
    mapa, fraccion_cambiada = honesty.mapa_de_diferencias(con_ia, sin_ia)
    diferencia_coherencia = honesty.coherencia(entrada, con_ia)

    progreso(0.85, desc="Buscando personas y objetos en ambas versiones (RT-DETR)...")
    deteccion = analysis.comparar(sin_ia, con_ia)

    ruta_salida = os.path.join(tempfile.mkdtemp(), "foto_ampliada_con_IA.png")
    honesty.exportar_con_aviso(con_ia, ruta_salida)

    informe = _armar_informe(entrada, con_ia, fue_achicada, fraccion_cambiada,
                             diferencia_coherencia, deteccion, tipo_de_foto, segundos)
    return (sin_ia, con_ia), mapa, deteccion["imagen_con_cajas"], informe, ruta_salida


def _armar_informe(entrada, con_ia, fue_achicada, fraccion, coherencia, deteccion, tipo, segundos):
    lineas = ["### Qué hizo el sistema"]
    if fue_achicada:
        lineas.append(f"- Tu foto era grande: la achiqué a **{entrada.width} × {entrada.height} px** antes de ampliarla, "
                      "para que entre en la memoria del equipo.")
    lineas.append(f"- Pasó de **{entrada.width} × {entrada.height}** a **{con_ia.width} × {con_ia.height} px**: "
                  "16 veces más píxeles. **Esa información nueva no estaba en tu foto.** "
                  f"(Tardó {segundos:.0f} s.)")
    lineas.append(f"- Comparado con una ampliación clásica sin IA (bicúbica), el modelo cambió de forma notable "
                  f"el **{fraccion:.0%}** de la imagen. En el mapa, **lo claro es detalle inventado**.")
    lineas.append(f"- Al volver a achicar el resultado al tamaño de tu foto, la diferencia promedio es "
                  f"**{coherencia:.1f}** (en una escala de 0 a 255): {_interpretar_coherencia(coherencia)}. "
                  "Esto dice si respetó tu foto, **no** si el detalle nuevo es real.")

    lineas.append("\n### Verificación con un detector de objetos (RT-DETR)")
    lineas.append(f"- Sin IA encontró: {deteccion['texto_sin_ia']}.")
    lineas.append(f"- Con IA encontró: {deteccion['texto_con_ia']}.")
    if deteccion["cambios"]:
        lineas.append(f"- **Atención: {'; '.join(deteccion['cambios'])}.** El modelo pudo haber agregado o borrado algo. "
                      "(También puede ser que el detector reconozca mejor una imagen más nítida.)")
    else:
        lineas.append("- No aparecieron ni desaparecieron objetos. Ojo: esto no garantiza que los detalles "
                      "(una cara, una letra) sean los verdaderos.")

    lineas.append("\n### Antes de usar la foto")
    lineas.append(f"- {AVISOS_POR_TIPO.get(tipo, AVISOS_POR_TIPO['Objetos u otra cosa'])}")
    lineas.append("- Si hay **letras o números**, el modelo puede inventarlos o deformarlos.")
    lineas.append("- El archivo que descargás lleva la leyenda *\"Ampliada con IA\"* para que nadie lo confunda con el original.")
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
                    "Subí una foto de baja resolución y la amplío **4 veces** con un modelo de IA. "
                    "Te muestro qué agregó el modelo por su cuenta.")
        gr.Markdown(AVISO_PRINCIPAL)

        with gr.Row():
            with gr.Column(scale=1):
                entrada = gr.Image(type="filepath", label="Tu foto (chica: hasta unos 200 × 200 px)")
                tipo = gr.Radio(list(diffusion.PROMPTS), value="Personas o retratos",
                                label="¿Qué muestra la foto?",
                                info="Le da una pista al modelo y ajusta los avisos.")
                libertad = gr.Slider(0, 100, value=20, step=5,
                                     label="Libertad del modelo para inventar",
                                     info="Bajo: se pega más a tu foto. Alto: inventa más detalle (y se equivoca más).")
                with gr.Accordion("Opciones avanzadas", open=False):
                    semilla = gr.Number(value=0, precision=0, label="Semilla",
                                        info="Con la misma semilla el resultado se repite. Cambiala para ver otra invención posible.")
                boton = gr.Button("Ampliar", variant="primary")
                ejemplos = _ejemplos()
                if ejemplos:
                    gr.Examples(ejemplos, inputs=[entrada, tipo], label="Ejemplos (fotos de dominio público)")

            with gr.Column(scale=2):
                comparador = gr.ImageSlider(label="Izquierda: ampliación clásica sin IA (bicúbica) · Derecha: ampliación con IA",
                                            type="pil")
                with gr.Row():
                    mapa = gr.Image(label="Mapa de lo inventado (claro = el modelo agregó o cambió detalle)", type="pil")
                    cajas = gr.Image(label="Lo que detecta RT-DETR en la versión con IA", type="pil")
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
