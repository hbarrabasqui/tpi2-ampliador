"""Ampliador honesto de fotos chicas.

Ejecutar con:  uv run app.py
En Google Colab se abre además un link público (share=True) automáticamente.

Todos los textos de la interfaz están escritos para una persona sin
conocimientos técnicos. Los nombres técnicos (difusión, bicúbica, RT-DETR)
se explican en el README, no en la app.
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

# ---------------------------------------------------------------------------
# Textos de la interfaz
# ---------------------------------------------------------------------------

PRESENTACION = """
# Agrandá una foto chica con inteligencia artificial

**¿Para qué sirve?** Si tenés una foto chica o de poca calidad (una foto vieja escaneada, una que te mandaron
por WhatsApp), esta herramienta la **agranda 4 veces** y la deja más nítida usando inteligencia artificial (IA).

**¿Cómo se usa?**
1. Subí tu foto en el recuadro de la izquierda (funciona mejor con fotos chicas), o tocá una de las **fotos de ejemplo** que están abajo.
2. Indicá qué se ve en la foto.
3. Tocá **Agrandar** y esperá unos segundos. Los resultados aparecen a la derecha.
"""

AVISO_PRINCIPAL = """
> **Antes de empezar, algo importante.** Cuando una foto es chica, los detalles finos *no están*: se perdieron.
> La IA no los recupera: **los inventa**, para que la foto se vea bien. A veces acierta y a veces no
> (por ejemplo, puede cambiar un poco una cara). Por eso, además de la foto agrandada, la herramienta
> te muestra **dónde inventó** y **te avisa si cambió algo importante**.
"""

TITULO_COMPARADOR = """
### 1. Compará: sin IA y con IA
Arrastrá la **línea naranja** hacia los costados.
- **Lado izquierdo:** tu foto agrandada *sin IA*, de la forma común (como cuando la estirás en cualquier programa). Se ve más borrosa, pero no agrega nada que no estuviera.
- **Lado derecho:** tu foto agrandada *con IA*. Se ve más nítida, pero **esa nitidez extra la dibujó la IA**.
"""

TITULO_MAPA = """
### 2. ¿Dónde inventó la IA?
Este dibujo marca las diferencias entre las dos versiones de arriba.
**Lo oscuro** quedó casi igual a tu foto. **Lo claro** es donde la IA dibujó detalles por su cuenta: ahí conviene mirar con más desconfianza.
"""

TITULO_DESCARGA = """
### 4. Descargá tu foto
La foto descargada lleva abajo la leyenda *"Ampliada con IA"*, para que nadie la confunda con la original.
"""

TITULO_TECNICO = """
Para chequear el trabajo de la IA que agranda, la herramienta usa **una segunda IA** (se llama *RT-DETR*)
que sabe reconocer personas y objetos. Mira la foto agrandada **sin IA** y la agrandada **con IA**, y compara lo que reconoce en cada una.
Si reconoce cosas distintas, es una señal de que la primera IA cambió algo.

Esta segunda IA también se equivoca: a veces confunde cosas (por ejemplo, puede tomar una cara modificada por otra cosa).
Abajo se ve, con recuadros verdes, lo que reconoció en la versión con IA.
"""

CONSEJOS_POR_TIPO = {
    "Personas o retratos": (
        "Las **caras** las rearmó la IA: pueden verse más jóvenes, distintas o incluso parecerse a otra persona. "
        "Compará con otras fotos de esa persona antes de compartirla."
    ),
    "Edificios o paisajes": (
        "Las ventanas, ladrillos, hojas y texturas finas las **dibujó la IA**. "
        "La forma general viene de tu foto; los detalles, no."
    ),
    "Objetos u otra cosa": "Los detalles finos (texturas, bordes nítidos) los **dibujó la IA**.",
}

# ---------------------------------------------------------------------------
# Procesamiento
# ---------------------------------------------------------------------------


def _abrir_imagen(ruta: str) -> Image.Image:
    """Abre la imagen con mensajes de error comprensibles."""
    if ruta is None:
        raise gr.Error("Primero subí una foto o tocá una de las fotos de ejemplo.")
    tamano_mb = os.path.getsize(ruta) / (1024 * 1024)
    if tamano_mb > TAMANO_MAXIMO_MB:
        raise gr.Error(f"La foto pesa {tamano_mb:.0f} MB. Probá con una de menos de {TAMANO_MAXIMO_MB} MB.")
    try:
        return Image.open(ruta).convert("RGB")
    except UnidentifiedImageError as error:
        raise gr.Error("No pude abrir ese archivo como foto. Probá con una foto en formato JPG o PNG.") from error


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
        raise gr.Error("No pude agrandar la foto. Probá con una foto más chica.") from error
    segundos = time.time() - inicio

    progreso(0.75, desc="Buscando dónde inventó la IA...")
    mapa, fraccion_cambiada = honesty.mapa_de_diferencias(con_ia, sin_ia)
    diferencia_coherencia = honesty.coherencia(entrada, con_ia)

    progreso(0.85, desc="Una segunda IA está revisando el resultado...")
    deteccion = analysis.comparar(sin_ia, con_ia)

    ruta_salida = os.path.join(tempfile.mkdtemp(), "foto_ampliada_con_IA.png")
    honesty.exportar_con_aviso(con_ia, ruta_salida)

    resumen = _armar_resumen(entrada, con_ia, fue_achicada, fraccion_cambiada,
                             diferencia_coherencia, deteccion, tipo_de_foto, segundos)
    detalle_tecnico = (f"- En la foto agrandada **sin IA** reconoció: {deteccion['texto_sin_ia']}.\n"
                       f"- En la foto agrandada **con IA** reconoció: {deteccion['texto_con_ia']}.")
    return (sin_ia, con_ia), mapa, resumen, ruta_salida, deteccion["imagen_con_cajas"], detalle_tecnico


def _texto_chequeo(deteccion: dict) -> str:
    """Resultado del chequeo con la segunda IA, en palabras simples y sin nombres de objetos raros."""
    antes, despues = deteccion["personas_sin_ia"], deteccion["personas_con_ia"]
    if antes != despues:
        if despues < antes:
            ahora = "ya no reconoce a ninguna persona" if despues == 0 else f"reconoce solo {_personas(despues)}"
            return (f"**⚠ Atención: la IA cambió mucho a las personas.** En tu foto agrandada sin IA se reconoce "
                    f"{_personas(antes)}; en la versión con IA, la segunda IA {ahora}. "
                    "Eso quiere decir que la cara o el cuerpo quedaron tan cambiados que ya no parecen una persona "
                    "para la computadora. **Mirá bien las caras antes de compartir la foto.**")
        return (f"**⚠ Atención:** en la versión con IA se reconocen más personas ({_personas(despues)}) que en tu "
                f"foto ({_personas(antes)}). La IA pudo haber dibujado algo que parece una persona. Revisá la foto.")
    if deteccion["cambios"]:
        return ("**⚠ Atención:** la segunda IA ve cosas distintas en las dos versiones. La IA que agrandó la foto "
                "pudo haber agregado o borrado algún detalle. Revisá la foto con cuidado.")
    if antes:
        return (f"En las dos versiones se reconoce {_personas(antes)}. Es una buena señal, pero no garantiza "
                "que las caras sean idénticas a las reales.")
    return "La segunda IA no encontró diferencias importantes entre las dos versiones."


def _personas(cantidad: int) -> str:
    if cantidad == 0:
        return "ninguna persona"
    return "1 persona" if cantidad == 1 else f"{cantidad} personas"


def _texto_coherencia(diferencia: float) -> str:
    if diferencia < 6:
        return "Los colores y las formas generales son los de tu foto."
    if diferencia < 15:
        return "La IA cambió un poco algunos colores o formas de tu foto."
    return "La IA se alejó bastante de tu foto: revisala con cuidado."


def _armar_resumen(entrada, con_ia, fue_achicada, fraccion, coherencia, deteccion, tipo, segundos):
    lineas = ["### 3. Resumen y consejos", "**Qué pasó con tu foto**"]
    if fue_achicada:
        lineas.append("- Tu foto era grande y primero la achiqué un poco, porque la IA trabaja mejor con fotos chicas.")
    lineas.append(f"- Ahora es **4 veces más grande** (pasó de {entrada.width} × {entrada.height} "
                  f"a {con_ia.width} × {con_ia.height} puntos). Tardó {segundos:.0f} segundos.")
    lineas.append(f"- La IA dibujó detalles nuevos en más o menos el **{fraccion:.0%}** de la foto "
                  "(las zonas claras del punto 2).")
    lineas.append(f"- {_texto_coherencia(coherencia)}")
    lineas.append("\n**Chequeo automático** (una segunda IA revisa si siguen estando las mismas personas)")
    lineas.append(f"- {_texto_chequeo(deteccion)}")
    lineas.append("\n**Antes de usar la foto**")
    lineas.append(f"- {CONSEJOS_POR_TIPO.get(tipo, CONSEJOS_POR_TIPO['Objetos u otra cosa'])}")
    lineas.append("- Si hay **letras o números**, la IA puede inventarlos o deformarlos.")
    lineas.append("- Usala para mirar o compartir, **no como prueba** de cómo era algo o alguien.")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------


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
    with gr.Blocks(title="Agrandá una foto chica con IA") as demo:
        gr.Markdown(PRESENTACION)
        gr.Markdown(AVISO_PRINCIPAL)

        with gr.Row():
            with gr.Column(scale=1):
                entrada = gr.Image(type="filepath", label="Tu foto")
                tipo = gr.Radio(list(diffusion.PROMPTS), value="Personas o retratos",
                                label="¿Qué se ve en la foto?",
                                info="Ayuda a la IA y nos permite darte mejores consejos.")
                libertad = gr.Slider(0, 100, value=20, step=5,
                                     label="¿Cuánto puede inventar la IA?",
                                     info="Poco: se parece más a tu foto. Mucho: más nítida, pero inventa más y se equivoca más. "
                                          "Si no sabés, dejalo como está.")
                with gr.Accordion("Opciones avanzadas", open=False):
                    semilla = gr.Number(value=0, precision=0, label="Variante",
                                        info="Cambiá el número para que la IA invente otra versión de la misma foto.")
                boton = gr.Button("Agrandar", variant="primary")
                ejemplos = _ejemplos()
                if ejemplos:
                    gr.Examples(ejemplos, inputs=[entrada, tipo],
                                label="Fotos de ejemplo (fotos antiguas de uso libre): tocá una y después Agrandar")

            with gr.Column(scale=2):
                gr.Markdown(TITULO_COMPARADOR)
                comparador = gr.ImageSlider(label="Izquierda: sin IA · Derecha: con IA", type="pil")
                gr.Markdown(TITULO_MAPA)
                mapa = gr.Image(label="Lo claro es lo que inventó la IA", type="pil")
                resumen = gr.Markdown()
                gr.Markdown(TITULO_DESCARGA)
                descarga = gr.File(label="Tu foto agrandada")
                with gr.Accordion("Detalles técnicos: cómo se hace el chequeo automático", open=False):
                    gr.Markdown(TITULO_TECNICO)
                    cajas = gr.Image(label="Lo que reconoció la segunda IA en la foto agrandada con IA", type="pil")
                    detalle = gr.Markdown()

        boton.click(procesar, [entrada, tipo, libertad, semilla],
                    [comparador, mapa, resumen, descarga, cajas, detalle])
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
