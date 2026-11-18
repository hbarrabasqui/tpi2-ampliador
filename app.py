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

TITULO_CHEQUEO = """
### 3. Chequeo con una segunda IA
Para controlar el trabajo de la IA que agrandó tu foto, usamos **otra IA distinta**. Esta no agranda ni cambia nada:
solo **reconoce personas y objetos** y les dibuja un **recuadro verde con su nombre**.

Le mostramos las dos versiones de tu foto. Si en las dos reconoce lo mismo, es una buena señal.
Si reconoce cosas distintas, quiere decir que la IA que agrandó la foto **cambió algo importante**.
Ojo: esta segunda IA también puede equivocarse.
"""

TITULO_DESCARGA = """
### 5. Descargá tu foto
La foto descargada lleva abajo la leyenda *"Ampliada con IA"*, para que nadie la confunda con la original.
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
    return ((sin_ia, con_ia), mapa, deteccion["imagen_sin_ia_cajas"], deteccion["imagen_con_cajas"],
            _texto_chequeo(deteccion), resumen, ruta_salida)


def _texto_chequeo(deteccion: dict) -> str:
    """Conclusión del chequeo con la segunda IA, en palabras simples."""
    antes, despues = deteccion["personas_sin_ia"], deteccion["personas_con_ia"]
    lineas = [f"- En la foto agrandada **sin IA** reconoce: **{deteccion['texto_sin_ia']}**.",
              f"- En la foto agrandada **con IA** reconoce: **{deteccion['texto_con_ia']}**."]
    if despues < antes:
        confusion = ""
        if deteccion["aparecidos"]:
            confusion = (f" En su lugar ve: {', '.join(deteccion['aparecidos'])}. **Eso no está en tu foto:** "
                         "la segunda IA se confundió porque la cara o el cuerpo quedaron muy cambiados.")
        lineas.append(f"\n**⚠ Atención: la IA cambió mucho a las personas.** En la versión con IA, la segunda IA "
                      f"ya no reconoce {'a la persona' if antes - despues == 1 else f'a {antes - despues} de las personas'} "
                      f"que sí reconoce en la versión sin IA.{confusion} "
                      "**Mirá bien las caras antes de compartir la foto.**")
    elif despues > antes:
        lineas.append("\n**⚠ Atención:** en la versión con IA reconoce más personas que en tu foto. "
                      "La IA pudo haber dibujado algo que parece una persona. Revisá la foto.")
    elif deteccion["cambios"]:
        lineas.append(f"\n**⚠ Atención:** en la versión con IA reconoce cosas que en tu foto no "
                      f"({', '.join(deteccion['aparecidos']) or 'o deja de reconocer algunas'}). "
                      "La IA que agrandó la foto pudo haber agregado o cambiado detalles, o la segunda IA se confundió. "
                      "Revisá esa parte de la foto.")
    else:
        lineas.append("\nReconoce lo mismo en las dos versiones: buena señal. Igual, eso no garantiza que "
                      "los detalles (una cara, una letra) sean los verdaderos.")
    return "\n".join(lineas)


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
    lineas = ["### 4. Resumen y consejos", "**Qué pasó con tu foto**"]
    if fue_achicada:
        lineas.append("- Tu foto era grande y primero la achiqué un poco, porque la IA trabaja mejor con fotos chicas.")
    lineas.append(f"- Ahora es **4 veces más grande** (pasó de {entrada.width} × {entrada.height} "
                  f"a {con_ia.width} × {con_ia.height} puntos). Tardó {segundos:.0f} segundos.")
    lineas.append(f"- La IA dibujó detalles nuevos en más o menos el **{fraccion:.0%}** de la foto "
                  "(las zonas claras del punto 2).")
    lineas.append(f"- {_texto_coherencia(coherencia)}")
    if deteccion["cambios"]:
        lineas.append("- **La segunda IA encontró diferencias** entre las dos versiones (ver el punto 3).")
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
                gr.Markdown(TITULO_CHEQUEO)
                with gr.Row():
                    cajas_sin_ia = gr.Image(label="Lo que reconoce en la foto agrandada SIN IA", type="pil")
                    cajas_con_ia = gr.Image(label="Lo que reconoce en la foto agrandada CON IA", type="pil")
                chequeo = gr.Markdown()
                resumen = gr.Markdown()
                gr.Markdown(TITULO_DESCARGA)
                descarga = gr.File(label="Tu foto agrandada")

        boton.click(procesar, [entrada, tipo, libertad, semilla],
                    [comparador, mapa, cajas_sin_ia, cajas_con_ia, chequeo, resumen, descarga])
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
