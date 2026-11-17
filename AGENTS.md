# Instrucciones para agentes de IA (Claude Code, Codex, etc.)

## Proyecto
Ampliador 4x de fotos chicas con un modelo de difusión, que muestra qué detalles inventó la IA.
Trabajo Integrador 2 — Procesamiento Digital de Imágenes, IFTS 24 (2026).

## Stack
- Python 3.12, gestionado con `uv` (no usar pip ni conda).
- Interfaz: Gradio (`app.py`).
- Difusión: `stabilityai/stable-diffusion-x4-upscaler` con `diffusers` (`src/diffusion.py`).
- Análisis visual: RT-DETR `PekingU/rtdetr_r50vd` con `transformers` (`src/analysis.py`).
- Honestidad: bicúbica, mapa de diferencias, coherencia y exportación con aviso (`src/honesty.py`).
- `torch==2.10.0` fijado a propósito: es la última versión de PyPI compilada para CUDA 12.8, compatible con los drivers de Colab.

## Comandos
- Instalar: `uv sync`
- Ejecutar: `uv run app.py` (en Colab abre un link público solo)
- Verificar sin descargar modelos: `uv run python tests/verificar.py`

## Reglas
- Explicar los cambios en palabras simples antes de hacerlos; el autor tiene que poder defender cada línea.
- Cambios chicos, un commit por cambio, mensajes en español.
- Nunca decir que el modelo "recupera" o "restaura" detalles: los **genera**. Todo texto visible tiene que respetar eso.
- No subir tokens, `.env`, ni cachés de modelos.
- Correr `tests/verificar.py` antes de dar algo por terminado.
