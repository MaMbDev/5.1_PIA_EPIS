#!/usr/bin/python3

import io
import os
import base64
import requests
import gradio as gr
from PIL import Image
from dotenv import load_dotenv

# Carga de variables de entorno desde archivo .env
load_dotenv()
API_URL = os.environ.get('API_URL', 'http://127.0.0.1:5000')

# Carpeta donde se guardan los ficheros recibidos para mostrarlos y descargarlos
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tema de la interfaz
TEAL_DARK, TEAL, MINT, YELLOW = '#287A74', '#55A9A0', '#AEEED3', '#FFF8B0'
theme = gr.themes.Default(
    primary_hue=gr.themes.Color(c50='#effaf5', c100=MINT, c200='#8fdcc0', c300='#72c6b2',
                                c400=TEAL, c500='#3f918a', c600=TEAL_DARK, c700='#22665f',
                                c800='#1b524d', c900='#143e3a', c950='#0d2a27'),
).set(
    button_primary_background_fill=TEAL_DARK,
    button_primary_background_fill_hover=TEAL,
    button_primary_text_color='white',
    button_primary_background_fill_dark=TEAL,
    button_primary_background_fill_hover_dark=TEAL_DARK,
    block_label_background_fill=MINT,
    block_label_text_color=TEAL_DARK,
    block_label_background_fill_dark=TEAL_DARK,
    block_label_text_color_dark='white',
    block_title_text_color=TEAL_DARK,
    checkbox_background_color_selected=TEAL_DARK,
    slider_color=TEAL,
    color_accent=TEAL,
    color_accent_soft=YELLOW,
)


def post(path, payload):
    """Envía una petición POST con JSON a la API y devuelve la respuesta."""
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=180)
    except requests.ConnectionError:
        raise gr.Error(f"No se puede conectar con la API en {API_URL}. "
                       "¿Está arrancado el server?")
    if response.status_code != 200:
        raise gr.Error(response.json().get('error', 'Error en la API'))
    return response.json()


def save(content_b64, filename):
    """Guarda en outputs/ un fichero recibido en base64 y devuelve su ruta."""
    if not content_b64:
        return None
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'wb') as f:
        f.write(base64.b64decode(content_b64))
    return path


def analyze(image_path, equipment, min_confidence, voice, with_pdf):
    if not image_path:
        raise gr.Error("Sube primero una foto de los trabajadores.")
    if not equipment:
        raise gr.Error("Marca al menos un EPI.")

    # Convierto la foto a JPEG (ya que Rekognition admite JPEG y PNG) y la reducimos si es grande
    image = Image.open(image_path).convert('RGB')
    image.thumbnail((2048, 2048))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    filename = os.path.splitext(os.path.basename(image_path))[0] + '.jpg'

    # Subo la foto al bucket origen y luego pido el análisis
    key = post('/api/upload', {'filename': filename,
                               'image': base64.b64encode(buffer.getvalue()).decode()})['key']
    result = post('/api/analyze', {'key': key, 'equipment': equipment,
                                   'min_confidence': min_confidence,
                                   'voice': voice, 'pdf': with_pdf})

    # Guardo los ficheros recibidos y se muestran
    name = result['name']
    summary = f"### 📝 Resumen ({result['summary_source']})\n\n{result['summary']}"
    return (save(result['image'], f"result_{name}.jpg"), summary,
            save(result['pdf'], f"informe_{name}.pdf"),
            save(result['audio'], f"audio_{name}.mp3"), result['rekognition'])


with gr.Blocks(title="Detector de EPIs - AWS Rekognition") as demo:
    gr.Markdown("# Detector de EPIs en trabajadores\n"
                "Sube una foto y **Amazon Rekognition** contará cuántas personas llevan "
                "**casco** y **mascarilla**. **OpenAI** redacta el resumen y el informe PDF, "
                "y **Amazon Polly** lo lee en voz alta.")
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(label="Foto de los trabajadores", type="filepath",
                                sources=["upload"])
            equipment_in = gr.CheckboxGroup(
                choices=[('Casco', 'HEAD_COVER'), ('Mascarilla', 'FACE_COVER')],
                value=['HEAD_COVER', 'FACE_COVER'], label="EPIs a revisar")
            conf_in = gr.Slider(50, 100, value=80, step=1, label="Confianza mínima (%)")
            voice_in = gr.Checkbox(value=True, label="🔊 Generar resumen en voz (Polly)")
            pdf_in = gr.Checkbox(value=True, label="📄 Generar informe PDF (OpenAI)")
            button = gr.Button("Analizar EPIs", variant="primary")
        with gr.Column():
            image_out = gr.Image(label="Resultado")
            summary_out = gr.Markdown()
            pdf_out = gr.File(label="📄 Informe PDF (generado por OpenAI)")
            audio_out = gr.Audio(label="Resumen en voz")
    with gr.Accordion("Respuesta JSON de Rekognition", open=False):
        json_out = gr.JSON()

    button.click(analyze,
                 inputs=[image_in, equipment_in, conf_in, voice_in, pdf_in],
                 outputs=[image_out, summary_out, pdf_out, audio_out, json_out])


if __name__ == "__main__":
    demo.launch(theme=theme, footer_links=[])   # Con footer_links he ocultado el footer de gradio
