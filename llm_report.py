#!/usr/bin/python3

import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

# Carga de variables de entorno desde archivo .env
load_dotenv()

# Hago Get de las variables de entorno
apiKey = os.environ.get('OPENAI_API_KEY')
model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')

# Creo el cliente de OpenAI
openai_client = OpenAI(api_key=apiKey) if apiKey else None

INSTRUCTIONS = """
Eres el técnico de prevención de riesgos laborales de una obra. Recibes en JSON el
resultado de analizar una foto con Amazon Rekognition: el número de personas y, para
el casco y la mascarilla, cuántas personas lo llevan ("Lleva"), cuántas no ("Falta"),
en cuántas es dudoso porque se detecta con poca confianza ("Dudoso", no es un
incumplimiento confirmado) y de cuántas no se ve esa parte del cuerpo ("No visible").
{pdf_task}
Responde SOLO con un resumen breve y natural en español (máximo 50 palabras), que se
leerá en voz alta. El campo "hechos" contiene los datos ya calculados: reescríbelos como
texto fluido, con una frase por EPI, incluyendo TODAS sus cifras y sin añadir otras.
- Escribe los números con letras y no repitas la misma idea con otras palabras.
- Di "todas" o "ninguna" solo si la cifra es igual al número total de personas.
- Si solo hay una persona, habla en singular.
Ejemplos de buen estilo:
"Se ha detectado una persona. Lleva casco, pero no lleva mascarilla."
"Se han detectado ocho personas. Cinco llevan casco, uno es dudoso y en dos no se ve
la cabeza. Tres no llevan mascarilla y en cinco no se ve la cara."
No copies el campo "conclusion" (es solo para el PDF). Sin markdown, listas, emojis ni
enlaces, y sin mencionar ningún PDF.
"""

PDF_TASK = """
Antes de responder, usa la herramienta de Python para crear /mnt/data/informe_epis.pdf
con la librería fpdf (fuente Helvetica, solo caracteres latinos, sin emojis):
- Título "Informe de detección de casco y mascarilla" y la fecha y hora de hoy.
- Número de personas detectadas.
- Una tabla con una fila por EPI y las columnas "EPI", "Lleva", "Falta", "Dudoso" y
  "No visible" (anchos 40, 36, 36, 36 y 36 mm).
- Debajo, el texto del campo "conclusion" copiado tal cual.
"""


def people(n):
    """'1 persona', '3 personas'."""
    return f"{n} persona" if n == 1 else f"{n} personas"


def conclusion(info):
    """
    Frase final del PDF, calculada aquí para que los números sean siempre exactos.
    "Conclusión: 1 persona sin mascarilla. Casos dudosos que conviene revisar: 2 personas con casco dudoso."
    """
    counts = info['equipment_counts'].items()
    missing = [f"{people(c['Falta'])} sin {epi.lower()}" for epi, c in counts if c['Falta']]
    doubtful = [f"{people(c['Dudoso'])} con {epi.lower()} dudoso" for epi, c in counts
                if c['Dudoso']]
    text = "Conclusión: " + (" y ".join(missing) or "ningún incumplimiento confirmado") + "."
    if doubtful:
        text += f" Casos dudosos que conviene revisar: {' y '.join(doubtful)}."
    return text


def local_summary(info):
    """Resumen sin IA, por si OpenAI no está disponible (una frase por EPI, sin ceros)."""
    total = info['total_persons']
    text = "Se ha detectado una persona." if total == 1 else f"Se han detectado {total} personas."
    for epi, c in info['equipment_counts'].items():
        # (singular, plural) de cada estado: "1 lo lleva", "3 lo llevan"
        labels = {'Lleva': ('lo lleva', 'lo llevan'), 'Falta': ('no lo lleva', 'no lo llevan'),
                  'Dudoso': ('dudoso', 'dudosos'), 'No visible': ('no se ve', 'no se ven')}
        parts = [f"{c[s]} {labels[s][c[s] > 1]}" for s in labels if c[s]]
        text += f" {epi}: {', '.join(parts)}."
    return text


def download_pdf(response):
    """Busca el PDF que ha creado el modelo en su contenedor y lo descarga."""
    for item in response.output:
        if item.type == 'code_interpreter_call':
            for file in openai_client.containers.files.list(container_id=item.container_id):
                if file.path.endswith('.pdf'):
                    return openai_client.containers.files.content.retrieve(
                        file.id, container_id=item.container_id).read()
    return None


def clean_summary(text):
    """Quita enlaces y frases sobre el PDF que a veces añade el modelo (se lee en voz alta)."""
    text = re.sub(r'\[[^\]]*\]\([^)]*\)', '', text).replace('\n', ' ')
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return ' '.join(s for s in sentences
                    if not re.search(r'pdf|informe|archivo|aquí tienes|conclusión', s, re.I)).strip()


def generate_report(info, with_pdf):
    if openai_client is None:
        return local_summary(info), None, "local (no hay OPENAI_API_KEY)"

    # El recuento (para la tabla del PDF), la conclusión del PDF y los hechos del resumen
    # se calculan aquí para que los números sean siempre exactos
    data = json.dumps({**info, 'conclusion': conclusion(info), 'hechos': local_summary(info)},
                      ensure_ascii=False)

    # Creación de parámetro y llamada al servicio
    try:
        response = openai_client.responses.create(
            model=model,
            instructions=INSTRUCTIONS.format(pdf_task=PDF_TASK if with_pdf else ''),
            tools=[{'type': 'code_interpreter', 'container': {'type': 'auto'}}]
            if with_pdf else [],
            input=data)
        summary = clean_summary(response.output_text) or local_summary(info)
        pdf = download_pdf(response) if with_pdf else None

        print("La tarea de reporte LLM ha concluido con éxito")
    except Exception as error:
        print(error)
        return local_summary(info), None, f"local (OpenAI no disponible: {error})"

    return summary, pdf, f"OpenAI {response.model}"
