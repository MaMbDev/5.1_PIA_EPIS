#!/usr/bin/python3

import os
import json
import base64
from datetime import datetime
import boto3
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from detect_ppe import detect_ppe
from ppe_compliance import analyze_compliance
from llm_report import generate_report
from speech_synthesis import speechSynthesis

# Carga de variables de entorno desde archivo .env
load_dotenv()

# Hago Get de las variables de entorno
accessKeyId = os.environ.get('ACCESS_KEY_ID')
secretKey = os.environ.get('ACCESS_SECRET_KEY')
region = os.environ.get('REGION')
bucket_source = os.environ.get('BUCKET_SOURCE')
bucket_dest = os.environ.get('BUCKET_DEST')

# Creación del servidor Flask
application = Flask(__name__)

# Creación de los servicios S3 y asignación de credenciales
s3 = boto3.Session(
    aws_access_key_id=accessKeyId,
    aws_secret_access_key=secretKey,
    region_name=region).resource('s3')


def b64(content):
    """Bytes -> texto base64, para poder enviar ficheros dentro de un JSON."""
    return base64.b64encode(content).decode() if content else None


# api/upload endpoint
@application.route('/api/upload', methods=['POST'])
def uploadImage():
    data = request.get_json()
    if not data or not data.get('image'):
        abort(400, description="Falta el campo 'image'")

    # Nombre único: fecha y hora + nombre original limpio
    key = f"{datetime.now():%Y%m%d_%H%M%S}_{secure_filename(data.get('filename', 'foto.jpg'))}"
    try:
        s3.Object(bucket_source, key).put(Body=base64.b64decode(data['image']))
    except Exception as error:
        print(error)
        abort(500, description=str(error))
    return jsonify({'key': key})


# api/analyze endpoint
@application.route('/api/analyze', methods=['POST'])
def analyzeImage():
    data = request.get_json()
    if not data or not data.get('key'):
        abort(400, description="Falta el campo 'key'")
    key = data['key']
    equipment = data.get('equipment', ['HEAD_COVER', 'FACE_COVER'])
    min_confidence = float(data.get('min_confidence', 80))

    try:
        # Rekognition detecta las personas y sus EPIs
        response = detect_ppe(key, equipment, min_confidence)

        # Recuento de EPIs e imagen anotada
        fileContent = s3.Object(bucket_source, key).get()['Body'].read()
        result_img, info = analyze_compliance(fileContent, response, equipment, min_confidence)

        # Resumen y, si se pide, informe PDF con OpenAI
        summary, pdf, summary_source = generate_report(info, data.get('pdf', True))

        # Resumen en voz con Amazon Polly
        audio = speechSynthesis(summary) if data.get('voice', True) else None

        # Guardo los resultados en el bucket destino
        name = os.path.splitext(key)[0]
        s3.Object(bucket_dest, f"result_{name}.jpg").put(Body=result_img)
        s3.Object(bucket_dest, f"result_{name}.json").put(
            Body=json.dumps({**info, 'summary': summary}, ensure_ascii=False))
        if pdf:
            s3.Object(bucket_dest, f"informe_{name}.pdf").put(Body=pdf)
        if audio:
            s3.Object(bucket_dest, f"audio_{name}.mp3").put(Body=audio)
    except Exception as error:
        print(error)
        abort(500, description=str(error))

    return jsonify({
        'name': name,
        'summary': summary,
        'summary_source': summary_source,
        'image': b64(result_img),
        'pdf': b64(pdf),
        'audio': b64(audio),
        'rekognition': {k: v for k, v in response.items() if k != 'ResponseMetadata'},
    })


# Errores como JSON, así la interfaz puede mostrar el mensaje.
@application.errorhandler(400)
@application.errorhandler(500)
def handle_error(error):
    return jsonify({'error': error.description}), error.code


# Corro la app
if __name__ == "__main__":
    application.debug = True
    application.run(port=int(os.environ.get('PORT', 5000)))
