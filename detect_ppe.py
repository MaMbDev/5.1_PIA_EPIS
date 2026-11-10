#!/usr/bin/python3

import os
import boto3
from dotenv import load_dotenv

# Carga de variables de entorno desde archivo .env
load_dotenv()

# Hago Get a las variables de entorno
accessKeyId = os.environ.get('ACCESS_KEY_ID')
secretKey = os.environ.get('ACCESS_SECRET_KEY')
bucket = os.environ.get('BUCKET_SOURCE')
region = os.environ.get('REGION')

# Creo el servicio Rekognition y asigno las credenciales
# al estar fuera de la función se crea una sola vez y se reutiliza en cada petición
rekognition_client = boto3.Session(
    aws_access_key_id=accessKeyId,
    aws_secret_access_key=secretKey,
    region_name=region).client('rekognition')


def detect_ppe(img, equipment, min_confidence):

    # Asignación de parámetros y llamada al servicio
    try:
        response = rekognition_client.detect_protective_equipment(
            Image={'S3Object': {'Bucket': bucket, 'Name': img}},
            SummarizationAttributes={'MinConfidence': min_confidence,
                                     'RequiredEquipmentTypes': equipment})

        print("La tarea de detección PPE a corrido con éxito")
    except Exception as error:
        raise Exception(f"Ha ocurrido un error inesperado detectando PPE: {error}")

    return response
