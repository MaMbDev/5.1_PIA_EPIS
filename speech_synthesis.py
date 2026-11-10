#!/usr/bin/python3

import os
import boto3
from dotenv import load_dotenv

# Carga de variables de entorno desde archivo .env
load_dotenv()

# Get env variables
accessKeyId = os.environ.get('ACCESS_KEY_ID')
secretKey = os.environ.get('ACCESS_SECRET_KEY')
region = os.environ.get('REGION')
voiceId = os.environ.get('POLLY_VOICE', 'Lucia')

# Hago Get de las variables de entorno
polly_client = boto3.Session(
    aws_access_key_id=accessKeyId,
    aws_secret_access_key=secretKey,
    region_name=region).client('polly')


def speechSynthesis(text):
    # Creación de parámetro y llamada al servicio.
    try:
        response = polly_client.synthesize_speech(
            VoiceId=voiceId,
            Engine='neural',
            OutputFormat='mp3',
            Text=text,
            TextType='text')

        print("Speech synthesis task has been successfully run")
    except Exception as error:
        raise Exception(f"An unexpected error was raised synthesizing speech: {error}")

    # synthesize_speech devuelve el audio como un flujo de datos para leerse entero.
    return response['AudioStream'].read()
