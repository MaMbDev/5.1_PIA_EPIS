#!/usr/bin/python3

import numpy as np
import cv2

# EPIs que se revisan: nombre y parte del cuerpo que deben cubrir
PPE_TYPES = {
    'HEAD_COVER': {'name': 'Casco', 'body_part': 'HEAD'},
    'FACE_COVER': {'name': 'Mascarilla', 'body_part': 'FACE'},
}
STATES = ['Lleva', 'Dudoso', 'Falta', 'No visible']

# Colores en BGR (formato de OpenCV)
GREEN, ORANGE, RED, WHITE = (60, 190, 60), (0, 150, 255), (40, 40, 230), (255, 255, 255)
EQUIPMENT_COLORS = {'HEAD_COVER': (0, 215, 255), 'FACE_COVER': (255, 200, 0)}


def equipment_state(person, ppe_type, min_confidence):
    """Estado de un EPI en una persona: 'Lleva', 'Dudoso', 'Falta' o 'No visible'."""
    for part in person['BodyParts']:
        # La parte del cuerpo debe verse con la confianza mínima (si la persona
        # está de espaldas, Rekognition da una "cara" con muy poca confianza)
        if (part['Name'] != PPE_TYPES[ppe_type]['body_part']
                or part['Confidence'] < min_confidence):
            continue
        # Confianza de los EPIs de este tipo que cubren la parte del cuerpo
        # (una mascarilla bajada a la barbilla se detecta, pero no cubre la cara)
        confidences = [eq['Confidence'] for eq in part['EquipmentDetections']
                       if eq['Type'] == ppe_type and eq['CoversBodyPart']['Value']]
        if not confidences:
            return 'Falta'
        return 'Lleva' if max(confidences) >= min_confidence else 'Dudoso'
    return 'No visible'


def to_pixels(box, width, height):
    """BoundingBox de Rekognition (0-1) -> (x, y, ancho, alto) en píxeles."""
    return (int(width * box['Left']), int(height * box['Top']),
            int(width * box['Width']), int(height * box['Height']))


def draw_label(img, text, x, y, color, scale, thickness):
    """Escribe un texto blanco sobre un rectángulo de color en la posición (x, y)."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y = max(y, th + 8)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y), color, -1)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thickness)


def analyze_compliance(buffer, response, equipment, min_confidence):

    # Lectura de imagen desde buffer
    img = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    thickness = max(2, width // 600)
    scale = max(0.5, width / 1800)

    counts = {PPE_TYPES[t]['name']: dict.fromkeys(STATES, 0) for t in equipment}

    for number, person in enumerate(response['Persons'], start=1):
        # Estado de cada EPI en la persona y recuento
        states = {t: equipment_state(person, t, min_confidence) for t in equipment}
        for t, state in states.items():
            counts[PPE_TYPES[t]['name']][state] += 1

        # Caja de cada EPI detectado (con su confianza si es dudoso)
        for part in person['BodyParts']:
            for eq in part['EquipmentDetections']:
                if eq['Type'] in equipment:
                    x, y, w, h = to_pixels(eq['BoundingBox'], width, height)
                    text = PPE_TYPES[eq['Type']]['name']
                    if eq['Confidence'] < min_confidence:
                        text += f" {eq['Confidence']:.0f}%?"
                    eq_color = EQUIPMENT_COLORS[eq['Type']]
                    cv2.rectangle(img, (x, y), (x + w, y + h), eq_color, thickness)
                    draw_label(img, text, x, y + h + int(30 * scale), eq_color, scale * 0.6, 1)


        #   Caja de la persona: colores definición EPIS
        problems = [f"{PPE_TYPES[t]['name']} {s.lower()}"
                    for t, s in states.items() if s != 'Lleva']
        if not problems:
            color = GREEN
        elif 'Falta' in states.values():
            color = RED
        else:
            color = ORANGE
        x, y, w, h = to_pixels(person['BoundingBox'], width, height)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        draw_label(img, f"P{number}: " + (", ".join(problems) or "OK"), x, y, color,
                   scale * 0.8, thickness)

    # Cabecera con el recuento total
    header = f"Personas: {len(response['Persons'])}" + "".join(
        f" | {name}: {c['Lleva']} si, {c['Falta']} no, {c['Dudoso']} dudoso"
        for name, c in counts.items())
    draw_label(img, header, 0, 0, (30, 30, 30), scale * 0.8, thickness)

    print("PPE compliance task has been successfully run")

    # Codifico la imagen y devuelvo la imagen anotada y contadores
    _, res_buffer = cv2.imencode('.jpg', img)
    info = {'total_persons': len(response['Persons']), 'equipment_counts': counts}
    return res_buffer.tobytes(), info
