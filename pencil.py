#!py -3.10
# main.py
# Versión: DepthAI + MediaPipe HandLandmarker
# + trails tipo "pincel" para dedos controlados manualmente con keyboard (F1..F5)

import os
import cv2
import depthai as dai
from collections import deque
import math
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions
)
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python import BaseOptions
import threading
import keyboard  # pip install keyboard

# --- CONFIG ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
NUM_HANDS = 2
MIN_DET_CONF = 0.7
MIN_TRACK_CONF = 0.7

# trail config
TRAIL_MAX_LEN = 10               # cuantos puntos mantener (~frames)
MIN_MOVEMENT = 0.006             # umbral en coords normalizadas para filtrar jitter
BACKGROUND_ALPHA = 0.85          # mezcla del overlay con la imagen (0..1)

# finger mapping (orden solicitado por el usuario: 1..5)
# 1 - meñique (tip 20), 2 - anular (16), 3 - medio (12), 4 - indice (8), 5 - pulgar (4)
FINGER_TIP_LM = [20, 16, 12, 8, 4]
FINGER_NAMES = ["Menique", "Anular", "Medio", "Indice", "Pulgar"]

# colores BGR para cada dedo (puedes ajustar)
FINGER_COLORS = [
    (120, 30, 200),   # meñique
    (255, 120, 0),    # anular
    (0, 200, 150),    # medio
    (0, 180, 255),    # indice
    (180, 90, 255)    # pulgar
]

# hand connections for drawing (opcional)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# --- helpers de dibujo y matemática ---
def pixel_from_norm_xy(norm_x, norm_y, frame_shape):
    h, w = frame_shape[0], frame_shape[1]
    return int(norm_x * w), int(norm_y * h)

def dist_norm_xy(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def draw_connections(frame, hand_landmarks):
    for start, end in HAND_CONNECTIONS:
        x1 = int(hand_landmarks[start].x * frame.shape[1])
        y1 = int(hand_landmarks[start].y * frame.shape[0])
        x2 = int(hand_landmarks[end].x * frame.shape[1])
        y2 = int(hand_landmarks[end].y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def draw_landmarks(frame, hand_landmarks):
    for lm in hand_landmarks:
        x = int(lm.x * frame.shape[1])
        y = int(lm.y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

def draw_trail(frame, pts, base_color):
    """
    Dibuja rastro tipo pincel sobre `frame` usando la lista de puntos pts (en píxels).
    pts: list de (x,y) con pts[0] el más antiguo y pts[-1] el más reciente.
    """
    if len(pts) < 2:
        return
    overlay = frame.copy()
    n = len(pts)
    for i in range(n-1):
        rel = (i / (n-2)) if n > 2 else 1.0
        recency = 1.0 - rel
        alpha = 0.08 + 0.92 * recency    # 0.08 .. 1.0
        thickness = int(1 + 12 * recency)  # 1 .. 13 px
        color = (
            int(base_color[0] * (0.6 + 0.4 * recency)),
            int(base_color[1] * (0.6 + 0.4 * recency)),
            int(base_color[2] * (0.6 + 0.4 * recency)),
        )
        cv2.line(overlay, pts[i], pts[i+1], color, thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, BACKGROUND_ALPHA, frame, 1 - BACKGROUND_ALPHA, 0, frame)

# --- estado de historiales ---
histories = [
    [deque(maxlen=TRAIL_MAX_LEN) for _ in range(len(FINGER_TIP_LM))]
    for _ in range(NUM_HANDS)
]

# para filtrar jitter: almacenar última coordenada normalizada por mano y dedo
last_norm = [
    [None for _ in range(len(FINGER_TIP_LM))]
    for _ in range(NUM_HANDS)
]

# --- MANUAL FOLLOW FLAGS (F1..F5) ---
# Si manual_follow[i] == True, seguiremos ese dedo (para todas las manos).
manual_follow = [False for _ in range(len(FINGER_TIP_LM))]

# Lock para sincronizar accesos entre hilos (keyboard callback / mediapipe callback)
state_lock = threading.Lock()

# exit flag
exit_flag = threading.Event()

# --- detección de dedo levantado (DESACTIVADA, mantenida como util) ---
THUMB_EXTENDED_THRESHOLD = 0.23  # distancia normalizada (ajustable)

def hand_bbox_and_palm_center(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_size = max(max_x - min_x, max_y - min_y)
    mcp_idxs = [5, 9, 13, 17]
    cx = sum(landmarks[i].x for i in mcp_idxs) / len(mcp_idxs)
    cy = sum(landmarks[i].y for i in mcp_idxs) / len(mcp_idxs)
    return bbox_size, (cx, cy)

def is_finger_up(landmarks, finger_index):
    tip_idx = FINGER_TIP_LM[finger_index]
    if tip_idx >= len(landmarks):
        return False
    if finger_index != 4:
        pip_idx = tip_idx - 2
        if pip_idx < 0 or pip_idx >= len(landmarks):
            return False
        return landmarks[tip_idx].y < landmarks[pip_idx].y
    else:
        bbox_size, palm_center = hand_bbox_and_palm_center(landmarks)
        tip = (landmarks[tip_idx].x, landmarks[tip_idx].y)
        d = dist_norm_xy(tip, palm_center)
        if bbox_size <= 0:
            return False
        normalized = d / bbox_size
        return normalized > THUMB_EXTENDED_THRESHOLD

# --- mediapipe callback ---
frame_for_display = None

def callback(result, output_image, timestamp_ms):
    global frame_for_display
    try:
        img_rgb = output_image.numpy_view().copy()
        h, w = img_rgb.shape[0], img_rgb.shape[1]

        present_hands = 0
        if result.hand_landmarks:
            present_hands = min(len(result.hand_landmarks), NUM_HANDS)

            # copia local de manual_follow protegida por lock
            with state_lock:
                local_manual = manual_follow.copy()

            for hand_index in range(present_hands):
                hand_landmarks = result.hand_landmarks[hand_index]

                for finger_idx, lm_index in enumerate(FINGER_TIP_LM):
                    if not local_manual[finger_idx]:
                        # si no está en modo manual para ese dedo, limpiar historial
                        # protección de estado
                        with state_lock:
                            histories[hand_index][finger_idx].clear()
                            last_norm[hand_index][finger_idx] = None
                        continue

                    # si está activado manualmente, añadimos punto si se mueve suficiente
                    tip = hand_landmarks[lm_index]
                    norm_pt = (tip.x, tip.y)
                    with state_lock:
                        prev = last_norm[hand_index][finger_idx]
                        if prev is None or dist_norm_xy(prev, norm_pt) >= MIN_MOVEMENT:
                            last_norm[hand_index][finger_idx] = norm_pt
                            px = int(tip.x * w)
                            py = int(tip.y * h)
                            histories[hand_index][finger_idx].append((px, py, timestamp_ms))

                # opcional: dibujar conexiones y landmarks sobre img_rgb
                draw_connections(img_rgb, hand_landmarks)
                draw_landmarks(img_rgb, hand_landmarks)

        # Si hay menos manos detectadas que NUM_HANDS, limpiar historiales de manos faltantes
        for missing_hand in range(present_hands, NUM_HANDS):
            with state_lock:
                for fi in range(len(FINGER_TIP_LM)):
                    histories[missing_hand][fi].clear()
                    last_norm[missing_hand][fi] = None

        # convertir a BGR para mostrar y dibujar trails
        frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # dibujar trails para todas las manos/dedos (si hay puntos)
        with state_lock:
            # hacemos copia ligera para iterar sin mantener lock demasiado tiempo
            copy_histories = [
                [list(histories[h][f]) for f in range(len(FINGER_TIP_LM))]
                for h in range(NUM_HANDS)
            ]
        for hand_index in range(NUM_HANDS):
            for finger_idx in range(len(FINGER_TIP_LM)):
                pts = [(p[0], p[1]) for p in copy_histories[hand_index][finger_idx]]
                if pts:
                    draw_trail(frame_bgr, pts, FINGER_COLORS[finger_idx])

        frame_for_display = frame_bgr

    except Exception:
        frame_for_display = None

# --- keyboard callbacks ---
def toggle_finger(fi):
    with state_lock:
        manual_follow[fi] = not manual_follow[fi]
        if not manual_follow[fi]:
            for h in range(NUM_HANDS):
                histories[h][fi].clear()
                last_norm[h][fi] = None
    print(f"Toggled {FINGER_NAMES[fi]} -> {'ON' if manual_follow[fi] else 'OFF'}")

def on_esc(event):
    exit_flag.set()
    print("Exit flag set via ESC")

# registrar teclas F1..F5 y Esc (no bloqueante)
keyboard.on_press_key('f1', lambda e: toggle_finger(0))
keyboard.on_press_key('f2', lambda e: toggle_finger(1))
keyboard.on_press_key('f3', lambda e: toggle_finger(2))
keyboard.on_press_key('f4', lambda e: toggle_finger(3))
keyboard.on_press_key('f5', lambda e: toggle_finger(4))
keyboard.on_press_key('esc', on_esc)

# --- util: dibujar estado ON/OFF en pantalla (manual flags) ---
def draw_toggle_status(frame):
    x0, y0 = 10, 20
    with state_lock:
        mf = manual_follow.copy()
    for i, name in enumerate(FINGER_NAMES):
        on = mf[i]
        status = "ON" if on else "OFF"
        color = (0, 200, 0) if on else (0, 0, 200)
        text = f"F{i+1} - {name}: {status}"
        cv2.putText(frame, text, (x0, y0 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "Presiona F1..F5 para toggle. Esc o Q para salir.", (10, y0 + (len(FINGER_NAMES)+1)*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

# --- main loop ---
def main():
    global frame_for_display

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=callback,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF
    )
    hand_landmarker = HandLandmarker.create_from_options(options)

    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        frame_id = 0

        while True:
            # salir si exit_flag fue activado por ESC
            if exit_flag.is_set():
                break

            in_rgb = q.get()
            frame_bgr = in_rgb.getCvFrame()
            frame_bgr = cv2.flip(frame_bgr, 1)  # espejo horizontal
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

            # envío asíncrono para detección
            hand_landmarker.detect_async(mp_image, timestamp_ms=int(frame_id * 33))

            # elegir frame a mostrar (callback actualiza frame_for_display)
            if frame_for_display is not None:
                display = frame_for_display.copy()
            else:
                display = frame_bgr.copy()

            # Dibujar estado manual ON/OFF
            draw_toggle_status(display)

            cv2.imshow("Hand Detection - Manual Finger Trails (F1..F5)", display)
            frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            # Q para salir desde la ventana
            if key == ord('q'):
                break

    # cleanup keyboard hooks antes de salir
    keyboard.unhook_all()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
