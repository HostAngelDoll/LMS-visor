# tracker.py
# Clase para gestionar el seguimiento de dedos y dibujo de estelas (trails).

import cv2
from collections import deque

class HandTracker:
    """
    Se encarga de mantener el historial de posiciones de los dedos y dibujarlos
    como un "pincel" en pantalla.
    """
    FINGER_COLORS = {
        4: (180, 90, 255),   # Pulgar
        8: (0, 180, 255),     # Índice
        12: (0, 200, 150),    # Medio
        16: (255, 120, 0),    # Anular
        20: (120, 30, 200)    # Meñique
    }

    def __init__(self, max_len=15, background_alpha=0.85):
        self.max_len = max_len
        self.background_alpha = background_alpha
        # hand_idx -> finger_idx -> deque of (x, y)
        self.histories = {}
        self.active_fingers = [] # Lista de IDs de dedos a seguir (4, 8, 12, 16, 20)

    def set_active_fingers(self, finger_ids):
        """Define qué dedos deben dejar rastro."""
        self.active_fingers = finger_ids

    def update(self, hand_landmarks, frame_shape, hand_idx=0):
        """Actualiza las posiciones de los dedos activos."""
        h, w = frame_shape[:2]

        if hand_idx not in self.histories:
            self.histories[hand_idx] = {fid: deque(maxlen=self.max_len) for fid in self.FINGER_COLORS}

        # Solo guardamos puntos para los dedos activos
        for fid in self.FINGER_COLORS:
            if fid in self.active_fingers and hand_landmarks:
                tip = hand_landmarks[fid]
                px, py = int(tip.x * w), int(tip.y * h)
                self.histories[hand_idx][fid].append((px, py))
            else:
                # Si el dedo ya no está activo, vamos vaciando su historial gradualmente
                if self.histories[hand_idx][fid]:
                    self.histories[hand_idx][fid].popleft()

    def draw_trails(self, frame):
        """Dibuja las estelas de todos los dedos seguidos en el frame."""
        overlay = frame.copy()
        for hand_idx, fingers in self.histories.items():
            for fid, pts in fingers.items():
                if len(pts) < 2:
                    continue

                color = self.FINGER_COLORS[fid]
                n = len(pts)
                for i in range(n - 1):
                    # Efecto de desvanecimiento
                    thickness = int(1 + 10 * (i / n))
                    cv2.line(overlay, pts[i], pts[i+1], color, thickness, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, self.background_alpha, frame, 1 - self.background_alpha, 0, frame)

    def clear_all(self):
        """Limpia todos los historiales."""
        self.histories.clear()
