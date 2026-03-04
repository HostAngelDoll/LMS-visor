#!py -3.10
# main.py
# Sistema de Reconocimiento y Grabación de Lenguaje de Señas (LSM)
# Este código está diseñado para ser estudiado por estudiantes de visión por computadora.

# NOTA: revisar los ultimos cambios en modulos en jules...

import cv2
import time
import os
import numpy as np
import threading
from training.train_static import train

# Importación de nuestros módulos personalizados
from camera_engine import CameraEngine
from hand_processor import HandProcessor
from gesture_logic import GestureLogic
from tracker import HandTracker
from recorder import GestureRecorder

class HandApp:
    """
    Clase principal que orquesta el funcionamiento del sistema.
    Combina la captura de video, procesamiento de manos, reconocimiento y grabación.
    """
    def __init__(self):
        # Configuración de rutas
        self.model_path = "hand_landmarker.task"

        # Inicialización de componentes
        self.camera = CameraEngine()
        self.processor = HandProcessor(self.model_path, num_hands=2)
        self.logic = GestureLogic()
        self.tracker = HandTracker()
        self.recorder = GestureRecorder()

        # Estado de la aplicación
        self.running = True
        self.show_landmarks = True
        self.only_one_hand = True  # Requerimiento: Seguir solo una mano

        # Variables para mostrar en UI
        self.current_static_letter = "---"
        self.recognition_source = "---"
        self.manual_letter = None # Letra seleccionada manualmente con el teclado
        self.target_motion_letter = None
        self.status_msg = "Listo"

        # Estado del entrenamiento
        self.training_in_progress = False
        self.training_status = ""

    def run(self):
        """Bucle principal de ejecución."""
        with self.camera as cam:
            frame_id = 0
            last_timestamp_ms = -1
            start_time_ns = time.perf_counter_ns()
            while self.running:
                # 1. Obtener frame de la cámara
                frame = cam.get_frame()
                if frame is None: continue

                # 2. Procesar mano con MediaPipe (Asíncrono)
                # MediaPipe requiere timestamps estrictamente crecientes en ms.
                curr_ms = (time.perf_counter_ns() - start_time_ns) // 1_000_000
                if curr_ms <= last_timestamp_ms:
                    curr_ms = last_timestamp_ms + 1
                last_timestamp_ms = curr_ms

                self.processor.detect(frame, curr_ms)

                # 3. Obtener resultados del procesamiento
                lands = self.processor.get_hand_landmarks(0) # Tomamos la primera mano detectada

                if lands:
                    # Extraer propiedades para reconocimiento
                    props = self.logic.extract_properties(lands, self.processor)

                    # Reconocer letra estática actual
                    detected, source = self.logic.recognize_static(props, lands)
                    self.current_static_letter = detected if detected else "---"
                    self.recognition_source = source if source else "---"

                    # Lógica de Seguimiento Automático (Triggers)
                    # Si la letra detectada tiene un mapeo de movimiento, activamos seguimiento
                    motion_target, fingers_to_track = self.logic.get_trigger_info(self.current_static_letter)

                    if motion_target:
                        self.target_motion_letter = motion_target
                        self.tracker.set_active_fingers(fingers_to_track)
                        self.status_msg = f"Trigger: {self.current_static_letter} -> Seguir para {motion_target}"
                    elif not self.recorder.recording:
                        # Si no estamos grabando y no hay trigger, limpiamos el objetivo
                        self.target_motion_letter = None
                        self.tracker.set_active_fingers([])
                        self.status_msg = "Esperando gesto disparador..."

                    # Actualizar estelas de los dedos
                    self.tracker.update(lands, frame.shape)

                    # Si estamos grabando, añadir datos al buffer
                    if self.recorder.recording:
                        record_data = {
                            "letter": self.recorder.current_letter,
                            "landmarks": [{"x": l.x, "y": l.y, "z": l.z} for l in lands],
                            "direction": props["direction"],
                            "rotation": props["rotation"],
                            "tracked_fingers": self.tracker.active_fingers,
                            "props": props
                        }
                        self.recorder.add_frame(record_data)

                    # Dibujar landmarks si está activado
                    if self.show_landmarks:
                        self._draw_hand_landmarks(frame, lands)

                # 4. Actualizar estado del grabador (Verificar tiempos y guardado)
                self.recorder.update()

                # 5. Dibujar estelas (pincel)
                self.tracker.draw_trails(frame)

                # 5. Dibujar interfaz de usuario (HUD)
                self._draw_hud(frame)

                # 6. Mostrar resultado
                cv2.imshow("Sistema de Señas LSM - Aprendizaje", frame)

                # 7. Manejar teclado
                self._handle_keys()
                frame_id += 1

        cv2.destroyAllWindows()

    def _handle_keys(self):
        """Gestiona las pulsaciones de teclas para grabación y control."""
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.running = False

        # Selección manual de letra (a-z)
        if ord('a') <= key <= ord('z'):
            self.manual_letter = chr(key).upper()
            self.status_msg = f"Letra manual establecida: {self.manual_letter}"

        # Enter: Grabar gesto estático (1.5 segundos)
        elif key in (13, 10):
            # Prioridad a la letra manual, si no a la detectada
            letter_to_save = self.manual_letter or (self.current_static_letter if self.current_static_letter != "---" else None)
            if letter_to_save:
                self.recorder.start_recording(letter_to_save, is_motion=False, duration=1.5)
            else:
                self.status_msg = "Error: Selecciona una letra con el teclado primero."

        # F11: Entrenar el modelo
        elif key == 122: # F11
            self.start_training()

        # F12: Grabar movimiento (activado por los triggers)
        elif key == 123: # F12
            if self.target_motion_letter:
                self.recorder.start_recording(self.target_motion_letter, is_motion=True)
            else:
                print("No hay un gesto disparador activo para grabar movimiento.")

    def _draw_hand_landmarks(self, frame, lands):
        """Dibuja los puntos y conexiones de la mano."""
        # Definición de conexiones MediaPipe
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20), (0,17)
        ]
        h, w = frame.shape[:2]
        for start, end in CONNECTIONS:
            p1 = (int(lands[start].x * w), int(lands[start].y * h))
            p2 = (int(lands[end].x * w), int(lands[end].y * h))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
        for lm in lands:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1)

    def start_training(self):
        """Inicia el entrenamiento en un hilo separado."""
        if self.training_in_progress:
            self.status_msg = "Entrenamiento ya en curso..."
            return

        def training_task():
            self.training_in_progress = True
            self.status_msg = "Entrenamiento iniciado..."

            def update_status(msg):
                self.training_status = msg
                self.status_msg = f"Entrenando: {msg}"

            try:
                train(progress_callback=update_status)
                self.status_msg = "¡Entrenamiento completado!"
            except Exception as e:
                self.status_msg = f"Error en entrenamiento: {e}"
                print(f"Error en entrenamiento: {e}")
            finally:
                self.training_in_progress = False

        thread = threading.Thread(target=training_task, daemon=True)
        thread.start()

    def _draw_hud(self, frame):
        """Dibuja la información en pantalla para el usuario."""
        y = 30
        cv2.putText(frame, f"Letra: {self.current_static_letter} ({self.recognition_source})", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        y += 30
        manual = self.manual_letter if self.manual_letter else "Ninguna"
        cv2.putText(frame, f"Letra Manual (Teclado): {manual}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y += 30
        target = self.target_motion_letter if self.target_motion_letter else "Ninguno"
        cv2.putText(frame, f"Grabacion Pendiente (F12): {target}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if self.recorder.recording:
            rem = self.recorder.get_remaining_time()
            cv2.putText(frame, f"GRABANDO {self.recorder.current_letter}: {rem:.1f}s", (10, y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        if self.training_in_progress:
            cv2.putText(frame, f"ENTRENANDO: {self.training_status}", (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, self.status_msg, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

if __name__ == "__main__":
    app = HandApp()
    app.run()
