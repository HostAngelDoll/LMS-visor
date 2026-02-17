# recorder.py
# Clase para gestionar la grabación de gestos estáticos y en movimiento.

import json
import time
import os

class GestureRecorder:
    """
    Maneja la lógica de grabación temporizada (5 segundos).
    Permite guardar gestos estáticos y gestos con movimiento en archivos JSON separados.
    """
    def __init__(self, static_path="gestures.json", motion_path="motion_gestures.json"):
        self.static_path = static_path
        self.motion_path = motion_path
        self.recording = False
        self.start_time = 0
        self.duration = 5.0
        self.buffer = [] # Para almacenar frames durante la grabación
        self.current_letter = ""
        self.is_motion = False

    def start_recording(self, letter, is_motion=False):
        """Inicia el proceso de grabación."""
        self.recording = True
        self.start_time = time.time()
        self.current_letter = letter
        self.is_motion = is_motion
        self.buffer = []
        print(f"Grabando {'MOVIMIENTO' if is_motion else 'ESTÁTICO'} para letra: {letter}")

    def add_frame(self, data):
        """Añade datos del frame actual al buffer de grabación."""
        if self.recording:
            self.buffer.append({
                "timestamp": time.time() - self.start_time,
                "data": data
            })

            if time.time() - self.start_time >= self.duration:
                self.stop_and_save()

    def stop_and_save(self):
        """Finaliza la grabación y guarda en el archivo correspondiente."""
        self.recording = False
        path = self.motion_path if self.is_motion else self.static_path

        # Cargar datos existentes
        data_all = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    data_all = json.load(f)
                except:
                    data_all = {}

        if self.current_letter not in data_all:
            data_all[self.current_letter] = []

        # Guardar la muestra
        sample = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": self.buffer
        }
        data_all[self.current_letter].append(sample)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_all, f, indent=2, ensure_ascii=False)

        print(f"Grabación guardada en {path} ({len(self.buffer)} frames)")

    def get_remaining_time(self):
        """Retorna el tiempo restante de grabación."""
        if not self.recording: return 0
        return max(0, self.duration - (time.time() - self.start_time))
