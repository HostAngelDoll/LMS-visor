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

    def _compute_aggregates(self):
        """Calcula promedios y proporciones de las propiedades grabadas."""
        if not self.buffer: return {}
        
        num_frames = len(self.buffer)
        # Extraer todas las props
        all_props = [f["data"]["props"] for f in self.buffer if "props" in f["data"]]
        if not all_props: return {}

        agg = {}
        # Promedios numéricos
        num_keys = ["d_thumb_index", "d_thumb_middle", "d_index_middle", "rotation"]
        for k in num_keys:
            vals = [p[k] for p in all_props if k in p]
            if vals:
                agg[f"avg_{k}"] = sum(vals) / len(vals)

        # Proporciones booleanas
        bool_keys = ["index", "middle", "ring", "pinky"]
        for k in bool_keys:
            vals = [1 if p["states"][k] else 0 for p in all_props if "states" in p and k in p["states"]]
            if vals:
                agg[f"prop_{k}_ext"] = sum(vals) / len(vals)
        
        return agg

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
            data_all[self.current_letter] = {"samples": []}
        
        # Manejar compatibilidad si por algún motivo es una lista (formato muy antiguo)
        if isinstance(data_all[self.current_letter], list):
            data_all[self.current_letter] = {"samples": data_all[self.current_letter]}

        # Guardar la muestra
        sample = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": self.buffer,
            "aggregates": self._compute_aggregates() if not self.is_motion else {}
        }
        data_all[self.current_letter]["samples"].append(sample)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_all, f, indent=2, ensure_ascii=False)
            
        print(f"Grabación guardada en {path} ({len(self.buffer)} frames)")

    def get_remaining_time(self):
        """Retorna el tiempo restante de grabación."""
        if not self.recording: return 0
        return max(0, self.duration - (time.time() - self.start_time))
