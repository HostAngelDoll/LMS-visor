# recorder.py
# Clase para gestionar la grabación de gestos estáticos y en movimiento.

import json
import time
import os

class GestureRecorder:
    """
    Maneja la lógica de grabación temporizada.
    Permite guardar gestos estáticos y gestos con movimiento en archivos JSON separados.
    """
    def __init__(self, static_path="gestures.json", motion_path="motion_gestures.json", log_callback=None):
        self.static_path = static_path
        self.motion_path = motion_path
        self.log_callback = log_callback
        self.recording = False
        self.start_time = 0
        self.duration = 5.0
        self.buffer = [] # Para almacenar frames durante la grabación
        self.current_letter = ""
        self.is_motion = False

    def start_recording(self, letter, is_motion=False, duration=None):
        """Inicia el proceso de grabación."""
        if self.recording:
            return # Evitar reinicio si ya está grabando

        self.recording = True
        self.start_time = time.time()
        self.current_letter = letter
        self.is_motion = is_motion
        # Duración por defecto: 5s para movimiento, 1.5s para estático
        self.duration = duration if duration is not None else (5.0 if is_motion else 1.5)
        self.buffer = []
        print(f"Iniciando grabación {'MOVIMIENTO' if is_motion else 'ESTÁTICA'} para letra: {letter} ({self.duration}s)")

    def add_frame(self, data):
        """Añade datos del frame actual al buffer de grabación si estamos grabando."""
        if self.recording:
            self.buffer.append({
                "timestamp": time.time() - self.start_time,
                "data": data
            })
            # El control de parada ahora se delega a update() para ser independiente de si hay detección

    def update(self):
        """Verifica el tiempo y finaliza la grabación si se cumple la duración."""
        if self.recording:
            if time.time() - self.start_time >= self.duration:
                return self.stop_and_save()
        return None

    def _compute_aggregates(self):
        """Calcula promedios y proporciones de las propiedades grabadas."""
        if not self.buffer: return {}
        
        # Extraer todas las props de los frames grabados
        all_props = [f["data"]["props"] for f in self.buffer if "props" in f["data"]]
        if not all_props: return {}

        agg = {}
        # Promedios numéricos
        num_keys = ["d_thumb_index", "d_thumb_middle", "d_index_middle", "rotation"]
        for k in num_keys:
            vals = [p[k] for p in all_props if k in p]
            if vals:
                agg[f"avg_{k}"] = sum(vals) / len(vals)

        # Proporciones booleanas (cuánto tiempo estuvo el dedo extendido)
        bool_keys = ["thumb", "index", "middle", "ring", "pinky"]
        for k in bool_keys:
            vals = [1 if p["states"].get(k, False) else 0 for p in all_props if "states" in p]
            if vals:
                agg[f"prop_{k}_ext"] = sum(vals) / len(vals)
        
        return agg

    def stop_and_save(self):
        """Finaliza la grabación y guarda en el archivo correspondiente."""
        if not self.recording: return None

        self.recording = False
        res_msg = ""
        res_type = "info"

        # Filtrar frames que no tengan landmarks válidos antes de guardar
        valid_buffer = [f for f in self.buffer if f["data"].get("landmarks")]

        if len(valid_buffer) < 5:
            res_msg = f"Grabación cancelada: Insuficientes frames válidos ({len(valid_buffer)})."
            res_type = "error"
            print(res_msg)
            if self.log_callback: self.log_callback(res_msg, res_type)
            return (res_msg, res_type)

        self.buffer = valid_buffer
        path = self.motion_path if self.is_motion else self.static_path
        
        # Cargar datos existentes con manejo de errores robusto
        data_all = {}
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        data_all = json.loads(content)
            except Exception as e:
                print(f"Error al cargar DB existente ({path}): {e}. Se creará un nuevo archivo.")
                data_all = {}

        if self.current_letter not in data_all:
            data_all[self.current_letter] = {"samples": []}
        
        # Compatibilidad con formatos antiguos
        if isinstance(data_all[self.current_letter], list):
            data_all[self.current_letter] = {"samples": data_all[self.current_letter]}

        # Guardar la muestra
        sample = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": self.buffer,
            "aggregates": self._compute_aggregates() if not self.is_motion else {}
        }
        data_all[self.current_letter]["samples"].append(sample)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data_all, f, indent=2, ensure_ascii=False)
            res_msg = f"Éxito: {len(self.buffer)} frames guardados para '{self.current_letter}' en {path}"
            res_type = "success"
            print(res_msg)
        except Exception as e:
            res_msg = f"Error fatal al guardar en {path}: {e}"
            res_type = "error"
            print(res_msg)

        if self.log_callback: self.log_callback(res_msg, res_type)
        return (res_msg, res_type)

    def get_remaining_time(self):
        """Retorna el tiempo restante de grabación."""
        if not self.recording: return 0
        return max(0, self.duration - (time.time() - self.start_time))
