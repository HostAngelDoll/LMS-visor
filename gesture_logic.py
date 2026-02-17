# gesture_logic.py
# Lógica para reconocer gestos estáticos y determinar disparadores de movimiento.

import math

class GestureLogic:
    """
    Contiene la lógica de reconocimiento de letras del alfabeto dactilológico
    y el mapeo para las grabaciones automáticas de movimiento.
    """

    # Mapeo de gestos estáticos que disparan seguimiento automático
    # Gesto estático -> (Letra a grabar, [Índices de dedos a seguir])
    TRIGGER_MAP = {
        "I": ("J", [20]),       # Seguir meñique para J
        "P": ("K", [8]),        # Seguir índice para K
        "N": ("Ñ", [8]),        # Seguir índice para Ñ
        "Q": ("Q", [8, 4]),     # Seguir índice y pulgar para Q moviéndose
        "X": ("X", [8, 4]),     # Seguir índice y pulgar para X moviéndose
        "D": ("Z", [8]),        # Seguir índice para Z
    }

    def __init__(self, gestures_db_path="gestures.json"):
        self.gestures_db_path = gestures_db_path
        self.gestures_db = self._load_db()
        self.threshold = 0.35 # Umbral de confianza

    def _load_db(self):
        """Carga la base de datos de gestos desde el archivo JSON."""
        import json
        import os
        if os.path.exists(self.gestures_db_path):
            try:
                with open(self.gestures_db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def recognize_static(self, hand_props):
        """
        Reconoce la letra estática comparando con la DB y aplicando heurísticas.
        """
        # 1. Intentar por heurística (Reglas rápidas)
        heuristic_res = self._recognize_heuristic(hand_props)
        if heuristic_res:
            return heuristic_res

        # 2. Intentar por comparación con base de datos JSON
        if not self.gestures_db:
            return None

        best_letter = None
        best_score = 1e9

        # Simplificación de la lógica de comparación del main.py original
        # Comparamos estados de dedos y distancias normalizadas
        for letter, info in self.gestures_db.items():
            samples = info.get("samples", []) if isinstance(info, dict) else []
            for samp in samples:
                agg = samp.get("aggregates", {})
                score = self._compute_score(hand_props, agg)
                if score < best_score:
                    best_score = score
                    best_letter = letter

        if best_score < self.threshold:
            return best_letter
        return None

    def _recognize_heuristic(self, hand_props):
        """Reglas lógicas para algunas letras básicas."""
        st = hand_props["states"]
        d_th_idx = hand_props["d_thumb_index"]
        d_idx_mid = hand_props["d_index_middle"]

        # Letra D
        if st["index"] and not st["middle"] and not st["ring"] and not st["pinky"] and not hand_props["thumb_left"]:
            return "D"
        # Letra I
        if st["pinky"] and not st["index"] and not st["middle"] and not st["ring"]:
            return "I"
        # Letra L
        if st["index"] and hand_props["thumb_left"] and not st["middle"]:
            return "L"
        # Letra P (Aproximación)
        if st["index"] and st["middle"] and not st["ring"] and not st["pinky"] and hand_props["rotation"] > 130:
            return "P"

        return None

    def _compute_score(self, props, agg):
        """Calcula una puntuación de diferencia entre las propiedades actuales y una muestra."""
        # Esta es una versión simplificada de la lógica de comparación original
        diff = 0
        # Comparar estados de dedos (Booleanos)
        keys = ["idx_ext", "mid_ext", "ring_ext", "pinky_ext"]
        prop_map = {"idx_ext": "index", "mid_ext": "middle", "ring_ext": "ring", "pinky_ext": "pinky"}

        for k in keys:
            current = 1 if props["states"][prop_map[k]] else 0
            expected = agg.get(f"prop_{k}", 0.5)
            diff += abs(current - expected)

        # Comparar distancias (Numéricas)
        d_keys = [("d_thumb_index", "d_thumb_index"), ("d_index_middle", "d_index_middle")]
        for pk, ak in d_keys:
            current = props[pk]
            avg_val = agg.get(f"avg_{ak}", 0.5)
            diff += abs(current - avg_val) * 0.5

        return diff / (len(keys) + len(d_keys))

    def get_trigger_info(self, static_letter):
        """Retorna la letra objetivo y los dedos a seguir según el disparo."""
        return self.TRIGGER_MAP.get(static_letter, (None, []))

    @staticmethod
    def extract_properties(lands, processor):
        """Prepara un diccionario de propiedades útiles para el reconocimiento."""
        hand_s = processor.get_palm_size(lands)
        return {
            "states": processor.get_finger_states(lands),
            "palm_size": hand_s,
            "d_thumb_index": processor.distance(lands[4], lands[8]) / hand_s,
            "d_index_middle": processor.distance(lands[8], lands[12]) / hand_s,
            "thumb_left": lands[4].x < lands[8].x, # Simplificado para mano derecha
            "direction": processor.get_hand_direction(lands),
            "rotation": processor.get_hand_rotation(lands)
        }
