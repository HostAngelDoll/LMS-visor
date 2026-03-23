# gesture_logic.py
# Lógica para reconocer gestos estáticos y determinar disparadores de movimiento.

import math
import numpy as np
import os
import json
import sys
import joblib

# Manejo de importación para scikit-learn
try:
    from sklearn.neural_network import MLPClassifier
    SKLEARN_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Error fatal: scikit-learn no pudo cargarse ({e}).")
    SKLEARN_AVAILABLE = False

class GestureLogic:
    """
    Contiene la lógica de reconocimiento de letras del alfabeto dactilológico
    y el mapeo para las grabaciones automáticas de movimiento.
    """

    # Gesto estático -> (Letra a grabar, [Índices de dedos a seguir])
    TRIGGER_MAP = {
        "I": ("J", [20]),       # Seguir meñique para J
        "P": ("K", [8]),        # Seguir índice para K
        "N": ("Ñ", [8]),        # Seguir índice para Ñ
        "Q": ("Q", [8, 4]),     # Seguir índice y pulgar para Q moviéndose
        "X": ("X", [8, 4]),     # Seguir índice y pulgar para X moviéndose
        "D": ("Z", [8]),        # Seguir índice para Z
    }

    def __init__(self, gestures_db_path="gestures.json", model_dir="models"):
        self.gestures_db_path = gestures_db_path
        self.model_dir = model_dir
        self.threshold = 0.35 # Umbral de confianza para heurística/DB
        self.mlp_threshold = 0.70 # Umbral para scikit-learn (predict_proba)

        # Cargar base de datos y modelo MLP
        self.gestures_db = {}
        self.model = None
        self.class_mapping = {}

        self.motion_model = None
        self.motion_class_mapping = {}

        self.reload()

    def reload(self):
        """Recarga la base de datos de gestos y los modelos MLP sin reiniciar la app."""
        self.gestures_db = self._load_db()
        if SKLEARN_AVAILABLE:
            self._load_mlp(self.model_dir)
            self._load_motion_mlp(self.model_dir)

    def _load_db(self):
        """Carga la base de datos de gestos desde el archivo JSON."""
        if os.path.exists(self.gestures_db_path):
            try:
                with open(self.gestures_db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _load_motion_mlp(self, model_dir):
        """Carga el modelo scikit-learn de movimiento si existe."""
        model_path = os.path.join(model_dir, "motion_model.joblib")
        mapping_path = os.path.join(model_dir, "motion_class_mapping.json")

        if os.path.exists(model_path) and os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    self.motion_class_mapping = json.load(f)

                self.motion_model = joblib.load(model_path)
                print(f"Modelo de movimiento cargado con {len(self.motion_class_mapping)} clases.")
            except Exception as e:
                print(f"Error al cargar el modelo de movimiento: {e}")

    def _load_mlp(self, model_dir):
        """Carga el modelo scikit-learn y el mapeo de clases si existen."""
        model_path = os.path.join(model_dir, "static_model.joblib")
        mapping_path = os.path.join(model_dir, "class_mapping.json")

        if os.path.exists(model_path) and os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    self.class_mapping = json.load(f)

                self.model = joblib.load(model_path)
                print(f"Modelo scikit-learn cargado correctamente con {len(self.class_mapping)} clases.")
            except Exception as e:
                print(f"Error al cargar el modelo scikit-learn: {e}")

    def recognize_static(self, hand_props, lands=None):
        """
        Reconoce la letra estática usando MLP (prioridad) o heurísticas (fallback).
        Retorna (Letra, Origen)
        """
        # 1. Intentar con MLP de scikit-learn
        if self.model and lands:
            mlp_res, confidence = self._recognize_mlp(lands)
            if mlp_res and confidence >= self.mlp_threshold:
                return mlp_res, f"MLP ({confidence:.2f})"

        # 2. Intentar por heurística
        heuristic_res = self._recognize_heuristic(hand_props)
        if heuristic_res:
            return heuristic_res, "Heurística"

        # 3. Intentar por comparación estadística
        if not self.gestures_db:
            return None, None

        best_letter = None
        best_score = 1e9

        for letter, info in self.gestures_db.items():
            samples = info.get("samples", []) if isinstance(info, dict) else []
            for samp in samples:
                agg = samp.get("aggregates", {})
                score = self._compute_score(hand_props, agg)
                if score < best_score:
                    best_score = score
                    best_letter = letter

        if best_score < self.threshold:
            return best_letter, "Base de Datos"

        return None, None

    def _recognize_mlp(self, lands):
        """Realiza la inferencia con el modelo scikit-learn."""
        if not SKLEARN_AVAILABLE:
            return None, 0.0
        try:
            # Normalización
            lms = np.array([[lm.x, lm.y, lm.z] for lm in lands])
            base = lms[0]
            lms = lms - base
            palm_size = np.linalg.norm(lms[9])
            if palm_size > 0:
                lms = lms / palm_size

            input_data = lms.flatten().reshape(1, -1)

            # Obtener probabilidades
            probs = self.model.predict_proba(input_data)[0]
            predicted_idx = np.argmax(probs)
            confidence = probs[predicted_idx]

            label_idx = str(predicted_idx)
            return self.class_mapping.get(label_idx), float(confidence)
        except Exception as e:
            # print(f"Error en inferencia scikit-learn: {e}")
            return None, 0.0

    def _recognize_heuristic(self, hand_props):
        """Reglas lógicas para algunas letras básicas."""
        st = hand_props["states"]
        d_th_idx = hand_props["d_thumb_index"]
        curls = hand_props["curls"]

        if st["index"] and st["middle"] and st["ring"] and st["pinky"]:
            if not st["thumb"] or d_th_idx < 0.3:
                return "B"
        if st["index"] and not st["middle"] and not st["ring"] and not st["pinky"] and not hand_props["thumb_left"]:
            return "D"
        if st["pinky"] and not st["index"] and not st["middle"] and not st["ring"]:
            return "I"
        if st["index"] and hand_props["thumb_left"] and not st["middle"]:
            return "L"
        if st["index"] and st["middle"] and not st["ring"] and not st["pinky"] and hand_props["rotation"] > 130:
            return "P"

        is_curled_all = all(curls[f] < 130 for f in ["index", "middle", "ring", "pinky"])
        if is_curled_all:
            if d_th_idx > 0.45:
                return "C"
            elif d_th_idx < 0.35:
                return "E"

        return None

    def _compute_score(self, props, agg):
        """Calcula puntuación de diferencia."""
        if not agg: return 1.0

        diff = 0
        keys = ["thumb", "index", "middle", "ring", "pinky"]
        for k in keys:
            current = 1 if props["states"].get(k, False) else 0
            expected = agg.get(f"prop_{k}_ext", 0.5)
            diff += abs(current - expected)

        d_keys = ["d_thumb_index", "d_thumb_middle", "d_index_middle"]
        for k in d_keys:
            if k in props:
                current = props[k]
                avg_val = agg.get(f"avg_{k}", 0.5)
                diff += abs(current - avg_val) * 0.5

        return diff / (len(keys) + len(d_keys))

    def get_trigger_info(self, static_letter):
        return self.TRIGGER_MAP.get(static_letter, (None, []))

    def recognize_motion(self, histories):
        """
        Reconoce el gesto de movimiento basado en las trayectorias de los dedos.
        histories: dict hand_idx -> finger_idx -> deque(maxlen=N) of (x, y)
        """
        if not self.motion_model or not histories:
            return None, 0.0

        # Usamos la mano 0 por defecto
        hand_0 = histories.get(0, {})
        if not hand_0:
            return None, 0.0

        # Extraer características (ej. desplazamientos relativos de los dedos activos)
        features = self.extract_motion_features(hand_0)
        if features is None:
            return None, 0.0

        try:
            input_data = features.reshape(1, -1)
            probs = self.motion_model.predict_proba(input_data)[0]
            predicted_idx = np.argmax(probs)
            confidence = probs[predicted_idx]

            label_idx = str(predicted_idx)
            return self.motion_class_mapping.get(label_idx), float(confidence)
        except Exception as e:
            # print(f"Error en inferencia de movimiento: {e}")
            return None, 0.0

    @staticmethod
    def extract_motion_features(hand_history, seq_len=15):
        """
        Extrae características de la trayectoria para el modelo de movimiento.
        Se basa en los desplazamientos (dx, dy) normalizados de los dedos seguidos.
        """
        # Identificar dedos activos (que tengan puntos)
        active_fids = [fid for fid, pts in hand_history.items() if len(pts) >= 2]
        if not active_fids:
            return None

        # Para que el modelo sea consistente, siempre tomamos una longitud fija
        # Si hay más puntos, los muestreamos o tomamos los últimos.
        # Si hay menos, retornamos None por ahora (insuficiente para reconocer).

        all_features = []
        # Para cada dedo, queremos representar su movimiento reciente
        # Usaremos los últimos 'seq_len' puntos.
        for fid in [4, 8, 12, 16, 20]: # Orden fijo
            pts = list(hand_history.get(fid, []))
            if len(pts) < 5: # Mínimo para considerar movimiento
                # Rellenar con ceros si no está activo o tiene pocos puntos
                # 2 (dx, dy) * (seq_len - 1)
                all_features.extend([0.0] * (2 * (seq_len - 1)))
                continue

            # Tomar los últimos seq_len puntos
            if len(pts) > seq_len:
                pts = pts[-seq_len:]

            # Si tiene menos de seq_len, "pad" con el primer punto (velocidad 0)
            if len(pts) < seq_len:
                pts = [pts[0]] * (seq_len - len(pts)) + pts

            # Calcular desplazamientos (dx, dy) normalizados
            # Normalizamos por el ancho/alto del frame no es ideal,
            # pero aquí los puntos ya están en píxeles.
            # Mejor normalizar por una escala interna si fuera posible.
            for i in range(len(pts) - 1):
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                # Podríamos normalizar por la distancia total o algo similar
                all_features.append(dx / 100.0) # Escala arbitraria para evitar valores enormes
                all_features.append(dy / 100.0)

        return np.array(all_features)

    @staticmethod
    def extract_properties(lands, processor):
        hand_s = processor.get_palm_size(lands)
        return {
            "states": processor.get_finger_states(lands),
            "curls": processor.get_finger_curls(lands),
            "palm_size": hand_s,
            "d_thumb_index": processor.distance(lands[4], lands[8]) / hand_s,
            "d_thumb_middle": processor.distance(lands[4], lands[12]) / hand_s,
            "d_index_middle": processor.distance(lands[8], lands[12]) / hand_s,
            "thumb_left": lands[4].x < lands[8].x,
            "direction": processor.get_hand_direction(lands),
            "rotation": processor.get_hand_rotation(lands)
        }
