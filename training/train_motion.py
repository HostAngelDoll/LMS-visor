import json
import os
import numpy as np
import sys
import joblib
from sklearn.neural_network import MLPClassifier

# Añadir el directorio raíz al path para permitir importaciones consistentes
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from gesture_logic import GestureLogic

class MotionDataset:
    def __init__(self, json_path, seq_len=15):
        self.samples = []
        self.labels = []
        self.classes = []
        self.seq_len = seq_len

        if not os.path.exists(json_path):
            print(f"Error: {json_path} no existe.")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Mapeo de letras a índices
        self.classes = sorted(list(data.keys()))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for letter, info in data.items():
            samples_list = info.get("samples", [])
            letter_samples_count = 0
            for sample in samples_list:
                frames = sample.get("frames", [])
                # Un "sample" de movimiento es una secuencia completa de 5 segundos.
                # Podemos extraer múltiples ventanas de seq_len para aumentar el dataset.

                # Extraer trayectorias de los frames
                finger_histories = {fid: [] for fid in [4, 8, 12, 16, 20]}

                # Necesitamos un frame de referencia (w, h) para convertir landmarks a pixeles (o algo similar)
                # O simplemente trabajar con landmarks normalizados directamente.
                # GestureLogic.extract_motion_features usa pixeles (histories del tracker).
                # Convertiremos los landmarks (0-1) a una escala de 640x480 para consistencia.

                for frame in frames:
                    lands = frame.get("data", {}).get("landmarks", [])
                    if len(lands) == 21:
                        for fid in finger_histories:
                            lm = lands[fid]
                            # Escala 640x480
                            finger_histories[fid].append((lm['x'] * 640, lm['y'] * 480))

                # Extraer ventanas de seq_len
                # Si tenemos 150 frames (5s @ 30fps), podemos sacar muchas ventanas
                step = 5
                for start_f in range(0, len(frames) - seq_len, step):
                    window_hist = {}
                    for fid, all_pts in finger_histories.items():
                        window_hist[fid] = all_pts[start_f : start_f + seq_len]

                    features = GestureLogic.extract_motion_features(window_hist, seq_len=seq_len)
                    if features is not None:
                        self.samples.append(features)
                        self.labels.append(self.class_to_idx[letter])
                        letter_samples_count += 1

            print(f"  Letra '{letter}': {letter_samples_count} ventanas de movimiento.")

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        print(f"Dataset de Movimiento cargado: {len(self.samples)} muestras, {len(self.classes)} clases.")

def train_motion(progress_callback=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(project_root, "motion_gestures.json")

    if progress_callback: progress_callback("Cargando dataset de movimiento...")
    dataset = MotionDataset(json_path)
    if len(dataset.samples) == 0:
        msg = "No hay suficientes datos de movimiento para entrenar."
        print(msg)
        if progress_callback: progress_callback(msg)
        return

    if progress_callback: progress_callback("Iniciando entrenamiento MLP (Movimiento)...")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=1000,
        learning_rate_init=0.001,
        verbose=False,
        random_state=42
    )

    try:
        model.fit(dataset.samples, dataset.labels)
    except Exception as e:
        msg = f"Error en fit de movimiento: {e}"
        print(msg)
        if progress_callback: progress_callback(msg)
        return

    # Guardar modelo y mapeo
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "motion_model.joblib")
    joblib.dump(model, model_path)

    mapping = {i: cls for i, cls in enumerate(dataset.classes)}
    mapping_path = os.path.join(models_dir, "motion_class_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    msg_fin = "Entrenamiento de movimiento completado. Modelo guardado."
    print(msg_fin)
    if progress_callback: progress_callback(msg_fin)

if __name__ == "__main__":
    train_motion()
