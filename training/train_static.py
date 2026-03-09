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

class GestureDataset:
    def __init__(self, json_path):
        self.samples = []
        self.labels = []
        self.classes = []

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
                # Descartar primeros frames para estabilidad
                start_idx = 3 if len(frames) > 15 else (1 if len(frames) > 5 else 0)
                for frame in frames[start_idx:]:
                    landmarks = frame.get("data", {}).get("landmarks", [])
                    if len(landmarks) == 21:
                        # Normalización
                        normalized = self.normalize_landmarks(landmarks)
                        self.samples.append(normalized)
                        self.labels.append(self.class_to_idx[letter])
                        letter_samples_count += 1
            print(f"  Letra '{letter}': {letter_samples_count} muestras de landmarks.")

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        print(f"Dataset cargado: {len(self.samples)} muestras, {len(self.classes)} clases.")

    def normalize_landmarks(self, landmarks):
        # Convertir a numpy para facilidad
        lms = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])

        # 1. Traslación: Restar muñeca (landmark 0)
        base = lms[0]
        lms = lms - base

        # 2. Escalado: Distancia muñeca (0) a base dedo medio (9)
        palm_size = np.linalg.norm(lms[9])
        if palm_size > 0:
            lms = lms / palm_size

        return lms.flatten()

def train(progress_callback=None):
    # Usar rutas absolutas basadas en la raíz del proyecto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(project_root, "gestures.json")
    
    if progress_callback: progress_callback("Cargando dataset...")
    dataset = GestureDataset(json_path)
    if len(dataset.samples) == 0:
        msg = "No hay suficientes datos para entrenar."
        print(msg)
        if progress_callback: progress_callback(msg)
        return

    if progress_callback: progress_callback("Iniciando entrenamiento MLP (scikit-learn)...")
    
    # Definición del modelo usando scikit-learn
    # Parámetros equivalentes al MLP previo: capas (128, 64), Adam, ReLU
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        learning_rate_init=0.001,
        verbose=False, # verbose=True imprimiría en consola, manejaremos por callback
        random_state=42
    )

    # El verbose de scikit-learn no es fácilmente capturable por callbacks línea a línea
    # así que simularemos etapas.
    try:
        model.fit(dataset.samples, dataset.labels)
    except Exception as e:
        msg = f"Error en fit: {e}"
        print(msg)
        if progress_callback: progress_callback(msg)
        return

    # Guardar modelo y mapeo
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "static_model.joblib")
    joblib.dump(model, model_path)

    mapping = {i: cls for i, cls in enumerate(dataset.classes)}
    mapping_path = os.path.join(models_dir, "class_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    msg_fin = "Entrenamiento scikit-learn completado. Modelo guardado."
    print(msg_fin)
    if progress_callback: progress_callback(msg_fin)

if __name__ == "__main__":
    train()
