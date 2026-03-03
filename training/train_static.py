import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.model_def import StaticGestureMLP

class GestureDataset(Dataset):
    def __init__(self, json_path):
        self.samples = []
        self.labels = []

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
                # Descartar primeros 2 frames para estabilidad si la grabación es larga
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

        self.samples = torch.FloatTensor(np.array(self.samples))
        self.labels = torch.LongTensor(self.labels)
        print(f"Dataset cargado: {len(self.samples)} muestras, {len(self.classes)} clases.")

    def normalize_landmarks(self, landmarks):
        # Convertir a numpy para facilidad
        lms = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])

        # 1. Traslación: Restar muñeca (landmark 0)
        base = lms[0]
        lms = lms - base

        # 2. Escalado: Distancia muñeca (0) a base dedo medio (9)
        # MediaPipe ya normaliza x,y entre 0 y 1, pero queremos independencia de escala
        palm_size = np.linalg.norm(lms[9]) # lms[9] ya tiene restado lms[0]
        if palm_size > 0:
            lms = lms / palm_size

        return lms.flatten()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def train():
    json_path = "gestures.json"
    dataset = GestureDataset(json_path)
    if len(dataset) == 0:
        print("No hay suficientes datos para entrenar.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_classes = len(dataset.classes)
    model = StaticGestureMLP(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # Guardar modelo y mapeo
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/static_model.pt")

    mapping = {i: cls for i, cls in enumerate(dataset.classes)}
    with open("models/class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print("Entrenamiento completado. Modelo guardado en models/")

if __name__ == "__main__":
    train()
