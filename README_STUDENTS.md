# Guía para Estudiantes - Sistema de Reconocimiento de LSM

![Interfaz del Sistema](https://github.com/user-attachments/assets/a9a0524e-b278-4ab4-acf3-1ddcb28737f3)

Este proyecto ha sido organizado para facilitar su estudio. A continuación se explica la arquitectura y el funcionamiento de cada componente, incluyendo las adiciones recientes de Inteligencia Artificial e Interfaces Gráficas.

## Estructura del Proyecto

El código se divide en módulos especializados:

1.  **`main.py` (Orquestador PyQt6)**: Es el punto de entrada. Utiliza la librería **PyQt6** para crear una interfaz con botones, logs y visualización. Los estudiantes pueden aprender aquí sobre la gestión de hilos (**QThread**) para que la IA no congele la interfaz.
2.  **`camera_engine.py` (Captura Dual)**: Gestiona tanto la cámara OAK-D (`depthai`) como Webcams estándar (`opencv`). Es un gran ejemplo de cómo abstraer el hardware.
3.  **`hand_processor.py` (Visión y Geometría)**: Utiliza **MediaPipe** para detectar los 21 puntos (landmarks). Contiene funciones para calcular:
    *   Ángulos entre dedos (curvatura).
    *   Distancias normalizadas.
    *   Dirección (Arriba, Abajo, Izquierda, Derecha).
4.  **`gesture_logic.py` (Cerebro del Sistema)**: Decide la letra detectada usando un sistema híbrido:
    *   **MLP (Machine Learning)**: Un Perceptrón Multicapa que aprende de los datos.
    *   **Heurística**: Reglas lógicas matemáticas.
    *   **Estadística**: Comparación directa con promedios en JSON.
5.  **`training/train_static.py`**: Script que toma los datos de `gestures.json` y entrena el modelo de scikit-learn.
6.  **`tracker.py` y `recorder.py`**: Gestionan el seguimiento visual y el guardado de datos en archivos JSON.

## Conceptos Clave para Estudiar

### 1. Clasificación con MLP (Machine Learning)
El sistema usa un modelo **MLP (Multi-Layer Perceptron)**.
*   **Entrada**: Los 21 puntos de la mano (x, y, z) aplanados en un vector de 63 valores.
*   **Normalización**: Antes de entrar al modelo, los puntos se centran en la muñeca y se escalan según el tamaño de la palma. Esto permite que el modelo funcione igual si la mano está cerca o lejos.

### 2. Procesamiento Asíncrono (Hilos)
Cuando presionas **F11**, el entrenamiento ocurre en un hilo separado (`TrainingThread`). Esto es vital para que la ventana de la aplicación siga respondiendo mientras la computadora realiza cálculos intensos.

### 3. Sistema de Disparadores (Triggers)
El sistema automático de seguimiento funciona así:
*   Se reconoce una letra estática base (ej. 'P').
*   Se busca en `TRIGGER_MAP` (en `gesture_logic.py`) si debe iniciar un seguimiento.
*   Si existe, se activan los colores de los dedos necesarios (ej. índice para la 'K') y se habilita la grabación de movimiento.

### 4. Persistencia de Datos
Los datos se guardan en formato **JSON**. Observa cómo `recorder.py` calcula "agregados" (promedios) al guardar, lo que facilita que luego podamos comparar una mano nueva contra esos promedios.

## Tareas Sugeridas para Estudiantes

1.  **Exploración de Datos**: Abre `gestures.json` y observa cómo se guardan las coordenadas. ¿Podrías crear un script sencillo que grafique estos puntos?
2.  **Ajuste de Hiperparámetros**: En `training/train_static.py`, cambia el número de neuronas en `hidden_layer_sizes=(128, 64)` y observa si la precisión mejora o empeora.
3.  **Nuevas Reglas**: Intenta añadir una regla en `_recognize_heuristic` dentro de `gesture_logic.py` que detecte la letra "V" (dedos índice y medio levantados en forma de V).
4.  **Personalización UI**: Cambia los colores de los logs en la clase `LogWidget` dentro de `main.py`.
5.  **Análisis de Trayectorias**: Usa `pencil.py` para dibujar letras en el aire y analiza cómo cambian las coordenadas en el tiempo.
