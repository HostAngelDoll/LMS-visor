# LMS-Visor

Aplicación de escritorio en Python para captura de video, detección de manos, reconocimiento de señas y registro de muestras para entrenamiento.

## Descripción

LMS-Visor está orientado al trabajo con Lengua de Señas Mexicana (LSM). La aplicación integra una interfaz de PyQt6 con captura de cámara, detección de landmarks de mano, lógica de reconocimiento y herramientas para grabar muestras estáticas y dinámicas.

El sistema puede trabajar con una cámara **OAK-D** o con una **webcam** convencional.

## Cómo está compuesto

### `main.py`
Es el punto de entrada de la aplicación. Coordina la interfaz, la cámara, el procesamiento de video, el reconocimiento, la grabación y el entrenamiento.

### `camera_engine.py`
Abstrae la fuente de video. Permite usar:
- OAK-D mediante DepthAI
- Webcam mediante `cv2.VideoCapture(0)`

### `hand_processor.py`
Ejecuta la detección de mano con MediaPipe Hand Landmarker y entrega los landmarks para el análisis posterior.

### `gesture_logic.py`
Contiene la lógica de reconocimiento:
- reconocimiento estático
- detección de movimiento
- disparadores entre letras
- comparación con datos almacenados en JSON
- uso de modelo MLP entrenado

### `tracker.py`
Mantiene el seguimiento visual de los dedos y ayuda a dibujar trayectorias.

### `recorder.py`
Se encarga de guardar muestras:
- muestras estáticas
- muestras de movimiento

### `pencil.py`
Herramienta auxiliar para pruebas visuales y manejo de trazos.

### Archivos de datos y modelos
- `gestures.json`: datos de gestos estáticos
- `motion_gestures.json`: datos de movimientos
- `models/`: modelos entrenados y recursos relacionados
- `hand_landmarker.task`: modelo requerido por MediaPipe

## Requisitos

- Python 3.10
- `opencv-python`
- `depthai`
- `mediapipe`
- `numpy`
- `keyboard`
- `PyQt6`
- `scikit-learn`
- `joblib`

## Instalación

1. Clona el repositorio.
2. Crea y activa un entorno virtual.
3. Instala las dependencias.
4. Verifica que el archivo `hand_landmarker.task` esté en la raíz del proyecto.
5. Ejecuta `main.py`.

Ejemplo en Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install opencv-python depthai mediapipe numpy keyboard PyQt6 scikit-learn joblib
python main.py
```

## Uso básico

1. Abre la aplicación.
2. Selecciona la fuente de cámara.
3. Conecta la cámara.
4. Coloca la mano frente al lente.
5. Observa la letra detectada en la interfaz.
6. Guarda muestras si deseas alimentar el sistema.
7. Entrena el modelo cuando tengas suficientes datos.

## Controles

- `A` a `Z`: selección manual de letra
- `Enter`: grabar muestra estática
- `F11`: entrenar el modelo
- `F12`: grabar muestra de movimiento
- `Q`: cerrar la aplicación

## Flujo de trabajo

1. El sistema captura video desde la cámara seleccionada.
2. `hand_processor.py` detecta la mano y extrae landmarks.
3. `gesture_logic.py` analiza los datos y propone una letra.
4. Si hay una letra válida, el sistema puede activar seguimiento o grabación.
5. `recorder.py` guarda la muestra si se inicia una captura.
6. El entrenamiento puede ejecutarse sin cerrar la aplicación.

## Reconocimiento

La lógica de reconocimiento combina varios criterios:
- modelo MLP
- heurísticas de decisión
- comparación estadística con datos existentes

Cuando la confianza no es suficiente, el sistema puede apoyarse en reglas adicionales antes de aceptar una predicción.

## Captura de pantalla

La aplicación puede guardar capturas completas en:

`Documentos/Capturas_LSM/`

El nombre del archivo se genera automáticamente con fecha y hora.

## Notas importantes

- La aplicación depende de `hand_landmarker.task`.
- El proyecto está diseñado para trabajar con detección visual en tiempo real.
- La documentación del repositorio menciona algunos atajos adicionales, pero el comportamiento que se debe tomar como referencia es el que implementa el código fuente.
- Si no hay cámara conectada o el archivo de modelo falta, la aplicación no podrá funcionar correctamente.

## Estructura resumida

```text
LMS-visor/
├─ main.py
├─ camera_engine.py
├─ hand_processor.py
├─ gesture_logic.py
├─ tracker.py
├─ recorder.py
├─ pencil.py
├─ gestures.json
├─ motion_gestures.json
├─ hand_landmarker.task
└─ models/
```

## Propósito del proyecto

Este programa sirve para:
- reconocer señas de manera interactiva
- registrar nuevas muestras
- entrenar el sistema con datos propios
- probar detección y seguimiento de manos en tiempo real

## Observación final

La app no es solo un visor de cámara: integra captura, análisis, almacenamiento y entrenamiento en una sola interfaz.
