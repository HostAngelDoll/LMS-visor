# LSM-Visor: Sistema de Reconocimiento de Lengua de Señas Mexicana

Este repositorio contiene un sistema avanzado de visión por computadora diseñado para el reconocimiento y grabación de la Lengua de Señas Mexicana (LSM). Utiliza la potencia del hardware de la cámara **OAK-D** o cualquier **Webcam** estándar, junto con la precisión de **MediaPipe** y un modelo de **Machine Learning (MLP)** para ofrecer una experiencia fluida de seguimiento y clasificación de gestos.

## 🚀 Características
- **Interfaz Moderna en PyQt6**: Panel de control intuitivo con visualización en tiempo real, logs detallados y controles de grabación.
- **Detección de Manos en Tiempo Real**: Procesamiento de 21 landmarks por mano con alta precisión mediante MediaPipe.
- **Reconocimiento Híbrido Avanzado**:
    1. **MLP (Perceptrón Multicapa)**: Clasificador entrenado con scikit-learn para máxima precisión.
    2. **Reglas Heurísticas**: Lógica geométrica para casos base.
    3. **Comparación Estadística**: Validación contra base de datos JSON.
- **Soporte Multicámara**: Selector dinámico para alternar entre cámaras OAK-D (Luxonis) y Webcams convencionales.
- **Entrenamiento Integrado**: Capacidad de re-entrenar el modelo de IA directamente desde la interfaz sin detener la aplicación.
- **Sistema de Grabación Inteligente**: Captura de gestos estáticos y dinámicos (trayectorias de 5 segundos) con generación automática de agregados estadísticos.
- **Feedback Visual (Pincel)**: Estelas de colores personalizadas para cada dedo que facilitan la visualización de gestos con movimiento.

## 🛠 Requisitos de Hardware
- **Cámara OAK-D (Opcional)**: El sistema está optimizado para el ecosistema DepthAI de Luxonis.
- **Webcam Estándar**: Soporte para cámaras integradas o USB mediante OpenCV.
- **Procesador**: Se recomienda un equipo capaz de mantener al menos 20-30 FPS para una detección fluida.

## 📦 Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repo>
   cd LSM-visor
   ```

2. **Instalar dependencias:**
   Se recomienda usar un entorno virtual de Python 3.10+.
   ```bash
   pip install opencv-python depthai mediapipe numpy keyboard PyQt6 scikit-learn joblib
   ```

3. **Modelo de MediaPipe:**
   Asegúrate de que el archivo `hand_landmarker.task` esté presente en el directorio raíz.

## 📂 Estructura del Repositorio

- **`main.py`**: El orquestador principal basado en PyQt6. Gestiona la UI, el ciclo de vida de la app y los eventos.
- **`camera_engine.py`**: Motor de captura que unifica el acceso a OAK-D y Webcams.
- **`hand_processor.py`**: Núcleo de procesamiento de visión. Calcula ángulos de flexión, distancias normalizadas y orientación espacial.
- **`gesture_logic.py`**: Cerebro del reconocimiento. Implementa el clasificador MLP y las reglas de negocio.
- **`training/`**: Scripts para el entrenamiento del modelo MLP (`train_static.py`).
- **`models/`**: Almacena el modelo entrenado (`static_model.joblib`) y el mapeo de clases.
- **`tracker.py`**: Implementa el seguimiento temporal de los dedos y las estelas visuales.
- **`recorder.py`**: Maneja la persistencia en `gestures.json` y `motion_gestures.json`.
- **`pencil.py`**: Herramienta de dibujo independiente para análisis de trayectorias.

## ⚙️ Funcionamiento Técnico

### Lógica de Reconocimiento
El sistema en `gesture_logic.py` prioriza el modelo **MLP (Neural Network)**. Si la confianza es baja (< 70%), recurre a reglas heurísticas y finalmente a la comparación estadística.

### Entrenamiento de la IA
Al presionar **F11**, el sistema toma todas las muestras almacenadas en `gestures.json`, las normaliza y entrena un nuevo modelo `MLPClassifier`. Una vez finalizado, el modelo se recarga en caliente sin necesidad de reiniciar la aplicación.

### Sistema de Disparadores (Triggers)
El flujo para gestos dinámicos es automático:
- Al detectar una base estática (ej. 'P'), se activa un "trigger" para la letra dinámica (ej. 'K').
- El `tracker` configura los dedos a seguir y el sistema queda listo para grabar con **F12**.

## ⌨️ Controles del Teclado

| Tecla | Acción |
| :--- | :--- |
| **`A - Z`** | Establece manualmente la letra objetivo para la grabación / navegación en el menú. |
| **`Enter`** | Captura y guarda una muestra **estática** (1.5 segundos de muestras). |
| **`F11`** | Inicia el **entrenamiento** del modelo de Machine Learning. |
| **`F12`** | Inicia una grabación de **movimiento** de 5 segundos. |
| **`Q`** / **`Esc`** | Sale de la aplicación de forma segura. |
| **`F1 - F5`** | (En `pencil.py`) Alterna rastros para: Meñique, Anular, Medio, Índice y Pulgar. |

## ❗ Solución de Problemas (Troubleshooting)

**1. Error de scikit-learn:**
Si el sistema indica que `scikit-learn` no pudo cargarse, el reconocimiento MLP se desactivará automáticamente. Asegúrate de tener instalada una versión compatible:
```bash
pip install --upgrade scikit-learn joblib
```

**2. Cámara no detectada:**
Si usas OAK-D, verifica la conexión USB-C. Para Webcams, asegúrate de que ninguna otra aplicación esté usando la cámara. El selector en la UI permite reintentar la conexión.

**3. Capturas de Pantalla:**
El botón de "Capturar Pantalla" guarda archivos PNG en la carpeta `Documentos/Capturas_LSM` de tu usuario.

---
Este proyecto busca cerrar la brecha entre la visión artificial y la accesibilidad, proporcionando una base sólida para herramientas de traducción de LSM.
