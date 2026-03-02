# LSM-Visor: Sistema de Reconocimiento de Lengua de Señas Mexicana

Este repositorio contiene un sistema avanzado de visión por computadora diseñado para el reconocimiento y grabación de la Lengua de Señas Mexicana (LSM). Utiliza la potencia del hardware de la cámara **OAK-D** junto con la precisión de **MediaPipe** para ofrecer una experiencia fluida de seguimiento y clasificación de gestos.

## 🚀 Características
- **Detección de Manos en Tiempo Real**: Procesamiento de 21 landmarks por mano con alta precisión.
- **Reconocimiento Híbrido**: Combina reglas heurísticas matemáticas con comparación estadística contra una base de datos JSON.
- **Sistema de Grabación Inteligente**: Grabación de gestos estáticos (landmarks) y dinámicos (trayectorias de 5 segundos) para entrenamiento.
- **Feedback Visual (Pincel)**: Estelas de colores personalizadas para cada dedo que facilitan la visualización de gestos con movimiento.
- **Arquitectura Modular**: Diseñado pedagógicamente para que tanto desarrolladores como estudiantes puedan entender y extender cada componente.

## 🛠 Requisitos de Hardware
- **Cámara OAK-D (Luxonis)**: El sistema está construido específicamente para el ecosistema DepthAI.
- **Conectividad**: Es necesario que la cámara esté conectada físicamente para que el flujo de video se inicie.

## 📦 Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repo>
   cd LMS-visor
   ```

2. **Instalar dependencias:**
   Se recomienda usar un entorno virtual de Python 3.10+.
   ```bash
   pip install opencv-python depthai mediapipe numpy keyboard
   ```

3. **Modelo de MediaPipe:**
   Asegúrate de que el archivo `hand_landmarker.task` esté presente en el directorio raíz.

## 📂 Estructura del Repositorio

El proyecto se organiza en módulos especializados:

- **`main.py`**: El orquestador principal. Gestiona el ciclo de vida de la aplicación, la interfaz de usuario (HUD) y la captura de eventos del teclado.
- **`camera_engine.py`**: Gestiona el pipeline de **DepthAI** y la comunicación con el hardware de la cámara OAK-D.
- **`hand_processor.py`**: Núcleo de procesamiento de visión. Calcula ángulos de flexión, distancias normalizadas (independientes de la escala) y orientación espacial de la mano.
- **`gesture_logic.py`**: Contiene la lógica de negocio para el reconocimiento. Define las reglas de clasificación y los disparadores (triggers) para gestos dinámicos.
- **`tracker.py`**: Implementa el seguimiento temporal de los dedos, gestionando las estelas visuales mediante colas (`deque`).
- **`recorder.py`**: Maneja la persistencia de datos, guardando muestras en `gestures.json` (estáticas) y `motion_gestures.json` (dinámicas).
- **`pencil.py`**: Herramienta especializada para el dibujo y seguimiento preciso de dedos, permitiendo alternar el rastro de cada dedo individualmente.

## ⚙️ Funcionamiento Técnico

### Lógica de Reconocimiento
El sistema en `gesture_logic.py` opera bajo dos modalidades:
1. **Reglas Heurísticas**: Identifica letras mediante la relación geométrica de los dedos (ej. si el meñique es el único extendido, se propone la 'I').
2. **Comparación Estadística**: Compara los "agregados" (promedios de distancias y estados) de la mano actual contra muestras grabadas previamente en el archivo JSON.

### Normalización de Datos
Para garantizar la robustez, el sistema utiliza la **distancia de la palma** (del punto 0 al 9 de MediaPipe) como factor de escala. Esto permite que el reconocimiento sea efectivo sin importar qué tan cerca o lejos esté la mano de la cámara.

### Sistema de Disparadores (Triggers)
El flujo para gestos dinámicos es automático:
- Al detectar una base estática (ej. 'P'), el sistema activa un "trigger" para una letra dinámica (ej. 'K').
- Esto configura automáticamente el `tracker` para seguir los dedos necesarios y prepara el grabador para la secuencia de movimiento.

## ⌨️ Controles del Teclado

| Tecla | Acción |
| :--- | :--- |
| **`A - Z`** | Establece manualmente la letra objetivo para la grabación. |
| **`Enter`** | Captura y guarda una muestra **estática** (instantánea). |
| **`F12`** | Inicia una grabación de **movimiento** de 5 segundos. |
| **`Q`** / **`Esc`** | Sale de la aplicación. |
| **`F1 - F5`** | (En `pencil.py`) Activa/Desactiva el rastro para Meñique, Anular, Medio, Índice y Pulgar respectivamente. |

---
Este proyecto busca cerrar la brecha entre la visión artificial y la accesibilidad, proporcionando una base sólida para herramientas de traducción de LSM.
