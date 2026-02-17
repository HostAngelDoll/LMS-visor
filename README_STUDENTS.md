# Guía para Estudiantes - Sistema de Reconocimiento de LSM

Este proyecto ha sido organizado para facilitar su estudio. A continuación se explica la arquitectura y el funcionamiento de cada componente.

## Estructura del Proyecto

El código se divide en módulos especializados:

1.  **`main.py` (Orquestador)**: Es el punto de entrada. Aquí se encuentra el bucle principal que conecta todos los componentes. Maneja la interfaz de usuario (HUD) y los eventos del teclado.
2.  **`camera_engine.py` (Captura)**: Gestiona la cámara OAK-D usando la librería `depthai`. Los estudiantes pueden aprender aquí cómo se configura un "pipeline" de procesamiento de hardware.
3.  **`hand_processor.py` (Visión y Geometría)**: Utiliza **MediaPipe** para detectar los 21 puntos (landmarks) de la mano. Contiene funciones matemáticas para calcular:
    *   Ángulos entre dedos.
    *   Distancias normalizadas (independientes de qué tan cerca esté la mano de la cámara).
    *   Dirección y rotación de la mano.
4.  **`gesture_logic.py` (IA y Reglas)**: Decide qué letra se está mostrando. Combina dos enfoques:
    *   **Heurística**: Reglas lógicas (ej. "si el índice está levantado y los demás no, es la D").
    *   **Comparación con Base de Datos**: Compara la mano actual con muestras grabadas previamente en `gestures.json`.
5.  **`tracker.py` (Seguimiento)**: Implementa el efecto de "estela" o "pincel". Muestra cómo usar estructuras de datos como `deque` para mantener un historial temporal de posiciones.
6.  **`recorder.py` (Persistencia)**: Maneja la lógica de grabación de 5 segundos y el guardado en archivos JSON.

## Conceptos Clave para Estudiar

### 1. MediaPipe Hand Landmarker
MediaPipe nos entrega coordenadas (x, y, z) normalizadas de 0 a 1.
*   **x, y**: Posición relativa al ancho y alto de la imagen.
*   **z**: Profundidad relativa (aproximada).

### 2. Normalización de Distancias
Para que el sistema reconozca gestos sin importar la distancia a la cámara, dividimos todas las distancias por el "tamaño de la palma" (distancia entre la muñeca y la base del dedo medio).

### 3. Sistema de Disparadores (Triggers)
El sistema automático de seguimiento funciona así:
*   Se reconoce una letra estática (ej. 'I').
*   Se busca en `TRIGGER_MAP` si esa letra debe iniciar un seguimiento.
*   Si existe, se activan los dedos específicos (ej. meñique para la 'J') y se prepara el sistema para grabar al pulsar F12.

### 4. Manejo de Eventos
El sistema reacciona a:
*   **Enter**: Graba el estado estático actual.
*   **F12**: Graba una secuencia de movimiento de 5 segundos.
*   **Q**: Sale de la aplicación.

## Tareas Sugeridas para Estudiantes
1.  Modificar los colores de las estelas en `tracker.py`.
2.  Agregar una nueva regla heurística en `gesture_logic.py` para una letra nueva.
3.  Implementar un suavizado diferente en `hand_processor.py`.
