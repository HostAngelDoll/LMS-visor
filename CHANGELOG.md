# Changelog - LSM-Visor

Todos los cambios notables en este proyecto serán documentados en este archivo.

## [1.2.0] - 2024-05-20
### Añadido
- Integración de modelo **MLP (Perceptrón Multicapa)** usando scikit-learn para mejorar la precisión del reconocimiento estático.
- Sistema de **entrenamiento en caliente** (F11) que permite re-entrenar la IA sin reiniciar la aplicación.
- Soporte para **Webcams convencionales** a través de OpenCV, permitiendo el uso del sistema sin hardware OAK-D.
- Funcionalidad de **Captura de Pantalla** automática guardada en la carpeta de Documentos.
- Navegación circular de letras en la interfaz y selección mediante teclado (A-Z).
- Logs detallados con colores en la UI para mejor depuración.

## [1.1.0] - 2024-04-15
### Añadido
- Nueva interfaz gráfica profesional construida con **PyQt6**.
- Implementación de **QThreads** para el procesamiento de cámara y entrenamiento, evitando bloqueos en la UI.
- Sistema de **Triggers automáticos** para gestos dinámicos (ej. 'P' activa seguimiento para 'K').
- Cálculo de agregados estadísticos automáticos al guardar muestras estáticas.

## [1.0.0] - 2024-03-01
### Añadido
- Versión inicial del sistema con soporte para cámara **OAK-D (DepthAI)**.
- Integración de **MediaPipe Hand Landmarker** para detección de 21 puntos.
- Lógica de reconocimiento basada en **Reglas Heurísticas** (geometría de dedos).
- Sistema de grabación de gestos estáticos y dinámicos (5 segundos).
- Herramienta **Pincel (pencil.py)** para visualización de trayectorias.
- Estructura modular del proyecto (camera, processor, logic, tracker, recorder).

---
*Nota: Las fechas son ilustrativas para representar el progreso del desarrollo.*
