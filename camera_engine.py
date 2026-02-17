# camera_engine.py
# Clase para gestionar la cámara OAK-D y el flujo de frames usando DepthAI.
# Diseñado para estudiantes: maneja la inicialización y obtención de imágenes.

import depthai as dai
import cv2

class CameraEngine:
    """
    Esta clase se encarga de configurar y gestionar la comunicación con la cámara OAK-D.
    Utiliza la librería depthai para crear un pipeline que captura video en color.
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = dai.Pipeline()
        self._setup_pipeline()
        self.device = None
        self.q_rgb = None

    def _setup_pipeline(self):
        """Configura los nodos de la cámara y la salida hacia el host (computadora)."""
        # Crear nodo de cámara de color
        self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        self.cam_rgb.setPreviewSize(self.width, self.height)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.cam_rgb.setFps(self.fps)

        # Crear salida XLink para enviar los frames al host
        self.xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        self.xout_rgb.setStreamName("rgb")
        self.cam_rgb.preview.link(self.xout_rgb.input)

    def start(self):
        """Inicia la conexión con el dispositivo físico OAK-D."""
        self.device = dai.Device(self.pipeline)
        # Cola de salida para obtener los frames. maxSize=4 y blocking=False para fluidez.
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
        print("Cámara OAK-D iniciada correctamente.")

    def get_frame(self):
        """
        Obtiene el frame más reciente de la cámara.
        Retorna el frame en formato BGR de OpenCV y una versión espejada.
        """
        in_rgb = self.q_rgb.get()
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            # Espejamos el frame para que actúe como un espejo (común en aplicaciones de UI)
            frame_mirrored = cv2.flip(frame, 1)
            return frame_mirrored
        return None

    def stop(self):
        """Cierra la conexión con la cámara."""
        if self.device:
            self.device.close()
            print("Cámara OAK-D detenida.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
