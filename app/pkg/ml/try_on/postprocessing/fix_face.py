import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
from app.pkg.logger import get_logger

logger = get_logger(__name__)

class FaceFixer:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        logger.info("FaceFixer inited")

    def fix_face(self, orig_image, result_image):
        orig_image_np = np.array(orig_image)
        mask = self.get_face_mask(orig_image_np)
        pil_mask = Image.fromarray(mask)
        result_image.paste(im=orig_image,
                           box=(0,0),
                           mask=pil_mask)
        return result_image

    def get_face_mask(self, image):

        face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    
        # Detect faces in the image
        results_detection = face_detection.process(image)

        # Initialize a blank mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if results_detection.detections:
            for detection in results_detection.detections:
                logger.info("Found face on the image")
                # Get the bounding box of the face
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * image.shape[1]), int(bbox.ymin * image.shape[0]), \
                            int(bbox.width * image.shape[1]), int(bbox.height * image.shape[0])

                # Extract the face region from the image
                face_image = image[y:y+h, x:x+w]

                # Process the face region with MediaPipe Face Mesh
                results_mesh = face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

                if results_mesh.multi_face_landmarks:
                    for face_landmarks in results_mesh.multi_face_landmarks:
                        # Get the landmark points for the face
                        landmarks = [(int(landmark.x * face_image.shape[1] + x), int(landmark.y * face_image.shape[0] + y))
                                    for landmark in face_landmarks.landmark]
                        landmarks = np.array(landmarks, dtype=np.int32)

                        # Create a convex hull from the landmark points
                        convexhull = cv2.convexHull(landmarks)

                        # Draw the convex hull on the mask
                        cv2.fillConvexPoly(mask, convexhull, 255)
        else:
            logger.info("Not found face on the image")
        return mask

