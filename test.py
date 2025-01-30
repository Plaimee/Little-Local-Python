import cv2
import os
import socketio
import mediapipe as mp

SERVER_URL = 'http://localhost:3000'
DETECTION_CONFIDENCE = 0.6
ANNOTATED_FOLDER = "annotated_images"

sio = socketio.Client()
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

@sio.on("sendtoAI")
def sendtoAI(data):
    print("Received image file path:", data)

    data = os.path.abspath("testCouple.png")
    if os.path.exists(data):
        image = cv2.imread(data)
        if image is None:
            print(f"Failed to load image from {data}")
            return

        with mp_face_detection.FaceDetection(min_detection_confidence=DETECTION_CONFIDENCE) as face_detector:
            results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            keypoints_data_list = []
            if results.detections:
                for face_id, face in enumerate(results.detections):
                    bounding_box = face.location_data.relative_bounding_box
                    width, height = image.shape[1], image.shape[0]
                    x = int(bounding_box.xmin * width)
                    y = int(bounding_box.ymin * height)
                    w = int(bounding_box.width * width)
                    h = int(bounding_box.height * height)

                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    landmarks = face.location_data.relative_keypoints
                    keypoints_data = {
                        'face_id': face_id,
                        'left_eye': {"x": int(landmarks[0].x * width), "y": int(landmarks[0].y * height)},
                        'right_eye': {"x": int(landmarks[1].x * width), "y": int(landmarks[1].y * height)},
                        'nose': {"x": int(landmarks[2].x * width), "y": int(landmarks[2].y * height)},
                        'mouth': {"x": int(landmarks[3].x * width), "y": int(landmarks[3].y * height)}
                    }
                    keypoints_data_list.append(keypoints_data)

                    for kp_name, kp in keypoints_data.items():
                        if isinstance(kp, dict):  
                            cv2.circle(image, (kp["x"], kp["y"]), 7, (0, 255, 0), -1)
                            text = f"{kp_name}: ({kp['x']}, {kp['y']})"
                            cv2.putText(image, text, (kp["x"] + 10, kp["y"] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                output_path = os.path.join(ANNOTATED_FOLDER, f"annotated_{os.path.basename(data)}")
                cv2.imwrite(output_path, image)
                full_output_path = os.path.abspath(output_path)
                print(f"Annotated image saved to: {full_output_path}")

                sio.emit('annotatedImage', full_output_path)
                sio.emit('keypointsData', keypoints_data_list)

            else:
                print("No faces detected.")

    else:
        print(f"Image file does not exist at: {data}")

# Connect to the server
try:
    sio.connect(SERVER_URL)  # Replace with your server URL
    print("Connected to server.")
    sio.wait()
except Exception as e:
    print(f"Failed to connect to server: {e}")
