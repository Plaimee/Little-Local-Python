import cv2
import os
import numpy as np
import socketio
import mediapipe as mp
from datetime import datetime

SERVER_URL = 'http://localhost:3001'
DETECTION_CONFIDENCE = 0.6
ANNOTATED_FOLDER = "images_annotated"
REMOVED_BG_FOLDER = "images_removed_bg"

sio = socketio.Client()
mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

# สร้างโฟลเดอร์ใหม่ทุกครั้งที่รันโปรแกรม
def setup_folders():
    for folder in [ANNOTATED_FOLDER, REMOVED_BG_FOLDER]:
        if os.path.exists(folder):
            # ลบไฟล์เก่าทั้งหมดในโฟลเดอร์
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            os.makedirs(folder)

def generate_unique_filename(base_path, prefix):
    # สร้างชื่อไฟล์ที่ไม่ซ้ำกันโดยใช้ timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.png"
    return os.path.join(base_path, filename)

def remove_background(image):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segment:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = segment.process(image_rgb)

        if result.segmentation_mask is None:
            print("Error: Segmentation mask is None")
            return image
        
        mask = result.segmentation_mask
        threshold = 0.8
        mask_3d = (mask > threshold).astype(np.uint8) * 255
        mask_3d = cv2.cvtColor(mask_3d, cv2.COLOR_GRAY2BGR)

        image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image_bgra[:, :, 3] = mask_3d[:, :, 0]

        return image_bgra

@sio.on("sendtoAI")
def sendtoAI(data):
    try:
        print("Received data: ", data)
        imagePath = data.get('path')
        max_results = data.get('max_results')
        
        if not (imagePath and isinstance(imagePath, str) and os.path.exists(imagePath)):
            print(f"Invalid image path: {imagePath}")
            return

        # อ่านรูปภาพใหม่ทุกครั้ง
        image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Failed to load image from {imagePath}")
            return
        
        # สร้างชื่อไฟล์ที่ไม่ซ้ำกัน
        removed_bg_path = generate_unique_filename(REMOVED_BG_FOLDER, "removedBG")
        
        # ประมวลผลรูปภาพและบันทึก
        image_no_bg = remove_background(image.copy())  # ใช้ copy เพื่อไม่ให้กระทบกับรูปต้นฉบับ
        cv2.imwrite(removed_bg_path, image_no_bg, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        full_removed_bg_path = os.path.abspath(removed_bg_path)
        print(f"Background removed and saved to: {full_removed_bg_path}")

        # ประมวลผลใบหน้า
        with mp_face_detection.FaceDetection(min_detection_confidence=DETECTION_CONFIDENCE) as face_detector:
            results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not results.detections:
                print("No faces detected.")
                sio.emit('nfd', 'No faces detected.')
                return

            annotated_image = image.copy()  # ใช้ copy สำหรับการวาดการตรวจจับ
            keypoints_data_list = []

            for face_id, face in enumerate(results.detections[:max_results]):
                bounding_box = face.location_data.relative_bounding_box
                width, height = image.shape[1], image.shape[0]
                
                x = int(bounding_box.xmin * width)
                y = int(bounding_box.ymin * height)
                w = int(bounding_box.width * width)
                h = int(bounding_box.height * height)

                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
                        cv2.circle(annotated_image, (kp["x"], kp["y"]), 7, (0, 255, 0), -1)
                        text = f"{kp_name}: ({kp['x']}, {kp['y']})"
                        cv2.putText(annotated_image, text, (kp["x"] + 10, kp["y"] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # บันทึกรูปที่มีการ annotate
            output_path = generate_unique_filename(ANNOTATED_FOLDER, "mark_green")
            cv2.imwrite(output_path, annotated_image)
            full_output_path = os.path.abspath(output_path)
            print(f"Annotated image saved to: {full_output_path}")

            # ส่งข้อมูลกลับไปยัง Unity
            sio.emit('removedBGImage', full_removed_bg_path)
            sio.emit('annotatedImage', full_output_path)
            sio.emit('keypointsData', keypoints_data_list)

    except Exception as e:
        print(f"Error processing image: {e}")

# เริ่มต้นโปรแกรม
if __name__ == "__main__":
    try:
        setup_folders()  # เรียกฟังก์ชันเพื่อเตรียมโฟลเดอร์
        sio.connect(SERVER_URL)
        print("Connected to server.")
        sio.wait()
    except Exception as e:
        print(f"Failed to connect to server: {e}")