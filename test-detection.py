import mediapipe as mp
import cv2

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Access the webcam
webcam = cv2.VideoCapture(0)

# Use Face Detection with valid parameters
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    frame_count = 0
    frame_skip = 3  # Skip every 2 frames
    while webcam.isOpened():
        success, img = webcam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Convert image to RGB for MediaPipe
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img)

        # Convert back to BGR for OpenCV rendering
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Image dimensions
        h, w, _ = img.shape

        # Draw the face detection annotations on the image
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)

                # Extract keypoints
                keypoints = detection.location_data.relative_keypoints
                if keypoints:
                    for i, keypoint in enumerate(keypoints):
                        x = int(keypoint.x * w)  # Convert normalized x to pixel x
                        y = int(keypoint.y * h)  # Convert normalized y to pixel y
                        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw keypoints
                        cv2.putText(img, f"Point {i}: ({x}, {y})", (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        print(f"Keypoint {i}: x={x}, y={y}")

        # Display the resulting frame
        cv2.imshow("Detection", img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

# Cleanup
webcam.release()
cv2.destroyAllWindows()

