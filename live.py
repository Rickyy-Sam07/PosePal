import cv2
import mediapipe as mp

# Initialize Mediapipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Mediapipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Mediapipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# Open the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect poses and hands.
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)

    # Draw pose annotations on the image.
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
        )

    # Draw hand annotations on the image.
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

    # Display the image.
    cv2.imshow('Pose and Hand Gesture Estimation', frame)

    # Break the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows.
cap.release()
cv2.destroyAllWindows()
