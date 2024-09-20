import cv2
import mediapipe as mp
import numpy as np

# Function to calculate similarity between two sets of pose landmarks
def calculate_similarity(landmarks1, landmarks2):
    """
    

    Returns:
    - similarity_score: float, similarity score between 0 and 1
    """

    assert landmarks1.shape == landmarks2.shape, "Landmark shapes must be the same"

    
    distances = np.sqrt(np.sum((landmarks1 - landmarks2)**2, axis=1))

   
    max_distance = np.sqrt(2)  
    similarity_score = np.mean(np.exp(-distances / max_distance))

    return similarity_score

# Load MediaPipe Pose and Gesture models
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
holistic = mp_holistic.Holistic(static_image_mode=True)


def detect_pose(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        
        landmarks = np.array([[lmk.x, lmk.y] for lmk in results.pose_landmarks.landmark])
        return landmarks, results.pose_landmarks
    else:
        return None, None


def detect_gestures(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    if results.pose_landmarks and results.left_hand_landmarks and results.right_hand_landmarks:
        
        left_hand_open = is_hand_open(results.left_hand_landmarks)
        right_hand_open = is_hand_open(results.right_hand_landmarks)
        
       
        if left_hand_open and right_hand_open:
            gesture = "Both hands open"
        elif left_hand_open:
            gesture = "Left hand open"
        elif right_hand_open:
            gesture = "Right hand open"
        else:
            gesture = "Both hands closed"
        
        return gesture
    else:
        return None


def is_hand_open(hand_landmarks):
   
    index_finger_tip = hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    index_finger_base = hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP]
    distance = np.linalg.norm(np.array(index_finger_tip) - np.array(index_finger_base))
    if distance > 0.1:  # Adjust threshold as needed
        return True
    else:
        return False


image1_path = r'C:\Users\DELL\Desktop\projects\pose estimation\human-pose-estimation-opencv-master\img1.jpg'
image2_path = r'C:\Users\DELL\Desktop\projects\pose estimation\human-pose-estimation-opencv-master\img2.jpg'

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)


landmarks1, pose1 = detect_pose(image1)
landmarks2, pose2 = detect_pose(image2)

# Check if pose estimation was successful for both images
if landmarks1 is not None and landmarks2 is not None:
    # Calculate similarity score
    similarity_score = calculate_similarity(landmarks1, landmarks2)
    
    # Define a threshold for similarity
    similarity_threshold = 0.85
    
    # Compare similarity score with threshold
    if similarity_score >= similarity_threshold:
        match_status = "Success: Poses match!"
    else:
        match_status = "Poses do not match."
    
    #  first image
    gesture1 = detect_gestures(image1)
    
   
    img1_output = image1.copy()
    img2_output = image2.copy()

    
    if pose1 is not None:
        mp_drawing = mp.solutions.drawing_utils
        # Draw the pose skeleton
        mp_drawing.draw_landmarks(
            img1_output, pose1, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,0), thickness=2))
        
    if pose2 is not None:
        mp_drawing.draw_landmarks(
            img2_output, pose2, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,0), thickness=2))


    cv2.imshow('Image 1 with Pose Estimation', img1_output)
    cv2.imshow('Image 2 with Pose Estimation', img2_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the match status and detected gesture
    print(match_status)
    if gesture1:
        print("Gesture in Image 1:", gesture1)
    else:
        print("Gesture detection failed for Image 1.")

else:
    print("Pose estimation failed for one or both images.")


pose.close()
holistic.close()
