from utils.mediapipe.defaults import mp_pose, mp_hands

pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5)

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7)