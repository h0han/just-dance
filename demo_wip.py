import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Get the number of landmarks
num_landmarks = len(mp.solutions.pose.PoseLandmark)

# Create a dictionary of landmark names
landmark_names = {}
for landmark in mp.solutions.pose.PoseLandmark:
    landmark_names[landmark] = landmark.name

# Scoring
def calculate_movement(joint, pose_landmarks, prev_pose_landmarks, score):
    if prev_pose_landmarks is not None:
        cur_joint = pose_landmarks.landmark[joint]
        prev_joint = prev_pose_landmarks.landmark[joint]
        if abs(cur_joint.x - prev_joint.x) > 0.005 or abs(cur_joint.y - prev_joint.y) > 0.005:
            score += 1
    return score, pose_landmarks

# Speed of the joint
def calculate_speed(joint, pose_landmarks, prev_pose_landmarks, speed):
    if prev_pose_landmarks is not None:
        cur_joint = pose_landmarks.landmark[joint]
        prev_joint = prev_pose_landmarks.landmark[joint]
        cur_time = time.time()
        prev_time = cur_time - 1
        dt = cur_time - prev_time
        dx = cur_joint.x - prev_joint.x
        dy = cur_joint.y - prev_joint.y
        dist = math.sqrt(dx**2 + dy**2)
        speed = dist / dt
        # print(dx, dy, dist, dt, speed)
    return speed, pose_landmarks

def calculate_similarity(video_pose_landmarks, webcam_pose_landmarks):
    similarity = 0
    
    # Check if webcam pose landmarks are not None
    if webcam_pose_landmarks is None:
        return similarity
    
    # Get the position of the left wrist joint in the video pose
    video_left_wrist = video_pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    
    # Get the position of the left wrist joint in the webcam pose
    webcam_left_wrist = webcam_pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    
    # Calculate the Euclidean distance between the two left wrist joint positions
    distance = math.sqrt((video_left_wrist.x - webcam_left_wrist.x)**2 + (video_left_wrist.y - webcam_left_wrist.y)**2)
    
    # Convert distance to similarity score using a linear function
    similarity = max(0, 1 - distance / 0.2)
    
    return similarity

# Joint info
def draw_joint_info(frame, score):
    # Joint ????????? score??? ????????? ????????? ??????
    joint_info = ''
    joint_info += f'l_w mv : {score} '
    # joint_info += f'left_wrist speed : {speed} px/sec'
    # joint_info += f'l_w speed : {m_count} mv/s'
    
    # ???????????? ????????? ?????? ??????
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)
    thickness = 2
    pos = (10, 30)
    cv2.putText(frame, joint_info, pos, font, font_scale, color, thickness, cv2.LINE_AA)

def w_joint_highlight(frame, webcam_pose_landmarks, width):
    if webcam_pose_landmarks is not None:
        webcam_left_wrist = webcam_pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        if webcam_left_wrist.visibility > 0.1:
            # Calculate the position of the left wrist joint in the frame
            left_wrist_x = int(width - webcam_left_wrist.x * frame.shape[1])
            left_wrist_y = int(webcam_left_wrist.y * frame.shape[0])
            # Draw a marker at the position of the left wrist joint
            cv2.drawMarker(frame, (left_wrist_x, left_wrist_y), (255, 255, 255), markerType=cv2.MARKER_STAR, markerSize=20, thickness=8, line_type=cv2.LINE_AA)

def v_joint_highlight(frame, video_pose_landmarks, width):
    if video_pose_landmarks is not None:
        video_left_wrist = video_pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        if video_left_wrist.visibility > 0.1:
            # Calculate the position of the left wrist joint in the frame
            left_wrist_x = int(width - video_left_wrist.x * frame.shape[1])
            left_wrist_y = int(video_left_wrist.y * frame.shape[0])
            # Draw a marker at the position of the left wrist joint
            cv2.drawMarker(frame, (left_wrist_x, left_wrist_y), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=8, line_type=cv2.LINE_AA)

import json

def calculate_similarity_joints(video_pose_landmarks, webcam_pose_landmarks, filename=None):
    similarities = {}
    
    # Check if webcam pose landmarks are not None
    if webcam_pose_landmarks is None:
        return similarities
    
    # Calculate the Euclidean distance between each pair of corresponding joints
    num_landmarks = len(mp.solutions.pose.PoseLandmark)
    for i in range(num_landmarks):
        landmark = mp.solutions.pose.PoseLandmark(i)
        if landmark not in landmark_names:
            continue
        video_joint = video_pose_landmarks.landmark[landmark]
        webcam_joint = webcam_pose_landmarks.landmark[landmark]
        distance = math.sqrt((video_joint.x - webcam_joint.x)**2 + (video_joint.y - webcam_joint.y)**2)
        # Convert distance to similarity score using a linear function
        similarity = max(0, 1 - distance / 0.2)
        similarities[landmark] = similarity
    
    if filename is not None:
        with open(filename, 'a') as f:
            json.dump(similarities, f)
            f.write('\n')
    
    return similarities



# ????????? ?????? ?????? ?????? ??????
cap_video = cv2.VideoCapture('OMG_sml.mp4')
cap_webcam = cv2.VideoCapture(0)

# ????????? ??????
cv2.namedWindow('2-screen display', cv2.WINDOW_NORMAL)

# MediaPipe Pose ?????? ??????
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ????????? ??????
joint = 19
score = 0
pose_landmarks = None
prev_pose_landmarks = None
speed = 0
prev_time = 0

while True:
    # ?????? ????????? ????????????
    ret, frame_video = cap_video.read()
    if not ret:
        cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # ?????? ????????? ????????????
    ret, frame_webcam = cap_webcam.read()
    if not ret:
        continue

    # fps ??????
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time

    cv2.putText(frame_video, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # MediaPipe Pose ??? ????????? skeleton estimation
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        # RGB??? ?????? ??? ?????????
        pose_results = pose.process(cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB))
        w_pose_results = pose.process(cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB))

        # Calculate the similarity score
        similarity = calculate_similarity(pose_results.pose_landmarks, w_pose_results.pose_landmarks)

        # Score ??????
        if prev_pose_landmarks is not None:
            pose_landmarks = pose_results.pose_landmarks
            score, prev_pose_landmarks = calculate_movement(joint, pose_landmarks, prev_pose_landmarks, score)
            # speed, prev_pose_landmarks = calculate_speed(joint, pose_landmarks, prev_pose_landmarks, speed)
            prev_pose_landmarks = pose_landmarks

        else:
            pose_landmarks = pose_results.pose_landmarks
            score, prev_pose_landmarks = calculate_movement(joint, pose_landmarks, pose_landmarks, score)
            # speed, prev_pose_landmarks = calculate_speed(joint, pose_landmarks, prev_pose_landmarks, speed)
            
        # Calculate similarities and save to JSON file
        similarities = calculate_similarity_joints(pose_results.pose_landmarks, w_pose_results.pose_landmarks, "sim.json")

        # cv2.putText(frame_video, f'Similarity: {similarity:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # cv2.putText(frame_video, f'MV/FPS: {int(score*(fps/4))}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Joint Info ??????
        draw_joint_info(frame_video, score)

        # Pose Landmark ?????????
        mp_drawing.draw_landmarks(frame_video, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame_webcam, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # webcam pose
            # ?????? ??????
        mp_drawing_styles = mp.solutions.drawing_styles
        line_color = (255, 0, 0)
        mp_drawing.draw_landmarks(frame_webcam, w_pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=line_color, thickness=2),)


    # ????????? ????????? ????????????
    video_width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ?????? ????????? ????????????
    webcam_width = int(cap_webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap_webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ?????? ???????????? ?????? ??????????????? ?????? ????????? ????????? ????????? ?????? ?????? ??????
    if webcam_width > webcam_height:
        new_width = int(video_height / webcam_height * webcam_width)
        new_height = video_height
    else:
        new_width = video_width
        new_height = int(video_width / webcam_width * webcam_height)

    frame_webcam_resized = cv2.resize(frame_webcam, (new_width, new_height))

    # ????????? ??? ????????? ??????????????? ?????????
    frame = np.zeros((video_height, video_width + new_width, 3), dtype=np.uint8)
    frame[:, :video_width, :] = frame_video
    frame[:, video_width:(video_width+new_width), :] = frame_webcam_resized[:, ::-1, :]

    # ?????? ?????? ?????? ?????? ??????
    w_joint_highlight(frame[:, video_width:(video_width+new_width), :], w_pose_results.pose_landmarks, new_width)
    v_joint_highlight(frame[:, video_width:(video_width+new_width), :], pose_results.pose_landmarks, new_width)

    # ???????????? ????????? ??????
    cv2.imshow('2-screen display', frame)

    # 'q' ?????? ????????? ??????
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ?????? ????????? ????????? ??????
cap_video.release()
cap_webcam.release()
cv2.destroyAllWindows()

# JSON ?????? ??????
"sim.json".close()