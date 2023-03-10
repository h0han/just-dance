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
    # Joint 이름과 score를 출력할 문자열 생성
    joint_info = ''
    joint_info += f'l_w mv : {score} '
    # joint_info += f'left_wrist speed : {speed} px/sec'
    # joint_info += f'l_w speed : {m_count} mv/s'
    
    # 문자열을 이미지 상에 표시
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


# 영상과 웹캠 캡처 객체 생성
cap_video = cv2.VideoCapture('OMG_sml.mp4')
cap_webcam = cv2.VideoCapture(0)

# 윈도우 생성
cv2.namedWindow('2-screen display', cv2.WINDOW_NORMAL)

# MediaPipe Pose 객체 생성
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 초기값 설정
joint = 19
score = 0
pose_landmarks = None
prev_pose_landmarks = None
speed = 0
prev_time = 0

while True:
    # 영상 프레임 읽어오기
    ret, frame_video = cap_video.read()
    if not ret:
        cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 웹캠 프레임 읽어오기
    ret, frame_webcam = cap_webcam.read()
    if not ret:
        continue

    # fps 계산
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time

    cv2.putText(frame_video, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # MediaPipe Pose 를 사용한 skeleton estimation
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        # RGB로 변환 후 넘겨줌
        pose_results = pose.process(cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB))
        w_pose_results = pose.process(cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB))

        # Calculate the similarity score
        similarity = calculate_similarity(pose_results.pose_landmarks, w_pose_results.pose_landmarks)

        # Score 계산
        if prev_pose_landmarks is not None:
            pose_landmarks = pose_results.pose_landmarks
            score, prev_pose_landmarks = calculate_movement(joint, pose_landmarks, prev_pose_landmarks, score)
            # speed, prev_pose_landmarks = calculate_speed(joint, pose_landmarks, prev_pose_landmarks, speed)
            prev_pose_landmarks = pose_landmarks

        else:
            pose_landmarks = pose_results.pose_landmarks
            score, prev_pose_landmarks = calculate_movement(joint, pose_landmarks, pose_landmarks, score)
            # speed, prev_pose_landmarks = calculate_speed(joint, pose_landmarks, prev_pose_landmarks, speed)
            
        cv2.putText(frame_video, f'Similarity: {similarity:.2f}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_video, f'MV/FPS: {int(score*(fps/4))}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Joint Info 표기
        draw_joint_info(frame_video, score)

        # Pose Landmark 그리기
        mp_drawing.draw_landmarks(frame_video, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame_webcam, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # webcam pose
            # 색상 구별
        mp_drawing_styles = mp.solutions.drawing_styles
        line_color = (255, 0, 0)
        mp_drawing.draw_landmarks(frame_webcam, w_pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=line_color, thickness=2),)


    # 비디오 해상도 가져오기
    video_width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 웹캠 해상도 가져오기
    webcam_width = int(cap_webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap_webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 웹캠 프레임의 비율 유지하면서 원본 비디오 프레임 크기에 맞게 크기 조정
    if webcam_width > webcam_height:
        new_width = int(video_height / webcam_height * webcam_width)
        new_height = video_height
    else:
        new_width = video_width
        new_height = int(video_width / webcam_width * webcam_height)

    frame_webcam_resized = cv2.resize(frame_webcam, (new_width, new_height))

    # 나머지 빈 영역을 검은색으로 채우기
    frame = np.zeros((video_height, video_width + new_width, 3), dtype=np.uint8)
    frame[:, :video_width, :] = frame_video
    frame[:, video_width:(video_width+new_width), :] = frame_webcam_resized[:, ::-1, :]

    # 왼쪽 손목 관절 좌표 강조
        # webcam : white // video : green
    w_joint_highlight(frame[:, video_width:(video_width+new_width), :], w_pose_results.pose_landmarks, new_width)
    v_joint_highlight(frame[:, video_width:(video_width+new_width), :], pose_results.pose_landmarks, new_width)

    # 윈도우에 프레임 출력
    cv2.imshow('2-screen display', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
cap_video.release()
cap_webcam.release()
cv2.destroyAllWindows()