import cv2
import mediapipe as mp
import numpy as np
import time
import math

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


# Joint info
def draw_joint_info(frame, score):
    # Joint 번호와 이름 매칭
    # JOINTS = {0: 'nose',
    #           1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
    #           4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
    #           7: 'left_ear', 8: 'right_ear',
    #           9: 'mouth_left', 10: 'mouth_right',
    #           11: 'left_shoulder', 12: 'right_shoulder',
    #           13: 'left_elbow', 14: 'right_elbow',
    #           15: 'left_wrist', 16: 'right_wrist',
    #           17: 'left_pinky', 18: 'right_pinky',
    #           19: 'left_index', 20: 'right_index',
    #           21: 'left_thumb', 22: 'right_thumb',
    #           23: 'left_hip', 24: 'right_hip',
    #           25: 'left_knee', 26: 'right_knee',
    #           27: 'left_ankle', 28: 'right_ankle',
    #           29: 'left_heel', 30: 'right_heel',
    #           31: 'left_foot_index', 32: 'right_foot_index'}
    
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

# 영상과 웹캠 캡처 객체 생성
cap_video = cv2.VideoCapture('OMG_sml.mp4')
cap_webcam = cv2.VideoCapture(0)

# 원본 영상의 크기 가져오기
video_width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 윈도우 생성
cv2.namedWindow('2-screen display', cv2.WINDOW_NORMAL)

# MediaPipe Pose 객체 생성
mp_pose = mp.solutions.pose

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

        # Pose Landmark 시각화
        mp_drawing = mp.solutions.drawing_utils
        
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
    frame[:, video_width:(video_width+new_width), :] = cv2.flip(frame_webcam_resized, 1)

    # 윈도우에 프레임 출력
    cv2.imshow('2-screen display', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
cap_video.release()
cap_webcam.release()
cv2.destroyAllWindows()