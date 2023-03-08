import cv2
import mediapipe as mp
import numpy as np

# Scoring
def calculate_score(pose_landmarks, prev_pose_landmarks, score):
    if prev_pose_landmarks is not None:
        for joint in [11, 12]:
            cur_joint = pose_landmarks.landmark[joint]
            prev_joint = prev_pose_landmarks.landmark[joint]
            if abs(cur_joint.x - prev_joint.x) > 0.05 or abs(cur_joint.y - prev_joint.y) > 0.05:
                score += 1
    return score, pose_landmarks

# Joint info
def draw_joint_info(frame, pose_landmarks, score):
    # Joint 번호와 이름 매칭
    JOINTS = {0: 'nose',
              1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
              4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
              7: 'left_ear', 8: 'right_ear',
              9: 'mouth_left', 10: 'mouth_right',
              11: 'left_shoulder', 12: 'right_shoulder',
              13: 'left_elbow', 14: 'right_elbow',
              15: 'left_wrist', 16: 'right_wrist',
              17: 'left_pinky', 18: 'right_pinky',
              19: 'left_index', 20: 'right_index',
              21: 'left_thumb', 22: 'right_thumb',
              23: 'left_hip', 24: 'right_hip',
              25: 'left_knee', 26: 'right_knee',
              27: 'left_ankle', 28: 'right_ankle',
              29: 'left_heel', 30: 'right_heel',
              31: 'left_foot_index', 32: 'right_foot_index'}
    
    # Joint 이름과 score를 출력할 문자열 생성
    joint_info = ''
    for idx, joint in JOINTS.items():
        if pose_landmarks.landmark[idx].visibility > 0.5:
            joint_info += f'{joint}: {score}'
            joint_info += '\n'
    
    # 문자열을 이미지 상에 표시
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 0, 0)
    thickness = 1
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
prev_pose_landmarks = None
score = 0

joint_names = ["Left shoulder", "Right shoulder"]
score = 0
prev_pose_landmarks = None

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

    # MediaPipe Pose 를 사용한 skeleton estimation
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # RGB로 변환 후 넘겨줌
        pose_results = pose.process(cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB))
        w_pose_results = pose.process(cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB))

        # Pose Landmark 시각화
        mp_drawing = mp.solutions.drawing_utils

        # Score 계산
        if prev_pose_landmarks is not None:
            score, prev_pose_landmarks = calculate_score(pose_results.pose_landmarks, prev_pose_landmarks)

        # draw joint info
        joint_scores = [0, 0]
        for i, joint in enumerate([0, 1]):
            if prev_pose_landmarks is not None:
                joint_scores[i] = draw_joint_info(frame_video, score, joint_names[i], prev_pose_landmarks.landmark[joint])
            
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
