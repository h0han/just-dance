import cv2
import mediapipe as mp
import numpy as np

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

# 각 관절별 score 초기화
joint_scores = {
    "left_shoulder": 0,
    "right_shoulder": 0
}

# score를 증가시키기 위한 threshold 설정
score_threshold = 0.1

# 관절 움직임 스코어링
def calculate_score(pose_landmarks, prev_pose_landmarks):
    joint_scores = {
        "left_shoulder": 0,
        "right_shoulder": 0
    }
    if prev_pose_landmarks:
        # 이전 프레임과 비교하여 어깨가 움직였는지 확인
        if abs(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x - prev_pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x) > score_threshold:
            joint_scores["left_shoulder"] += 1
        if abs(pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - prev_pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) > score_threshold:
            joint_scores["right_shoulder"] += 1
    return joint_scores, pose_landmarks

# 관절 스코어 표시
def draw_joint_info(frame, joint_scores):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_color = (255, 255, 255)
    y_offset = 20
    for i, (joint_name, score) in enumerate(joint_scores.items()):
        text = f"{joint_name}: {score}"
        cv2.putText(frame, text, (10, (i+1)*y_offset), font, font_scale, text_color, thickness, cv2.LINE_AA)


# # 각 관절 이름 리스트
# joint_names = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

# # 각 관절의 스코어 초기값 0으로 설정
# joint_scores = [0.0] * len(joint_names)


# def draw_joints(frame, results):
#     h, w, _ = frame.shape

#     # 각 관절에 대해 스코어 저장
#     for i, landmark in enumerate(results.pose_landmarks.landmark):
#         joint_scores[i] = landmark.visibility

#     # 각 관절의 이름과 스코어 출력
#     for i, joint_name in enumerate(joint_names):
#         joint_score = joint_scores[i]
#         if joint_score > 0.5:
#             text = f"{joint_name}: {joint_score:.2f}"
#             text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)
#             x_pos = 10
#             y_pos = 10 + (text_size[1] + 5) * i
#             cv2.putText(frame, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=1)


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
        # webcam pose
        w_pose_results = pose.process(cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB))

        # Pose Landmark 시각화
        mp_drawing = mp.solutions.drawing_utils

        # frame_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(frame_video, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # calculate score
        score, prev_pose_landmarks = calculate_score(pose_results.pose_landmarks, prev_pose_landmarks)

        # draw joint info
        draw_joint_info(frame_video, score)
        
        # webcam pose
            # 색상 구별
        mp_drawing_styles = mp.solutions.drawing_styles
        line_color = (0, 255, 0)
        mp_drawing.draw_landmarks(frame_webcam, w_pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  connection_drawing_spec=mp_drawing_styles.DrawingSpec(color=line_color, thickness=2),
                                  )
        
        
        # TODO : skeleton 좌우 반전
        mp_drawing.draw_landmarks(frame_webcam, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


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
