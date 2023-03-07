import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# mediapipe를 사용하여 skeleton 추출
def get_skeleton(image):
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        # BGR을 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 추출을 위해 이미지를 처리
        image.flags.writeable = False
        results = pose.process(image)
        
        # skeleton을 이미지에 표시
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # skeleton의 좌표를 반환
        landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks

# 댄스 영상을 열어서 skeleton을 추출
dance_video_path = "OMG_sml.mp4"
cap_dance = cv2.VideoCapture(dance_video_path)
frames_dance = []
loading = 0
while True:
    print("Loading : " + str(loading))
    ret, frame = cap_dance.read()
    if not ret:
        break
    landmarks = get_skeleton(frame)
    frames_dance.append((frame, landmarks))
    loading += 1

# webcam 캡처
cap_webcam = cv2.VideoCapture(0)

while True:
    # 댄스 영상에서 프레임을 캡처
    ret_dance, frame_dance = cap_dance.read()
    ret_webcam, frame_webcam = cap_webcam.read()
    if not ret_dance:
        break
    
    # skeleton 추출
    landmarks_dance = get_skeleton(frame_dance)
    landmarks_webcam = get_skeleton(frame_webcam)
    
    # 좌우 분할
    height, width, _ = frame_dance.shape
    frame = cv2.resize

    # skeleton과 웹캠 이미지 비교
    if landmarks_dance and landmarks_webcam:
        # landmark가 존재하는 경우
        score = 0
        for i in range(len(landmarks_dance)):
            # Euclidean distance 계산
            dist = ((landmarks_dance[i][0] - landmarks_webcam[i][0]) ** 2
                    + (landmarks_dance[i][1] - landmarks_webcam[i][1]) ** 2
                    + (landmarks_dance[i][2] - landmarks_webcam[i][2]) ** 2) ** 0.5
            score += dist
        
        # 일치하는 정도 계산
        score /= len(landmarks_dance)  # 평균값 계산
        score = 1 - score  # 거리 차이가 적을수록 일치하는 정도가 높음
        
        # score에 따라 적절한 이미지 또는 색상으로 표시
        if score >= 0.8:
            cv2.putText(frame, f"Perfect! Score: {score:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif score >= 0.6:
            cv2.putText(frame, f"Good! Score: {score:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Not bad. Score: {score:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 좌우 화면 분할
    h, w, _ = frame.shape
    frame = cv2.resize(frame, (2 * w, h))
    frame_dance = cv2.resize(frame_dance, (w, h))
    frame[:, w:] = frame_dance
    
    # 프레임 화면에 표시
    cv2.imshow("Dance game", frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap_dance.release()
    cap_webcam.release()
    cv2.destroyAllWindows()