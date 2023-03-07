import cv2
import numpy as np

# 영상과 웹캠 캡처 객체 생성
cap_video = cv2.VideoCapture('OMG_sml.mp4')
cap_webcam = cv2.VideoCapture(0)

# 원본 영상의 크기 가져오기
video_width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 윈도우 생성
cv2.namedWindow('2-screen display', cv2.WINDOW_NORMAL)

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

    # 좌우 분할
    h, w, _ = frame_webcam.shape
    new_w = int(video_height / h * w)
    frame = np.zeros((video_height, video_width + new_w, 3), dtype=np.uint8)
    frame[:, :video_width, :] = cv2.resize(frame_video, (video_width, video_height))
    frame[:, video_width:, :] = cv2.flip(cv2.resize(frame_webcam, (new_w, video_height)), 1)

    # 윈도우에 프레임 출력
    cv2.imshow('2-screen display', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
cap_video.release()
cap_webcam.release()
cv2.destroyAllWindows()
