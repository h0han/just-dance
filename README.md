# Just Dance (Work in Progress)

## Features implemented
- View user screen by splitting video and webcam (streaming)
- Perform pose estimation for reference video
- Perform pose estimation for webcam (streaming)
- View the pose estimation result on the webcam
- Perform movement calculation and viewing for specific joints
- Compare the similarity between the skeleton of reference video and the webcam

## Features to be implemented
- Calculate the sensitivity of specific joints to movement


## Prerequisites
- A video to be used for demonstration (if you use the CPU, it's better to use shorter videos)
- Put your video path in `cap_video = cv2.VideoCapture('video.mp4')`

## Run
```
$ pyenv virtualenv 3.6 <the name of virtualenv>
```


```
$ pip3 install requirements.txt
```


```
$ python3 demo.py
```

## Result
<img width="100%" src="result/result.gif"/>
