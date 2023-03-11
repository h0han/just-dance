# Just Dance (Work in Progress)

## Features implemented
- Split video and webcam streams to view the user's screen.
- Perform pose estimation on the reference video.
- Perform pose estimation on the webcam stream.
- View the pose estimation results on the webcam stream.
- Calculate and view movement for specific joints.
- Compare the similarity between the reference video's skeleton and the webcam stream.

## Features to be implemented
- Calculate the sensitivity of particular joints to movement.


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
