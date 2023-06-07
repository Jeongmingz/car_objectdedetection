# car_objectdedetection
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgjbae1212%2Fhit-counter)](https://hits.seeyoufarm.com)                    
#### 융합소프트웨어학과 융합프로젝트 팀프로젝트 (사조참치)
#### 개발 : 이정민 (Jeongmingz)

## 사용 환경 
#### Pycharm : Python3.10, opencv
#### [Roboflow API](https://universe.roboflow.com/project-jik0c/car-models-08rnv/model/4), RoboFlow Model (Coustom Dataset Used, 75 images + 122 Augmentation images)
---
## Our Object Detaction Model API (in Python)
### Code
``` python
from roboflow import Roboflow
rf = Roboflow(api_key="CFD5OGXqiYxYtTQoEBGo")
project = rf.workspace().project("car-models-08rnv")
model = project.version(4).model

# ===== "your_image.jpg" insert own image flie path ===== 
# infer on a local image
print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")
```


## 1. base_img_coordinate.py
``` python
import cv2
import json

# 전역 변수
ref_points = []
data = []
cnt = 0  # id값을 위한 cnt

# Base 이미지 로드
image = cv2.imread('imgs/base.png')

# 마우스 이벤트 처리를 위한 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global ref_points
    global cnt

    # 왼쪽 마우스 버튼이 눌려진 경우
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points = [(x, y)]
        cnt += 1

    # 왼쪽 마우스 버튼이 놓여진 경우
    elif event == cv2.EVENT_LBUTTONUP:
        ref_points.append((x, y))
        cv2.rectangle(image, ref_points[0], ref_points[1], (0, 255, 0), 2)
        cv2.imshow('image', image)

        # 사각형의 꼭짓점 좌표 출력
        print("Rectangle Corner Points:")
        print("Top Left: {}".format(ref_points[0]))
        print("Bottom Right: {}".format(ref_points[1]))

        # 좌표값을 JSON 파일에 저장
        save_coordinates(ref_points)

def save_coordinates(coordinates):
    global data
    # JSON 데이터
    data +=[{"id": cnt, "coordinate": [coordinates[0],coordinates[1]]}]
    print(data)
    # JSON 데이터를 파일로 저장


# coordinates값이 들어있는 Json 파일을 Read해서 
with open('coordinate/coordinates.json', 'r') as file:
    data = json.load(file)

# 만약 그 data가 아무것도 없다면 좌표값을 생성하는 함수들을 실행한다.
if data == '':
    # 윈도우 생성 및 마우스 콜백 함수 등록
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    # 이미지를 화면에 표시
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    with open('coordinate/coordinates.json', 'w') as file:
        json.dump(data, file)
# 그렇지 않고 데이터들이 존재하면 그 좌표값들을 처리하는 코드를 실행한다.
else:
    base_img_label = []
    with open('coordinate/img_labeling_num.txt', 'w') as file:
        file.write("")
    for i, coordinates in enumerate(data):
        coordinates_list = coordinates.get('coordinate')
        x_list, y_list = zip(*coordinates_list)

        txt = str(x_list[0]) + " " + str(x_list[1]) + " " + str(y_list[0]) + " " + str(y_list[1]) + "\n"
        with open('coordinate/img_labeling_num.txt', 'a') as file:
            file.write(txt)

        base_img_label += [{"id": i+1, "x": x_list, "y": y_list}]
    with open('coordinate/img_labeling_list.json', 'w') as file:
        json.dump(base_img_label, file)
```

## 2. api.py
``` python
from roboflow import Roboflow
import json
import numpy as np

# Roboflow에서 발행한 api_key로 접근한다.
rf = Roboflow(api_key="CFD5OGXqiYxYtTQoEBGo")
project = rf.workspace().project("car-models-08rnv")
# 현재 car model의 버전이 4버전까지 만들어져 있다.
model = project.version(4).model

# 이미지를 Object Detection을 한 Data값을 Json 형태로 리턴한다.
data = model.predict("imgs/test.png", confidence=40, overlap=30).json()
# data => x, y, class, image_path ...
print(data)

# 이미지를 Object Detection을 한 결과를 jpg 파일로 리턴한다.
model.predict("imgs/test.png", confidence=40, overlap=30).save("prediction.jpg")

# data를 파일에 dump 시킨다.
with open('data/data.json', 'w') as file:
	json.dump(data, file)

# base_img_coordinate.py 파일에서 생성한 Base 좌표값 리스트를 들고 온다.
with open('coordinate/img_labeling_list.json', 'r') as file:
	coordinate = json.load(file)

# 총 23개의 주차자리에 차량이 탐지되었는지 확인하기 위한 0으로 구성된 ndarray 생성
is_parked = np.zeros(24, dtype="int")

# data에 있는 x, y좌표를 얻어오기 위한 dict 접근
for i, datas in enumerate(data.get("predictions")):
    # 객체의 x 좌표값
	x_coordinate = datas.get('x')
    # 객체의 ㅛ 좌표값
	y_coordinate = datas.get('y')
    
    # Base 좌표값 불러오기
	for i, coordinates in enumerate(coordinate):
		x_list, y_list = coordinates.get('x'), coordinates.get('y')
		# Data의 x, y좌표가 존재하는 공간의 id값에 해당하는 값을 True(1)로 변경
		if x_list[0] <= x_coordinate <= x_list[1]:
			if y_list[0] <= y_coordinate <= y_list[1]:
				is_parked[coordinates.get('id')] = True

print(is_parked)
dict_ = {"parkingInfo": is_parked.tolist()}
with open('result/result.json', 'w') as file:
	json.dump(dict_, file)
```
