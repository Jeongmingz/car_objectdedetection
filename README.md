# car_objectdedetection
#### 융합소프트웨어학과 융합프로젝트 팀프로젝트 (사조참치)
#### 개발 : 이정민 (Jeongmingz)

## 사용 환경 
#### Pycharm : Python3.10, opencv
#### [Roboflow API](https://universe.roboflow.com/project-jik0c/car-models-08rnv/model/4), RoboFlow Model (Coustom Dataset Used, 75 images + 122 Augmentation images)

### Our Object Detaction Model API (in Python)
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
