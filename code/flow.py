import picamera, json, cv2
from roboflow import Roboflow
import numpy as np
from time import sleep

# 카메라 객체 생성
camera = picamera.PiCamera()

# 변수 선언
file_path = 'result/imgs/origin.jpg'
save_path = 'result/imgs/warpping.jpg'
w = 640
h = 480
is_parked = np.zeros(24, dtype="int")


# 카메라 설정

camera.resolution = (w, h)  # 해상도 설정
camera.rotation = 180  # 카메라 회전 설정

# Car Object Detaction Model init - api.py
rf = Roboflow(api_key="CFD5OGXqiYxYtTQoEBGo")
project = rf.workspace().project("car-models-08rnv")
model = project.version(4).model

# 이미지 캡처
while 1:
    try:
        # camera img save
        camera.capture(file_path)
        sleep(1)

        # img warpping - houghline
        img = cv2.imread(file_path)

        location = np.array([[113, 91], [520, 91], [603, 414], [28, 421]], np.float32)
        location2 = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

        pers = cv2.getPerspectiveTransform(location, location2)
        dst = cv2.warpPerspective(img, pers, (w, h))

        cv2.imwrite(save_path, dst)

        # img modeling - api.py

        data = model.predict("save_path", confidence=40, overlap=30).json()

        with open('coordinate/img_labeling_list.json', 'r') as file:
            coordinate = json.load(file)

        for i, datas in enumerate(data.get("predictions")):
            x_coordinate = datas.get('x')
            y_coordinate = datas.get('y')
            for i, coordinates in enumerate(coordinate):
                x_list, y_list = coordinates.get('x'), coordinates.get('y')

                if x_list[0] <= x_coordinate <= x_list[1]:
                    if y_list[0] <= y_coordinate <= y_list[1]:
                        is_parked[coordinates.get('id')] = True

        dict_ = {"parkingInfo": is_parked.tolist()}
        with open('result/result.json', 'w') as file:
            json.dump(dict_, file)
    except KeyboardInterrupt:
        break
    break

# 카메라 객체 종료
camera.close()
