import cv2
import numpy as np
import os

directory = "/Users/ljeongmin/Documents/GitHub/school/car_objectdedetection/code/imgs"  # 파일명을 받을 디렉토리 경로

file_list = os.listdir(directory)

print("디렉토리 내 파일 목록:")
for file_name in file_list:
    file_path = "imgs/" + file_name
    save_path = "warp/" + file_name
    img = cv2.imread(file_path)
    print()

    # 사진 크기 설정
    w = 640
    h = 480
    # TR, TL, BL, BR
    location = np.array([[113, 91], [520, 91], [603, 414], [28, 421]], np.float32)
    location2 = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    pers = cv2.getPerspectiveTransform(location, location2)
    dst = cv2.warpPerspective(img, pers, (w,h))
    cv2.imshow('origin', img)
    cv2.imshow('result', dst)

    cv2.imwrite(save_path, dst)

    cv2.waitKey()
