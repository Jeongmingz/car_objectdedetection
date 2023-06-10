import cv2

def on_image_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Clicked coordinates: ({}, {})".format(x, y))

image_path = "imgs/base.jpg"  # 클릭할 이미지 파일 경로
image = cv2.imread(image_path)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", on_image_click)

while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()