import cv2

image = cv2.imread("image/Co2.jpg")
image_width = image.shape[1]
image_height = image.shape[0]
print("中心点像素坐标", image_width/2, image_height/2)

def get_coordinate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("像素坐标 (x, y):", x, y)

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", get_coordinate)


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
