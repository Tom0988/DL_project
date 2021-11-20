import cv2  # open_cv


def split(video, time):
    cap = cv2.VideoCapture(video)
    image = []  # 圖片的list
    c = 1
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    while ret:
        ret, frame = cap.read()

        if c % time == 0:
            image.append(frame)
        c += 1

    cap.release()
    return image


time_F = 15
video_name = input('enter video dec: ')
images = split('video_set/'+video_name, time_F)

n = 1
for i in images:
    cv2.imwrite('image_set/image_2/' + str(n) + '.jpg', i)
    n += 1
