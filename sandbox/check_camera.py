import cv2
import datetime


def check_camera_connection():
    """
    Check the connection between any camera and the PC.

    """

    print(f'[{datetime.datetime.now()}]', 'searching any camera...')
    true_camera_is = []  # 空の配列を用意

    # カメラ番号を0～9まで変えて、COM_PORTに認識されているカメラを探す
    for camera_number in range(0, 10):
        cap = cv2.VideoCapture(camera_number)
        ret, frame = cap.read()

        if ret is True:
            true_camera_is.append(camera_number)
            print("camera_number", camera_number, "Find!")

        else:
            print("camera_number", camera_number, "None")
    print("接続されているカメラは", len(true_camera_is), "台です。")


if __name__ == "__main__":
    check_camera_connection()