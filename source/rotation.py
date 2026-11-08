"""画像を回転する。

Date: 2017/8/2
Authors: masaniwa
"""

from timer import timer

import cv2


def rotate(frame, angle):
    """画像を回転する。

    Params:
        frame = 画像。
        angle = 回転角度。

    Returns: 回転した画像。
    """

    size = frame.shape[1], frame.shape[0]

    center = size[0] / 2, size[1] / 2

    matrix = cv2.getRotationMatrix2D(center, angle, scale=1)

    return cv2.warpAffine(frame, matrix, size, flags=cv2.INTER_CUBIC)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    angle = 0

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        angle = (angle + 5) % 360

        rotated = timer(rotate, frame, angle)

        cv2.imshow("Input", frame)
        cv2.imshow("Rotated", rotated)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
