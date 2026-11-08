"""画像を色反転する。

Date: 2017/8/2
Authors: masaniwa
"""

from timer import timer

import cv2


def invert(frame):
    """画像を色反転する。

    Params:
        frame = 画像。

    Returns: 色反転した画像。
    """

    return 255 - frame


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        inverse = timer(invert, frame)

        cv2.imshow("Input", frame)
        cv2.imshow("Inverse", inverse)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
