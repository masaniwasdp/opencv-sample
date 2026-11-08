"""画像を二値化する。

Date: 2017/8/2
Authors: masaniwa
"""

from timer import timer

import cv2


def make_binary(frame):
    """画像を二値化する。

    Params:
        frame = 画像。

    Returns: 二値画像。
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return cv2.adaptiveThreshold(
        src=gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=3,
        C=3,
    )


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        binary = timer(make_binary, frame)

        cv2.imshow("Input", frame)
        cv2.imshow("Binary", binary)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
