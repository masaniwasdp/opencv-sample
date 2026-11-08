"""外郭を抽出する。

Date: 2017/8/2
Authors: masaniwa
"""

from edgedetection import apply_canny
from timer import timer

import cv2


def extract_outer_contours(frame):
    """外郭を抽出する。

    Params:
        frame = 画像。

    Returns: 抽出した外郭の点情報。
    """

    edge = apply_canny(frame)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        contours = timer(extract_outer_contours, frame)

        cv2.drawContours(frame, contours, contourIdx=-1, color=(0, 255, 0))

        cv2.imshow("Frame", frame)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
