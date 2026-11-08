"""画像をガンマ補正する。

Date: 2017/8/2
Authors: masaniwa
"""

from numpy import uint8

from timer import timer

import cv2


def apply_gamma(frame):
    """画像をガンマ補正する。

    Params:
        frame = 画像。

    Returns: 補正した画像。
    """

    table = uint8([calc_gamma(x) for x in range(256)])

    return cv2.LUT(frame, table)


def calc_gamma(x):
    """ガンマ補正のルックアップテーブルの値を計算する。

    Params:
        x = x。

    Returns: 計算値。
    """

    return 255 * (x / 255) ** (1 / 2)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        corrected = timer(apply_gamma, frame)

        cv2.imshow("Input", frame)
        cv2.imshow("Corrected", corrected)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
