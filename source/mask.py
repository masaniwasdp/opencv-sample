"""画像にマスクをかける。

Date: 2017/8/2
Authors: masaniwa
"""

from binary import make_binary
from inverse import invert
from timer import timer

import cv2


def apply_mask(frame, mask):
    """画像にマスクをかける。

    Params:
        frame = 画像。
        mask = マスク。

    Returns: マスクをかけた画像。
    """

    return cv2.bitwise_and(frame, frame, mask=mask)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        # 入力画像にその反転二値画像をマスクする。
        inverse = invert(make_binary(frame))

        masked = timer(apply_mask, frame, inverse)

        cv2.imshow("Input", frame)
        cv2.imshow("Masked", masked)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
