"""画像を合成する。

Date: 2017/8/2
Authors: masaniwa
"""

from binary import make_binary
from timer import timer

import cv2


def blend(lhs, rhs):
    """画像を合成する。

    Params:
        lhs = 画像 A。
        rhs = 画像 B。

    Returns: 合成した画像。
    """

    return cv2.addWeighted(src1=lhs, alpha=0.5, src2=rhs, beta=0.5, gamma=0)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        # 入力画像とその二値画像を合成する。
        binary = cv2.cvtColor(make_binary(frame), cv2.COLOR_GRAY2BGR)

        blended = timer(blend, frame, binary)

        cv2.imshow("Input", frame)
        cv2.imshow("Blended", blended)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
