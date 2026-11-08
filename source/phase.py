"""位相画像を得る。

Date: 2017/8/2
Authors: masaniwa
"""

from numpy import float32

from amplitude import calc_magnitude, optimaize
from timer import timer

import cv2


def make_phase(frame):
    """画像をフーリエ変換して位相画像を得る。

    Params:
        frame = 画像。

    Returns: 位相画像。
    """

    dft = cv2.dft(float32(frame), flags=cv2.DFT_COMPLEX_OUTPUT)

    magnitude = calc_magnitude(dft)

    dft[:, :, 0] /= magnitude
    dft[:, :, 1] /= magnitude

    phase = calc_magnitude(cv2.idft(dft))

    return cv2.convertScaleAbs(phase)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        optimaized = optimaize(gray)

        phase = timer(make_phase, optimaized)

        cv2.imshow("Input", frame)
        cv2.imshow("Phase", phase)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
