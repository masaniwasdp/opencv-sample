"""画像の振幅スペクトルを得る。

Date: 2017/8/2
Authors: masaniwa
"""

from numpy import float32, hamming, log10
from numpy.fft import fftshift

from timer import timer

import cv2


def optimaize(frame):
    """画像をフーリエ変換に適するサイズにリサイズする。

    Params:
        frame = 画像。

    Returns: リサイズした画像。
    """

    h, w = map(cv2.getOptimalDFTSize, frame.shape[0:2])

    return cv2.resize(frame, (w, h))


def make_spectrum(frame):
    """画像をフーリエ変換して振幅スペクトルを得る。

    Params:
        frame = 画像。

    Returns: 振幅スペクトル。
    """

    windowed = apply_hamming(float32(frame))

    dft = cv2.dft(windowed, flags=cv2.DFT_COMPLEX_OUTPUT)

    shift = fftshift(dft)

    amplitude = calc_magnitude(shift)

    return cv2.convertScaleAbs(20 * log10(amplitude))


def apply_hamming(frame):
    """画像に Hamming 窓を適用する。

    Params:
        frame = 画像。

    Returns: 窓を適用した画像。
    """

    rows, cols = frame.shape[0:2]

    window = make_hamming(rows, cols)

    return frame * window


def make_hamming(rows, cols):
    """二次元の Hamming 窓を作る。

    Params:
        rows = 行数。
        cols = 列数。

    Returns: 作成した窓。
    """

    row = hamming(cols)
    col = hamming(rows)

    return float32([row * i for i in col])


def calc_magnitude(dft):
    """強度を計算する。

    Params:
        dft = 離散フーリエ変換の結果。

    Returns: 強度。
    """

    re = dft[:, :, 0]
    im = dft[:, :, 1]

    return cv2.magnitude(re, im)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        optimaized = optimaize(gray)

        spectrum = timer(make_spectrum, optimaized)

        cv2.imshow("Input", frame)
        cv2.imshow("Spectrum", spectrum)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
