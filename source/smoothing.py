"""画像を平滑化する。

Date: 2017/8/2
Authors: masaniwa
"""

from numpy import float32
from numpy.random import normal

from timer import timer

import cv2


def blur(frame):
    """正規化したボックスフィルタを適用する。

    Params:
        frame = 画像。

    Returns: フィルタを適用した画像。
    """

    return cv2.blur(frame, ksize=(3, 3))


def apply_gaussian(frame):
    """Gaussian フィルタを適用する。

    Params:
        frame = 画像。

    Returns: フィルタを適用した画像。
    """

    return cv2.GaussianBlur(frame, ksize=(3, 3), sigmaX=0, sigmaY=0)


def apply_median(frame):
    """Median フィルタを適用する。

    Params:
        frame = 画像。

    Returns: フィルタを適用した画像。
    """

    return cv2.medianBlur(frame, ksize=3)


def apply_bilateral(frame):
    """Bilateral フィルタを適用する。

    Params:
        frame = 画像。

    Returns: フィルタを適用した画像。
    """

    return cv2.bilateralFilter(frame, d=-1, sigmaColor=25, sigmaSpace=25)


def apply_nl_means(frame):
    """Non-local Means フィルタを適用する。

    Params:
        frame = 画像。

    Returns: フィルタを適用した画像。
    """

    return cv2.fastNlMeansDenoisingColored(
        src=frame,
        dst=None,
        h=5,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def add_noise(frame, sigma):
    """ノイズを付加する。

    Params:
        frame = 画像。
        sigma = ノイズの強さ。

    Returns: ノイズを付加した画像。
    """

    noise = normal(0, sigma, frame.shape)

    return cv2.convertScaleAbs(float32(frame) + noise)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        noised = add_noise(frame, sigma=10)

        blured = timer(blur, noised)
        gaussian = timer(apply_gaussian, noised)
        median = timer(apply_median, noised)
        bilateral = timer(apply_bilateral, noised)
        nl_means = timer(apply_nl_means, noised)

        cv2.imshow("Noised", noised)
        cv2.imshow("Blured", blured)
        cv2.imshow("Gaussian", gaussian)
        cv2.imshow("Median", median)
        cv2.imshow("Bilateral", bilateral)
        cv2.imshow("NonLocalMeans", nl_means)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
