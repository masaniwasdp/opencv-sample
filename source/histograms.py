"""カラー画像のヒストグラムを取得する。

Date: 2017/8/2
Authors: masaniwa
"""

from functools import reduce
from numpy import uint8, zeros

from blending import blend
from timer import timer

import cv2


def get_histogram(frame):
    """グレースケール画像のヒストグラムを得る。

    Params:
        frame = グレースケール画像。

    Returns: 0-128 に正規化された長さ 256 のヒストグラム。
    """

    histogram = cv2.calcHist([frame], [0], mask=None, histSize=[256], ranges=[0, 256])

    cv2.normalize(histogram, histogram, alpha=0, beta=128, norm_type=cv2.NORM_MINMAX)

    return histogram


def get_histograms(frame):
    """カラー画像のヒストグラムを得る。

    Params:
        frame = カラー画像。

    Returns: 0-128 に正規化された長さ 256 のヒストグラム。
    """

    histogram_b = get_histogram(frame[:, :, 0])
    histogram_g = get_histogram(frame[:, :, 1])
    histogram_r = get_histogram(frame[:, :, 2])

    return histogram_b, histogram_g, histogram_r


def draw(histogram, frame, color):
    """画像にヒストグラムを描画する。

    Params:
        histogram = ヒストグラム。
        frame = 描画する画像。
        color = ヒストグラムの色。
    """

    _, max_value, _, _ = cv2.minMaxLoc(histogram)

    h, w = frame.shape[0:2]

    frame *= 0

    for i, value in enumerate(histogram):
        x = int(i / histogram.size * w)
        y = int((1 - value / max_value) * h)

        cv2.line(frame, (x, h), (x, y), color)


def create_color_frame(w, h):
    """空の画像を作成する。

    Params:
        w = 幅。
        h = 高さ。

    Returns: 空の画像。
    """

    return zeros((h, w, 3), uint8)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_b = create_color_frame(w=256, h=128)
    frame_g = create_color_frame(w=256, h=128)
    frame_r = create_color_frame(w=256, h=128)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        hist_b, hist_g, hist_r = timer(get_histograms, frame)

        timer(draw, hist_b, frame_b, (255, 0, 0))
        timer(draw, hist_g, frame_g, (0, 255, 0))
        timer(draw, hist_r, frame_r, (0, 0, 255))

        frame_bgr = reduce(blend, [frame_b, frame_g, frame_r])

        cv2.imshow("Input", frame)
        cv2.imshow("Histogram(Blue)", frame_b)
        cv2.imshow("Histogram(Green)", frame_g)
        cv2.imshow("Histogram(Red)", frame_r)
        cv2.imshow("Histogram", frame_bgr)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
