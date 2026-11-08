"""カラー画像をヒストグラム平坦化する。

Date: 2017/8/2
Authors: masaniwa
"""

from functools import reduce

from blending import blend
from histograms import create_color_frame, draw, get_histograms
from timer import timer

import cv2


def apply_histogram(frame):
    """カラー画像をヒストグラム平坦化する。

    Params:
        frame = カラー画像。

    Returns: 平坦化した画像。
    """

    # BGR の各成分を個別に平坦化すると色のバランスが崩れるので、
    # HSV の V についてヒストグラム平坦化して BGR に戻す。
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_b = create_color_frame(w=256, h=128)
    frame_g = create_color_frame(w=256, h=128)
    frame_r = create_color_frame(w=256, h=128)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        corrected = timer(apply_histogram, frame)

        hist_b, hist_g, hist_r = get_histograms(corrected)

        draw(hist_b, frame_b, color=(255, 0, 0))
        draw(hist_g, frame_g, color=(0, 255, 0))
        draw(hist_r, frame_r, color=(0, 0, 255))

        frame_bgr = reduce(blend, [frame_b, frame_g, frame_r])

        cv2.imshow("Input", frame)
        cv2.imshow("Corrected", corrected)
        cv2.imshow("Histogram(Blue)", frame_b)
        cv2.imshow("Histogram(Green)", frame_g)
        cv2.imshow("Histogram(Red)", frame_r)
        cv2.imshow("Histogram", frame_bgr)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
