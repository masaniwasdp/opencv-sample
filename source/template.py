"""複数物体のテンプレートマッチングを行う。

Date: 2017/8/2
Authors: masaniwa
"""

from sys import argv
from numpy import where

from timer import timer

import cv2


def find_patterns(frame, template):
    """画像からテンプレートと似たパターンを探す。

    Params:
        frame = 画像。
        template = テンプレート画像。

    Returns: パターンの座標のリスト。
    """

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

    return [(x, y) for y, x in zip(*where(result > 0.97))]


def draw_rects(frame, points, size, color):
    """複数の矩形を描画する。

    Params:
        frame = 描画する画像。
        points = 矩形の左上の点のリスト。
        size = 矩形のサイズ。
        color = 矩形の色。
    """

    for upper_left in points:
        lower_right = upper_left[0] + size[0], upper_left[1] + size[1]

        cv2.rectangle(frame, upper_left, lower_right, color)


def main():
    if len(argv) < 3:
        return

    frame = cv2.imread(argv[1])
    template = cv2.imread(argv[2])

    if frame is None or template is None:
        return

    points = timer(find_patterns, frame, template)

    size = template.shape[1], template.shape[0]

    draw_rects(frame, points, size, color=(0, 255, 0))

    cv2.imshow("Frame", frame)
    cv2.imshow("Template", template)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
