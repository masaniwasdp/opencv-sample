"""画像からカードを抽出する。

Date: 2017/8/2
Authors: masaniwa
"""

from numpy import int32
from numpy.linalg import norm

from carddetection import close, detect_card
from edgedetection import apply_canny
from timer import timer

import cv2


def extract_card(frame, contour):
    """画像からカードを抽出する。

    Params:
        frame = 画像。
        contour = カードの輪郭の頂点情報。

    Returns: 抽出した画像。
    """

    h, w = frame.shape[0:2]

    arranged = arrange(contour)

    rect = create_rect(0, 0, w, h)

    matrix, _ = cv2.findHomography(arranged, rect)

    card = cv2.warpPerspective(frame, matrix, (w, h))

    return cv2.resize(card, calc_size(arranged))


def arrange(contour):
    """四角形の輪郭の頂点を左上から反時計回りに並べ替える。

    Params:
        contour = 輪郭の頂点情報。

    Returns: 並べ替えた輪郭の頂点情報。
    """

    body = contour[:, 0, :]

    arranged = sorted(body, key=lambda x: x[0])

    l_pts = sorted(arranged[:2], key=lambda x: x[1])
    r_pts = sorted(arranged[2:], key=lambda x: x[1], reverse=True)

    return int32(l_pts + r_pts)


def create_rect(x, y, w, h):
    """頂点が左上から反時計回りに並んだ矩形を作る。

    Params:
        x = 左上のx座標。
        y = 左上のy座標。
        w = 幅。
        h = 高さ。

    Returns: 矩形の頂点情報。
    """

    u = x + w
    v = y + h

    return int32([[x, y], [x, v], [u, v], [u, y]])


def calc_size(points):
    """左上から反時計回りに並んだ頂点から矩形のサイズを計算する。

    Params:
        points = 頂点のリスト。

    Returns: サイズ。
    """

    w = norm(points[0] - points[3])
    h = norm(points[0] - points[1])

    return int(w), int(h)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        area = frame.shape[0] * frame.shape[1]

        area_range = area * 0.025, area * 0.25

        edge = close(apply_canny(frame))

        contour = detect_card(edge, area_range)

        if contour is not None:
            card = timer(extract_card, frame, contour)

            cv2.polylines(frame, [contour], True, color=(0, 255, 0))

            cv2.imshow("Input", frame)
            cv2.imshow("Card", card)

        else:
            cv2.imshow("Input", frame)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
