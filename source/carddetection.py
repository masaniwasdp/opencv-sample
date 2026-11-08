"""カードを検出する。

Date: 2017/8/2
Authors: masaniwa
"""

from edgedetection import apply_canny
from timer import timer

import cv2


def close(frame):
    """画像をクロージングする。

    Params:
        frame = 画像。

    Returns: クロージングした画像。
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3))

    return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=1)


def detect_card(edge, area_range):
    """カードを検出して直線近似する。

    Params:
        edge = エッジ画像。
        area_range = カードの面積の範囲。

    Returns: 直線近似した輪郭の頂点情報。検出できなければ None。
    """

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contour_list:
        e = 0.025 * cv2.arcLength(contour, True)

        approx_contour = cv2.approxPolyDP(contour, e, True)

        if is_card(approx_contour, area_range):
            return approx_contour

    return None


def is_card(approx_contour, area_range):
    """輪郭がカードかどうかを判定する。

    Params:
        approx_contour = 輪郭の点情報。
        area_range = カードの面積の範囲。

    Returns: そうであれば True。そうでなければ False。
    """

    if len(approx_contour) != 4:
        return False

    if not cv2.isContourConvex(approx_contour):
        return False

    area = cv2.contourArea(approx_contour)

    if not (area_range[0] <= area <= area_range[1]):
        return False

    return True


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        area = frame.shape[0] * frame.shape[1]

        area_range = area * 0.025, area * 0.25

        edge = close(apply_canny(frame))

        card = timer(detect_card, edge, area_range)

        if card is not None:
            cv2.polylines(frame, [card], True, color=(0, 255, 0))

        cv2.imshow("Input", frame)
        cv2.imshow("Edge", edge)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
