"""画像のエッジ検出をする。

Date: 2017/8/2
Authors: masaniwa
"""

from timer import timer

import cv2


def apply_sobel_x(frame):
    """Sobel フィルタで横方向のエッジを検出する。

    Params:
        frame = 画像。

    Returns: 横方向のエッジ画像。
    """

    sobel = cv2.Sobel(frame, cv2.CV_32F, dx=1, dy=0, ksize=3)

    return cv2.convertScaleAbs(sobel)


def apply_sobel_y(frame):
    """Sobel フィルタで縦方向のエッジを検出する。

    Params:
        frame = 画像。

    Returns: 縦方向のエッジ画像。
    """

    sobel = cv2.Sobel(frame, cv2.CV_32F, dx=0, dy=1, ksize=3)

    return cv2.convertScaleAbs(sobel)


def apply_laplacian(frame):
    """Laplacian フィルタでエッジを検出する。

    Params:
        frame = 画像。

    Returns: エッジ画像。
    """

    laplacian = cv2.Laplacian(frame, cv2.CV_32F, ksize=3)

    return cv2.convertScaleAbs(laplacian)


def apply_morphology(frame):
    """モルフォロジー勾配でエッジを検出する。

    Params:
        frame = 画像。

    Returns: エッジ画像。
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3))

    return cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel, iterations=1)


def apply_canny(frame):
    """Canny 法でエッジを検出する。

    Params:
        frame = 画像。

    Returns: エッジ画像。
    """

    return cv2.Canny(frame, threshold1=75, threshold2=100)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sobel_x = timer(apply_sobel_x, gray)
        sobel_y = timer(apply_sobel_y, gray)
        laplacian = timer(apply_laplacian, gray)
        morphology = timer(apply_morphology, gray)
        canny = timer(apply_canny, frame)

        cv2.imshow("Input", frame)
        cv2.imshow("Sobel X", sobel_x)
        cv2.imshow("Sobel Y", sobel_y)
        cv2.imshow("Laplacian", laplacian)
        cv2.imshow("Morphology", morphology)
        cv2.imshow("Canny", canny)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
