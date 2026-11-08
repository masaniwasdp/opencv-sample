"""画像を減色する。

Date: 2017/8/2
Authors: masaniwa
"""

from numpy import float32, uint8

from timer import timer

import cv2


def reduce_colors(frame, k):
    """画像を減色する。

    Params:
        frame = 画像。
        k = 色数。

    Returns: 減色した画像。
    """

    array = float32(frame.reshape(-1, 3))

    # K-平均法で画像をクラスタリングすることで減色する。
    # ピクセルを k 色にクラスタリングする。
    _, labels, centers = cv2.kmeans(
        data=array,
        K=k,
        bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1),
        attempts=1,
        flags=cv2.KMEANS_RANDOM_CENTERS,
    )

    # クラスタリング結果から減色画像を生成する。
    reduced = uint8(centers)[labels.flatten()]

    return reduced.reshape(frame.shape)


def main():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while video.isOpened():
        result, frame = video.read()

        if not result:
            break

        reduced = timer(reduce_colors, frame, 32)

        cv2.imshow("Input", frame)
        cv2.imshow("Reduced", reduced)

        if cv2.waitKey(16) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
