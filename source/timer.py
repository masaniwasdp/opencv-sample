"""処理時間の計測。

Date: 2017/8/1
Authors: masaniwa
"""

from time import time


class TimerError(Exception):
    """処理時間計測に失敗した場合の例外。"""

    def __init__(self, func, msg):
        """初期化する。

        Params:
            func = 処理時間を計測しようとしていた関数。
            msg = 例外メッセージ。
        """

        self._func = func
        self._msg = msg

    def __str__(self):
        return "{}: {}".format(self._func.__name__, self._msg)


class Timer:
    """処理時間を計測するクラス。"""

    def __init__(self, name):
        """初期化する。

        Params:
            name = 処理の名前。
        """

        self._start = 0
        self._name = name

    def __enter__(self):
        self._start = time()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        proc_time = time() - self._start

        print("{}: {:.8f} sec.".format(self._name, proc_time))

        return False


def timer(func, *args):
    """関数の処理時間を計測する。

    Params:
        func = 関数。
        args = 関数に与える引数。

    Returns: 関数の戻り値。

    Throws:
        TimerError 関数が例外を投げた場合。
    """

    start = time()

    try:
        value = func(*args)

        print("{}: {:.8f} sec.".format(func.__name__, time() - start))

        return value

    except Exception as e:
        raise TimerError(func, "The function raised an exception.") from e
