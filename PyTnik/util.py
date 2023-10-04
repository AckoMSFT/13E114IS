import ctypes
import time
from threading import Timer, Thread


class Timeout(Exception):
    pass


def send_thread_exception(*args):
    for t_id in args:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(t_id), ctypes.py_object(Timeout))
        if not res:
            print(f'ERR: Thread {t_id} not found')
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(t_id, 0)
            print(f'ERR: Failed to send exception to thread {t_id}')


class TimedFunction(Thread):
    def __init__(self, parent_id, queue, max_time_sec, method, *args):
        super().__init__()
        self.parent_id = parent_id
        self.queue = queue
        self.max_time_sec = max_time_sec
        self.method = method
        self.args = args

    def get_id(self):
        return self.ident

    def run(self) -> None:
        timer = Timer(interval=self.max_time_sec,
                      function=send_thread_exception, args=[self.ident, self.parent_id])
        timer.start()
        try:
            start_time = time.time()
            result = self.method(*self.args)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.queue.put((result, elapsed_time), block=False)
        except (Timeout, Exception):
            pass
        finally:
            timer.cancel()

