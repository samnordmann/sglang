import socket
import threading
import time
import unittest

from clocks import ClockUnqualified, map_event, respond, sample


class Tests(unittest.TestCase):
    def test_asymmetry_and_unrelated_boot_times(self):
        offset = 19_000_000_000
        before = [[0, offset+300, offset+310, 350], [360, offset+365, offset+369, 599]]
        after = [[1000, offset+1040, offset+1050, 1090]]
        low, high = map_event(offset+820, before, after, 600, 950, 0, 50)
        self.assertEqual((low, high), (815, 860))

    def test_drift_and_reject(self):
        # B runs 500 ppm faster, and the two hosts booted at different times.
        b = lambda a: a + a//2000 + 10**12
        before = [[0,b(5000),b(5500),8000]]
        after = [[10_000_000,b(10_005_000),b(10_005_500),10_010_000]]
        args = (b(5_933_000),before,after,1_000_000,9_000_000)
        lo, hi = map_event(*args, drift_ppm=1000, max_width_ns=20_000)
        self.assertLessEqual(lo,5_933_000)
        self.assertGreaterEqual(hi,5_933_000)
        with self.assertRaises(ClockUnqualified):
            map_event(*args, drift_ppm=1000,max_width_ns=50)
        with self.assertRaises(ClockUnqualified):
            map_event(b(4),before,after,1_000_000,9_000_000)
        with self.assertRaises(ClockUnqualified):
            map_event(b(5_933_000),before,after,1,9_000_000)
        with self.assertRaises(ClockUnqualified):
            map_event(b(5_933_000),before,[[10_000_000,b(20_005_000),b(20_005_500),10_010_000]],1_000_000,9_000_000)

    def test_wire_format(self):
        # No network needed. 'respond' uses TCP_NODELAY, so use a socket wrapper.
        class NoDelay:
            def __init__(self, sock):
                self.sock=sock
            def setsockopt(self,*_):
                pass
            def __getattr__(self, name):
                return getattr(self.sock,name)
        a,b=socket.socketpair()
        with a,b:
            thread=threading.Thread(target=respond,args=(NoDelay(b),))
            thread.start()
            try:
                before=sample(a,3)
                start=y=time.monotonic_ns()
                end=time.monotonic_ns()
                after=sample(a,3)
                lo,hi=map_event(y,before,after,start,end,1000,20_000_000)
                self.assertLessEqual(lo,y)
                self.assertGreaterEqual(hi,y)
            finally:
                a.shutdown(socket.SHUT_WR)
                thread.join(2)
            self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
