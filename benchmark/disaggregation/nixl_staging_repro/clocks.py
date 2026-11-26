"""Bound a B-clock event in A's monotonic clock without assuming path symmetry."""
import socket
import struct
import time


class ClockUnqualified(ValueError):
    pass


def recv_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        part = sock.recv(count - len(data))
        if not part:
            raise EOFError("clock channel closed")
        data.extend(part)
    return bytes(data)


def respond(conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        tag = conn.recv(1)
        if not tag:
            return
        t2 = time.monotonic_ns()
        if tag != b"T":
            raise ValueError("unexpected clock command")
        t3 = time.monotonic_ns()
        conn.sendall(struct.pack("!QQ", t2, t3))


def sample(conn, count=20):
    result = []
    for _ in range(count):
        t1 = time.monotonic_ns()
        conn.sendall(b"T")
        reply = recv_exact(conn, 16)
        t4 = time.monotonic_ns()
        t2, t3 = struct.unpack("!QQ", reply)
        result.append([t1, t2, t3, t4])
    return result


def map_event(y, before, after, start, end, drift_ppm=1000, max_width_ns=200_000):
    """Guaranteed only conditional on |d(B−A)/dB| <= drift_ppm / 1e6."""
    if not before or not after or end < start or drift_ppm < 0 or max_width_ns <= 0:
        raise ClockUnqualified("invalid bookends or bounds")
    if max(q[3] for q in before) > start or min(q[0] for q in after) < end:
        raise ClockUnqualified("A exchanges do not bracket traffic")
    if max(q[2] for q in before) > y or min(q[1] for q in after) < y:
        raise ClockUnqualified("B exchanges do not bracket decode")
    lo, hi = None, None
    for t1, t2, t3, t4 in [*before, *after]:
        if t4 < t1 or t3 < t2:
            raise ClockUnqualified("nonmonotonic exchange")
        # At receipt, t2-A(receipt) <= t2-t1. At send, t3-A(send) >= t3-t4.
        # Propagate each one-sided offset bound to the event under the drift budget.
        lower = t3 - t4 - (abs(y-t3)*drift_ppm + 999_999)//1_000_000
        upper = t2 - t1 + (abs(y-t2)*drift_ppm + 999_999)//1_000_000
        lo = lower if lo is None else max(lo, lower)
        hi = upper if hi is None else min(hi, upper)
    if hi < lo or hi-lo > max_width_ns:
        raise ClockUnqualified(f"offset interval width {hi-lo} ns")
    return y-hi, y-lo
