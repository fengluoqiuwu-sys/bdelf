"""后台合并写日志：训练线程只入队，热路径上不 wait 文件系统。

覆盖 CSV 追加、stdout（.out）、stderr（.err，含 tqdm ``\\r`` 刷新）。
本机 TTY 不包装 stdio。checkpoint 落盘仍同步。
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, TextIO

_QUEUE_MAX = 4096
_FLUSH_BYTES = 64 * 1024
_FLUSH_SEC = 0.05
_SHUTDOWN_SEC = 5.0

_KIND_FD = "fd"
_KIND_PATH = "path"
_KIND_CLOSE = "close"
_KIND_STOP = "stop"

# (kind, dest, data, is_cr)
_Item = tuple[str, str, bytes, bool]

_lock = threading.Lock()
_cv = threading.Condition(_lock)
_queue: deque[_Item] = deque()
_pending_cr: dict[str, bytes] = {}
_close_waiters: list[tuple[str, threading.Event]] = []
_dropped = 0
_thread: threading.Thread | None = None
_running = False
_atexit_registered = False
_orig_stdout: TextIO | None = None
_orig_stderr: TextIO | None = None
_wrapped = False


def _encode(data: str | bytes) -> bytes:
    if isinstance(data, bytes):
        return data
    return data.encode("utf-8", errors="replace")


def _is_cr_only(data: bytes) -> bool:
    return b"\r" in data and b"\n" not in data


def _dest_fd(fd: int) -> str:
    return f"fd:{fd}"


def _dest_path(path: str | Path) -> str:
    return f"path:{os.fspath(path)}"


def is_running() -> bool:
    return _running


def dropped_count() -> int:
    return _dropped


def enqueue_fd(fd: int, data: str | bytes) -> None:
    """写入 stdout/stderr 对应 fd；训练线程不等待。"""
    raw = _encode(data)
    if not raw:
        return
    _enqueue(_KIND_FD, _dest_fd(fd), raw, _is_cr_only(raw))


def enqueue_file(path: str | Path, data: str | bytes) -> bool:
    """追加到文件。写线程未启动时返回 False，调用方自行同步写。"""
    raw = _encode(data)
    if not raw:
        return True
    if not _running:
        return False
    _enqueue(_KIND_PATH, _dest_path(path), raw, False)
    return True


def close_path(
    path: str | Path, *, wait: bool = False, timeout: float = _SHUTDOWN_SEC,
) -> None:
    """关掉该 path 的后台句柄。schema 改写 / resume 截断前须 ``wait=True``。"""
    if not _running:
        return
    done = threading.Event()
    dest = _dest_path(path)
    with _cv:
        _queue.append((_KIND_CLOSE, dest, b"", False))
        _close_waiters.append((dest, done))
        _cv.notify()
    if wait:
        done.wait(timeout=timeout)


def _enqueue(kind: str, dest: str, data: bytes, is_cr: bool) -> None:
    global _dropped
    with _cv:
        if not _running:
            return
        if is_cr and kind == _KIND_FD:
            _pending_cr[dest] = data
            _cv.notify()
            return
        if len(_queue) >= _QUEUE_MAX and kind != _KIND_STOP:
            if is_cr:
                _dropped += 1
                return
            kept: deque[_Item] = deque()
            for old in _queue:
                if old[3] and old[0] == _KIND_FD:
                    _dropped += 1
                    continue
                kept.append(old)
            _queue.clear()
            _queue.extend(kept)
            if len(_queue) >= _QUEUE_MAX:
                _dropped += 1
                return
        _queue.append((kind, dest, data, is_cr))
        _cv.notify()


class _AsyncStdIO:
    """write 入队；flush 为空操作，不等待写线程。"""

    def __init__(self, raw: TextIO) -> None:
        self._raw = raw
        self._fd = raw.fileno()
        self.encoding = getattr(raw, "encoding", None) or "utf-8"
        self.errors = getattr(raw, "errors", None) or "replace"
        self.newlines = getattr(raw, "newlines", None)
        self.buffer = getattr(raw, "buffer", None)
        self.name = getattr(raw, "name", "<async-stdio>")
        self.mode = getattr(raw, "mode", "w")

    def write(self, s: Any) -> int:
        if not s:
            return 0
        if isinstance(s, str):
            enqueue_fd(self._fd, s)
            return len(s)
        raw = bytes(s)
        enqueue_fd(self._fd, raw)
        return len(raw)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        return

    def isatty(self) -> bool:
        try:
            return bool(self._raw.isatty())
        except OSError:
            return False

    def fileno(self) -> int:
        return self._fd

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        return

    @property
    def closed(self) -> bool:
        return False

    def reconfigure(self, **kwargs: Any) -> None:
        reconf = getattr(self._raw, "reconfigure", None)
        if callable(reconf):
            reconf(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def _emit_fd(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            n = os.write(fd, view)
        except InterruptedError:
            continue
        except OSError:
            return
        if n <= 0:
            return
        view = view[n:]


def _writer_loop() -> None:
    bufs: dict[str, bytearray] = {}
    path_fds: dict[str, int] = {}
    last_flush = time.monotonic()

    def buf_of(dest: str) -> bytearray:
        slot = bufs.get(dest)
        if slot is None:
            slot = bytearray()
            bufs[dest] = slot
        return slot

    def open_path(dest: str) -> int | None:
        fd = path_fds.get(dest)
        if fd is not None:
            return fd
        path = dest[len("path:"):]
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        except OSError:
            return None
        path_fds[dest] = fd
        return fd

    def emit(dest: str, data: bytes) -> None:
        if not data:
            return
        if dest.startswith("fd:"):
            _emit_fd(int(dest[3:]), data)
            return
        fd = open_path(dest)
        if fd is None:
            return
        view = memoryview(data)
        while view:
            try:
                n = os.write(fd, view)
            except InterruptedError:
                continue
            except OSError:
                return
            if n <= 0:
                return
            view = view[n:]

    def flush_dest(dest: str) -> None:
        slot = bufs.get(dest)
        if slot:
            emit(dest, bytes(slot))
            slot.clear()

    def close_dest(dest: str) -> None:
        flush_dest(dest)
        fd = path_fds.pop(dest, None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        bufs.pop(dest, None)

    stop = False
    while True:
        with _cv:
            if not _queue and not _pending_cr and not stop:
                _cv.wait(timeout=_FLUSH_SEC)
            batch: list[_Item] = []
            while _queue:
                batch.append(_queue.popleft())
            cr = dict(_pending_cr)
            _pending_cr.clear()
            waiters = list(_close_waiters)
            _close_waiters.clear()
        now = time.monotonic()
        closed: set[str] = set()
        for kind, dest, data, is_cr in batch:
            if kind == _KIND_STOP:
                stop = True
                continue
            if kind == _KIND_CLOSE:
                close_dest(dest)
                closed.add(dest)
                continue
            if is_cr:
                cr[dest] = data
                continue
            buf_of(dest).extend(data)
        for dest, data in cr.items():
            if dest in closed:
                continue
            buf_of(dest).extend(data)
        due = stop or (now - last_flush) >= _FLUSH_SEC
        for dest, slot in list(bufs.items()):
            if slot and (due or len(slot) >= _FLUSH_BYTES):
                emit(dest, bytes(slot))
                slot.clear()
        if due:
            last_flush = now
        for dest, ev in waiters:
            ev.set()
        if not stop:
            continue
        with _cv:
            leftover = list(_queue)
            _queue.clear()
            cr_left = dict(_pending_cr)
            _pending_cr.clear()
        for kind, dest, data, is_cr in leftover:
            if kind == _KIND_CLOSE:
                close_dest(dest)
                continue
            if kind == _KIND_STOP:
                continue
            if is_cr:
                cr_left[dest] = data
                continue
            buf_of(dest).extend(data)
        for dest, data in cr_left.items():
            buf_of(dest).extend(data)
        for dest in list(bufs):
            flush_dest(dest)
        for dest in list(path_fds):
            close_dest(dest)
        return


def install(*, wrap_stdio: bool | None = None) -> None:
    """启动写线程；非 TTY 时包装 stdout/stderr。"""
    global _running, _thread, _wrapped, _orig_stdout, _orig_stderr, _dropped
    with _cv:
        if _running:
            return
        _dropped = 0
        _running = True
        _thread = threading.Thread(
            target=_writer_loop, name="bdelf-async-log", daemon=True,
        )
        _thread.start()
    _register_atexit()
    if wrap_stdio is None:
        try:
            wrap_stdio = not (sys.stdout.isatty() and sys.stderr.isatty())
        except OSError:
            wrap_stdio = True
    if not wrap_stdio or _wrapped:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except OSError:
        pass
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    try:
        if not sys.stdout.isatty():
            sys.stdout = _AsyncStdIO(_orig_stdout)
        if not sys.stderr.isatty():
            sys.stderr = _AsyncStdIO(_orig_stderr)
        _wrapped = True
    except OSError:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _wrapped = False


def shutdown(timeout: float = _SHUTDOWN_SEC) -> None:
    """刷完队列并停线程；仅退出/中断路径调用。可重复。"""
    global _running, _thread
    with _cv:
        if not _running and _thread is None:
            _restore_stdio()
            return
        if _running:
            _queue.append((_KIND_STOP, "", b"", False))
            _running = False
            _cv.notify_all()
    th = _thread
    if th is not None:
        th.join(timeout=timeout)
    _thread = None
    n = _dropped
    err = _orig_stderr
    _restore_stdio()
    if n:
        sink = err if err is not None else sys.stderr
        try:
            sink.write(f"[train] async_log dropped {n} records\n")
            sink.flush()
        except OSError:
            pass


def _restore_stdio() -> None:
    global _wrapped, _orig_stdout, _orig_stderr
    if not _wrapped:
        return
    if _orig_stdout is not None:
        sys.stdout = _orig_stdout
    if _orig_stderr is not None:
        sys.stderr = _orig_stderr
    _wrapped = False


def _register_atexit() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    atexit.register(shutdown)
    _atexit_registered = True
