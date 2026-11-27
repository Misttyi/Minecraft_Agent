"""无显示器机器上给 Minecraft 起一个 Xvfb。

优先用 PATH 里的 Xvfb；否则用 MC_XVFB 指向的二进制（服务器上是从 apt 包解出来的用户态 Xvfb）。
每个进程起一个独立显示号，进程退出时一起关掉。
"""

import atexit
import os
import shutil
import subprocess
import time

DEFAULT_XVFB = "/usr/bin/Xvfb"


def _find_xvfb():
    for candidate in (os.environ.get("MC_XVFB"), shutil.which("Xvfb"), DEFAULT_XVFB):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def ensure_display(force_new=False):
    if os.environ.get("DISPLAY") and not force_new:
        return os.environ["DISPLAY"]
    binary = _find_xvfb()
    if binary is None:
        raise RuntimeError("没有 DISPLAY，也找不到 Xvfb；设置 MC_XVFB 指向 Xvfb")
    root = os.path.dirname(os.path.dirname(os.path.dirname(binary)))
    xkb = os.path.join(root, "usr", "share", "X11", "xkb")
    for number in range(100 + os.getpid() % 300, 1000):
        if os.path.exists(f"/tmp/.X{number}-lock") or os.path.exists(f"/tmp/.X11-unix/X{number}"):
            continue
        cmd = [binary, f":{number}", "-screen", "0", "1024x768x24", "-nolisten", "tcp", "-ac"]
        if os.path.isdir(xkb):
            cmd += ["-xkbdir", xkb]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        if proc.poll() is None:
            os.environ["DISPLAY"] = f":{number}"
            atexit.register(proc.terminate)
            return os.environ["DISPLAY"]
    raise RuntimeError("Xvfb 起不来")
