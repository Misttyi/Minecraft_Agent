"""修复 minerl 0.4.4 与 numpy>=1.24 的兼容问题（np.bool/int/float 已删除）。

在 mc 环境里执行一次；会在该环境的 site-packages 写入 sitecustomize.py，
之后每次 Python 启动自动注入别名。不修改任何已安装的包，删除该文件即可回滚。

    python scripts/install_compat.py

验证：

    python -c "import numpy as np; print(np.bool is bool, np.int is int)"
"""

import site
import sys
from pathlib import Path


SHIM = '''import warnings
warnings.filterwarnings("ignore", message=".*np\\.(bool|int|float).*deprecated.*")
import numpy as _np
for _name, _type in (
    ("bool", bool),
    ("int", int),
    ("float", float),
    ("object", object),
    ("str", str),
    ("complex", complex),
):
    if not hasattr(_np, _name):
        setattr(_np, _name, _type)
'''


def main():
    targets = [Path(p) for p in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        targets.append(Path(user_site))
    written = []
    for target in targets:
        if not target.is_dir():
            continue
        path = target / "sitecustomize.py"
        if path.exists():
            print(f"skip (already exists): {path}")
            continue
        path.write_text(SHIM, encoding="utf-8")
        written.append(path)
        print(f"wrote: {path}")
    if not written:
        print("No new sitecustomize.py written. If compatibility errors persist, "
              "check that you are using the same interpreter as your training environment.")
    print("Verify with: python -c \"import numpy as np; print(np.bool is bool, np.int is int)\"")


if __name__ == "__main__":
    main()
