#!/usr/bin/env python3
"""把 ``dist/`` 里的 PyInstaller 产物整理成可以挂到 Release 的附件。

    python .github/scripts/make_assets.py

产出到 ``release/``：

    radar-sim-<版本>-win64.zip         文件夹版（启动快，templates/ 可直接改）
    radar-sim-<版本>-win64-single.exe  单文件版
    SHA256SUMS.txt                     前两者的校验值

附件名用 ASCII，避免下载链接、CI 日志和 ``gh`` 命令行各处的编码坑；
压缩包里仍是原来的中文文件名（``zipfile`` 会自动置 UTF-8 标志位）。

版本号优先取 CI 的 git tag（``GITHUB_REF_TYPE=tag``），本地运行时回落到
``src/radar_sim/__init__.py`` 的 ``__version__``。
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DIST = PROJECT / "dist"
OUT = PROJECT / "release"

# 同 packaging/build.py：Windows 控制台默认 GBK，重定向时中文日志会变乱码。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                          # pragma: no cover
    pass

FOLDER_BUILD = DIST / "雷达仿真器"
ONEFILE_BUILD = DIST / "雷达仿真器-单文件版.exe"


def declared_version() -> str:
    init = (PROJECT / "src" / "radar_sim" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.MULTILINE)
    if match is None:
        raise SystemExit("[assets] src/radar_sim/__init__.py 里找不到 __version__")
    return match.group(1)


def project_version() -> str:
    """包名用 tag 的版本（Release 才是权威），但代码里的版本号得对得上。"""
    declared = declared_version()
    if os.environ.get("GITHUB_REF_TYPE") != "tag":
        return declared

    tag = os.environ.get("GITHUB_REF_NAME", "").strip()
    version = tag[1:] if tag.startswith("v") else tag
    if not version:
        return declared
    if version != declared:
        print(f"[assets] 警告：tag 是 {tag}，但 __version__ 还是 {declared}；"
              f"exe 的 --version 会打出 {declared}", file=sys.stderr)
    return version


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_folder(src: Path, dst: Path) -> None:
    """zip 的根就是程序目录本身，解压后不用再套一层同名文件夹。"""
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(src).as_posix())


def main() -> int:
    missing = [p for p in (FOLDER_BUILD, ONEFILE_BUILD) if not p.exists()]
    if missing:
        for path in missing:
            print(f"[assets] 缺少构建产物：{path}", file=sys.stderr)
        print("[assets] 先跑 python packaging/build.py", file=sys.stderr)
        return 1

    version = project_version()
    OUT.mkdir(exist_ok=True)
    for stale in OUT.iterdir():                 # 重跑时不留上一版的附件
        if stale.is_file():
            stale.unlink()

    folder_zip = OUT / f"radar-sim-{version}-win64.zip"
    print(f"[assets] 压缩 {FOLDER_BUILD.name}/ ...", flush=True)
    zip_folder(FOLDER_BUILD, folder_zip)

    single_exe = OUT / f"radar-sim-{version}-win64-single.exe"
    shutil.copy2(ONEFILE_BUILD, single_exe)

    with zipfile.ZipFile(folder_zip) as archive:
        entries = archive.namelist()
    if not any(name.endswith("雷达仿真器.exe") for name in entries):
        print("[assets] 压缩包里没有 雷达仿真器.exe，产物不完整", file=sys.stderr)
        return 1

    assets = [folder_zip, single_exe]
    sums = OUT / "SHA256SUMS.txt"
    sums.write_text(
        "\n".join(f"{sha256(p)}  {p.name}" for p in assets) + "\n", encoding="utf-8"
    )

    print(f"\n[assets] 版本 {version}（压缩包 {len(entries)} 个文件）")
    for path in assets + [sums]:
        print(f"  {path.name:<40} {path.stat().st_size / 1024 / 1024:8.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
