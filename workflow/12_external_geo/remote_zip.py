"""List or extract members of a large public zip over HTTP byte ranges, without downloading the whole archive.

usage: python -I remote_zip.py <url> list [pattern]
       python -I remote_zip.py <url> get <pattern> <out_dir>

Members are written under <out_dir> with their archive path flattened ('/' -> '__'); zipfile checks each member's
CRC-32 on extraction. Nothing from the archive is executed. Used for the van Keken & Wilson (2023) model archive
(3.4 GB), of which only the Cascadia files are needed (D-B114).
"""
from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

UA = "Mozilla/5.0 (compatible; generic-client)"


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"User-Agent": UA, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def main(argv):
    url, cmd = argv[1], argv[2]
    pat = re.compile(argv[3]) if len(argv) > 3 else re.compile(".")
    zf = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=1 << 20))
    names = [i for i in zf.infolist() if pat.search(i.filename)]
    if cmd == "list":
        for i in names:
            print(f"{i.file_size:>12d} {i.filename}")
        print(f"{len(names)} members", file=sys.stderr)
        return
    out = Path(argv[4])
    out.mkdir(parents=True, exist_ok=True)
    for i in names:
        if i.is_dir():
            continue
        dst = out / i.filename.replace("/", "__")
        with zf.open(i) as src, open(dst, "wb") as f:
            while chunk := src.read(1 << 20):
                f.write(chunk)
        print(f"ok {i.file_size:>12d} {dst.name}")


if __name__ == "__main__":
    main(sys.argv)
