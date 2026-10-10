"""极简 multipart/form-data 解析（只用标准库）。"""

from __future__ import annotations

import re

_PARAM_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')


def _params(header: str) -> dict[str, str]:
    return {key.lower(): value for key, value in _PARAM_RE.findall(header)}


def parse_multipart(body: bytes, boundary: str) -> dict:
    """把请求体解析成 {"fields": {...}, "files": {...}}。"""
    if not boundary:
        return {"fields": {}, "files": {}}
    delimiter = b"--" + boundary.encode("latin-1")
    fields: dict[str, str] = {}
    files: dict[str, dict] = {}

    for chunk in body.split(delimiter):
        if not chunk or chunk in (b"--", b"--\r\n", b"--\n", b"\r\n"):
            continue
        if chunk.startswith(b"--"):
            continue
        chunk = chunk.lstrip(b"\r\n")
        header_blob, sep, data = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]

        headers: dict[str, str] = {}
        for line in header_blob.split(b"\r\n"):
            key, _, value = line.partition(b":")
            if _:
                headers[key.decode("latin-1").strip().lower()] = (
                    value.decode("latin-1").strip()
                )

        disposition = headers.get("content-disposition", "")
        params = _params(disposition)
        name = params.get("name", "")
        filename = params.get("filename")
        if not name:
            continue
        if filename is None:
            fields[name] = data.decode("utf-8", "replace")
        else:
            files[name] = {
                "filename": filename,
                "content_type": headers.get("content-type", "application/octet-stream"),
                "data": data,
            }
    return {"fields": fields, "files": files}
