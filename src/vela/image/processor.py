from __future__ import annotations

import base64
import io
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

from vela.image.clipboard import ClipboardImageResult, grab_clipboard_image

IMAGE_PATTERN = re.compile(
    r"@image:(?P<path><[^>]+>|[^\s<>\u2010-\u206f\u3000-\u303f\uff00-\uffef]+)"
    r"|(?P<clipboard>@clipboard)(?![\w])"
)
ClipboardGrabber = Callable[[], ClipboardImageResult]


def parse_image_references(
    message: str,
    cwd: str,
    *,
    clipboard_grabber: ClipboardGrabber = grab_clipboard_image,
) -> str | list[dict]:
    matches = list(IMAGE_PATTERN.finditer(message))
    if not matches:
        return message

    content: list[dict] = []
    image_count = 0
    cursor = 0
    for match in matches:
        _append_text_part(content, message[cursor : match.start()])
        raw_path = match.group("path")
        reference = "@clipboard" if raw_path is None else _strip_angle_brackets(raw_path)
        try:
            part = _image_part(reference, cwd, clipboard_grabber)
            content.append(part)
            image_count += 1
        except Exception as exc:  # noqa: BLE001 - invalid attachment remains a visible text note
            label = "剪贴板" if reference == "@clipboard" else reference
            _append_text_part(content, f"[图片引用无效: {label}，原因: {exc}]")
        cursor = match.end()
    _append_text_part(content, message[cursor:])

    if not image_count:
        return "".join(part["text"] for part in content).strip()
    if not any(part["type"] == "text" and part["text"].strip() for part in content):
        content.insert(0, {"type": "text", "text": "请分析以下图片。"})
    _append_text_part(
        content,
        "\n\n[图片已作为本轮附件附加。请直接观察图片内容；如果无法看图，请明确说明，"
        "不要根据文件路径或历史上下文猜测。]",
    )
    return content


def _image_part(
    reference: str,
    cwd: str,
    clipboard_grabber: ClipboardGrabber,
) -> dict:
    if reference == "@clipboard":
        grabbed = clipboard_grabber()
        if not grabbed.ok or grabbed.path is None:
            raise ValueError(grabbed.error or "剪贴板里没有图片")
        reference = str(grabbed.path)
    if reference.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": reference}}
    path = _resolve_image_path(reference, cwd)
    data_url, width, height = _encode_image(path)
    return {
        "type": "image_url",
        "image_url": {"url": data_url},
        "metadata": {"source": str(path), "width": width, "height": height},
    }


def _resolve_image_path(reference: str, cwd: str) -> Path:
    if reference.startswith("file://"):
        parsed = urlparse(reference)
        path = Path(unquote(parsed.path))
    else:
        path = Path(reference)
    if not path.is_absolute():
        path = Path(cwd).resolve() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError("不是普通文件")
    return path


def _encode_image(path: Path, max_side: int = 1568) -> tuple[str, int, int]:
    with Image.open(path) as image:
        image.thumbnail((max_side, max_side))
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}", image.width, image.height


def _strip_angle_brackets(reference: str) -> str:
    if reference.startswith("<") and reference.endswith(">"):
        return reference[1:-1]
    return reference


def _append_text_part(content: list[dict], text: str) -> None:
    if not text:
        return
    if content and content[-1]["type"] == "text":
        content[-1]["text"] += text
    else:
        content.append({"type": "text", "text": text})
