from __future__ import annotations

from PIL import Image

from vela.image import ClipboardImageResult, parse_image_references
from vela.llm.openai_compatible import OpenAICompatibleClient


def test_parse_local_image_reference(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 128)).save(image_path)

    content = parse_image_references(f"look @image:{image_path.name}", str(tmp_path))

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_parse_angle_wrapped_absolute_image_path_with_spaces(tmp_path):
    image_path = tmp_path / "outside folder" / "screen shot.png"
    image_path.parent.mkdir()
    Image.new("RGB", (12, 8), "blue").save(image_path)

    content = parse_image_references(f"分析 @image:<{image_path}>", str(tmp_path / "project"))

    assert isinstance(content, list)
    image_part = next(part for part in content if part["type"] == "image_url")
    assert image_part["metadata"]["source"] == str(image_path)


def test_parse_clipboard_reference_uses_saved_image(tmp_path):
    image_path = tmp_path / "clip.png"
    Image.new("RGB", (9, 7), "green").save(image_path)

    content = parse_image_references(
        "帮我看看 @clipboard",
        str(tmp_path),
        clipboard_grabber=lambda: ClipboardImageResult.success(image_path),
    )

    assert isinstance(content, list)
    image_part = next(part for part in content if part["type"] == "image_url")
    assert image_part["metadata"]["source"] == str(image_path)
    assert "@clipboard" not in content[0]["text"]


def test_parse_clipboard_failure_becomes_visible_note(tmp_path):
    content = parse_image_references(
        "帮我看看 @clipboard",
        str(tmp_path),
        clipboard_grabber=lambda: ClipboardImageResult.failure("剪贴板里没有图片"),
    )

    assert isinstance(content, str)
    assert "图片引用无效: 剪贴板" in content
    assert "剪贴板里没有图片" in content


def test_invalid_image_reference_becomes_visible_note(tmp_path):
    content = parse_image_references("看看 @image:<missing image.png>", str(tmp_path))

    assert isinstance(content, str)
    assert "图片引用无效" in content
    assert "missing image.png" in content


def test_image_reference_stops_before_chinese_punctuation(tmp_path):
    image_path = tmp_path / "shot.png"
    Image.new("RGB", (10, 10), "red").save(image_path)

    content = parse_image_references("看看 @image:shot.png。这是什么？", str(tmp_path))

    assert isinstance(content, list)
    text = "".join(part["text"] for part in content if part["type"] == "text")
    assert "这是什么" in text


def test_multiple_images_preserve_interleaved_text_order(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (10, 10), "red").save(first)
    Image.new("RGB", (10, 10), "blue").save(second)

    content = parse_image_references(
        "第一张 @image:first.png 和第二张 @image:second.png 有什么区别？",
        str(tmp_path),
    )

    assert isinstance(content, list)
    assert [part["type"] for part in content[:5]] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert content[0]["text"] == "第一张 "
    assert content[2]["text"] == " 和第二张 "
    assert "有什么区别" in content[4]["text"]


def test_non_vision_model_omits_image_payload(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (10, 10), "red").save(image_path)
    content = parse_image_references(f"look @image:{image_path.name}", str(tmp_path))
    client = OpenAICompatibleClient(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://example.com/v1",
    )

    formatted = client._format_content(content)

    assert isinstance(formatted, str)
    assert "Image omitted" in formatted
