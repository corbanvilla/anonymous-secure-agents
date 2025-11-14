import io
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

# Import module normally; SCREENSHOT_DIR value will be patched in the test
import src.gradio.qual.trajectory_visualizer as vis


def test_data_to_image_fallback(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        # Create simple image in temporary screenshot directory
        img_path = Path(tmpdir) / "test.png"
        Image.new("RGB", (1, 1)).save(img_path)

        monkeypatch.setattr(vis, "SCREENSHOT_DIR", Path(tmpdir))

        result = vis._data_to_image("test.png")
        assert isinstance(result, Image.Image)


def test_data_to_image_bytes():
    img = Image.new("RGB", (1, 1))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    result = vis._data_to_image(img_bytes)
    assert isinstance(result, Image.Image)
