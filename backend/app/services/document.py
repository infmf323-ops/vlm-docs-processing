from io import BytesIO

import fitz
from PIL import Image


def count_pages(content: bytes, content_type: str) -> int:
    """Число страниц документа (для PDF — реальное, для изображений — 1)."""
    if content_type == "application/pdf":
        try:
            with fitz.open(stream=content, filetype="pdf") as doc:
                return max(1, doc.page_count)
        except Exception:
            return 1
    return 1


def load_all_page_images(
    content: bytes, content_type: str, max_pages: int = 20
) -> list[Image.Image]:
    """Все страницы документа как изображения (для PDF — до max_pages)."""
    if content_type == "application/pdf":
        try:
            images: list[Image.Image] = []
            with fitz.open(stream=content, filetype="pdf") as doc:
                for i in range(min(doc.page_count, max_pages)):
                    pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(2, 2))
                    images.append(
                        Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
                    )
            if images:
                return images
        except Exception as exc:  # noqa: BLE001
            raise ValueError("не удалось прочитать страницы PDF") from exc
    image, _ = load_first_page_image(content, content_type)
    return [image]


def load_first_page_image(content: bytes, content_type: str) -> tuple[Image.Image, bytes]:
    try:
        if content_type == "application/pdf":
            doc = fitz.open(stream=content, filetype="pdf")
            if doc.page_count == 0:
                raise ValueError("PDF не содержит страниц")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
            preview_bytes = pix.tobytes("png")
            return image, preview_bytes

        image = Image.open(BytesIO(content)).convert("RGB")
        preview_buffer = BytesIO()
        image.save(preview_buffer, format="PNG")
        return image, preview_buffer.getvalue()
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 — любые ошибки декодирования приводим к ValueError
        raise ValueError(
            "файл не является поддерживаемым изображением или PDF"
        ) from exc
