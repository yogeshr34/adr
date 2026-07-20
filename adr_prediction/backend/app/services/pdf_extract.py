"""PDF text extraction: embedded text via PyMuPDF, OCR fallback via Tesseract."""

import io
import logging

import fitz  # PyMuPDF

from app.config import MIN_EMBEDDED_TEXT_CHARS, OCR_DPI

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when no text could be extracted from the uploaded PDF."""


def _extract_embedded_text(doc: "fitz.Document") -> str:
    parts = [page.get_text("text") for page in doc]
    return "\n".join(parts).strip()


def _extract_via_ocr(doc: "fitz.Document") -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise PDFExtractionError(
            "PDF has no embedded text and OCR dependencies (pytesseract/Pillow) "
            "are not installed."
        ) from exc

    zoom = OCR_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)
    ocr_parts = []

    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            ocr_parts.append(pytesseract.image_to_string(img))
        except Exception as exc:  # pytesseract.TesseractNotFoundError, etc.
            raise PDFExtractionError(
                "PDF appears to be scanned (no embedded text) and the Tesseract "
                "OCR engine is not available on this server."
            ) from exc

    return "\n".join(ocr_parts).strip()


def extract_text(file_bytes: bytes) -> dict:
    """Extract text from a PDF's bytes.

    Returns a dict: {"text": str, "method": "embedded" | "ocr", "pages": int}
    Raises PDFExtractionError if no usable text could be recovered.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError(f"Could not open file as a PDF: {exc}") from exc

    try:
        pages = doc.page_count
        embedded_text = _extract_embedded_text(doc)

        if len(embedded_text) >= MIN_EMBEDDED_TEXT_CHARS:
            return {"text": embedded_text, "method": "embedded", "pages": pages}

        logger.info("Embedded text too short (%d chars); falling back to OCR", len(embedded_text))
        ocr_text = _extract_via_ocr(doc)

        if len(ocr_text) < MIN_EMBEDDED_TEXT_CHARS:
            raise PDFExtractionError(
                "Could not extract readable text from this PDF, even with OCR. "
                "The document may be blank, corrupted, or image quality too low."
            )

        return {"text": ocr_text, "method": "ocr", "pages": pages}
    finally:
        doc.close()
