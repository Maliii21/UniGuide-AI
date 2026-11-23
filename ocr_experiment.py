
from pathlib import Path
import fitz
import pytesseract
from PIL import Image, ImageOps, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

pytesseract.pytesseract.tesseract_cmd = (
    r"D:\Tesseract-OCR\tesseract.exe"
)

TEST_PAGES = [
    {
        "filename": "5-Course-Registration-Policy.pdf",
        "page": 4,
        "language": "eng"
    },
    {
        "filename": "15-Withdraw-from-AIU-Policy.pdf",
        "page": 3,
        "language": "ara"
    }
]


def preprocess_image(image):

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # Improve text sharpness
    image = image.filter(ImageFilter.SHARPEN)

    return image


for item in TEST_PAGES:

    pdf_path = RAW_DIR / item["filename"]

    with fitz.open(pdf_path) as pdf:

        page = pdf[item["page"] - 1]

        # Normal extraction
        normal_text = page.get_text("text")

        # Render page at high resolution
        pix = page.get_pixmap(dpi=400, alpha=False)

        image = Image.frombytes(
            "RGB",
            (pix.width, pix.height),
            pix.samples
        )

        # Image preprocessing
        processed_image = preprocess_image(image)

        # OCR extraction
        ocr_text = pytesseract.image_to_string(
            processed_image,
            lang=item["language"],
            config="--oem 3 --psm 11"
        )

        print("\n" + "=" * 60)
        print("Document:", item["filename"])
        print("Page:", item["page"])
        print("Language:", item["language"])

        print("\n--- NORMAL EXTRACTION ---\n")
        print(normal_text)

        print("\n--- IMPROVED OCR EXTRACTION ---\n")
        print(ocr_text)

        print("\n--- COMPARISON ---")
        print("Normal words:", len(normal_text.split()))
        print("OCR words:", len(ocr_text.split()))
