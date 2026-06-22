import pdfplumber
from pdf2image import convert_from_path

class PDFProcessor:

    def extract_text(self, pdf_path):
        text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        return text

    def convert_to_images(self, pdf_path):
        images = convert_from_path(pdf_path)
        image_paths = []

        for i, img in enumerate(images):
            path = f"temp_page_{i}.png"
            img.save(path, "PNG")
            image_paths.append(path)

        return image_paths