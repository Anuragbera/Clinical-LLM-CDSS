from modules.ocr_engine import ClinicalOCR
from modules.global_parser import GlobalClinicalParser
from modules.llm_extraction import extract_structured_data
from modules.normalization import normalize_lab_keys

ocr = ClinicalOCR(
    tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

parser = GlobalClinicalParser()


# ==========================
# 📷 IMAGE PIPELINE
# ==========================
def process_image(image_path):

    text = ocr.extract(image_path)

    print("\n[DEBUG OCR TEXT]:\n", text)

    # IMPORTANT: use original text (not cleaned)
    labs = parser.extract_key_value_pairs(text)

    print("\n[DEBUG PARSED LABS]:", labs)

    # APPLY NORMALIZATION HERE
    labs_numeric, labs_display = normalize_lab_keys(labs)

    print("\n[DEBUG NORMALIZED LABS]:", labs_numeric)

    return {
        "labs_numeric": labs_numeric,
        "labs": labs_display
    }

# ==========================
# 📝 TEXT PIPELINE
# ==========================
def process_text(text):

    structured_output = extract_structured_data(text)

    return structured_output