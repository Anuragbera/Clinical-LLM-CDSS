import cv2
import easyocr


class ClinicalOCR:
    def __init__(self):
        self.reader = easyocr.Reader(
            ['en'],
            gpu=True,          # 🔥GPU ENABLED
            quantize=False     # better GPU performance
        )

    def preprocess(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 🔹 OPTIONAL: Denoising (good for scanned reports)
        img_processed = cv2.fastNlMeansDenoisingColored(
            img_rgb, None, 10, 10, 7, 21
        )

        return img_processed

        # ⚡ FAST MODE (use this if needed)
        # return img_rgb

    def extract(self, image_path):

        try:
            processed_img = self.preprocess(image_path)

            results = self.reader.readtext(
                processed_img,
                detail=0,
                paragraph=True   # 🔥 important for lab reports
            )

            final_text = "\n".join(results)
            return final_text

        except Exception as e:
            return f"OCR Error: {str(e)}"