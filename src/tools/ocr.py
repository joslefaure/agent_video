import os
from typing import List, Dict
import logging

# Try importing easyocr, else mock
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

class OCRTool:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.reader = None
        if HAS_EASYOCR:
            # Initialize reader for English
            # In a real scenario, we might want to load this lazily or globally
            print("Initializing EasyOCR...")
            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
        else:
            logging.warning("EasyOCR not found. OCR will return dummy data.")

    def extract_text(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            return ""
        
        if self.reader:
            try:
                result = self.reader.readtext(image_path, detail=0)
                return " ".join(result)
            except Exception as e:
                logging.error(f"OCR failed for {image_path}: {e}")
                return ""
        else:
            # Dummy fallback for smoke tests if dependencies missing
            return "OCR_TEXT_PLACEHOLDER"

    def batch_ocr(self, image_paths: List[str]) -> Dict[str, str]:
        results = {}
        for path in image_paths:
            results[path] = self.extract_text(path)
        return results
