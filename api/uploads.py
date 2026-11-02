import os
from rest_framework.exceptions import APIException, ValidationError
from config import settings
from pypdf import PdfReader

class PayloadTooLarge(APIException):
    status_code = 413
    default_detail = "Payload too large"
    default_code = "payload_too_large"

def extract_text(uploaded):
    if uploaded.size > settings.MAX_UPLOAD_BYTES:
        mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise PayloadTooLarge(f"File exceeds the {mb} MB size limit")

    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_UPLOAD_EXTENSIONS))
        raise ValidationError(f"Unsupported File type. Allowed: {allowed}")

    if ext == '.txt':
        try:
            text = uploaded.read().decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValidationError({"file": "TXT file must be UTF-8 encoded."})

    else:
        try:
            reader = PdfReader(uploaded)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            raise ValidationError({"file": "Could not read the PDF file."})

    text = text.replace("\x00", "").strip()
    if not text:
        raise ValidationError({"file": "No extractable text found in the file."})
    return text

