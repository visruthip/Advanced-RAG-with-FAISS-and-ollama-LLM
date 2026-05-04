from pypdf import PdfReader

class PDFReader:
    def __init__(self, path):
        self.path = path

    def read(self):
        reader = PdfReader(self.path)
        text = []

        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)

        return ["\n".join(text)]