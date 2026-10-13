import os, json, sys
from PyPDF2 import PdfReader

cwd = r"C:\\Users\\gshan\\OneDrive\\Documents\\PrenatalAnemia"
output_dir = os.path.join(cwd, "extracted")
os.makedirs(output_dir, exist_ok=True)

summary = []
for filename in os.listdir(cwd):
    if filename.lower().endswith('.pdf'):
        path = os.path.join(cwd, filename)
        try:
            reader = PdfReader(path)
            text = ''
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    text += txt + "\n"
            txt_path = os.path.join(output_dir, filename + '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            summary.append({'pdf': filename, 'pages': len(reader.pages), 'txt': txt_path})
        except Exception as e:
            summary.append({'pdf': filename, 'error': str(e)})
print(json.dumps(summary, indent=2))
