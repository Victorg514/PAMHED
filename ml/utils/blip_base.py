"""
Run BLIP captioning on every image file and save one .txt per image.
"""
import pathlib, torch, tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

IMG_DIR = pathlib.Path("data/images")
CAP_DIR = pathlib.Path("data/captions")
CAP_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base").to(device)

def caption(img_path):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20)
    return processor.decode(out[0], skip_special_tokens=True)

for img in tqdm.tqdm(list(IMG_DIR.glob("*.*"))):
    txt_file = CAP_DIR / (img.stem + ".txt")
    if txt_file.exists():
        continue
    try:
        txt_file.write_text(caption(img))
    except Exception as e:
        print("FAIL", img, e)
