# img/core.py

import sys
from loguru import logger
import clip
import easyocr
from PIL import Image, ImageDraw, ImageFont
from textblob import TextBlob
from typing import List, Tuple
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from scripts.setup_logging import setup_logging

logger.remove()

# Configure Loguru to log to standard output and a file
logger.add(sys.stdout, level="INFO", format="{time} {level} {message}", backtrace=True, diagnose=True)
logger.add("app.log", level="DEBUG", format="{time} {level} {message}", rotation="10 MB")

# Example of adding contextual information
logger = logger.bind(application="text-generation")


def load_uploaded_image(image_path: str) -> Image.Image:
    try:
        image = Image.open(image_path)
        logger.info("Image loaded", extra={"image_path": image_path})
        return image
    except Exception as e:
        logger.error("Failed to load image", extra={"error": str(e)})
        raise

def interpret_image(image: Image.Image, additional_prompt: str = "") -> str:
    try:
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        image_tensor = preprocess(image).unsqueeze(0).to("cpu")
        
        if additional_prompt:
            prompts = [additional_prompt]
        else:
            prompts = ["a diagram that describes a graph infrastructure emulating the human brain"]

        text_tensor = clip.tokenize(prompts).to("cpu")
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tensor)
            
        similarity = torch.cosine_similarity(image_features, text_features)
        description = prompts[similarity.argmax()]
        logger.info("Image interpreted", extra={"description": description})
        return description
    except Exception as e:
        logger.error("Failed to interpret image", extra={"error": str(e)})
        raise

def get_bounding_boxes(image: Image.Image, expansion: int = 5) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    try:
        reader = easyocr.Reader(['en'])
        image_np = np.array(image)
        results = reader.readtext(image_np)
        
        bounding_boxes = [
            (
                text,
                (max(left - expansion, 0), max(top - expansion, 0), min(right + expansion, image_np.shape[1]) - max(left - expansion, 0), min(bottom + expansion, image_np.shape[0]) - max(top - expansion, 0))
            )
            for box, text, _ in results
            for (left, top), (right, bottom) in [(box[0], box[2])]
        ]
        logger.info("Bounding boxes detected", extra={"bounding_boxes": bounding_boxes})
        return bounding_boxes
    except Exception as e:
        logger.error("Failed to get bounding boxes", extra={"error": str(e)})
        raise

def correct_text(text: str, description: str) -> str:
    try:
        blob = TextBlob(text)
        corrected_text = str(blob.correct())
        logger.info("Text corrected", extra={"original_text": text, "corrected_text": corrected_text})
        return corrected_text
    except Exception as e:
        logger.error("Failed to correct text", extra={"error": str(e)})
        raise

def inpaint_image(image: Image.Image, bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]]) -> Image.Image:
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        inpainted_image = image.copy()
        mask = Image.new("L", (image.width, image.height), 0)
        draw = ImageDraw.Draw(mask)

        for _, (x, y, w, h) in bounding_boxes:
            draw.rectangle([x, y, x+w, y+h], fill=255)

        image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).permute(0, 3, 1, 2)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

        result = pipe(prompt="", image=image_tensor, mask_image=mask_tensor)
        inpainted_image_np = result.images[0].cpu().numpy().astype(np.uint8)
        inpainted_image = Image.fromarray(inpainted_image_np)
        logger.info("Image inpainted")
        return inpainted_image
    except Exception as e:
        logger.error("Failed to inpaint image", extra={"error": str(e)})
        raise

def compose_image(original_image: Image.Image, corrected_texts: List[str], bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]]) -> Image.Image:
    try:
        image = original_image.copy()
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        for (text, (x, y, w, h)), corrected_text in zip(bounding_boxes, corrected_texts):
            draw.text((x + w // 10, y + h // 10), corrected_text, font=font, fill="black")
        logger.info("Image composed with corrected text")
        return image
    except Exception as e:
        logger.error("Failed to compose image", extra={"error": str(e)})
        raise
