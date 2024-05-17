# img/core.py

import clip
import easyocr
from PIL import Image, ImageDraw, ImageFont
from textblob import TextBlob
from typing import List, Tuple
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline

def load_uploaded_image(image_path: str) -> Image.Image:
    """
    Load an image from the given file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Image.Image: Loaded image.
    """
    image = Image.open(image_path)
    return image

def interpret_image(image: Image.Image, additional_prompt: str = "") -> str:
    """
    Interpret the image and provide a description using CLIP.

    Args:
        image (Image.Image): Image to interpret.
        additional_prompt (str): Additional prompt to describe the image better.

    Returns:
        str: Description of the image.
    """
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
    
    return description

def get_bounding_boxes(image: Image.Image, expansion: int = 5) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Detect text in the image and return slightly larger bounding boxes using EasyOCR.

    Args:
        image (Image.Image): Image in which to detect text.
        expansion (int): Number of pixels to expand each bounding box.

    Returns:
        List[Tuple[str, Tuple[int, int, int, int]]]: List of tuples containing detected text and their expanded bounding boxes.
    """
    reader = easyocr.Reader(['en'])  # Initialize with the language you need
    image_np = np.array(image)  # Convert PIL Image to NumPy array
    results = reader.readtext(image_np)
    
    bounding_boxes = [
        (
            text,
            (max(left - expansion, 0), max(top - expansion, 0), min(right + expansion, image_np.shape[1]) - max(left - expansion, 0), min(bottom + expansion, image_np.shape[0]) - max(top - expansion, 0))
        )
        for box, text, _ in results
        for (left, top), (right, bottom) in [(box[0], box[2])]
    ]
    
    return bounding_boxes

def correct_text(text: str, description: str) -> str:
    """
    Correct the given text using TextBlob, with additional context from image description.

    Args:
        text (str): Text to correct.
        description (str): Description of the image for better context.

    Returns:
        str: Corrected text.
    """
    blob = TextBlob(text)
    corrected_text = str(blob.correct())

    # Additional correction logic can be added here using the description

    return corrected_text

def inpaint_image(image: Image.Image, bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]]) -> Image.Image:
    """
    Perform inpainting on the image to fill white rectangles with suitable background.

    Args:
        image (Image.Image): Original image.
        bounding_boxes (List[Tuple[str, Tuple[int, int, int, int]]]): List of bounding boxes to inpaint.

    Returns:
        Image.Image: Inpainted image.
    """
    # Initialize the Stable Diffusion inpainting pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    inpainted_image = image.copy()
    mask = Image.new("L", (image.width, image.height), 0)
    draw = ImageDraw.Draw(mask)

    for _, (x, y, w, h) in bounding_boxes:
        draw.rectangle([x, y, x+w, y+h], fill=255)

    # Convert images to required format
    image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    mask_np = np.array(mask).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).unsqueeze(0).permute(0, 3, 1, 2)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

    # Perform inpainting
    inpainted_image_tensor = pipe(prompt="", image=image_tensor, mask_image=mask_tensor).images[0]
    inpainted_image = Image.fromarray((inpainted_image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    return inpainted_image

def compose_image(original_image: Image.Image, corrected_texts: List[str], bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]]) -> Image.Image:
    """
    Integrate corrected text back into the original image within the expanded bounding boxes.

    Args:
        original_image (Image.Image): Original image.
        corrected_texts (List[str]): List of corrected texts.
        bounding_boxes (List[Tuple[str, Tuple[int, int, int, int]]]): List of tuples containing original text and their expanded bounding boxes.

    Returns:
        Image.Image: Image with corrected text integrated.
    """
    image = original_image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for (text, (x, y, w, h)), corrected_text in zip(bounding_boxes, corrected_texts):
        draw.text((x + w // 10, y + h // 10), corrected_text, font=font, fill="black")
    
    return image
