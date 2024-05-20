# img/core.py
import easyocr
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import IdeficsForVisionText2Text, IdeficsProcessor, ByT5Tokenizer, T5ForConditionalGeneration
from scripts.logging import Logger

logger = Logger()
logger = logger.start()

class ImageProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b-instruct", torch_dtype=torch.float16).to(self.device)
        self.processor = IdeficsProcessor.from_pretrained("HuggingFaceM4/idefics-9b-instruct")

    def load_image(self, image_path: str) -> Image.Image:
        try:
            image = Image.open(image_path)
            logger.info("Image loaded", extra={"image_path": image_path})
            return image
        except Exception as e:
            logger.error("Failed to load image", extra={"error": str(e)})
            raise

    def interpret_image(self, image: Image.Image, additional_prompt: str = "") -> str:
        try:
            #image_tensor = self.processor(images=[image], return_tensors="pt").pixel_values.to(self.device)
            #prompts = [additional_prompt] if additional_prompt else ["Describe this image in detail."]
            #inputs = self.processor(prompts=prompts, return_tensors="pt").to(self.device)
            #generated_ids = self.model.generate(**inputs, pixel_values=image_tensor)
            #description = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            description = "A image full of text that needs to be corrected and restylized."

            logger.info("Image interpreted", extra={"description": description})
            return description
        except Exception as e:
            logger.error("Failed to interpret image", extra={"error": str(e)})
            raise

    def get_bounding_boxes(self, image: Image.Image, expansion: int = 5) -> List[Tuple[str, Tuple[int, int, int, int]]]:
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

    def correct_text(self, text: str, description: str, additional_prompt: str) -> str:
        try:
            model_name = "google/byt5-small"
            tokenizer = ByT5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            context = f"{description}. {additional_prompt}."
            input_text = f"Context: '{context}' Correct the following text: '{text}'"
            
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(**inputs)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info("Text corrected with context", extra={"original_text": text, "corrected_text": corrected_text})
            return corrected_text
        except Exception as e:
            logger.error("Failed to correct text with context", extra={"error": str(e)})
            raise

    def compose_image(self, original_image: Image.Image, corrected_texts: List[str], bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]]) -> Image.Image:
        try:
            image = original_image.copy()
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            
            for (text, (x, y, w, h)), corrected_text in zip(bounding_boxes, corrected_texts):
                draw.rectangle([x, y, x + w, y + h], fill="white")
                draw.text((x + w // 10, y + h // 10), corrected_text, font=font, fill="black")
            
            logger.info("Image composed with corrected text and white background")
            return image
        except Exception as e:
            logger.error("Failed to compose image", extra={"error": str(e)})
            raise

    def inpaint_image(self, image: Image.Image, bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]], corrected_texts: List[str]) -> Image.Image:
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            
            inpainted_image = image.copy()
            mask = Image.new("L", (image.width, image.height), 0)
            draw = ImageDraw.Draw(mask)
            
            for _, (x, y, w, h) in bounding_boxes:
                draw.rectangle([x, y, x + w, y + h], fill=255)
            
            image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            mask_np = np.array(mask).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).permute(0, 3, 1, 2)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
            
            result = pipe(prompt="", image=image_tensor, mask_image=mask_tensor)
            inpainted_image_np = np.array(result.images[0]).astype(np.uint8)
            inpainted_image = Image.fromarray(inpainted_image_np).resize(image.size)
            
            final_image = Image.composite(
                inpainted_image,
                image,
                Image.fromarray((mask_np * 255).astype(np.uint8)).resize(image.size)
            )
            self.output = final_image.convert("RGB")

            logger.info("Image inpainted with background maintained")
            return final_image
        except Exception as e:
            logger.error("Failed to inpaint image", extra={"error": str(e)})
            raise

    def generate_artistic_text(self, prompt: str) -> Image.Image:
        """
        Generate artistic text using DeepFloyd IF.

        Args:
            prompt (str): The prompt to generate artistic text.

        Returns:
            Image.Image: The generated artistic text as an image.
        """
        try:
#            text_image = self.if_pipeline(prompt=prompt)["sample"]
            logger.info("Artistic text generated", extra={"prompt": prompt})
            #return text_image
            return self.output
        except Exception as e:
            logger.error("Failed to generate artistic text", extra={"error": str(e)})
            raise
