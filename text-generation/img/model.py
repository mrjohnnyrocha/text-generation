# img/model.py

import tensorflow as tf
from transformers import T5Tokenizer, T5ForConditionalGeneration
from diffusers import StableDiffusionInpaintPipeline
from tensorflow.keras import Model
from PIL import Image, ImageDraw
import numpy as np
import torch
from typing import List, Tuple
from scripts.logging import Logger
from accelerate import init_empty_weights

logger = Logger()
logger = logger.start()


class TrainableModel(Model):
    def __init__(self):
        super(TrainableModel, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("google/byt5-small")
        self.text_correction_model = T5ForConditionalGeneration.from_pretrained(
            "google/byt5-small"
        )
        self.inpainting_model = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        )
        self.inpainting_model.to("cuda" if torch.cuda.is_available() else "cpu")

    def call(self, inputs, training=False):
        # Define your forward pass
        pass

    def correct_text(self, text: str, context: str) -> str:
        try:
            input_text = f"Context: '{context}' Correct the following text: '{text}'"
            inputs = self.tokenizer(input_text, return_tensors="pt")
            outputs = self.text_correction_model.generate(**inputs)
            corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(
                "Text corrected with context",
                extra={"original_text": text, "corrected_text": corrected_text},
            )
            return corrected_text
        except Exception as e:
            logger.error("Failed to correct text with context", extra={"error": str(e)})
            raise

    def inpaint_image(
        self,
        image: Image.Image,
        bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]],
        corrected_texts: List[str],
    ) -> Image.Image:
        try:
            mask = Image.new("L", (image.width, image.height), 0)
            draw = ImageDraw.Draw(mask)

            for _, (x, y, w, h) in bounding_boxes:
                draw.rectangle([x, y, x + w, y + h], fill=255)

            image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            mask_np = np.array(mask).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0).permute(0, 3, 1, 2)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

            result = self.inpainting_model(
                prompt="", image=image_tensor, mask_image=mask_tensor
            )

            inpainted_image_np = np.array(result.images[0]).astype(np.uint8)
            inpainted_image = Image.fromarray(inpainted_image_np).resize(image.size)

            final_image = Image.composite(
                inpainted_image,
                image,
                Image.fromarray((mask_np * 255).astype(np.uint8)).resize(image.size),
            )
            logger.info("Image inpainted with background maintained")
            return final_image
        except Exception as e:
            logger.error("Failed to inpaint image", extra={"error": str(e)})
            raise

    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, labels = data
        predictions = self(inputs, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}
