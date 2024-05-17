# main.py

import os
import sys
from img.core import load_uploaded_image, interpret_image, get_bounding_boxes, correct_text, inpaint_image, compose_image
from PIL import Image
from typing import List, Tuple

def process_image(image_path: str, additional_prompt: str = "") -> None:
    """
    Process the image to detect, correct text, and save the final image.

    Args:
        image_path (str): Path to the input image file.
        additional_prompt (str): Additional prompt to describe the image better.
    """
    # Load the uploaded image
    uploaded_image: Image.Image = load_uploaded_image(image_path)

    # Interpret the image and get description
    description: str = interpret_image(uploaded_image, additional_prompt)
    
    # Step 2: Detect text using OCR and get expanded bounding boxes
    bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = get_bounding_boxes(uploaded_image)
    
    # Step 3: Correct the text using the description
    corrected_texts: List[str] = [correct_text(text, description) for text, _ in bounding_boxes]
    
    # Step 4: Inpaint the image to fill white rectangles with suitable background
    inpainted_image: Image.Image = inpaint_image(uploaded_image, bounding_boxes)
    
    # Step 5: Integrate corrected text back into the image
    final_image: Image.Image = compose_image(inpainted_image, corrected_texts, bounding_boxes)
    
    # Ensure the output directory exists
    output_dir: str = 'mnt/data/generated'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the final image
    final_image_path: str = os.path.join(output_dir, 'example.png')
    final_image.save(final_image_path)
    final_image.show()

if __name__ == "__main__":
    additional_prompt = ""
    if len(sys.argv) > 2:
        additional_prompt = sys.argv[2]
    
    # Replace 'uploaded_image_path' with the actual path to the uploaded image
    process_image('mnt/data/source/example1.webp', additional_prompt)
