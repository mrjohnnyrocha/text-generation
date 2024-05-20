# main.py
import os
import sys
from img.core import ImageProcessor
from PIL import Image
from typing import List, Tuple

def process_image(image_path: str, additional_prompt: str = "") -> None:
    """
    Process the image to detect, correct text, and save the final image.

    Args:
        image_path (str): Path to the input image file.
        additional_prompt (str): Additional prompt to describe the image better.
    """
    processor = ImageProcessor()

    # Load the uploaded image
    uploaded_image: Image.Image = processor.load_image(image_path)

    # Interpret the image and get description
    description: str = processor.interpret_image(uploaded_image, additional_prompt)
    print("Description:", description)

    # Step 2: Detect text using OCR and get expanded bounding boxes
    bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = processor.get_bounding_boxes(uploaded_image)

    # Step 3: Correct the text using the description
    corrected_texts: List[str] = [processor.correct_text(text, description, additional_prompt) for text, _ in bounding_boxes]

    # Step 4: Integrate corrected text back into the image
    final_image: Image.Image = processor.inpaint_image(uploaded_image, bounding_boxes, corrected_texts)

    # Generate artistic text
    artistic_text_image: Image.Image = processor.generate_artistic_text(additional_prompt)

    # Ensure the output directory exists
    output_dir: str = "mnt/data/generated"
    os.makedirs(output_dir, exist_ok=True)

    # Save the final image
    final_image_path: str = os.path.join(output_dir, "example2.png")
    final_image.save(final_image_path)
    final_image.show()

    # Save the artistic text image
    artistic_text_image_path: str = os.path.join(output_dir, "artistic_text.png")
    artistic_text_image.save(artistic_text_image_path)
    artistic_text_image.show()

if __name__ == "__main__":
    additional_prompt = ""
    if len(sys.argv) > 2:
        additional_prompt = sys.argv[2]

    process_image("mnt/data/source/example2.webp", additional_prompt)
