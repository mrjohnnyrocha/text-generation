
## Text Generation and Inpainting Pipeline

This project provides a comprehensive pipeline for processing images containing text, correcting the text, and generating artistic text using advanced AI models. The pipeline includes functionalities for image loading, text detection, text correction, image inpainting, and text generation in an artistic style.


## Installation

Installation
To set up the project, follow these steps:

Clone the repository:

```bash
git clone https://github.com/your-repo/text-generation.git
cd text-generation
Install dependencies:
```


## Usage

```bash
poetry install
Activate the virtual environment:
```

```bash
poetry shell
Usage
To run the pipeline, use the following command:
```

```bash
python main.py <path_to_image> [additional_prompt]
<path_to_image>: Path to the image file to be processed.
[additional_prompt] (optional): Additional prompt to describe the image better.
Example:
```

```bash
python main.py "mnt/data/source/example2.webp" "A detailed description of the image."
```

## Components

The main.py script handles the overall processing flow:

Loads the uploaded image.
Interprets the image and retrieves a description.
Detects text using OCR and obtains expanded bounding boxes.
Corrects the text using the description.
Inpaints the image to integrate corrected text.
Generates artistic text based on the additional prompt.
Saves the final image and artistic text image.

- **EasyOCR**: For text detection in images.
- **T5 Model**: For correcting the detected text.
- **Stable Diffusion**: For image inpainting.
- **DeepFloyd IF**: For generating artistic text.

```bash
ImageProcessor Class
load_image(image_path: str) -> Image.Image: Loads an image from the specified path.
interpret_image(image: Image.Image, additional_prompt: str = "") -> str: Interprets the image and provides a description.
get_bounding_boxes(image: Image.Image, expansion: int = 5) -> List[Tuple[str, Tuple[int, int, int, int]]]: Detects text and returns expanded bounding boxes.
correct_text(text: str, description: str, additional_prompt: str) -> str: Corrects the detected text using contextual information.
compose_image(original_image: Image.Image, corrected_texts: List[str], bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]]) -> Image.Image: Integrates corrected text back into the image.
inpaint_image(image: Image.Image, bounding_boxes: List[Tuple[str, Tuple[int, int, int, int]]], corrected_texts: List[str]) -> Image.Image: Inpaints the image to replace text regions with appropriate backgrounds.
generate_artistic_text(prompt: str) -> Image.Image: Generates artistic text based on the provided prompt.
```
## Features

- Comprehensive pipeline for processing images containing text.
- Advanced AI models for text correction and generation.
- Image inpainting capabilities.


# Copyright

Copyright (c) 2024 mrjohnnyrocha

# License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to:

- Use
- Copy
- Modify
- Merge

And to permit persons to whom the Software is furnished to do so, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 

IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

