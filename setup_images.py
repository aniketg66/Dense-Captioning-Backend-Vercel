import os
import shutil
from PIL import Image
import numpy as np

def create_sample_image(filename, size=(800, 600)):
    """Create a sample landscape image if none exists."""
    img = Image.new('RGB', size)
    pixels = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(pixels)
    img.save(filename)

def setup_sample_images():
    """Set up sample images in the static/images directory."""
    images_dir = "static/images"
    os.makedirs(images_dir, exist_ok=True)
    
    sample_images = ["landscape1.jpg", "landscape2.jpg", "landscape3.jpg", "text.jpg"]
    
    for image in sample_images:
        image_path = os.path.join(images_dir, image)
        if not os.path.exists(image_path):
            print(f"Creating sample image: {image}")
            create_sample_image(image_path)
        else:
            print(f"Sample image already exists: {image}")

if __name__ == "__main__":
    setup_sample_images()
    print("Sample images setup complete!") 