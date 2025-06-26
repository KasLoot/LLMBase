from PIL import Image
import numpy as np

def generate_modelcard_kernel():
    w, h = 400, 68
    # Create a horizontal gradient from 255 (white) to 0 (black)
    gradient = np.linspace(255, 0, w, dtype=np.uint8)
    # Repeat the gradient for all rows and all 3 color channels
    image = np.stack([gradient]*h, axis=0)
    image = np.stack([image]*3, axis=2)

    image = Image.fromarray(image, 'RGB')
    image.save('demo_model_cover.png')
