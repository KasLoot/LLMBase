import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
import time

def test_vision_model():

    # Load the model and processor
    ckpt = "/home/yuxin/models/siglip2-large-patch16-256"
    model = AutoModel.from_pretrained(ckpt, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(ckpt)


    total_time = 0

    for i in range(10):

        # Load a single image
        image = load_image("/home/yuxin/Downloads/DemoModel_cover.png")
        inputs = processor(images=[image for _ in range(64)], return_tensors="pt").to(model.device)

        # --- The Key Step ---
        # Call the vision_model directly to get detailed outputs
        with torch.no_grad():
            t1 = time.time()
            vision_outputs = model.vision_model(**inputs)
            # print(vision_outputs.shape)

            # Extract the last_hidden_state from the outputs
            # This contains all patch tokens embeddings
            last_hidden_state = vision_outputs.last_hidden_state
            t2 = time.time()
            total_time += t2 - t1

            # Print the shape to see the full sequence: torch.Size([1, 256, 1024])
            # print(last_hidden_state.shape)


    print(f"average batch time: {total_time / 10}")
    # batch = 32: average batch time: 0.04168238639831543
    # batch = 64: average batch time: 0.07072806358337402


