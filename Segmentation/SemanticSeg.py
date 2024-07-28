import torch
import torchvision.transforms as T
from PIL import Image

# Load DeepLabV3 model
model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Prepare image
input_image = Image.open('path/to/your/image.jpg')
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# Perform inference
with torch.no_grad():
    output = model(input_batch)['out'][0]

output_predictions = output.argmax(0)

# Create a color-coded segmentation map
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

# Save the segmentation map
r.save('semantic_segmentation_output.png')
