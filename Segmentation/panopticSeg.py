import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Load DETR model with panoptic segmentation head
model = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True)
model_panoptic, postprocessor = model
model_panoptic.eval()

# Prepare image
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img = Image.open('path/to/your/image.jpg')
img_tensor = transform(img).unsqueeze(0)

# Perform inference
with torch.no_grad():
    outputs = model_panoptic(img_tensor)

# Post-process the results
result = postprocessor(outputs, torch.as_tensor(img.size[::-1]).unsqueeze(0))[0]

# Get panoptic segmentation
panoptic_seg = result['panoptic_seg'].cpu().numpy()
segments_info = result['segments_info']

# Visualize results
plt.figure(figsize=(20,20))
plt.imshow(panoptic_seg[0])
plt.axis('off')
plt.savefig('panoptic_segmentation_output.png')
plt.close()

# Print segment information
for segment in segments_info:
    print(f"ID: {segment['id']}, Label: {segment['label']}, Area: {segment['area']}")
