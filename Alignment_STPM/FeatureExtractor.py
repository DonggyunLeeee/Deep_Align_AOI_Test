import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.feature = nn.Sequential(*list(base.children())[:-2])

    def forward(self, x):
        x = self.feature(x)
        return x.flatten(1)

def preprocess(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)

if __name__ == "__main__":
    model = FeatureExtractor()
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "feature_extractor.onnx",
                      input_names=["input"], output_names=["features"],
                      opset_version=11)
