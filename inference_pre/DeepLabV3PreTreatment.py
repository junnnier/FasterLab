import torch
from PIL import Image
from torchvision import transforms as T


class PreTreatment(object):
    def __init__(self):
        self.transform = T.Compose([T.Resize(513),
                                    T.CenterCrop(513),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

    def __call__(self, image_path, image_size, force_resize, device):
        image = Image.open(image_path).convert('RGB')
        image_shapes = (image.size, 1, (0,0))
        image = self.transform(image).unsqueeze(0)  # To tensor of NCHW
        image = image.to(device, dtype=torch.float32)

        return image, image_shapes