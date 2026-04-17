import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class CircleMaskCrop:
    def __call__(self, img):
        """
        img: Tensor of shape (C, H, W)
        returns: masked image (C, H, W)
        """
        C, H, W = img.shape
        Y, X = torch.meshgrid(
            torch.arange(H, device=img.device),
            torch.arange(W, device=img.device),
            indexing="ij"
        )
        center_y, center_x = H // 2, W // 2
        radius = min(center_y, center_x)

        dist = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = (dist <= radius).float()[None, :, :]   # shape (1, H, W)
        return img * mask
    

class RandomRotationFromList:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = self.angles[torch.randint(0, len(self.angles), (1,)).item()]
        return TF.rotate(img, angle)
