import torchvision.transforms.functional as TF


class ResizeCenterCropMixin:
    def __init__(self, resize=None, center_crop=None):
        # if isinstance(resize, (int, float)):
        # 	self.resize = (resize, resize)
        self.resize = resize
        self.center_crop = center_crop

    def resize_centercrop(self, img):
        if self.resize:
            img = TF.resize(img, self.resize)
        if self.center_crop:
            img = TF.center_crop(img, self.center_crop)
        return img
