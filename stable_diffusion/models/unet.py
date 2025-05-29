from diffusers import UNet2DConditionModel

class UNet:
    def __init__(self, model_path):
        self.model = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")