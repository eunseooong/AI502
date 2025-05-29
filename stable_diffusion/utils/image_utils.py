from torchvision.transforms.functional import to_pil_image

def latents_to_image(image_tensor):
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_tensor = image_tensor[0].cpu()
    return to_pil_image(image_tensor)