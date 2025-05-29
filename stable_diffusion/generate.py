from config import Config
from diffusion.stable_diffusion import StableDiffusionPipeline

if __name__ == "__main__":
    config = Config()
    prompt = input("Enter prompt: ")
    pipeline = StableDiffusionPipeline(config)
    image = pipeline.generate(prompt)
    image.show()