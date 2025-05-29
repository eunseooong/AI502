from config import Config
from diffusion.stable_diffusion import StableDiffusionPipeline

def main():
    config = Config()
    pipeline = StableDiffusionPipeline(config)
    prompt = "a cat"
    image = pipeline.generate(prompt)
    image.save("output.png")

if __name__ == "__main__":
    main()