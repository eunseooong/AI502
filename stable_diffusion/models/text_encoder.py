from transformers import CLIPTokenizer, CLIPTextModel
import torch
import os

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class TextEncoder:
    def __init__(self, model_path):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", token=HF_TOKEN)
        self.encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", token=HF_TOKEN)

    def encode(self, prompt):
        tokens = self.tokenizer(
            prompt,
            padding="max_length",  # ajoute du padding pour aligner
            max_length=77,        # max length fixe souvent 77 en SD
            truncation=True,
            return_tensors="pt"
        ).to(self.encoder.device)

        with torch.no_grad():
            return self.encoder(**tokens).last_hidden_state
