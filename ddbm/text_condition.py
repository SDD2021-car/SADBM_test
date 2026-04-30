import os
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class TextConditionEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', device='cpu', freeze=True, local_files_only=True):
        super().__init__()
        model_id = os.environ.get('CLIP_TEXT_MODEL_PATH', model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, local_files_only=local_files_only)
        self.text_encoder = self.text_encoder.to(device)
        if freeze:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, answers):
        tokens = self.tokenizer(
            answers,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        out = self.text_encoder(**tokens)
        return out.pooler_output