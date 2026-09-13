import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL", "aisingapore/Apertus-SEA-LION-v4-8B-IT")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/models/sealion-audit-adapter")

class ModelService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
        if torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
        self.model = PeftModel.from_pretrained(base, ADAPTER_PATH) if os.path.exists(ADAPTER_PATH) else base
        self.model.eval()

    def generate(self, messages, max_new_tokens=256):
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @staticmethod
    def extract_json(text):
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError(f"Model did not return JSON: {text}")
            return json.loads(match.group(0))
