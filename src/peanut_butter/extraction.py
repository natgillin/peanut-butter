import torch
import json
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class PhraseExtractor:
    def __init__(self, config):
        self.cfg = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None

    def load_model(self):
        """Loads a pretrained model solely for identifying phrases."""
        print("[Extraction] Loading pretrained model (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

    def unload_model(self):
        """Frees VRAM so we can load the training model next."""
        print("[Extraction] Unloading model to free VRAM...")
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        self.model = None

    def extract(self, dataset):
        if self.model is None:
            self.load_model()
            
        phrase_data = []
        print(f"[Extraction] Processing {len(dataset)} sentences...")
        
        for i, item in enumerate(dataset):
            # Handle WMT14 nested structure
            src = item['translation']['de']
            tgt = item['translation']['en']
            
            prompt = (
                f"Align the following German and English sentences into phrase pairs.\n"
                f"Format: JSON list of objects with keys 'src' and 'tgt'.\n"
                f"German: {src}\nEnglish: {tgt}\nJSON Output:"
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.cfg.device)
            
            with torch.no_grad():
                # Greedy decoding (deterministic), explicitly silencing sampling params
                ids = self.model.generate(
                    **inputs, max_new_tokens=256, 
                    do_sample=False, temperature=None, top_p=None
                )
            
            text = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            phrase_data.extend(self._parse_json(text))
            
        return phrase_data

    def _parse_json(self, text):
        try:
            s = text.find('[')
            e = text.rfind(']') + 1
            if s != -1 and e != -1:
                data = json.loads(text[s:e])
                return [p for p in data if 'src' in p and 'tgt' in p]
        except:
            pass
        return []
