import torch

class Scorer:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.device = model_wrapper.cfg.device

    def get_scores(self, src_phrase, candidate):
        """
        Returns (TM_Score, LM_Score)
        """
        # --- 1. TM Score (P(e|f)) ---
        prompt = f"Translate Phrase: {src_phrase} => "
        full_text = prompt + candidate + self.tokenizer.eos_token
        
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        labels = inputs.input_ids.clone()
        
        # Mask the prompt
        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        labels[:, :prompt_len] = -100
        
        with torch.no_grad():
            tm_out = self.model(**inputs, labels=labels)
        tm_score = -tm_out.loss.item()
        
        # --- 2. LM Score (P(e)) ---
        lm_inputs = self.tokenizer(candidate, return_tensors="pt").to(self.device)
        with torch.no_grad():
            lm_out = self.model(**lm_inputs, labels=lm_inputs.input_ids)
        lm_score = -lm_out.loss.item()
        
        return tm_score, lm_score
