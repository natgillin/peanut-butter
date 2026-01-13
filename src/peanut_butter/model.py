import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader
from .data import PhraseDataset

class TranslationModel:
    def __init__(self, config):
        self.cfg = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None

    def setup(self):
        print(f"[Model] Setup: Init={self.cfg.weight_init}, Type={self.cfg.training_type}")
        
        if self.cfg.weight_init == "random":
            # --- Initialize from Scratch (Architecture only) ---
            print("[Model] Initializing RANDOM weights from config...")
            conf = AutoConfig.from_pretrained(self.cfg.model_id)
            self.model = AutoModelForCausalLM.from_config(conf)
            # Must cast to appropriate dtype and move to device
            self.model.to(dtype=torch.float16, device=self.cfg.device) 
        else:
            # --- Load Pretrained Weights ---
            print("[Model] Loading PRETRAINED weights...")
            if self.cfg.training_type == "lora":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_id, quantization_config=bnb_config, device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.cfg.model_id, torch_dtype=torch.float16, device_map="auto"
                )

        # Apply LoRA if requested
        if self.cfg.training_type == "lora":
            print("[Model] Applying LoRA adapters...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                r=16, lora_alpha=32, 
                target_modules=["q_proj", "v_proj"]
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        else:
            # Full training
            self.model.train()
            if self.cfg.weight_init == "pretrained":
                self.model.gradient_checkpointing_enable()

    def train(self, phrase_data):
        dataset = PhraseDataset(phrase_data)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        
        print(f"[Training] Starting loop for {self.cfg.epochs} epochs...")
        self.model.train()
        
        for epoch in range(self.cfg.epochs):
            total_loss = 0
            for batch in loader:
                # Prompt: "Translate Phrase: SRC => TGT"
                prompts = [f"Translate Phrase: {s} => {t}{self.tokenizer.eos_token}" 
                           for s, t in zip(batch['src'], batch['tgt'])]
                
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, 
                                        truncation=True, max_length=128).to(self.cfg.device)
                
                # Standard Causal Loss
                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            
            print(f"  Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")
        
        self.model.eval()
