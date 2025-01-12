from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

class Llama3Lora:
    def __init__(self, mode_path, lora_path, config, device='cuda'):
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(mode_path)
        model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

    def inference(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=1,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response