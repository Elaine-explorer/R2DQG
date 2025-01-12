import argparse

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch
from peft import LoraConfig, TaskType, get_peft_model

def TemplateGeneratorTrainer(args):
    df = pd.read_json(args.train_template_path)

    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path + 'LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def process_func(example):
        MAX_LENGTH = 500
        instruction = tokenizer(
            f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False)
        response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(args.model_path + 'LLM-Research/Meta-Llama-3-8B-Instruct',
                                                 device_map="auto", torch_dtype=torch.bfloat16)

    model.enable_input_require_grads()

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)

    args = TrainingArguments(
        output_dir=args.model_path + f"/{args.lora_name}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=6,
        save_steps=500,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', default="./outputData/WQ")
    parser.add_argument('--train_template_path', default="./outputData/WQ/trainTemplates.json")
    parser.add_argument('--model_path', default="/data/rym/premodel/Meta-Llama-3-8B-Instruct/")
    parser.add_argument('--lora_name', default="TemplateGenerator")

    args = parser.parse_args()

    TemplateGeneratorTrainer(args)

if __name__ == '__main__':
    main()
