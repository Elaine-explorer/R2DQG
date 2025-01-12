import random
import openai
from tqdm import tqdm
import argparse
import json
import os
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

openai.api_key = "sk-xxxx"

class LlamaInference:
    def __init__(self, model_base_path, finetune_checkpoint, cuda_visible_devices="0"):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        self.model_base_path = model_base_path
        self.finetune_path = os.path.join(model_base_path, finetune_checkpoint)

        self.tokenizer = AutoTokenizer.from_pretrained(model_base_path)

        self.model = AutoModelForCausalLM.from_pretrained(model_base_path, device_map="auto", torch_dtype=torch.bfloat16)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=True,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )

        self.model = PeftModel.from_pretrained(self.model, model_id=self.finetune_path, config=lora_config)
        self.model.to("cuda")

    def generate_question(self, triples_info, answers):

        prompt = (
            f"Generate question template based on the provided triples and answer entity. Ensure the logical relationships between entities are clearly represented in each template. The important entities should be replaced with _.\n"
            f'Subgraph:{",".join(triples_info)}\n'
            f'Answer:{",".join(answers)}\n'
            f'Your generated question template:'
            )

        model_inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to("cuda")

        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def generate_corrector_question(self, triples_info, answers, draftQuestion):
        corrector_prompt = (
            f"Please refine the given initial question to generate a detailed and specific complex refined question using the information in the triples and the answer entities.  \n"
            f'Subgraph:{",".join(triples_info)}\n'
            f'Answer:{",".join(answers)}\n'
            f'Initial Question: {draftQuestion}  '
            f'Your refined question:'
        )

        model_inputs = self.tokenizer([corrector_prompt], return_tensors="pt", padding=True).to("cuda")

        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

def ChatGPTResponse(content, default="Default response", model="gpt-4-turbo"):
    temperature = 1
    max_tokens = 500
    messages = []
    messages.append({'role': 'assistant', 'content': "You are a useful assistant."})
    messages.append({"role": "user",
                     "content": content})

    res = ""
    try:
        output = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        res = output.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        res = default
    return res

def getNovelSubgraph(subkg):
    g_nodes = subkg['g_node_names']
    g_adj = subkg['g_adj']
    relations = []
    seq = []
    triples = []
    for key, value in g_adj.items():
        subject = g_nodes[key]
        if subject == "none":
            continue
        else:
            for k in list(value.keys()):
                obj = g_nodes[k]
                if obj == "none" and k in list(g_adj.keys()):
                    value.update(g_adj[k])

    for key, value in g_adj.items():
        subject = g_nodes[key]
        if subject == "none":
            continue
        else:
            for k, relation in value.items():
                obj = g_nodes[k]
                if obj == "none":
                    continue
                else:
                    subject = subject.strip().lower()
                    obj = obj.strip().lower()
                    relation = relation.strip().lower()
                    relations.append(relation)
                    relation = relation.strip().split('/')[-1]
                    if relation.find('_')!=-1:
                        relation = relation.split('_')
                        relation = ' '.join(relation).strip()
                    fact = "<{}, {}, {}>".format(subject, relation, obj)
                    seq.append(fact)
                    triples.append({"subject": subject, "relation": relation, "object": obj})  # 添加三元组
    subkg = ", ".join(seq)
    return triples

def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def genrateInitQuestion(args):

    generator = LlamaInference(args.model_path, args.genertor_finetune_checkpoint, cuda_visible_devices="0,1")

    corrector = LlamaInference(args.model_path, args.corrector_finetune_checkpoint, cuda_visible_devices="0,1")

    train_template_data = load_data(args.gold_template_path)

    prefix_prompts = []
    for data in train_template_data:
        question = data["question"]
        template_info = data["template_info"]
        answers = data["answers"]
        triples = data["triples"]

        triples_info = []
        for triple in triples:
            triples_info.append(f"<{triple['subject']},{triple['relation']},{triple['object']}>")

        if template_info != []:
            for template in template_info:
                prefix_prompt = (
                    f'Subgraph:{",".join(triples_info)}\n'
                    f'Answer:{",".join(answers)}\n'
                    f'Question template: {template}'
                    f'Your generated question:'
                )
                prefix_prompts.append(prefix_prompt)

    contents = []
    input_datas = []
    try:
        with open(args.data_path, 'r') as f:
            for line in f:
                input_datas.append(json.loads(line.strip()))
    except Exception as e:
        raise e

    for i, data in tqdm(enumerate(input_datas)):
        qId = data['qId']
        answers = data['answers']
        subkg = data['inGraph']
        triples = getNovelSubgraph(subkg)
        triples_info = []
        for triple in triples:
            triples_info.append(f"<{triple['subject']},{triple['relation']},{triple['object']}>")

        for i in range(args.k):
            template = generator.generate_question(triples_info, answers)

            prefix_sub_prompts = random.sample(prefix_prompts, 10)

            prompt = (
                f"As a question generator, your task is to generate the question draft solely based on the provided information: the question, the answer, the question template. Please generate a detailed and specific complex question using the provided template and the information in the triples related to the answer. The question should include all relevant details from the triples while avoiding directly mentioning the answer in the question itself.\n"
                f"{''.join(prefix_sub_prompts)}"
                f'Subgraph:{",".join(triples_info)}\n'
                f'Answer:{",".join(answers)}\n'
                f'Question template: {template}'
                f'Your generated question:'
                )

            draftQuestion = ChatGPTResponse(prompt)

            Question = corrector.generate_corrector_question(triples_info, answers, draftQuestion)

            temp = {"qId": f"{qId}_{i+1}",
                    "draftQuestion": draftQuestion,
                    "Question": Question
                    }
            contents.append(temp)

    with open(os.path.join(args.output_dir, f"{args.model_name}@{args.k}.json"), 'w', encoding='utf-8') as outfile:
        json.dump(contents, outfile, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default="./outputData/WQ")
    parser.add_argument('--model_name', default="draftQuestion")
    parser.add_argument('--gold_template_path', default="./outputData/WQ/gold_template_question.json")
    parser.add_argument('--template_path', default="./outputData/WQ/templates@10.json")
    parser.add_argument('--data_path', default="./data/WQ/test.json")
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--model_path', default="/data/rym/premodel/Meta-Llama-3-8B-Instruct/")
    parser.add_argument('--genertor_finetune_checkpoint', default="TemplateGenerator/checkpoint-10000")
    parser.add_argument('--corrector_finetune_checkpoint', default="DraftCorrector/checkpoint-10000")
    args = parser.parse_args()

    genrateInitQuestion(args)

if __name__ == '__main__':
    main()