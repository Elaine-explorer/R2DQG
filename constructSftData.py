import argparse
from tqdm import tqdm
import json
import warnings
from tools import ACVTreeSimilarity, extract_template_str_info
import openai

warnings.filterwarnings("ignore")

openai.api_key = "sk-xxxx"

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
                    relation = relation[0].strip().lower()
                    relations.append(relation)
                    relation = relation.strip().split('/')[-1]
                    if relation.find('_')!=-1:
                        relation = relation.split('_')
                        relation = ' '.join(relation).strip()
                    fact = "<{}, {}, {}>".format(subject, relation, obj)
                    seq.append(fact)
                    triples.append({"subject": subject, "relation": relation, "object": obj})
    return triples

def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

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

def genrateTrainData(args):
    acvt_similarity_calculator = ACVTreeSimilarity()
    contents = []
    corrector_contents = []
    template_contents = []
    input_datas = []
    try:
        with open(args.data_path, 'r') as f:
            for line in f:
                input_datas.append(json.loads(line.strip()))
    except Exception as e:
        raise e

    for i, data in tqdm(enumerate(input_datas)):

        qId = data["qId"]
        question = data['outSeq']
        answers = data['answers']
        subkg = data['inGraph']
        triples = getNovelSubgraph(subkg)
        triples_info = []
        for triple in triples:
            triples_info.append(f"<{triple['subject']},{triple['relation']},{triple['object']}>")

        prompt = (f"I want you to act as a language expert. You'll receive a subgraph, an answer, and a grount-truth question. Your task is produce {args.num} templates exhibit diverse linguistic structures and phrasing styles while maintaining semantic consistency with the input based on the question and proposed answer. Ensure the logical relationships between entities are clearly represented in each template. The important entities should be replaced with _. Please only reply with the question template and nothing else.\n"
                  f'Subgraph:{",".join(triples_info)}\n'
                  f'Answer:{",".join(answers)}\n'
                  f"Ground-truth question: {question}\n"
                  )

        for t_num in range(args.num):
            prompt += f"Template {t_num+1}: {{template}}\n"

        res_info = ChatGPTResponse(prompt)

        template_str = res_info.strip().replace("\n", "").replace('#', "").replace("*", "").strip()

        template_info = extract_template_str_info(template_str)

        templates = []

        for i, template in enumerate(template_info):

            if acvt_similarity_calculator.compute_similarity(template, question) < args.th:
                continue

            templates.append(template)

            prompt = (f"Generate question template based on the provided triples and answer entity. Ensure the logical relationships between entities are clearly represented in each template. The important entities should be replaced with _.\n"
                      f'Subgraph:{",".join(triples_info)}\n'
                      f'Answer:{",".join(answers)}\n'
                      f'Your generated question template:'
                      )

            temp = {
                "instruction": prompt,
                "input": "",
                "output": template
            }

            contents.append(temp)

            prompt = (f"As a question generator, your task is to generate the question draft solely based on the provided information: the question, the answer, the question template. Please generate a detailed and specific complex question using the provided template and the information in the triples related to the answer. The question should include all relevant details from the triples while avoiding directly mentioning the answer in the question itself.\n"
                      f'Subgraph:{",".join(triples_info)}\n'
                      f'Answer:{",".join(answers)}\n'
                      f'Question template: {template}'
                      f'Your generated question:'
                      )
            draftQuestion = ChatGPTResponse(prompt)

            corrector_prompt = (
                f"Please refine the given initial question to generate a detailed and specific complex refined question using the information in the triples and the answer entities.  \n"
                f'Subgraph:{",".join(triples_info)}\n'
                f'Answer:{",".join(answers)}\n'
                f'Initial Question: {draftQuestion}  '
                f'Your refined question:'
                )

            temp = {
                "instruction": corrector_prompt,
                "input": "",
                "output": question
            }

            corrector_contents.append(temp)

        temp = {
            "qId": qId,
            "question": question,
            "templates": templates,
            "res_info": res_info,
            "answers": answers,
            "triples": triples
        }
        template_contents.append(temp)

    with open(args.train_template_path, 'w', encoding='utf-8') as outfile:
        json.dump(contents, outfile, indent=4, ensure_ascii=False)

    with open(args.train_corrector_path, 'w', encoding='utf-8') as outfile:
        json.dump(corrector_contents, outfile, indent=4, ensure_ascii=False)

    with open(args.gold_template_path, 'w', encoding='utf-8') as outfile:
        json.dump(template_contents, outfile, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=f"./outputData/WQ")
    parser.add_argument('--data_path', default=f"./data/WQ/train.json")
    parser.add_argument('--num', type=int, default=3, help="Number of templates")
    parser.add_argument('--th', type=float, default=0.3, help="Similarity threshold")
    parser.add_argument('--train_template_path', default=f"./outputData/WQ/trainTemplates.json")
    parser.add_argument('--gold_template_path', default=f"./outputData/WQ/gold_template_question.json")
    parser.add_argument('--train_corrector_path', default=f"./outputData/WQ/trainCorrector.json")

    args = parser.parse_args()

    genrateTrainData(args)

if __name__ == '__main__':
    main()
