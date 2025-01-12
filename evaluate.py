import argparse
import json
import os
from evaluation_methods import TextEvaluator, DiversityCalculator
from collections import defaultdict, Counter
import statistics


def load_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def create_dictionary(data, gts):
    return {
        item["qId"].split("_")[0]: [item["qText"].lower().replace("\n", "").replace("?", "").strip()]
        for item in data
        if item["qId"].split("_")[0] in gts
    }

def process_files(data, gts):
    return create_dictionary(data, gts)

def distinct_n_score(texts, n):
    ngram_counts = Counter()
    total_ngrams = 0

    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        ngram_counts.update(ngrams)
        total_ngrams += len(ngrams)

    if total_ngrams == 0:
        return 0.0

    distinct_ngrams = len(ngram_counts)
    return distinct_ngrams / total_ngrams

def compute_distinct_n_scores(diverse_data, n=1):
    diverse_dic = defaultdict(list)
    for data in diverse_data:
        qId = data["qId"].split("_")[0]
        diverse_dic[qId].append(data["qText"].lower().replace("\n", "").replace("?", "").strip())

    dis_1_vals = []
    for qId, generated_questions in diverse_dic.items():
        dis_1_vals.append(distinct_n_score(generated_questions, n))
    return sum(dis_1_vals) / len(dis_1_vals)

def compute_diverse_k_scores(diverse_data, gts, diverse_evaluator, k=3):
    diverse_dic = defaultdict(list)

    for data in diverse_data:
        qId = data["qId"].split("_")[0]
        diverse_dic[qId].append(data["qText"].lower().replace("\n", "").replace("?", "").strip())

    diverse_k_scores = []

    for qId, generated_questions in diverse_dic.items():
        if f"{qId}" not in gts:
            continue
        ground_truth = gts[f"{qId}"][0]
        diverse_k_score = diverse_evaluator.compute_diverse_k(generated_questions, ground_truth, k)
        diverse_k_scores.append(diverse_k_score)

    return sum(diverse_k_scores) / len(diverse_k_scores) if diverse_k_scores else 0, diverse_k_scores

def evaluate(args):
    model_save_path = args.output_dir
    k = args.k
    model_name = args.model_name

    evaluator = TextEvaluator()
    diverse_evaluator = DiversityCalculator()

    base_path = os.path.join(model_save_path, "BaseAll.json")
    with open(base_path, "r") as file:
        data = json.load(file)

    gts = {}
    for item in data:
        qId = item["qId"]
        truth_question = item["truth_question"].lower().replace("\n", "").replace("?", "").strip()
        gts[f"{qId}"] = [truth_question]

    data_path = os.path.join(model_save_path, f"{model_name}.json")
    with open(data_path, "r") as file:
        sgsh_diverse_data = json.load(file)

    dis_1 = compute_distinct_n_scores(sgsh_diverse_data, 1)
    print(f"dis-1: {dis_1:.4f}")

    diverse_k_scores, diverse_k_scores_list = compute_diverse_k_scores(sgsh_diverse_data, gts, diverse_evaluator, k)
    print(f"Diverse@{k}: {diverse_k_scores:.4f}")

    data = load_data(data_path)
    scores_dic = {}
    for i in range(k):
        test_sentences = []

        id = str(i + 1)
        sub_data = []
        sub_gts = {}
        for item in data:
            qId = item["qId"]
            qText = item["qText"].lower().replace("\n", "").replace('\\"', "").replace("?", "").strip()
            if id == qId.split('_')[1]:
                temp = {
                    "qId": qId.split('_')[0],
                    "qText": qText
                }
                sub_data.append(temp)
                sub_gts[qId.split('_')[0]] = gts[qId.split('_')[0]]
                test_sentences.append(f"{qText}|{gts[qId.split('_')[0]]}")
        data_dic = process_files(sub_data, gts)

        scores = evaluator.evaluate(sub_gts, data_dic)

        if scores_dic == {}:
            for metric in scores:
                scores_dic[metric] = [scores[metric]]
        else:
            for metric in scores:
                scores_dic[metric].append(scores[metric])

    for metric in scores_dic:
        print(f"{metric}: {statistics.mean(scores_dic[metric]):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default="./outputData/WQ")
    parser.add_argument('--model_name', default="R2DQG@10")
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()