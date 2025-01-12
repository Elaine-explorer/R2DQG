from bleu.bleu import Bleu
from meteor.meteor import Meteor
from typing import List, Set
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import torch

class DiversityCalculator:
    def __init__(self):
        model_path = '/data/rym/premodel/all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def compute_embeddings(self, texts: List[str]):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def token_set(self, question: str) -> Set[str]:
        return set(question.split())

    def compute_diverse_k(self, generated_questions: List[str], ground_truth: str, k: int, threshold: float = 0.7) -> float:
        ground_truth_embedding = self.compute_embeddings([ground_truth])
        question_embeddings = self.compute_embeddings(generated_questions[:k])

        relevances = cosine_similarity(question_embeddings, ground_truth_embedding).flatten()

        relevant_questions = [generated_questions[i] for i in range(k) if relevances[i] >= threshold]

        diverse_k_score = 0.0
        n = len(relevant_questions)

        for i in range(n):
            for j in range(i + 1, n):
                ti = self.token_set(relevant_questions[i])
                tj = self.token_set(relevant_questions[j])
                diverse_score = (len(ti - tj) + len(tj - ti)) / len(ti.union(tj))
                diverse_k_score += diverse_score

        if n > 1:
            diverse_k_score /= (n * (n - 1)) / 2

        return diverse_k_score


class TextEvaluator:
    def __init__(self):
        self.met = Meteor()
        self.bleu = Bleu(4)

    def evaluate(self, gts, res):
        results = {}

        meteor_score = self.met.compute_score(gts, res)[0]
        results['METEOR'] = meteor_score

        bleu_scores = self.bleu.compute_score(gts, res)[0]
        results['BLEU-1'] = bleu_scores[0]
        return results