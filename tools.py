import re

from scipy.spatial.distance import cosine
import nltk
from nltk import pos_tag, word_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class ACVTreeSimilarity:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    class ACVNode:
        def __init__(self, label, vector=None, weight=1.0, children=None):
            self.label = label
            self.vector = vector
            self.weight = weight
            self.children = children if children else []

        def add_child(self, child):
            self.children.append(child)

    def calculate_similarity(self, node1, node2):
        if node1.vector is not None and node2.vector is not None:
            sim = 1 - cosine(node1.vector, node2.vector)
            return node1.weight * node2.weight * sim
        return 0

    def build_acv_tree(self, sentence):
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)

        nodes = []
        for word, pos in pos_tags:
            vector = self.model.encode(word)
            weight = 1.0
            node = self.ACVNode(label=pos, vector=vector, weight=weight)
            nodes.append(node)

        root = self.ACVNode(label="S", children=nodes)
        return root

    def calculate_tree_similarity(self, node1, node2):
        if not node1.children and not node2.children:
            return self.calculate_similarity(node1, node2)

        if not node1.children or not node2.children:
            return 0

        total_similarity = 0
        for child1 in node1.children:
            for child2 in node2.children:
                total_similarity += self.calculate_tree_similarity(child1, child2)

        return total_similarity

    def compute_similarity(self, sentence1, sentence2):
        tree1 = self.build_acv_tree(sentence1)
        tree2 = self.build_acv_tree(sentence2)
        return self.calculate_tree_similarity(tree1, tree2)


def extract_template_str_info(template_str):
    template_info = []

    matches = re.findall(r"Template (\d+):(.*?)(?=Template \d+:|$)", template_str, re.DOTALL)

    for match in matches:
        template_content = match[1]
        template_info.append(template_content)
    return template_info

