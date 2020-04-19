import os
from tqdm import tqdm
from transformers import AutoTokenizer


def process(directory, mode='train', format='bag-word'):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    positive_dir = os.path.join(directory, 'pos')
    negative_dir = os.path.join(directory, 'neg')
    vectors = []
    labels = []
    for label_dir, label in {positive_dir: 1, negative_dir: 0}.items():
        print(label_dir, label)
        for file_name in tqdm(os.listdir(label_dir)):
            file_path = os.path.join(label_dir, file_name)
            f = open(file_path, "r")
            vector = tokenizer.encode(f.read(), max_length=256)
            vector = [0 for i in range(256-len(vector))] + vector
            vectors.append(vector)
            labels.append(label)
            f.close()
    print(vectors[0], labels[0])
    return vectors, labels

def tokenize(texts, format='bag-word'):
    print(1)
