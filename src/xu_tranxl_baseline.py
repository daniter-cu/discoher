import argparse
import csv
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel


csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser(description='Baseline eval for xu insertion.')
parser.add_argument('--data_path', type=str, default='../../cross_domain_coherence/data/parsed_wsj/test_perm.tsv',
                    help='Name of save file')
args = parser.parse_args()

def get_paragraphs():
    paragraphs = []
    with  open(args.data_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            paragraphs.append(row[1].split("<PUNC>"))
    return paragraphs

def get_tok_and_model(device):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
    model.eval()
    model.to(device)

    return tokenizer, model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paragraphs = get_paragraphs()
    tokenizer, model = get_tok_and_model(device)

    all_acc = []
    for para in tqdm(paragraphs):
        indexed_tokens = tokenizer.encode(" ".join(para), return_tensors="pt", truncation=True, max_length=512)
        indexed_tokens = indexed_tokens[:, :512].to(device)
        with torch.no_grad():
            loss = model(indexed_tokens, labels=indexed_tokens, return_dict=True).loss
        true_loss = loss
        false_losses = []
        for i in range(len(para)):
            for j in range(len(para)):
                if i == j:
                    continue
                moved_sent = para[i]
                other_sents = para[:i] + para[i+1:]
                before_sents = other_sents[:j]
                after_sents = other_sents[j:]
                new_order = before_sents + [moved_sent] + after_sents
                indexed_tokens = tokenizer.encode(" ".join(new_order), return_tensors="pt", truncation=True, max_length=512)
                indexed_tokens = indexed_tokens[:, :512].to(device)
                with torch.no_grad():
                    loss = model(indexed_tokens, labels=indexed_tokens, return_dict=True).loss
                false_losses.append(loss)
        for loss in false_losses:
            all_acc.append(1 if loss > true_loss else 0)
        print("Currently at", np.mean(all_acc))
    print(np.mean(all_acc))

if __name__ == "__main__":
    main()
