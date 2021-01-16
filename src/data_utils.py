import logging

import tensorflow_datasets as tfds
from spacy.lang.en import English
import numpy as np
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from allennlp.predictors.predictor import Predictor


SPECIAL_CHARS = ['-', '.', '\'', ',', '’']
TAGS = ['V', 'ARG0','ARG1','ARG2','ARG3','ARG4','ARGM']


def get_wiki_data(num_examples):
    split_str = 'train[:' + str(num_examples) + ']'
    ds = tfds.load('wiki40b', split=split_str, shuffle_files=True, try_gcs=True)
    data = []
    for ex in ds:
        data.append(ex)
    return data

# Map token vectors from BERT to 'words' from SRL
def map_words_to_vectors(words, tokenized_words):
    words = [w.lower() for w in words]
    logging.debug(words)
    logging.debug(tokenized_words)
    word2veclist = defaultdict(list)
    index = 0
    for token_index, tword in enumerate(tokenized_words):
        if tword in  ['[SEP]', '[CLS]']:
            logging.debug("skipping" + tword)
            continue
        if tword == words[index]:
            logging.debug("Adding " + tword  + " for "+ words[index])
            word2veclist[index].append(token_index)
            index += 1
        elif tword.startswith("##") or words[index].startswith(tword) or tword in SPECIAL_CHARS:
            word2veclist[index].append(token_index)
            logging.debug("Adding "+ tword+ " for "+ words[index])
            #print(words[index], tword[2:], words[index].endswith(tword[:2]))
            if tword.startswith("##") and words[index].endswith(tword[2:]):
                index += 1
            if tword == '.' and words[index].endswith(tword): # mr.
                index += 1
        elif tokenized_words[token_index - 1] in SPECIAL_CHARS:
            word2veclist[index].append(token_index)
            logging.debug("Adding " + tword + " for " + words[index])
            if words[index].endswith(tword[2:]):
                index += 1            
        else:
            logging.debug("Word " + words[index])
            logging.debug("Token " + tword)
            raise Exception("Tokens and words don't line up.")
    return word2veclist

def preproc_text(ex):
    ex = ex[10:]
    ex = ex.replace("_NEWLINE_", "\n")
    return ex

def build_repr(parse, word2veclist, vectors):
    unique_tags = set()
    sent_repr = []
    for verb in parse:
        tag_2_vec = defaultdict(list)
        assert len(verb['tags']) == len(word2veclist.keys())
        for i, tag in enumerate(verb['tags']):
            if tag == 'O' or tag[2] == "R" or tag[2] == "C":
                continue
            unique_tags.add(tag[2:])
            for vec_index in word2veclist[i]:
                tag_2_vec[tag.split("-")[1]].append(vectors[vec_index])
        for key in tag_2_vec.keys():
            assert key in TAGS, key +  str(list(unique_tags))
        verb_repr = []
        for tag in TAGS:
            if tag not in tag_2_vec:
                continue
            vecs = tag_2_vec[tag]
            verb_repr.append((tag, np.mean(np.array(vecs), axis=0)))
        sent_repr.append(verb_repr)
    return sent_repr

def create_nlp():
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    return nlp

def make_lm_encoder():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tokenizer, model

def make_allennlp():
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    return predictor

def encode_data(data):
    nlp = create_nlp()
    tokenizer, model = make_lm_encoder()
    predictor = make_allennlp()
    reprs = []
    for d in data:
        text = d['text'].numpy().decode('utf-8')
        for para in text.split("_START_"):
            if not para.startswith("PARAGRAPH_"):
                continue
            example = preproc_text(para)
            doc = nlp(example)
            for sent in doc.sents:
                span = sent.text
                lm_tokenized_input = tokenizer(span, return_tensors='pt')
                output = model(**lm_tokenized_input)
                token_vecs = output['last_hidden_state'][0].detach().numpy()
                srl_parse = predictor.predict(sentence=span)
                token_strs = tokenizer.convert_ids_to_tokens(lm_tokenized_input['input_ids'].numpy()[0])
                word2veclist = map_words_to_vectors(srl_parse['words'], token_strs)
                sent_repr = build_repr(srl_parse['verbs'], word2veclist, token_vecs)
                reprs.append(sent_repr)
    return reprs
