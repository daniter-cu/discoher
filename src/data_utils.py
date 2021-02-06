import logging

import tensorflow_datasets as tfds
from spacy.lang.en import English
import numpy as np
import unicodedata
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from allennlp.predictors.predictor import Predictor


SPECIAL_CHARS = ['-', '.', '\'', ',', '’', '/', ':', '–', '(', '&']
TAGS = ['V', 'ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARGM']


def get_wiki_data(num_examples):
    split_str = 'train[:' + str(num_examples) + ']'
    ds = tfds.load('wiki40b', split=split_str, shuffle_files=True, try_gcs=True)
    data = []
    for ex in ds:
        data.append(ex)
    return data

def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def special_compare(raw_t1, bert_t2):
    if raw_t1.encode('ascii', 'ignore').decode("utf-8") == bert_t2:
        return True
    if _run_strip_accents(raw_t1) == bert_t2:
        return True
    return False

class NoStopException(Exception):
    pass

# Map token vectors from BERT to 'words' from SRL
def count_chars(toks):
    count = 0
    for t  in toks:
        if t in ['[SEP]', '[CLS]']:
            continue
        count += len(t)
        if t.startswith("##"):
            count -= 2
    return count
            
def new_map_words_to_vectors(allen_srl_words, bert_tokenized_words):
    # TODO: Add a error catcher so  if  we mess up we can just skip the example
    word2veclist = defaultdict(list)
    allen_srl_words = [w.lower() for w in allen_srl_words]
    all_srl_start_map = {}
    bert_tokenized_start_map = {}
#     print(allen_srl_words)
#     print(bert_tokenized_words)
    for i, word in enumerate(allen_srl_words):
        all_srl_start_map[count_chars(allen_srl_words[:i])] = (i, word)
    for i, word in enumerate(bert_tokenized_words):
        bert_tokenized_start_map[count_chars(bert_tokenized_words[:i])] = (i, word)
    # align all tokens with the same char_count and string
    for k, (srl_index, srl_word) in all_srl_start_map.items():
        if k in bert_tokenized_start_map and bert_tokenized_start_map[k][1] == srl_word:
            word2veclist[srl_index].append(bert_tokenized_start_map[k][0])
    
    for k, (srl_index, srl_word) in all_srl_start_map.items():
        if srl_index in word2veclist or k not in bert_tokenized_start_map:
            continue
        # matching start char_count but terms are different
        #print(srl_word, bert_tokenized_start_map[k][1])
        word2veclist[srl_index].append(bert_tokenized_start_map[k][0])
    completed_tokenized_indexes = set([v for vals in word2veclist.values() for v in vals])
    inverse_word2veclist = {}
    for k, v in word2veclist.items():
        for vv in v:
            inverse_word2veclist[vv] = k
    for bert_tok_index, bert_tok_word in enumerate(bert_tokenized_words):
        if bert_tok_index in completed_tokenized_indexes or bert_tok_word in ['[SEP]', '[CLS]']:
            continue
        # print("#"*20)
        # print(bert_tok_index, bert_tok_word)
        # if token before this token is completed and word index after is in word2veclist
        # just add this to the prev tok
        if (bert_tok_index - 1) in completed_tokenized_indexes:
#             print("Prev tok index is complete")
#             print("Prev Tok", bert_tok_index - 1, bert_tokenized_words[bert_tok_index - 1])
            srl_word_of_prev_tok = inverse_word2veclist[bert_tok_index - 1]
            # print("SRL word of prev tok:", allen_srl_words[srl_word_of_prev_tok])
            if (srl_word_of_prev_tok + 1) in word2veclist or srl_word_of_prev_tok + 1 ==  len(allen_srl_words):
                word2veclist[srl_word_of_prev_tok].append(bert_tok_index)
                inverse_word2veclist[bert_tok_index] = srl_word_of_prev_tok
                completed_tokenized_indexes.add(bert_tok_index)
                # print("Adding ", bert_tokenized_words[bert_tok_index], "to", allen_srl_words[srl_word_of_prev_tok])
    # At this point, the idea is that if there are still any srl_words not in the word2veclist
    # if their neighbors are in, then you can just map all the tokens not used between those two
    for srl_index, srl_word in enumerate(allen_srl_words):
        if srl_index not in word2veclist:
            if srl_index -1 in word2veclist and srl_index +1 in word2veclist:
                start = max(word2veclist[srl_index -1])
                end = min(word2veclist[srl_index +1])
                tok_indexes = list(range(start+1, end))
                if tok_indexes:
                    word2veclist[srl_index].extend(tok_indexes)
                else:
                    word2veclist[srl_index] = []
    
    try:
        if not len(word2veclist.keys()) == len(allen_srl_words):
            raise Exception
        if not len([v for vals in word2veclist.values() for v in vals]) == len(bert_tokenized_words) - 2:
            raise Exception("Er mergahd")
    except Exception as e:
        print(word2veclist)
        print(all_srl_start_map)
        print(bert_tokenized_start_map)
        raise e
    return  word2veclist

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
                try:
                    word2veclist = map_words_to_vectors(srl_parse['words'], token_strs)
                except Exception as e:
                    print(e)
                    print(token_strs)
                    raise e
                sent_repr = build_repr(srl_parse['verbs'], word2veclist, token_vecs)
                reprs.append(sent_repr)
    return reprs

def encode_data_srl_only(data):
    nlp = create_nlp()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
                srl_parse = predictor.predict(sentence=span)
                token_strs = tokenizer.convert_ids_to_tokens(lm_tokenized_input['input_ids'].numpy()[0])
                reprs.append((srl_parse, token_strs))
                # try:
                #     word2veclist = map_words_to_vectors(srl_parse['words'], token_strs)
                # except Exception as e:
                #     print(e)
                #     print(token_strs)
                #     continue #raise e
                # sent_repr = build_repr(srl_parse['verbs'], word2veclist, token_vecs)
                # reprs.append(sent_repr)
    return reprs
