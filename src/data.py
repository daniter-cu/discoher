import pickle
from pickle import UnpicklingError
import torch
from torch.utils.data import Dataset
from data_utils import make_lm_encoder

class HierLMExample(object):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.text_spans = []
        self.reprs = []

    def current_len(self):
        return len([r for span_repr in self.reprs for r in span_repr])

    def add_span(self, span):
        '''Add the span to this  example.
        return False if there is more space.
        return True if example is full.'''
        assert self.current_len() <= self.seq_len
        text, sent_repr = span
        # sent_repr is a list of verb_repr
        # verb_repr is a list of tags
        # example length is this flattened list
        flat_repr = [r for vec_repr in sent_repr for r in vec_repr]
        flat_repr = flat_repr[:self.seq_len - self.current_len()]
        self.reprs.append(flat_repr)
        self.text_spans.append(text)
        return self.current_len() != self.seq_len

class HierLMDataset(Dataset):
    def __init__(self, filenames, seq_len, device):
        self.seq_len = seq_len
        self.device = device
        # load data from all filenames
        self.all_spans = []
        for fname in filenames:
            with open(fname, 'rb') as f:
                unpickler = pickle.Unpickler(f)
                while True:
                    try:
                        self.all_spans.append(unpickler.load())
                    except (EOFError, UnpicklingError):
                        break
        # pack into sequence length
        self.tokenizer, self.encoder = make_lm_encoder()
        self.encoder.eval()
        self.encoder.to(self.device)
        self.examples = self.pack_examples(self.all_spans)

    def __len__(self):
        # total devided by seq_len
        return len(self.examples)

    def __getitem__(self, idx):
        # return packed sequence and run through bert model to encode
        # be careful with BERT attention masking
        example = self.examples[idx]
        tokenized_spans = self.tokenizer(example.text_spans, return_tensors="pt", padding=True,
                                         truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            bert_vecs = self.encoder(**tokenized_spans)['last_hidden_state']
        embedding_size = bert_vecs.size(2)
        inputs = [torch.zeros(embedding_size).to(self.device)]
        for i, sent_repr in  enumerate(example.reprs):
            for _, indexes in sent_repr:
                inputs.append(torch.mean(bert_vecs[i][indexes], axis=0))
        inputs = torch.stack(inputs)
        return inputs.detach().data

    def pack_examples(self, all_spans):
        '''Add spans to an example until it's full.'''
        all_examples = []
        example = HierLMExample(self.seq_len)
        for span in all_spans:
            ret = example.add_span(span)
            if ret is False:
                all_examples.append(example)
                example = HierLMExample(self.seq_len)
        return all_examples

class BSODataset(HierLMDataset):
    def __getitem__(self, idx):
        # return packed sequence and run through bert model to encode
        # be careful with BERT attention masking

        # This is a BSO Example
        example = self.examples[idx]
        tokenized_spans = self.tokenizer(example.text_spans, return_tensors="pt", padding=True,
                                         truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            bert_vecs = self.encoder(**tokenized_spans)['last_hidden_state']
        embedding_size = bert_vecs.size(2)
        inputs_list = []
        for order in range(2):
            inputs = [torch.zeros(embedding_size).to(self.device)]
            for i, sent_repr in enumerate(example.reprs[order]):
                for (_, indexes) in sent_repr:
                    inputs.append(torch.mean(bert_vecs[i if order == 0 else 1-i][indexes], axis=0))
            inputs = torch.stack(inputs)
            inputs_list.append(inputs.detach().data)
        return {True: inputs_list[0], False: inputs_list[1]}

    def pack_examples(self, all_spans):
        '''Add spans to an example until it's full.'''
        all_examples = []
        for i in range(len(all_spans) - 1):
            example = BSOExample(all_spans[i], all_spans[i+1], self.seq_len)
            all_examples.append(example)
        return all_examples

class BSOExample(object):
    def __init__(self, span1, span2, seq_len):
        self.seq_len = seq_len
        self.span1 = span1
        self.span2 = span2
        self.reprs, self.text_spans = self.add_spans(span1, span2)

    def add_spans(self, span1, span2):
        text1, sent_repr1 = span1
        text2, sent_repr2 = span2
        # sent_repr is a list of verb_repr
        # verb_repr is a list of tags
        # example length is this flattened list
        flat_repr1 = [r for vec_repr in sent_repr1 for r in vec_repr][:self.seq_len]
        flat_repr2 = [r for vec_repr in sent_repr2 for r in vec_repr][:self.seq_len]

        reprs = ([flat_repr1, flat_repr2], [flat_repr2 + flat_repr1])
        text_spans = [text1, text2]
        return reprs, text_spans

class InsertionDataset(HierLMDataset):
    def __getitem__(self, idx):
        # return packed sequence and run through bert model to encode
        # be careful with BERT attention masking
        example = self.examples[idx]
        tokenized_spans = self.tokenizer(example.text_spans, return_tensors="pt", padding=True,
                                         truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            bert_vecs = self.encoder(**tokenized_spans)['last_hidden_state']
        all_sents = []
        for i, sent_repr in enumerate(example.reprs):
            inputs = []
            if len(sent_repr) == 0:
                sent_repr = [('Dummy', list(range(bert_vecs[i].shape[0] - 2)))]
            for _, indexes in sent_repr:
                inputs.append(torch.mean(bert_vecs[i][indexes], axis=0))
            inputs = torch.stack(inputs).detach().data
            all_sents.append(inputs)
        return all_sents

    def pack_examples(self, all_spans):
        '''Add spans to an example until it's full.'''
        all_examples = []
        for picked_repr in all_spans:
            example = InsertionExample(picked_repr, self.seq_len)
            all_examples.append(example)
        return all_examples

class InsertionExample(object):
    def __init__(self, picked_repr, seq_len):
        self.seq_len = seq_len
        self.text_spans = []
        self.reprs = []
        for span, rpr in picked_repr:
            self.text_spans.append(span)
            self.reprs.append([r for vec_repr in rpr for r in vec_repr])
