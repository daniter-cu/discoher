import pickle
from pickle import UnpicklingError
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, filenames, seq_len):

        self.seq_len = seq_len
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
        self.examples = self.pack_examples(self.all_spans)

    def __len__(self):
        # total devided by seq_len
        return len(self.examples)

    def __getitem__(self, idx):
        # return packed sequence and run through bert model to encode
        # be careful with BERT attention masking
        # TODO: bert, return array of vecs
        # TODO: add targets
        return self.examples[idx]

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

# TODO: Define collate_fn to batch correctly
