import argparse
import glob
import msgpack
import numpy as np
import torch

from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import XLNetTokenizer, XLNetLMHeadModel
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel

parser = argparse.ArgumentParser(description='Collect per positions losses.')
parser.add_argument('--save', type=str, default='../results/results_books.npy',
                    help='Name of save file')
parser.add_argument('--data', type=str, default='../data/wiki40b/',
                    help='Path to data')
parser.add_argument('--batch', type=int, default=8,
                    help='Batch size for SRL parser.')
parser.add_argument('--dataset', type=str, default='books',
                    choices=['books', 'wiki', 'litbank'], help='Dataset type')
args = parser.parse_args()

def get_gpt():
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    model.eval()
    return tokenizer, model

def get_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    return tokenizer, model

def get_tranxl():
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
    model.eval()
    return tokenizer, model

def get_xlnet(): # Doesn't work
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
    model.eval()
    return tokenizer, model

def get_wiki_data(path):
    files = glob.glob(path+"/*")
    for fname in files:
        with open(fname, "rb") as buf:
            unpacker = msgpack.Unpacker(buf, raw=False)
            for unpacked in unpacker:
                yield unpacked

def get_books_data(path):
    data = []
    collect = True
    with open(path, "r") as f:
        one_book = []
        for i, line in enumerate(f):
            if collect:
                one_book.append(line)
                if len(one_book) == 50:
                    data.append(one_book)
                    one_book = []
                    collect = False
            if i % 10000 == 0:
                collect = True
    all_data = []
    for book in data:
        lines = " ".join([l.strip() for l in book])
        all_data.append(lines)
    return all_data[1:]

def get_litbank_data(path):
    litbank_file = glob.glob(path + "*")
    data = []
    for fname in litbank_file:
        book = []
        section = []
        with open(fname, "r") as f:
            spaces = 0
            for line in f:
                if line.isspace():
                    spaces += 1
                else:
                    if spaces < 2:
                        section.append(line.strip())
                    else:
                        book.append(section)
                        section = []
                        section.append(line.strip())
                    spaces = 0
            if section:
                book.append(section)
        data.append(book)
    
    all_data = []
    for book in data:
        new_book = [" ".join(sec) for sec in book if len(sec) > 50]
        all_data.append(new_book)
        
    all_data = [sec for book in all_data for sec in book]
    return all_data

def  main(args):
    # TODO: check uninitialized weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "books":
        get_data = get_books_data
    elif args.dataset == "litbank":
        get_data = get_litbank_data
    else:
        get_data = get_wiki_data

    configs = {"gpt": get_gpt,
               "gpt2": get_gpt2,
               "tranxl": get_tranxl} #,
               # "xlnet": get_xlnet}
    for name, get_config in configs.items():
        print("Running", name)
        paragraphs = get_data(args.data)

        tokenizer, model = get_config()
        model = model.to(device)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        done = False
        count = 0
        while not done:
            tokenized_paras = []
            for para in paragraphs:
                toked = tokenizer.encode(para, return_tensors="pt", truncation=True, max_length=512)
                if toked.shape[1] >= 512:
                    tokenized_paras.append(toked[:, :512])
                    if len(tokenized_paras) == args.batch:
                        break
            if len(tokenized_paras) < args.batch:
                done = True
            if not tokenized_paras:
                continue
            tokenized_paras = torch.cat(tokenized_paras, axis=0)
            tokenized_paras = tokenized_paras.to(device)

            with torch.no_grad():
                out = model(tokenized_paras, labels=tokenized_paras, return_dict=True)
                if name == "tranxl":
                    loss = out.losses.cpu()
                else:
                    shift_logits = out.logits[..., :-1, :].contiguous()
                    shift_labels = tokenized_paras[..., 1:].contiguous()
                    # Flatten the tokens
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                save_results(loss.cpu().numpy(), args.save, name)
                count += 1
                if count > 3000:
                    break
    return

def save_results(loss, savefile, key_name):
    filename = savefile.replace(".npy", "-" + key_name + ".npy")
    print("saving to filename...", filename)
    with open(filename, "ab") as f:
        np.save(f, loss)

if __name__ == "__main__":
    main(args)
