# coding: utf-8
import argparse
import time
import math
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

DATANAME_WILDCARD = "*.pkl"

def get_batch(batch):
    # target data is the input data offset by 1
    input_data = batch[:, :-1, :]
    target_data = batch[:, 1:, :]
    # change shape to (seq_len, batch, embed_size)
    input_data = input_data.permute(1, 0, 2)
    target_data = target_data.permute(1, 0, 2)
    return input_data, target_data


class ModelRunner():
    def __init__(self, cmd_line_args=[]):  # pylint: disable=dangerous-default-value
        self.args = parser.parse_args(cmd_line_args)
        # Set the random seed manually for reproducibility.
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        self.device = torch.device("cuda" if self.args.cuda else "cpu")

        # Build the model
        self.model = model.TransformerModel(self.args.emsize, self.args.nhead,
                                            self.args.nhid, self.args.nlayers,
                                            self.args.dropout).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Data loader
        data_files = glob.glob(self.args.data+"/"+DATANAME_WILDCARD)
        self.training_dataset = data.HierLMDataset(data_files, self.args.bptt)
        self.valid_dataset = data.HierLMDataset([data_files[0]], self.args.bptt)
        self.data_loader = DataLoader(self.training_dataset, batch_size=self.args.batch_size,
                                      shuffle=False, num_workers=0) # TODO: shuffle!!!
        self.valid_data_loader = DataLoader(self.valid_dataset, batch_size=self.args.batch_size,
                                            num_workers=0)


    def evaluate(self):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total = self.args.batch_size * self.args.bptt
        batch_size = self.args.batch_size
        losses = []
        all_acc = []

        with torch.no_grad():
            for i, batch in enumerate(self.valid_data_loader):
                input_data, target_data = get_batch(batch)
                output = self.model(input_data)
                logits = torch.matmul(output.reshape(-1, 768), target_data.reshape(-1, 768).t())

                # Mask logits that are dot product between the same values
                tmp = torch.diag_embed(torch.tensor([float('-inf')]*(total - 1)), offset=-1)  # pylint: disable=not-callable
                index = torch.LongTensor([0]+ list(range(total-batch_size+1, total))+ list(range(1, total-batch_size+1)))
                mask_self_dot = tmp[index]
                logits = logits + mask_self_dot

                labels = torch.tensor(list(range(total))) # pylint: disable=not-callable
                loss = self.criterion(logits, labels)
                acc = torch.sum(torch.argmax(logits, axis=1) == labels) / total
                all_acc.append(acc.numpy())
                losses.append(loss.item())
        return np.mean(losses), np.mean(all_acc)


    def train(self, epoch, max_batches=float('inf')):
        # Turn on training mode which enables dropout.
        self.model.train()
        total_loss = 0.
        total = self.args.batch_size * self.args.bptt
        batch_size = self.args.batch_size
        all_acc = []

        start_time = time.time()
        for batch_index, batch in enumerate(self.data_loader):
            batch = batch.to(self.device)
            input_data, target_data = get_batch(batch)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.model.zero_grad()
            output = self.model(input_data)
            logits = torch.matmul(output.reshape(-1, 768), target_data.reshape(-1, 768).t())

            # Mask logits that are dot product between the same values
            tmp = torch.diag_embed(torch.tensor([float('-inf')]*(total - 1)), offset=-1)  # pylint: disable=not-callable
            tmp = tmp.to(self.device)
            index = torch.LongTensor([0]+ list(range(total-batch_size+1, total))+ list(range(1, total-batch_size+1)))
            index = index.to(self.device)
            mask_self_dot = tmp[index]
            logits = logits + mask_self_dot

            labels = torch.tensor(list(range(total))) # pylint: disable=not-callable
            labels = labels.to(self.device)
            loss = self.criterion(logits, labels)

            acc = torch.sum(torch.argmax(logits, axis=1) == labels) / total
            all_acc.append(acc.cpu().numpy())
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            for p in self.model.parameters():
                p.data.add_(p.grad, alpha=-self.args.lr)

            total_loss += loss.item()

            if batch_index % self.args.log_interval == 0 and batch_index > 0:
                cur_loss = total_loss / self.args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | acc {:1.5}'.format(
                          epoch, batch_index, len(self.training_dataset) // self.args.bptt, self.args.lr,
                          elapsed * 1000 / self.args.log_interval, cur_loss, np.mean(all_acc)))
                total_loss = 0
                all_acc = []
                start_time = time.time()
            if self.args.dry_run:
                break
            if batch_index >= max_batches:
                break

    def run(self):
        # Loop over epochs.
        lr = self.args.lr
        best_val_loss = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, self.args.epochs+1):
                epoch_start_time = time.time()
                self.train()
                val_loss = self.evaluate(val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    pass
                    # Model checkpointing
                    # with open(args.save, 'wb') as f:
                    #     torch.save(model, f)
                    # best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 4.0
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        # # Load the best saved model.
        # with open(args.save, 'rb') as f:
        #     model = torch.load(f)

        # # Run on test data.
        # test_loss = evaluate(test_data)
        # print('=' * 89)
        # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        #     test_loss, math.exp(test_loss)))
        # print('=' * 89)

def other():
    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data)

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)



    ###############################################################################
    # Training code
    ###############################################################################

    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    def get_batch(source, i):
        seq_len = min(args.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target
