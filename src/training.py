# coding: utf-8
import argparse
import time
import glob
import os
import sys
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--train_data', type=str, default='../data/train/',
                    help='location of the data corpus')
parser.add_argument('--val_data', type=str, default='../data/val/',
                    help='location of the data corpus')
parser.add_argument('--contrasts', type=int, default=4,
                    help='Number of options in multiple choice.')
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
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='../checkpoints/model.pt',
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

        # create checkpoints dir
        checkpoints_dir = os.path.dirname(os.path.realpath(self.args.save))
        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)


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
        train_data_files = glob.glob(self.args.train_data+"/"+DATANAME_WILDCARD)
        val_data_files = glob.glob(self.args.val_data+"/"+DATANAME_WILDCARD)
        self.training_dataset = data.HierLMDataset(train_data_files, self.args.bptt, self.device)
        self.valid_dataset = data.HierLMDataset(val_data_files, self.args.bptt, self.device)
        self.data_loader = DataLoader(self.training_dataset, batch_size=self.args.batch_size,
                                      shuffle=True, num_workers=0)
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
            for _, batch in enumerate(self.valid_data_loader):
                batch = batch.to(self.device)
                input_data, target_data = get_batch(batch)
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
            # print("Time to load data:", time.time() - start_time)
            batch = batch.to(self.device)
            input_data, target_data = get_batch(batch)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.model.zero_grad()
            # start_model_time = time.time()
            output = self.model(input_data)
            # print("Run model", time.time() - start_model_time)
            logits = torch.matmul(output.reshape(-1, 768), target_data.reshape(-1, 768).t())

            # Mask logits that are dot product between the same values
            tmp = torch.diag_embed(torch.tensor([float('-inf')]*(total - 1)), offset=-1)  # pylint: disable=not-callable
            tmp = tmp.to(self.device)
            index = torch.LongTensor([0]+ list(range(total-batch_size+1, total))+ list(range(1, total-batch_size+1)))
            index = index.to(self.device)
            mask_self_dot = tmp[index]
            logits = logits + mask_self_dot

            # labels = torch.tensor(list(range(total))) # pylint: disable=not-callable
            labels = torch.zeros(total, dtype=torch.long)
            labels = labels.to(self.device)

            # Downsample logits for more reasonable window
            np_negs = []
            for i in range(total):
                choices = list(range(total))
                del choices[i]
                np_negs.append(np.random.choice(choices, self.args.contrasts, replace=False))
            np_negs = np.stack(np_negs)
            np_trues = np.arange(total).reshape(total, 1)
            all_options = np.concatenate([np_trues, np_negs], 1)
            all_options = torch.tensor(all_options).to(self.device) # pylint: disable=not-callable

            # downsample logits
            logits = torch.gather(logits, 1, all_options)
            loss = self.criterion(logits, labels)

            acc = torch.sum(torch.argmax(logits, axis=1) == labels) / total
            # print("Run model + logits", time.time() - start_model_time)

            all_acc.append(acc.cpu().numpy())
            # start_backprop = time.time()
            loss.backward()
            # print("backprop", time.time() - start_backprop)

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
                          epoch, batch_index, len(self.training_dataset) // self.args.batch_size, self.args.lr,
                          elapsed * 1000 / self.args.log_interval, cur_loss, np.mean(all_acc)))
                total_loss = 0
                all_acc = []
                start_time = time.time()
            if batch_index % self.args.save_interval == 0 and batch_index > 0:
                with open(self.args.save, 'wb') as f:
                        
                        torch.save(model, f)
            if self.args.dry_run:
                break
            if batch_index >= max_batches:
                break

    def checkpoint_name(self, epoch, batch_index, save_name):
        name = "chkpt-%d-%d-" % (epoch, batch_index)
        path, base = os.path.split(save_name)
        return path + "/" + name + base

    def run(self):
        # Loop over epochs.
        best_val_loss = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, self.args.epochs+1):
                epoch_start_time = time.time()
                self.train(epoch)
                val_loss, val_acc = self.evaluate()
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid acc {:2.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, val_acc))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    # Model checkpointing
                    with open(self.args.save, 'wb') as f:
                        torch.save(model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    self.args.lr /= 4.0
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

if __name__ == "__main__":
    print("args:", sys.argv)
    runner = ModelRunner(sys.argv[1:])
    runner.run()
