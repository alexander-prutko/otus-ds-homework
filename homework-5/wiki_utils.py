import os
import torch


class Alphabet(object):
    def __init__(self):
        self.symbol2idx = {}
        self.idx2symbol = []
        self._len = 0
        
    def add_symbol(self, s):
        if s not in self.symbol2idx:
            self.idx2symbol.append(s)
            self.symbol2idx[s] = self._len
            self._len += 1
    
    def __len__(self):
        return self._len

    
class Texts(object):
    def __init__(self, path):
        self.dictionary = Alphabet()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add symbol to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                tokens += len(line)
                for s in line:
                    self.dictionary.add_symbol(s)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                for s in line:
                    ids[token] = self.dictionary.symbol2idx[s]
                    token += 1

        return ids
    

class TextLoader(object):
    def __init__(self, dataset, batch_size=128, sequence_length=30):
        self.data = dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self._batchify()
        
    def _batchify(self):
        # Work out how cleanly we can divide the dataset into batch_size parts.
        self.nbatch = self.data.size(0) // self.batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = self.data.narrow(0, 0, self.nbatch * self.batch_size)
        # Evenly divide the data across the batch_size batches.
        self.batch_data = data.view(self.batch_size, -1).t().contiguous()
    
    def _get_batch(self, i):
        seq_len = min(self.sequence_length, len(self.batch_data) - 1 - i)
        data = self.batch_data[i:i+seq_len]
        target = self.batch_data[i+1:i+1+seq_len].view(-1)
        return data, target
    
    def __iter__(self):
        for i in range(0, self.batch_data.size(0) - 1, self.sequence_length):
            data, targets = self._get_batch(i)
            yield data, targets
    
    def __len__(self):
        return self.batch_data.size(0)