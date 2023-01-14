import os

import torch

from util.text import symbols


class Vocab:
    def __init__(self,
                 file):
        self.vocab = {idx: word.strip() for idx, word in enumerate(open(file, encoding="utf-8").readlines())}
        self.length = len(self.vocab)

    @property
    def i2s(self):
        return self.vocab

    @property
    def s2i(self):
        return {word: idx for idx, word in self.vocab.items()}


class Iter:
    def __init__(self,
                 data,
                 vocab,
                 batch_size,
                 device):
        self._data = data
        self.vocab = vocab
        self.batch_size = batch_size
        self.batches = None
        self.device = device

    def normalize(self, arr):
        try:
            return torch.tensor([[self.vocab.s2i[word] for word in batch] for batch in arr], dtype=torch.long)
        except:
            raise

    def build_batches(self):
        minibatch = []
        for offset, _data in enumerate(self._data):
            if minibatch and offset % self.batch_size == 0:
                yield minibatch
                minibatch = []
            minibatch.append(_data)

        if minibatch:
            yield minibatch

    def __iter__(self):
        batches = self.build_batches()
        for minibatch in batches:
            yield self.normalize(minibatch).to(self.device)


class Batch:
    def __init__(self, source, target):
        self.source = source
        self.target = target


class Data:
    def __init__(self,
                 source_data,
                 target_data,
                 source_vocab,
                 target_vocab,
                 batch_size,
                 device):
        self.source = Iter(source_data, source_vocab, batch_size, device)
        self.target = Iter(target_data, target_vocab, batch_size, device)
        self.device = device

    def __iter__(self):
        for source, target in iter(zip(self.source, self.target)):
            yield Batch(source, target)


def _symbol_replace(text: str) -> str:
    for symbol in symbols:
        text = text.replace(symbol, f" {symbol} ")
    return text


def _load_data(path):
    with open(path, encoding="utf-8") as f:
        data = [_symbol_replace(line.strip()).split() for line in f.readlines()]
    max_len = max([len(line) for line in data])
    return data, max_len


class DataLoader:
    def __init__(self,
                 *lang,
                 data_path,
                 batch_size,
                 padding=False,
                 device,
                 sos="[SOS]",
                 pad="[PAD]",
                 eos="[EOS]"):
        """
        DataLoader("en","de","data/", 20)
        """
        assert len(lang) == 2, "language accept 2 parameters"
        source_lang, target_lang = lang
        self.batch_size = batch_size
        self.device = device

        self.train_source_path = os.path.join(data_path, f"train.{source_lang}")
        self.train_target_path = os.path.join(data_path, f"train.{target_lang}")

        self.test_source_path = os.path.join(data_path, f"test.{source_lang}")
        self.test_target_path = os.path.join(data_path, f"test.{target_lang}")

        self.val_source_path = os.path.join(data_path, f"val.{source_lang}")
        self.val_target_path = os.path.join(data_path, f"val.{target_lang}")

        self.source_vocab_path = os.path.join(data_path, f"vocab.{source_lang}")
        self.target_vocab_path = os.path.join(data_path, f"vocab.{target_lang}")

        self.padding = padding
        self.sos = sos
        self.pad = pad
        self.eos = eos

        self.source_vocab = None
        self.target_vocab = None

    def _pad(self, data, max_len):
        return [[self.sos] + line_list + [self.eos] +
                (([self.pad] * (max_len - len(line_list))) if len(line_list) < max_len else [])
                for line_list in data]

    def load_vocab(self):
        source_vocab = Vocab(self.source_vocab_path)
        target_vocab = Vocab(self.target_vocab_path)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        return source_vocab, target_vocab

    def _make_data(self, src_path, trg_path, src_vocab, trg_vocab):
        source, source_max_len = _load_data(src_path)
        target, target_max_len = _load_data(trg_path)
        if self.padding:
            source = self._pad(source, source_max_len)
            target = self._pad(target, target_max_len)
        return Data(source_data=source, target_data=target, source_vocab=src_vocab, target_vocab=trg_vocab,
                    batch_size=self.batch_size, device=self.device)

    def make_data(self):
        src_voc, trg_voc = self.load_vocab()

        train_iter = self._make_data(self.train_source_path, self.train_target_path, src_voc, trg_voc)
        test_iter = self._make_data(self.test_source_path, self.test_target_path, src_voc, trg_voc)
        val_iter = self._make_data(self.val_source_path, self.val_target_path, src_voc, trg_voc)

        return train_iter, test_iter, val_iter


if __name__ == '__main__':
    data_ = DataLoader("en", "de", data_path="..\\data", batch_size=20, padding=True, device="cpu")
    train, test, val = data_.make_data()
    for batch in train:
        print(batch.source.unsqeeze())
