from util.dataloader import Iter, Vocab, pad, normalize
from util.text import chinese_text
import torch
from model.transformer import Transformer
from util.config import Config


max_len = 200

trg_vocab = Vocab("data/vocab.ja")
src_vocab = Vocab("data/vocab.zh")


text = "今天天气真好啊"  # input("=:")
text_list = chinese_text(text).split()
text_padded = pad(text_list, max_len)
text_tokenized = torch.tensor([normalize(src_vocab, text_padded)], dtype=torch.long)

conf = Config()

model = Transformer(
    dim_model=conf.dim_model,
    max_seq_len=conf.max_seq_len,
    encoder_vocab_size=src_vocab.length,
    decoder_vocab_size=trg_vocab.length,
    n_head=conf.n_head,
    n_layers=conf.n_layers,
    feed_hidden=conf.feed_hidden,
    pad_idx=conf.pad_idx,
    drop=conf.drop,
    device=conf.device
).to(conf.device)

model.load_state_dict(torch.load("save/DNT_zh2ja_500.pt"))
model.eval()
out = model(text_padded, text_padded)
print(out)

# data = Iter(text, Vocab(open("data/vocab.zh")), 1, False, None, None, "cpu")
#
# trg = Iter("")



