import jieba
import torch

from model.transformer import Transformer
from util.config import Config
from util.dataloader import DataLoader
from util import greedy_decode
from util.dataloader import vec2text

conf = Config()
data = DataLoader(source_lang=conf.source_lang,
                  target_lang=conf.target_lang,
                  data_path=conf.dataset,
                  batch_size=conf.batch_size,
                  save_normal=conf.save_normal,
                  device=conf.device,
                  padding=conf.padding,
                  pad=conf.pad,
                  sos=conf.sos,
                  eos=conf.eos)


def predict(text, model):
    src = jieba.lcut(text)
    src = torch.tensor([[data.source_vocab.s2i[s] for s in src]], dtype=torch.long, device=conf.device)
    src = src.transpose(0, 1)[:1]
    src_mask = (src != data.source_vocab.s2i[conf.pad]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, conf.max_seq_len,
                        data.target_vocab.s2i[conf.sos],
                        data.target_vocab.s2i[conf.eos])
    output = vec2text(out[0], data.target_vocab)
    return output


if __name__ == '__main__':
    print("make dataset")
    train_iter, test_iter, valid_iter = data.make_data()
    print("make model")
    model = Transformer.make_model(data.source_vocab.length,
                                   data.target_vocab.length,
                                   conf.max_seq_len,
                                   conf.pad_idx,
                                   conf.n_layers,
                                   conf.dim_model,
                                   conf.feed_hidden,
                                   conf.n_head,
                                   conf.drop)
    model.load_state_dict(torch.load("save/DNT_20.pt"))
    model.to(conf.device)
    model.eval()
    with torch.no_grad():
        while 1:
            res = predict(input("input:"), model)
            print("translation:", res)
