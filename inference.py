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


def text2vec(text, pad_len=20):
    vec = ["[SOS]"] + jieba.lcut(text) + ["[EOS]"]
    vec += ["[PAD]"] * (pad_len - len(vec))
    vec = torch.tensor([[data.source_vocab.s2i[s] for s in vec]], dtype=torch.long, device=conf.device)
    return vec


def predict(src, model):  # src = src.transpose(0, 1)[:1]
    src_mask = (src != data.source_vocab.s2i[conf.pad]).unsqueeze(-2)
    output = greedy_decode(model, src, src_mask, conf.max_seq_len,
                           data.target_vocab.s2i[conf.sos],
                           data.target_vocab.s2i[conf.eos])
    # output =
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
    model.load_state_dict(torch.load("save/DNT_200.pt"))
    model.to(conf.device)
    model.eval()
    with torch.no_grad():
        # src = text2vec("这是一段测试文本")
        # print(src.size())
        # print("source:", vec2text(src[0], data.source_vocab))
        # res = predict(src, model)
        # print(res)
        # print("predict:", vec2text(res[0], data.target_vocab))
        # while 1:
        #     text = input("input:")  # "这是一段测试文本"  # input("input:")
        #     res = predict(text, model)
        #     print("translation:", res)
        for batch in test_iter:
            for i in range(conf.batch_size):
                src = batch.source[i, :].repeat(1, 1)
                print("source:", vec2text(src[0], data.source_vocab))
                print("target:", vec2text(batch.target[i, :], data.target_vocab))
                res = predict(src, model)
                print("predict:", vec2text(res[0], data.target_vocab))
                print()
            break
