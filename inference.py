from util.dataloader import Iter, Vocab, pad, normalize, vec2text
from util.text import chinese_text
import torch
from model.transformer import Transformer
from util.config import Config
from torch.nn.functional import softmax


class Translator:
    def __init__(self):
        self.conf = Config()
        self.trg_vocab = Vocab("data/vocab.ja")
        self.src_vocab = Vocab("data/vocab.zh")
        self.max_len = 128
        self.pad_idx = self.conf.pad_idx
        self.sos_idx = self.trg_vocab.s2i[self.conf.sos]
        self.eos_idx = self.trg_vocab.s2i[self.conf.eos]
        self.model = Transformer(
            dim_model=self.conf.dim_model,
            max_seq_len=self.conf.max_seq_len,
            encoder_vocab_size=self.src_vocab.length,
            decoder_vocab_size=self.trg_vocab.length,
            n_head=self.conf.n_head,
            n_layers=self.conf.n_layers,
            feed_hidden=self.conf.feed_hidden,
            pad_idx=self.conf.pad_idx,
            drop=self.conf.drop,
            device=self.conf.device
        ).to(self.conf.device)

        self.init_seq = torch.LongTensor([[2]])

    def main(self):
        text = "没有证书要求，能正确听译一集动画即可"  # input("=:")
        self.model.load_state_dict(torch.load("save/DNT_zh2ja_500.pt"))
        self.model.eval()
        with torch.no_grad():
            source_seq = torch.tensor([[self.src_vocab.s2i[s] for s in chinese_text(text).split()]],
                                      dtype=torch.long).to(self.conf.device)
            source_mask = self.model.pad_mask(source_seq, source_seq)
            enc_out = self.model.encoder(source_seq, source_mask)

            last = torch.tensor([[self.pad_idx] * self.max_len], dtype=torch.long).unsqueeze(1).to(
                self.conf.device)
            last[0] = self.sos_idx

            step = 1

            for i in range(self.max_len):
                dec_mask = self.model.pad_mask(last, last) * self.model.no_peat_mask(last, last)
                source_target_mask = self.model.pad_mask(last, source_seq)
                print(last.size(), enc_out.size(), dec_mask.size(), source_target_mask.size())
                dec_out = self.model.decoder(last, enc_out, dec_mask, source_target_mask)
                out = torch.argmax(softmax(dec_out), dim=-1)
                last_id = out[0][i].item()

                if i < self.max_len - 1:
                    last[i + 1] = last_id
                    step += 1

                if last_id == self.eos_idx:
                    break
            dec_out = last[1:step].tolist() if last[-1].item() == self.pad_idx else last[1:].tolist()
            print(dec_out)


if __name__ == '__main__':
    t = Translator()
    t.main()
