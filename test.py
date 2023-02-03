import torch
from util.bleu import get_bleu
from util.dataloader import DataLoader, vec2text
from util.config import Config
from model.transformer import Transformer

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
train_data, test_data, val_data = data.make_data()

model = Transformer(
        dim_model=conf.dim_model,
        max_seq_len=conf.max_seq_len,
        encoder_vocab_size=data.source_vocab.length,
        decoder_vocab_size=data.target_vocab.length,
        n_head=conf.n_head,
        n_layers=conf.n_layers,
        feed_hidden=conf.feed_hidden,
        pad_idx=conf.pad_idx,
        drop=conf.drop,
        device=conf.device
    ).to(conf.device)
model.load_state_dict(torch.load("save/DNT_zh2ja_500.pt"))

batch_bleu = []
for batch in test_data:
    src = batch.source
    trg = batch.target

    output = model(src, trg[:, :-1])
    total_bleu = []
    for j in range(conf.batch_size):
        try:
            src_words = vec2text(src[j], data.source_vocab)
            trg_words = vec2text(trg[j], data.target_vocab)
            output_words = output[j].max(dim=1)[1]
            output_words = vec2text(output_words, data.target_vocab)

            print('source :', src_words)
            print('target :', trg_words)
            print('predicted :', output_words)
            bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
            total_bleu.append(bleu)
        except:
            pass

    total_bleu = sum(total_bleu) / len(total_bleu)
    print('BLEU SCORE = {}'.format(total_bleu))
    batch_bleu.append(total_bleu)

