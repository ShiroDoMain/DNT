import os.path

from model.transformer import Transformer
from util.config import Config
from util.dataloader import DataLoader
import torch
from torch import optim, nn
from tqdm import tqdm
from util.bleu import get_bleu

conf = Config()
data = DataLoader(source_lang=conf.source_lang,
                  target_lang=conf.target_lang,
                  data_path=conf.dataset,
                  batch_size=conf.batch_size,
                  device=conf.device,
                  padding=conf.padding,
                  pad=conf.pad,
                  sos=conf.sos,
                  eos=conf.eos)
print("load dataset")


def init_model(model):
    print("model initialize")
    model.apply(
        lambda m: torch.nn.init.kaiming_uniform_(m.weight)
        if hasattr(m, 'weight') and m.weight.dim() > 1 else None
    )
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')


def vec2text(vec, vocab):
    return "".join(vocab.i2s[i] for i in vec.tolist()
                   if vocab.i2s[i] not in [conf.pad, conf.eos, conf.sos])


def train(model, train_data, optimizer, criterion):
    model.train()
    loss = 0
    progress = tqdm(train_data, total=len(train_data))
    for step, batch in enumerate(progress):
        source = batch.source
        target = batch.target

        optimizer.zero_grad()
        output = model(source, target[:, :-1])
        output_ = output.contiguous().view(-1, output.size(-1))
        target = target[:, 1:].contiguous().view(-1)

        loss_ = criterion(output_, target)
        loss_.backward()

        nn.utils.clip_grad_norm_(model.parameters(), conf.clip)
        optimizer.step()

        loss += loss_.item()
        progress.set_postfix_str(f"step: {step}, loss: {loss_.item():.5f}")

    return loss / len(train_data)


def evaluate(model, val_data, criterion):
    model.eval()
    loss = 0
    batch_bleu = []

    with torch.no_grad():
        progress = tqdm(val_data, total=len(val_data))
        for step, batch in enumerate(progress):
            source = batch.source
            target = batch.target

            output = model(source, target[:, :-1])
            output_ = output.contiguous().view(-1, output.size(-1))
            target = target[:, 1:].contiguous().view(-1)

            loss_ = criterion(output_, target)
            loss += loss_.item()

            total_bleu = []

            for i in range(conf.batch_size):
                try:
                    target_words = vec2text(batch.target[i], data.target_vocab)
                    output_words = vec2text(output[i].max(dim=1)[1], data.target_vocab)
                    bleu = get_bleu(output_words.split(), target_words.split())
                    total_bleu.append(bleu)
                except:
                    pass
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)
            progress.set_postfix_str(f"evaluation: {step}, loss: {loss_.item():.5f}, bleu: {total_bleu}")

        total_bleu = sum(total_bleu) / len(batch_bleu)
        return loss / len(batch), total_bleu


def main():
    train_data, test_data, val_data = data.make_data()

    print("build model")
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

    init_model(model)

    optimizer = optim.Adam(model.parameters(),
                           lr=conf.init_lr,
                           weight_decay=conf.weight_decay,
                           eps=conf.eps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     verbose=True,
                                                     factor=conf.factor,
                                                     patience=conf.patience)
    criterion = nn.CrossEntropyLoss(ignore_index=conf.pad_idx)

    best = float("inf")
    for epoch in range(conf.epochs):
        train_loss = train(model, train_data, optimizer, criterion)

        if epoch % conf.eval_interval == 0 and epoch:
            val_loss, bleu = evaluate(model, val_data, criterion)

            if epoch > conf.warmup:
                scheduler.step(val_loss)
            if val_loss < best:
                print("save model by best loss:")
                best = val_loss
                torch.save(model.static_dict(), os.path.join(conf.save_path, "best.pt"))
            print(f"Epoch:{epoch} | train loss: {train_loss:.5f} | evaluation loss: {val_loss:.5f} | bleu: {bleu:.5f}")
        print(f"Epoch:{epoch} | train loss: {train_loss:.5f}")

        if epoch % conf.save_interval and epoch:
            torch.save(model.state_dict(), os.path.join(conf.save_path, f"checkpoint_{epoch}.pt"))


if __name__ == '__main__':
    main()
