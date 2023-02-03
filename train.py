import os.path
import pickle
from model.transformer import Transformer
from util.config import Config
from util.dataloader import DataLoader, vec2text
import torch
from torch import optim, nn
from tqdm import tqdm
from util.bleu import get_bleu
from matplotlib import pyplot


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
print("load dataset")

if conf.fp16:
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()


def init_model(model):
    print("model initialize")
    model.apply(
        lambda m: torch.nn.init.kaiming_uniform_(m.weight)
        if hasattr(m, 'weight') and m.weight.dim() > 1 else None
    )


def _train_batch(model, criterion, x, y):
    output = model(x, y[:, :-1])
    output = output.contiguous().view(-1, output.size(-1))

    target = y[:, 1:].contiguous().view(-1)
    loss_ = criterion(output, target)
    return output, loss_


def train(model, train_data, optimizer, criterion):
    model.train()
    loss = 0
    progress = tqdm(train_data, total=len(train_data), desc="train: ")
    for step, batch in enumerate(progress):
        source = batch.source
        target = batch.target

        optimizer.zero_grad()
        if conf.fp16:
            with autocast():
                output, loss_ = _train_batch(model, criterion, source, target)
            scaler.scale(loss_).backward()

            nn.utils.clip_grad_norm_(model.parameters(), conf.clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            output, loss_ = _train_batch(model, criterion, source, target)

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
        progress = tqdm(val_data, total=len(val_data), desc="evaluate: ")
        for step, batch in enumerate(progress):
            source = batch.source
            target = batch.target

            output, loss_ = _train_batch(model, criterion, source, target)

            loss += loss_.item()

            total_bleu = []

            for i in range(conf.batch_size):
                try:
                    target_words = vec2text(batch.target[i], data.target_vocab)
                    output_words = vec2text(output[i].max(dim=1)[1], data.target_vocab)
                    bleu = get_bleu(output_words.split(), target_words.split())
                    total_bleu.append(bleu)
                except IndexError:
                    pass
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)
            progress.set_postfix_str(f"evaluation: {step}, loss: {loss_.item():.5f}, bleu: {total_bleu}")

        total_bleu = sum(batch_bleu) / len(batch_bleu)
        return loss / len(val_data), total_bleu


def main():
    train_data, test_data, val_data = data.make_data()

    print("build model")
    print(f"use device {conf.device}")
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

    if conf.load_model is not None:
        print("load model from", conf.load_model)
        model.load_state_dict(torch.load(os.path.join(conf.save_path, conf.load_model)))
    else:
        init_model(model)

    print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

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
    model_name = f"DNT_{conf.source_lang}2{conf.target_lang}"
    for epoch in range(conf.resume_epoch, conf.epochs):
        train_loss = train(model, train_data, optimizer, criterion)

        val_loss, bleu = evaluate(model, val_data, criterion)

        if epoch > conf.warmup:
            scheduler.step(val_loss)
        if val_loss < best:
            print("save model by best loss:", val_loss)
            best = val_loss
            torch.save(model.state_dict(), os.path.join(conf.save_path, model_name + "_best.pt"))
        print(f"Epoch:{epoch} | train loss: {train_loss:.5f} | evaluation loss: {val_loss:.5f} | bleu: {bleu:.5f}")

        if epoch % conf.save_interval == 0 and epoch:
            torch.save(model.state_dict(), os.path.join(conf.save_path, model_name + f"_{epoch}.pt"))
        train_loss_record.append(train_loss)
        evaluation_loss_record.append(val_loss)
    torch.save(model.state_dict(), os.path.join(conf.save_path, model_name + f"_latest.pt"))


if __name__ == '__main__':
    train_loss_record, evaluation_loss_record = [], []
    try:
        main()
    except:
        raise
    #     pickle.dump(train_loss_record, open("train_loss.pkl", "wb"))
    #     pickle.dump(evaluation_loss_record, open("val_loss.pkl", "wb"))
    #
    #
    # pyplot.plot(train_loss_record, label="train")
    # pyplot.plot(evaluation_loss_record, label="val")
    # pyplot.legend()
    # pyplot.show()
