import os
import pickle

from matplotlib import pyplot

from model.transformer import Transformer, subsequent_mask
from util.config import Config
from util.dataloader import DataLoader
from tqdm import tqdm
import torch
from torch.autograd import Variable
from util.optimizer import LabelSmoothing, NoamOpt
from util.loss import SimpleLossCompute


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
                  eos=conf.eos,
                  vocab_size=conf.vocab_size)


if conf.fp16:
    from torch.cuda.amp import GradScaler, autocast

    scaler = None  # GradScaler()
else:
    scaler = None


def run_batch(model, batch, loss_compute):
    out = model.forward(batch.source, batch.target[:, :-1])
    ntokens = (batch.target[:, 1:] != conf.pad_idx).data.sum()
    loss = loss_compute(out, batch.target[:, 1:], ntokens, conf.fp16, scaler)
    return out, loss, ntokens



def run_epoch(train_iter, model, loss_compute, desc):
    """Standard Training and Logging Function"""
    total_tokens = 0
    total_loss = 0
    progress = tqdm(enumerate(train_iter), desc=desc, total=len(train_iter))
    for i, batch in progress:
        if conf.fp16:
            with autocast():
                out, loss, ntokens = run_batch(model, batch, loss_compute)
        else:
            out, loss, ntokens = run_batch(model, batch, loss_compute)
        # trg = batch.target[:, :-1]
        # out = model.forward(batch.source, trg)
        # ntokens = (batch.target[:, 1:] != conf.pad_idx).data.sum()
        # loss = loss_compute(out, batch.target[:, 1:], ntokens)
        total_loss += loss
        total_tokens += ntokens
        progress.set_postfix_str(f"Loss: {(loss / ntokens).item():.5f}"+f"| lr: {loss_compute.opt.lr:.8f}" if loss_compute.opt is not None else "")
    return (total_loss / total_tokens).item()


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
    if conf.load_model:
        print(f"load model {conf.load_model}")
        model.load_state_dict(torch.load(conf.load_model))

    model.to(conf.device)
    print("use device ", conf.device)
    print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    optimizer = NoamOpt(conf.dim_model, 1, 100, torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=conf.eps
    ))
    criterion = LabelSmoothing(data.target_vocab.length, conf.pad_idx, smoothing=.1).to(conf.device)
    train_history = []
    valid_history = []
    best = float("inf")
    print("start train")
    for epoch in range(conf.epochs):
        model.train()
        train_loss = run_epoch(train_iter, model, SimpleLossCompute(
            model.generator, criterion, optimizer
        ), "Train")
        model.eval()
        with torch.no_grad():
            valid_loss = run_epoch(valid_iter, model, SimpleLossCompute(
                model.generator, criterion, None
            ), "Evaluate")
        print("Epoch: %d| Train Loss: %.5f| Evaluation Loss: %.5f" % (epoch, train_loss, valid_loss))
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        if (epoch+1) % conf.save_interval == 0 and epoch:
            torch.save(model.state_dict(), os.path.join(conf.save_path, conf.model_name + f"_{epoch+1}.pt"))
        if valid_loss < best:
            print("save model for best loss:", valid_loss)
            best = valid_loss
            torch.save(model.state_dict(), os.path.join(conf.save_path, conf.model_name + "_best.pt"))
    pickle.dump(train_history, open("train_loss.pkl", "wb"))
    pickle.dump(valid_history, open("val_loss.pkl", "wb"))
    pyplot.plot(train_history, label="train")
    pyplot.plot(valid_history, label="val")
    pyplot.legend()
    pyplot.show()
