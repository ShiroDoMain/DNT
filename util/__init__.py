import torch
from torch.autograd import Variable

from model.transformer import subsequent_mask


def greedy_decode(model, src, src_mask, max_len, sos, eos):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(sos).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == eos:
            break

    return ys


def beam_search(model, src, src_mask, max_len, device, sos, eos, beam_size=5):
    seq = torch.tensor([[sos]], device=device)
    enc_out = model.encode(src, src_mask)
    dec_out = model.decode(enc_out, )

