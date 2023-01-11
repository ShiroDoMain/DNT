import torch

from model.embedding import Embedding, PositionEmbedding, TokenEmbedding

# data shape [10, 20, 128]
max_seq_len = 512
voc_size = 1024
bs = 10
seq_len = 20
dim_model = 512
drop = .1

device = "cpu"

data = torch.Tensor([[[2, 3, 4, 56, 23, 4], [2, 3, 2, 4, 3, 4]], [[5, 45, 4, 6, 3, 7], [2, 3, 12, 5, 6, 3]]]).int()
emb = Embedding(voc_size, dim_model, max_seq_len, drop, device)
pos_emb = PositionEmbedding(dim_model, max_seq_len, device)
tok_emb = TokenEmbedding(voc_size, dim_model)
for batch in data:
    # out = emb(batch)
    print("input: ", batch.size())
    out = emb(batch.to(device))
    print("output: ", out.size())
