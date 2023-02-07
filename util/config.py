import json
import os.path


class Config:
    model_name: str
    epochs: int
    source_lang: str
    target_lang: str
    fp16: bool
    load_model: str
    save_interval: int
    save_path: str
    resume_epoch: int
    save_best_model: bool
    dataset: str
    padding: bool
    device: str
    batch_size: int
    dim_model: int
    max_seq_len: int
    drop: float
    n_layers: int
    n_head: int
    feed_hidden: int
    pad_idx: int
    init_lr: float
    weight_decay: float
    factor: float
    save_normal: bool
    patience: int
    clip: float
    warmup: int
    eps: float
    pad: str
    sos: str
    vocab_size: int
    eos: str

    def __init__(self):
        config = json.load(open("conf/config.json"))
        for k, v in config.items():
            setattr(self, k, v)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
