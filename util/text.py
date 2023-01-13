import jieba
from typing import List
from collections import Counter
import string
import re


symbol_half = string.punctuation
symbol_full = r"""！@#￥%……&*（）——+{}：“”‘’；【】《》，。？/·~"""
symbol = symbol_full + symbol_half


def chinese_text(line_list, min_freq):
    return generate_vocab([" ".join(jieba.lcut(line)) for line in line_list], min_freq)


def generate_vocab(line_list: List[str], min_freq):
    word_list = []
    for line in line_list:
        line = re.sub(rf"[{symbol}]", "", line)
        word_list += line.split()
    return {word: freq for word, freq in Counter(word_list).items() if freq >= min_freq}


vocab_func = {
    "en": generate_vocab,
    "de": generate_vocab,
    "zh": chinese_text
}
