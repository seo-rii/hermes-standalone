"""Owned Sudachi/Nani bindings for preserved original Japanese reading rules."""
from pathlib import Path
from threading import local

from ..nani import NaniModel
from ..sudachi_tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent.parent / 'data'
_local = local()


def get_sudachi_tokenizer():
    tokenizer = getattr(_local, 'tokenizer', None)
    if tokenizer is None:
        tokenizer = Tokenizer(ROOT / 'sudachi/system.dic', ROOT / 'sudachi/resources')
        _local.tokenizer = tokenizer
    return tokenizer


def predict(input_njd):
    model = getattr(_local, 'nani', None)
    if model is None:
        model = NaniModel(ROOT / 'openjtalk/yomi_model')
        _local.nani = model
    return model.predict(input_njd[0])
