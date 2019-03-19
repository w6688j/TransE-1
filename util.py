import csv
import os

from sklearn.externals import joblib

from vocab import *

PROCESSED_DATA_PATH = 'KBdata/PDTB'


def _tokenize(text):
    if not isinstance(text, str):
        return []
    else:
        return [x.lower() for x in text.split()]


def create_word_index(glove_path, embedding_size, min_samples):
    if os.path.exists(glove_path):
        v_builder = GloveVocabBuilder(path_glove=glove_path)
        d_word_index, embed = v_builder.get_word_index()
        ed_size = embed.size(1)
    else:
        v_builder = VocabBuilder(path_file=PROCESSED_DATA_PATH + '/train.txt')
        d_word_index, embed = v_builder.get_word_index(min_sample=min_samples)
        ed_size = embedding_size

    joblib.dump(d_word_index, 'result/glove/' + ed_size + '/d_word_index.pkl', compress=3)

    return (v_builder, d_word_index, embed, ed_size)
