import os

from sklearn.externals import joblib

from vocab import VocabBuilder, GloveVocabBuilder

PROCESSED_DATA_PATH = 'KBdata/PDTB'


def create_word_index(glove_path, embedding_size, min_samples):
    if os.path.exists(glove_path):
        v_builder = GloveVocabBuilder(path_glove=glove_path)
        d_word_index, embed = v_builder.get_word_index()
        ed_size = embed.size(1)
        is_glove = True
    else:
        v_builder = VocabBuilder(path_file=PROCESSED_DATA_PATH + '/train_pdtb.tsv')
        d_word_index, embed = v_builder.get_word_index(min_sample=min_samples)
        ed_size = embedding_size
        is_glove = False

    results_path = get_results_path(is_glove, ed_size)
    joblib.dump(d_word_index, results_path + '/d_word_index.pkl', compress=3)

    return (v_builder, d_word_index, embed, ed_size)


def get_word_index(glove_path, embedding_size):
    if os.path.exists(glove_path):
        v_builder = GloveVocabBuilder(path_glove=glove_path)
        d_word_index, embed = v_builder.get_word_index()
        ed_size = embed.size(1)
        is_glove = True
    else:
        ed_size = embedding_size
        is_glove = False

    d_word_index = None
    results_path = get_results_path(is_glove, ed_size)
    if os.path.exists(results_path + '/d_word_index.pkl'):
        d_word_index = joblib.load(results_path + '/d_word_index.pkl')

    return d_word_index


def get_results_path(is_glove, embedding_size):
    if is_glove:
        results_path = 'result/glove_' + str(embedding_size) + 'v'
    else:
        results_path = 'result/no_glove_' + str(embedding_size) + 'v'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    return results_path


def get_sentence_by_key(key):
    with open('KBdata/PDTB/sentence_entity.txt', 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    for item in sentences:
        item_list = item.split('\t')
        if key == item_list[0]:
            return item_list[1]


def get_sentence_len_by_key(key):
    return len(get_sentence_by_key(key))
