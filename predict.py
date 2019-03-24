import numpy as np


def openTextFile(dir):
    list = []
    with open(dir, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line_split = line.strip()
            line_list = line_split.split('|||')
            list.append({
                'arg1': line_list[1].lower(),
                'arg2': line_list[2].lower(),
                'relation': line_list[0]
            })

    return list


def load_word_dict(dir):
    dict = {}
    dict['__UNK__'] = np.zeros((100))
    with open(dir, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line_list = line.split()
            dict[line_list[0]] = line_list[1]

    return dict


# 根据句子键值获取句子向量, 单词向量取平均
def get_vector_by_sentence(sentence, wordVectorDict):
    sentenceVector = np.zeros((100))
    word_list = sentence.split()
    for word in word_list:
        if word not in wordVectorDict.keys():
            word = '__UNK__'
        sentenceVector += wordVectorDict[word]

    sentenceVector = sentenceVector / len(word_list)

    return sentenceVector


testList = openTextFile('KBdata/PDTB/test_pdtb.txt')
wordVectorDict = load_word_dict('result/wordEntityVector_pdtb.txt')