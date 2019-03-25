import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


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


def load_relation_dict(dir):
    dict = {}
    with open(dir, 'r', encoding='utf-8') as file:
        for line in file:
            line_split = line.strip().split('\t')
            relation_name = line_split[0]
            relation_vec = line_split[1][1:-1].split(',')
            relationVector = np.array(relation_vec).astype(float)
            dict[relation_name] = relationVector

    return dict


def load_word_dict(dir):
    dict = {}
    dict['__UNK__'] = np.zeros((100))
    with open(dir, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line_list = line.strip().split('\t')
            dict[line_list[0]] = line_list[1]

    return dict


# 根据句子键值获取句子向量, 单词向量取平均
def get_vector_by_sentence(sentence, wordVectorDict):
    sentenceVector = np.zeros((100))
    word_list = sentence.split()
    for word in word_list:
        if word not in wordVectorDict.keys():
            word = '__UNK__'

        wordVector = np.array(wordVectorDict[word][1:-1].split(',')).astype(float)
        sentenceVector += wordVector

    sentenceVector = sentenceVector / len(word_list)

    return sentenceVector


# 获取句子向量列表
def getSentenceVectorList(testList, wordVectorDict):
    list = []
    for sentence in testList:
        sentence1Vector = get_vector_by_sentence(sentence['arg1'], wordVectorDict)
        sentence2Vector = get_vector_by_sentence(sentence['arg2'], wordVectorDict)
        list.append({
            'arg1': sentence1Vector,
            'arg2': sentence2Vector,
            'relation': sentence['relation'],
        })

    return list


testList = openTextFile('KBdata/PDTB/test_pdtb.txt')
wordVectorDict = load_word_dict('result/wordEntityVector_pdtb.txt')
relationVectorDict = load_relation_dict('result/relationVector_pdtb.txt')
sentenceVectorList = getSentenceVectorList(testList, wordVectorDict)

count = 0
sum = 0
for sentenceVector in sentenceVectorList:
    preRelationVector = sentenceVector['arg1'] - sentenceVector['arg2']
    preRelationVector = preRelationVector.reshape((1, 100))

    ComparisonVector = relationVectorDict['Comparison'].reshape((1, 100))
    ContingencyVector = relationVectorDict['Contingency'].reshape((1, 100))
    TemporalVector = relationVectorDict['Temporal'].reshape((1, 100))
    ExpansionVector = relationVectorDict['Expansion'].reshape((1, 100))

    cos_dict = {
        'Comparison': cosine_similarity(preRelationVector, ComparisonVector)[0, 0],
        'Contingency': cosine_similarity(preRelationVector, ContingencyVector)[0, 0],
        'Temporal': cosine_similarity(preRelationVector, TemporalVector)[0, 0],
        'Expansion': cosine_similarity(preRelationVector, ExpansionVector)[0, 0],
    }

    pre_label = max(cos_dict, key=cos_dict.get)

    print('label:', sentenceVector['relation'])
    print('pre_label:', pre_label)

    if sentenceVector['relation'] == pre_label:
        count += 1
    sum += 1

    print('准确数' + str(count))
    print('准确度' + str(count / sum))
