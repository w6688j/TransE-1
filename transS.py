from copy import deepcopy
from random import uniform, sample

import numpy as np


class TransS:
    def __init__(self, wordList, relationList, tripleList, sentenceDict, margin=1, learingRate=0.00001, dim=10,
                 L1=True):
        self.margin = margin
        self.learingRate = learingRate
        self.dim = dim  # 向量维度
        self.wordList = wordList  # 一开始，entityList是entity的list；初始化后，变为字典，key是entity，values是其向量（使用narray）。
        self.relationList = relationList  # 理由同上
        self.tripleList = tripleList  # 理由同上
        self.sentenceDict = sentenceDict  # 理由同上
        self.loss = 0
        self.L1 = L1

    def initWordVectorList(self):
        wordVectorList = {}
        for word in self.wordList:
            n = 0
            wordVector = []
            while n < self.dim:
                ram = init(self.dim)  # 初始化的范围
                wordVector.append(ram)
                n += 1
            wordVector = norm(wordVector)  # 归一化
            wordVectorList[word] = wordVector

        self.wordVectorList = wordVectorList
        print("wordVectorList初始化完成，数量是%d" % len(wordVectorList))

    def initSentenceEntityList(self):
        sentenceEntityList = {}
        for item in self.sentenceDict.items():
            key = item[0]
            sentence = item[1]
            sentenceEntityList[key] = sentence

        self.sentenceEntityList = sentenceEntityList
        print("sentenceEntityList初始化完成，数量是%d" % len(sentenceEntityList))

    def initRelationList(self):
        relationVectorList = {}
        for relation in self.relationList:
            n = 0
            relationVector = []
            while n < self.dim:
                ram = init(self.dim)  # 初始化的范围
                relationVector.append(ram)
                n += 1
            relationVector = norm(relationVector)  # 归一化
            relationVectorList[relation] = relationVector
        print("relationVectorList初始化完成，数量是%d" % len(relationVectorList))

        self.relationList = relationVectorList

    def initialize(self):
        '''
        初始化向量
        '''
        self.initWordVectorList()
        self.initSentenceEntityList()
        self.initRelationList()
        self.entityList = self.get_entity_vector_list()

    def transS(self, cI=20):
        print("训练开始")
        for cycleIndex in range(cI):
            Sbatch = self.getSample(150)
            Tbatch = []  # 元组对（原三元组，打碎的三元组）的列表 ：{((h,r,t),(h',r,t'))}
            for sbatch in Sbatch:
                tripletWithCorruptedTriplet = (sbatch, self.getCorruptedTriplet(sbatch))
                if (tripletWithCorruptedTriplet not in Tbatch):
                    Tbatch.append(tripletWithCorruptedTriplet)
            self.update(Tbatch)
            if cycleIndex % 100 == 0:
                print("第%d次循环" % cycleIndex)
                print(self.loss)
                self.writeRelationVector("result/relationVector_pdtb.txt")
                self.writeEntilyVector("result/sentenceEntityVector_pdtb.txt")
                self.writeWordEntilyVector("result/wordEntityVector_pdtb.txt")
                print("wordVectorList数量是%d" % len(self.wordVectorList))
                self.loss = 0

    def getSample(self, size):
        return sample(self.tripleList, size)

    def getCorruptedTriplet(self, triplet):
        '''
        training triplets with either the head or tail replaced by a random entity (but not both at the same time)
        :param triplet:
        :return corruptedTriplet:
        '''
        i = uniform(-1, 1)
        if i < 0:  # 小于0，打坏三元组的第一项
            while True:
                entityTemp = sample(self.sentenceEntityList.keys(), 1)[0]
                if entityTemp != triplet[0]:
                    break
            corruptedTriplet = (entityTemp, triplet[1], triplet[2])
        else:  # 大于等于0，打坏三元组的第二项
            while True:
                entityTemp = sample(self.sentenceEntityList.keys(), 1)[0]
                if entityTemp != triplet[1]:
                    break
            corruptedTriplet = (triplet[0], entityTemp, triplet[2])
        return corruptedTriplet

    # 根据句子键值获取句子向量, 单词向量取平均
    def get_vector_by_sentence(self, sentence):
        sentenceVector = np.zeros((self.dim))
        word_list = sentence.split()
        for word in word_list:
            if word not in self.wordVectorList.keys():
                word = '__UNK__'
            sentenceVector += self.wordVectorList[word]

        sentenceVector = sentenceVector / len(word_list)

        return sentenceVector

    # 获取句子实体的向量
    def get_entity_vector_list(self):
        entityVectorList = {}
        for key in self.sentenceEntityList:
            sentence = self.sentenceEntityList[key]
            entityVectorList[key] = self.get_vector_by_sentence(sentence)

        return entityVectorList

    def update(self, Tbatch):
        copyEntityList = deepcopy(self.entityList)
        copyRelationList = deepcopy(self.relationList)

        for tripletWithCorruptedTriplet in Tbatch:
            (
                headEntityKey,
                tailEntityKey,
                relationVector,
                headEntityKeyWithCorruptedTriplet,
                tailEntityKeyWithCorruptedTriplet,
                headEntityVectorBeforeBatch,
                tailEntityVectorBeforeBatch,
                relationVectorBeforeBatch,
                headEntityVectorWithCorruptedTripletBeforeBatch,
                tailEntityVectorWithCorruptedTripletBeforeBatch
            ) = self.getData(tripletWithCorruptedTriplet, copyRelationList)

            distTriplet, distCorruptedTriplet = self.calculateDist(
                headEntityVectorBeforeBatch,
                tailEntityVectorBeforeBatch,
                relationVectorBeforeBatch,
                headEntityVectorWithCorruptedTripletBeforeBatch,
                tailEntityVectorWithCorruptedTripletBeforeBatch)

            eg = self.margin + distTriplet - distCorruptedTriplet
            if eg > 0:  # [function]+ 是一个取正值的函数
                self.loss += eg
                tempPositive, tempNegtative = self.calculateTemp(
                    tailEntityVectorBeforeBatch,
                    headEntityVectorBeforeBatch,
                    relationVectorBeforeBatch,
                    tailEntityVectorWithCorruptedTripletBeforeBatch,
                    headEntityVectorWithCorruptedTripletBeforeBatch)

                relationVector = relationVector + tempPositive - tempNegtative

                headEntityVector, tailEntityVector, headEntityVectorWithCorruptedTriplet, tailEntityVectorWithCorruptedTriplet = self.updateVector(
                    tempPositive,
                    tempNegtative,
                    headEntityKey,
                    tailEntityKey,
                    headEntityKeyWithCorruptedTriplet,
                    tailEntityKeyWithCorruptedTriplet)

                copyEntityList, copyRelationList = self.updateEntityAndRelationList(
                    copyEntityList,
                    copyRelationList,
                    tripletWithCorruptedTriplet,
                    headEntityVector,
                    tailEntityVector,
                    relationVector,
                    headEntityVectorWithCorruptedTriplet,
                    tailEntityVectorWithCorruptedTriplet
                )

        self.entityList = copyEntityList
        self.relationList = copyRelationList

    def getData(self, tripletWithCorruptedTriplet, copyRelationList):
        headEntityKey = tripletWithCorruptedTriplet[0][0]
        tailEntityKey = tripletWithCorruptedTriplet[0][1]

        relationVector = copyRelationList[tripletWithCorruptedTriplet[0][2]]

        headEntityKeyWithCorruptedTriplet = tripletWithCorruptedTriplet[1][0]
        tailEntityKeyWithCorruptedTriplet = tripletWithCorruptedTriplet[1][1]

        headEntityVectorBeforeBatch = self.entityList[
            tripletWithCorruptedTriplet[0][0]]  # tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple
        tailEntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][1]]
        relationVectorBeforeBatch = self.relationList[tripletWithCorruptedTriplet[0][2]]
        headEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][0]]
        tailEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][1]]

        return (
            headEntityKey,
            tailEntityKey,
            relationVector,
            headEntityKeyWithCorruptedTriplet,
            tailEntityKeyWithCorruptedTriplet,
            headEntityVectorBeforeBatch,
            tailEntityVectorBeforeBatch,
            relationVectorBeforeBatch,
            headEntityVectorWithCorruptedTripletBeforeBatch,
            tailEntityVectorWithCorruptedTripletBeforeBatch
        )

    # distTriplet distCorruptedTriplet
    def calculateDist(self,
                      headEntityVectorBeforeBatch,
                      tailEntityVectorBeforeBatch,
                      relationVectorBeforeBatch,
                      headEntityVectorWithCorruptedTripletBeforeBatch,
                      tailEntityVectorWithCorruptedTripletBeforeBatch):
        if self.L1:
            distTriplet = distanceL1(headEntityVectorBeforeBatch, tailEntityVectorBeforeBatch,
                                     relationVectorBeforeBatch)
            distCorruptedTriplet = distanceL1(headEntityVectorWithCorruptedTripletBeforeBatch,
                                              tailEntityVectorWithCorruptedTripletBeforeBatch,
                                              relationVectorBeforeBatch)
        else:
            distTriplet = distanceL2(headEntityVectorBeforeBatch, tailEntityVectorBeforeBatch,
                                     relationVectorBeforeBatch)
            distCorruptedTriplet = distanceL2(headEntityVectorWithCorruptedTripletBeforeBatch,
                                              tailEntityVectorWithCorruptedTripletBeforeBatch,
                                              relationVectorBeforeBatch)
        return (distTriplet, distCorruptedTriplet)

    # 计算tempPositive tempNegtative
    def calculateTemp(self,
                      tailEntityVectorBeforeBatch,
                      headEntityVectorBeforeBatch,
                      relationVectorBeforeBatch,
                      tailEntityVectorWithCorruptedTripletBeforeBatch,
                      headEntityVectorWithCorruptedTripletBeforeBatch):
        if self.L1:
            tempPositive = 2 * self.learingRate * (
                    tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch)
            tempNegtative = 2 * self.learingRate * (
                    tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)
            tempPositiveL1 = []
            tempNegtativeL1 = []
            for i in range(self.dim):  # 不知道有没有pythonic的写法（比如列表推倒或者numpy的函数）？
                if tempPositive[i] >= 0:
                    tempPositiveL1.append(1)
                else:
                    tempPositiveL1.append(-1)
                if tempNegtative[i] >= 0:
                    tempNegtativeL1.append(1)
                else:
                    tempNegtativeL1.append(-1)
            tempPositive = np.array(tempPositiveL1)
            tempNegtative = np.array(tempNegtativeL1)

        else:
            tempPositive = 2 * self.learingRate * (
                    tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch)
            tempNegtative = 2 * self.learingRate * (
                    tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)

        return (tempPositive, tempNegtative)

    # 更新向量
    def updateVector(self,
                     tempPositive,
                     tempNegtative,
                     headEntityKey,
                     tailEntityKey,
                     headEntityKeyWithCorruptedTriplet,
                     tailEntityKeyWithCorruptedTriplet):

        self.updateWordVector(self.sentenceEntityList[headEntityKey], tempPositive, 'plus')
        self.updateWordVector(self.sentenceEntityList[tailEntityKey], tempPositive, 'reduce')
        self.updateWordVector(self.sentenceEntityList[headEntityKeyWithCorruptedTriplet],
                              tempNegtative,
                              'reduce')
        self.updateWordVector(self.sentenceEntityList[tailEntityKeyWithCorruptedTriplet],
                              tempNegtative,
                              'plus')

        # 重新计算sentence的word向量平均值
        headEntityVector = self.get_vector_by_sentence(self.sentenceEntityList[headEntityKey])
        tailEntityVector = self.get_vector_by_sentence(self.sentenceEntityList[tailEntityKey])
        headEntityVectorWithCorruptedTriplet = self.get_vector_by_sentence(
            self.sentenceEntityList[headEntityKeyWithCorruptedTriplet])
        tailEntityVectorWithCorruptedTriplet = self.get_vector_by_sentence(
            self.sentenceEntityList[tailEntityKeyWithCorruptedTriplet])

        return (
            headEntityVector,
            tailEntityVector,
            headEntityVectorWithCorruptedTriplet,
            tailEntityVectorWithCorruptedTriplet)

    # 更新单词向量
    def updateWordVector(self, sentence, data, type):
        for word in sentence.split():
            if word not in self.wordVectorList.keys():
                word = '__UNK__'

            wordVector = self.wordVectorList[word]
            if type == 'plus':
                wordVector += data
            else:
                wordVector -= data

            self.wordVectorList[word] = wordVector

    # 更新copyEntityList copyRelationList
    def updateEntityAndRelationList(self,
                                    copyEntityList,
                                    copyRelationList,
                                    tripletWithCorruptedTriplet,
                                    headEntityVector,
                                    tailEntityVector,
                                    relationVector,
                                    headEntityVectorWithCorruptedTriplet,
                                    tailEntityVectorWithCorruptedTriplet):
        # 只归一化这几个刚更新的向量，而不是按原论文那些一口气全更新了
        copyEntityList[tripletWithCorruptedTriplet[0][0]] = norm(headEntityVector)
        copyEntityList[tripletWithCorruptedTriplet[0][1]] = norm(tailEntityVector)
        copyRelationList[tripletWithCorruptedTriplet[0][2]] = norm(relationVector)
        copyEntityList[tripletWithCorruptedTriplet[1][0]] = norm(headEntityVectorWithCorruptedTriplet)
        copyEntityList[tripletWithCorruptedTriplet[1][1]] = norm(tailEntityVectorWithCorruptedTriplet)

        return (copyEntityList, copyRelationList)

    # 写入Word实体
    def writeWordEntilyVector(self, dir):
        print("写入Word实体")
        wordEntityVectorFile = open(dir, 'w')
        for word in self.wordVectorList.keys():
            wordEntityVectorFile.write(word + "\t")
            wordEntityVectorFile.write(str(self.wordVectorList[word].tolist()))
            wordEntityVectorFile.write("\n")
        wordEntityVectorFile.close()

    # 写入Sentence实体
    def writeEntilyVector(self, dir):
        print("写入Sentence实体")
        entityVectorFile = open(dir, 'w')
        for entity in self.entityList.keys():
            entityVectorFile.write(entity + "\t")
            entityVectorFile.write(str(self.entityList[entity].tolist()))
            entityVectorFile.write("\n")
        entityVectorFile.close()

    # 写入关系
    def writeRelationVector(self, dir):
        print("写入关系")
        relationVectorFile = open(dir, 'w')
        for relation in self.relationList.keys():
            relationVectorFile.write(relation + "\t")
            relationVectorFile.write(str(self.relationList[relation].tolist()))
            relationVectorFile.write("\n")
        relationVectorFile.close()


def init(dim):
    return uniform(-6 / (dim ** 0.5), 6 / (dim ** 0.5))


def distanceL1(h, t, r):
    s = h + r - t
    sum = np.fabs(s).sum()
    return sum


def distanceL2(h, t, r):
    s = h + r - t
    sum = (s * s).sum()
    return sum


def norm(list):
    '''
    归一化
    :param 向量
    :return: 向量的平方和的开方后的向量
    '''
    var = np.linalg.norm(list)
    i = 0
    while i < len(list):
        list[i] = list[i] / var
        i += 1
    return np.array(list)


def openDetailsAndId(dir, sp="\t"):
    idNum = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(DetailsAndId[0])
            idNum += 1
    return idNum, list


def openTrain(dir, sp="\t"):
    num = 0
    list = []
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if (len(triple) < 3):
                continue
            list.append(tuple(triple))
            num += 1
    return num, list


def openSentenceEntity(dir, sp="\t"):
    dict = {}
    with open(dir) as file:
        lines = file.readlines()
        for line in lines:
            line_sp = line.strip().split(sp)
            if len(line_sp) > 1:
                dict[line_sp[0]] = line_sp[1]

    return dict


if __name__ == '__main__':
    dirEntity = "KBdata/PDTB/word_list.txt"
    entityIdNum, wordList = openDetailsAndId(dirEntity)

    dirRelation = "KBdata/PDTB/relation2id.txt"
    relationIdNum, relationList = openDetailsAndId(dirRelation)

    dirTrain = "KBdata/PDTB/train.txt"
    tripleNum, tripleList = openTrain(dirTrain)

    dirSentenceEntity = "KBdata/PDTB/sentence_entity.txt"
    sentenceDict = openSentenceEntity(dirSentenceEntity)

    print("打开TransS")
    transS = TransS(wordList, relationList, tripleList, sentenceDict, margin=1, dim=100)
    print("TransS初始化")
    transS.initialize()
    transS.transS(15000)
    transS.writeRelationVector("result/relationVector_pdtb.txt")
    transS.writeEntilyVector("result/sentenceEntityVector_pdtb.txt")
    transS.writeWordEntilyVector("result/wordEntityVector_pdtb.txt")
