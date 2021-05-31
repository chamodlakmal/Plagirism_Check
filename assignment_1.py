from nltk.corpus import stopwords
import pandas as pd
from scipy import spatial


def removePunctuation(doc):
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in doc:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct


doc1 = removePunctuation(
    open("1", "r", encoding="utf-8").read().lower().strip()).split(" ")
doc2 = removePunctuation(
    open("2", "r", encoding="utf-8").read().lower().strip()).split(" ")
doc3 = removePunctuation(
    open("3", "r", encoding="utf-8").read().lower().strip()).split(" ")
doc4 = removePunctuation(
    open("4", "r", encoding="utf-8").read().lower().strip()).split(" ")
doc5 = removePunctuation(
    open("5", "r", encoding="utf-8").read().lower().strip()).split(" ")

query = removePunctuation(
    open("query", "r", encoding="utf-8").read().lower().strip()).split(" ")

doc1_without_sw = [word for word in doc1 if not word in stopwords.words()]
doc2_without_sw = [word for word in doc2 if not word in stopwords.words()]
doc3_without_sw = [word for word in doc3 if not word in stopwords.words()]
doc4_without_sw = [word for word in doc4 if not word in stopwords.words()]
doc5_without_sw = [word for word in doc5 if not word in stopwords.words()]

query_without_sw = [word for word in query if not word in stopwords.words()]

word_set = set(doc1_without_sw).union(set(doc2_without_sw), set(
    doc3_without_sw), set(doc4_without_sw), set(doc5_without_sw), set(query_without_sw))


doc1_word_dic = dict.fromkeys(word_set, 0)
doc2_word_dic = dict.fromkeys(word_set, 0)
doc3_word_dic = dict.fromkeys(word_set, 0)
doc4_word_dic = dict.fromkeys(word_set, 0)
doc5_word_dic = dict.fromkeys(word_set, 0)
query_word_dic = dict.fromkeys(word_set, 0)


for word in doc1_without_sw:
    doc1_word_dic[word] += 1

for word in doc2_without_sw:
    doc2_word_dic[word] += 1

for word in doc3_without_sw:
    doc3_word_dic[word] += 1

for word in doc4_without_sw:
    doc4_word_dic[word] += 1

for word in doc5_without_sw:
    doc5_word_dic[word] += 1

for word in query_without_sw:
    query_word_dic[word] += 1


def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict


tfDoc1 = computeTF(doc1_word_dic, doc1_without_sw)
tfDoc2 = computeTF(doc2_word_dic, doc2_without_sw)
tfDoc3 = computeTF(doc3_word_dic, doc3_without_sw)
tfDoc4 = computeTF(doc4_word_dic, doc4_without_sw)
tfDoc5 = computeTF(doc5_word_dic, doc5_without_sw)
tfQuery = computeTF(query_word_dic, doc5_without_sw)


def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict


idfs = computeIDF([doc1_word_dic, doc2_word_dic, doc3_word_dic,
                  doc4_word_dic, doc5_word_dic, query_word_dic])


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf


tfidfDoc1 = computeTFIDF(tfDoc1, idfs)
tfidfDoc2 = computeTFIDF(tfDoc2, idfs)
tfidfDoc3 = computeTFIDF(tfDoc3, idfs)
tfidfDoc4 = computeTFIDF(tfDoc4, idfs)
tfidfDoc5 = computeTFIDF(tfDoc5, idfs)
tfidfQuery = computeTFIDF(tfQuery, idfs)


doc1_result = 1 - \
    spatial.distance.cosine(list(tfidfDoc1.values()),
                            list(tfidfQuery.values()))
doc2_result = 1 - \
    spatial.distance.cosine(list(tfidfDoc2.values()),
                            list(tfidfQuery.values()))
doc3_result = 1 - \
    spatial.distance.cosine(list(tfidfDoc3.values()),
                            list(tfidfQuery.values()))
doc4_result = 1 - \
    spatial.distance.cosine(list(tfidfDoc4.values()),
                            list(tfidfQuery.values()))
doc5_result = 1 - \
    spatial.distance.cosine(list(tfidfDoc5.values()),
                            list(tfidfQuery.values()))
query_result = 1 - \
    spatial.distance.cosine(list(tfidfQuery.values()),
                            list(tfidfQuery.values()))


print(doc1_result, doc2_result, doc3_result,
      doc4_result, doc5_result, query_result)
