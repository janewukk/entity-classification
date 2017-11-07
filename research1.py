
# coding: utf-8

# In[1]:

from pyjarowinkler import distance
from math import*
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from sklearn.metrics import euclidean_distances
import gensim
import pymysql
import pymysql.cursors
from gensim import corpora, models
import math
from textblob import TextBlob as tb
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import requests
import json
import time


# In[2]:

def macro_precision(result, y, n_cluster):
    groupDict = {}
    i= 0
    for item in result:
        if item in groupDict:
            groupDict[item].append(i)
        else:
            groupDict[item] = [i]
        i = i + 1

    trueDict = {}
    i = 0
    for item in y:
        if item not in trueDict:
            trueDict[item] = [i]
        else:
            trueDict[item].append(i)
        i = i + 1

    inverseTrue = {}
    for key in trueDict.keys():
        for item in trueDict[key]:
            inverseTrue[item] = key
    
    totalNotSame = 0
    for key in groupDict.keys():
        temp = groupDict[key]
        f_id = ""
        i = 0
        control = True
        while i < len(temp) and control == True:
            f_id_cur = inverseTrue[temp[i]]
            if f_id == "":
                f_id = f_id_cur
            else:
                if f_id_cur != f_id:
                    totalNotSame += 1
                    control = False
            i += 1
                    
    return (n_cluster-totalNotSame) / n_cluster

def macro_recall(result, y, n_cluster):
    return macro_precision(y, result, n_cluster)

def micro_precision(result, y):
    lenC = len(result)
    groupDict = {}
    i= 0
    for item in result:
        if item in groupDict:
            groupDict[item].append(i)
        else:
            groupDict[item] = [i]
        i = i + 1

    trueDict = {}
    i = 0
    for item in y:
        if item not in trueDict:
            trueDict[item] = [i]
        else:
            trueDict[item].append(i)
        i = i + 1

    inverseTrue = {}
    for key in trueDict.keys():
        for item in trueDict[key]:
            inverseTrue[item] = key
    
    totalSame = 0
    for key in groupDict.keys():
        temp = groupDict[key]
        output = {}
        maxO = 0
        maxKey = ""
        for item in temp:
            f_id = inverseTrue[item]
            if f_id not in output:
                output[f_id] = 1
            else:
                output[f_id] += 1
            
            if output[f_id] > maxO:
                maxO = output[f_id]
                maxKey = key
                
        totalSame += maxO
    return  totalSame / lenC
                
def micro_recall(result, y):
    return micro_precision(y, result)


def pairwise_precision(result, y):
    lenC = len(result)
    groupDict = {}
    i= 0
    for item in y:
        if item in groupDict:
            groupDict[item].append(i)
        else:
            groupDict[item] = [i]
        i = i + 1

    trueDict = {}
    i = 0
    for item in result:
        if item not in trueDict:
            trueDict[item] = [i]
        else:
            trueDict[item].append(i)
        i = i + 1

    inverseTrue = {}
    for key in trueDict.keys():
        for item in trueDict[key]:
            inverseTrue[item] = key
    
    totalPair = 0
    totalHit = 0
    for key in groupDict.keys():
        temp = groupDict[key]
        i = 0
        j = 0
        while i < len(temp):
            while j < len(temp):
                f_id_i = inverseTrue[temp[i]]
                f_id_j = inverseTrue[temp[j]]
                if f_id_i == f_id_j:
                    totalHit += 1
                j += 1
                totalPair += 1
            i += 1
        
    return  totalHit / totalPair


def pairwise_recall(result, y):
    lenC = len(result)
    groupDict = {}
    i= 0
    for item in y:
        if item in groupDict:
            groupDict[item].append(i)
        else:
            groupDict[item] = [i]
        i = i + 1

    trueDict = {}
    i = 0
    for item in result:
        if item not in trueDict:
            trueDict[item] = [i]
        else:
            trueDict[item].append(i)
        i = i + 1

    inverseGroup = {}
    for key in groupDict.keys():
        for item in groupDict[key]:
            inverseGroup[item] = key
    
    totalPair = 0
    totalHit = 0
    for key in trueDict.keys():
        temp = trueDict[key]
        i = 0
        j = 0
        while i < len(temp):
            while j < len(temp):
                f_id_i = inverseGroup[temp[i]]
                f_id_j = inverseGroup[temp[j]]
                if f_id_i == f_id_j:
                    totalHit += 1
                j += 1
                totalPair += 1
            i += 1
        
    return  totalHit / totalPair


# In[3]:

def connect_to_database():
    options = {
        'user': "root",
        'passwd': "root",
        'db': "KnowBase",
        'cursorclass' : pymysql.cursors.DictCursor
    }
    db = pymysql.connect(**options)
    db.autocommit(True)
    return db

# data clean, exclude stop word, need to to lower
def exclude_stop_word(bloblist):
    stop = set(stopwords.words('english'))
    filtered_words = [i for i in bloblist[0].lower().split() if i not in stop]
    return filtered_words


def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

# helper for type_overlap
def lda_model(x):
    # Latent Dirichlet Allocation (LDA) for X
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(x)
    p_stemmer = PorterStemmer()
    texts = [p_stemmer.stem(i) for i in tokens]
    dictionary = corpora.Dictionary(texts)
    
    ################## errors here ##################
    
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("corpus")
    print(corpus)
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)

    return ldamodel

# extract the coefficient from ldamodel
def type_overlap(m1, m2):
    pred1 = m1["A"][0]
    pred2 = m2["A"][0]
    ldamodel_x = lda_model(pred1)
    ldamodel_y = lda_model(pred2)
    print(ldamodel_x)
    print(ldamodel_y)
    jaccard_similarity(ldamodel_x, ldamodel_y)


# In[19]:

def entity_overlap(m1, listB):
    input1 = {}
    input1["arg1"] = m1["entity"]
    input1["arg1Id"] = m1["f_id"]
    
    input2 = []
    for item in listB:
        temp = {}
        temp["arg1"] = item["entity"]
        temp["arg1Id"] = item["f_id"]
        input2.append(temp)
    
        
    payload1 = {
        "name": "entityOverlap",
        "timestamp": time.time(),
        "mention1": input1,
        "mention2": input2
    }
    r1 = requests.post("http://141.212.110.52:8181/service/entityOverlap", data = json.dumps(payload1))
    
    out = []
    result = json.loads(r1.text)["result"]
    for key in result.keys():
        out.append(np.exp(-result[key]))
    
    print(out)
    return out

    
def build_entity_overlap_matrix(wordList):
    out = []
    for i in range(len(wordList)):
        out.append(entity_overlap(wordList[i], wordList))

    return out
    
def apply_entity_overlap(wordList, y, n_cluster_in, entity_matrix):
    model = AgglomerativeClustering(n_clusters=n_cluster_in, affinity="precomputed", linkage="average")
    model.fit(entity_matrix)
    result = model.labels_
    Out_to_print = {}
    Out_to_print["micro precision"] = micro_precision(result, y)
    Out_to_print["micro recall"] = micro_recall(result, y)
    Out_to_print["macro precision"] = macro_precision(result, y, n_cluster_in)
    Out_to_print["macro recall"] = macro_recall(result, y, n_cluster_in)
    Out_to_print["pairwise precision"] = pairwise_precision(result, y)
    Out_to_print["pairwise recall"] = pairwise_recall(result, y)
    return Out_to_print


# In[5]:

def work_overlap_v1(m1, listB):
    input1 = {}
    input1["arg1"] = m1["entity"]
    input1["arg1Id"] = m1["f_id"]
    
    input2 = []
    for item in listB:
        temp = {}
        temp["arg1"] = item["entity"]
        temp["arg1Id"] = item["f_id"]
        input2.append(temp)
    
        
    payload1 = {
        "name": "workOverlap",
        "timestamp": time.time(),
        "mention1": input1,
        "mention2": input2
    }
    r1 = requests.post("http://141.212.110.52:8181/service/workOverlap", data = json.dumps(payload1))
    
    out = []
    result = json.loads(r1.text)["result"]
    for key in result.keys():
        out.append(np.exp(-result[key]))

    return out
    
def build_work_overlap_matrix(wordList):
    out = []
    
    for item in wordList:
        out.append(word_overlap(item, wordList))
    return out
    
def apply_work_overlap(wordList, y, n_cluster_in, work_matrix):
    model = AgglomerativeClustering(n_clusters=n_cluster_in, affinity="precomputed", linkage="average")
    model.fit(work_matrix)
    result = model.labels_
    Out_to_print = {}
    Out_to_print["micro precision"] = micro_precision(result, y)
    Out_to_print["micro recall"] = micro_recall(result, y)
    Out_to_print["macro precision"] = macro_precision(result, y, n_cluster_in)
    Out_to_print["macro recall"] = macro_recall(result, y, n_cluster_in)
    Out_to_print["pairwise precision"] = pairwise_precision(result, y)
    Out_to_print["pairwise recall"] = pairwise_recall(result, y)
    return Out_to_print


# In[7]:

db = connect_to_database()
cur = db.cursor()
string = "select b.freebase_id, b.entity, b.relation, b.value, b.link_am_score, b.link_scroe, b.base_id, "
string += "b.freebase_entity from EntitySelect b"
cur.execute(string)
results = cur.fetchall()
groundTrue = {}
label = []
entity = []
entityDict = {}
y = []
i= 0
wordList = []
allWord = []
file_t = open("data.txt", 'w')
for result in results:
    string = result["base_id"] + " " + result["entity"] + " " + result["relation"] + " " + result["value"] + " "
    string += result["freebase_id"] +  " " + result["freebase_entity"] + " " + result["link_scroe"] + " "
    string += result["link_am_score"]
    file_t.write(string + "\n")
    if result["freebase_id"] not in groundTrue:
        groundTrue[result["freebase_id"]] = i
        label.append(result["freebase_entity"].lower())
        i = i + 1
    if result["entity"] not in entityDict:
        entityDict[result["entity"].lower()] = 1
    entity.append(result["entity"].lower())
    y.append(groundTrue[result["freebase_id"]])
    
    temp = {}
    temp["n"] = result["entity"].lower().split()
    list_A = result["relation"].lower().split()
    list_A.extend(result["value"].lower().split())
    temp["A"] = list_A
    temp["f_id"] = result["freebase_id"]
    temp["f_entity"] = result["freebase_entity"]
    temp["score"] = result["link_scroe"]
    temp["am_score"] = result["link_am_score"]
    temp["relation"] = result["relation"]
    temp["value"] = result["value"]
    temp["id"] = result["base_id"]
    temp["entity"] = result["entity"]
    wordList.append(temp)
    allWord.extend(result["entity"].lower().split())
    allWord.extend(result["relation"].lower().split())
    allWord.extend(result["value"].lower().split())



# In[ ]:

entity_overlap_matrix = build_entity_overlap_matrix(wordList)
n_cluster = 150
print("finish building")
print(apply_entity_overlap(wordList, y, n_cluster, entity_overlap_matrix))


# In[ ]:

n_cluster = 150
work_overlap_matrix = build_work_overlap_matrix(wordList)
print(apply_work_overlap(wordList, y, n_cluster))


# In[ ]:
'''
def tf(word, blob):
	return blob.count(word) / len(blob)

def idf_func(word, blob):
	return np.log(1 / float(blob.count(word) + 1))

def tfidf(word, blob):
	return tf(word, blob) * idf(word, blob)

# input document 
def idf_token_overlap(m1, m2, blob):
	entity1 = m1["n"]
	entity2 = m2["n"]

	intersection_word = set.intersection(*[set(entity1), set(entity2)])
	union_word = set.union(*[set(entity1), set(entity2)])
	
	numerator = 0
	denominator = 0
	for word in intersection_word:
		numerator += idf_func(word, blob)
	if numerator == 0:
		return 0
	else:
		for word in union_word:
			denominator += idf_func(word, blob)
		return (numerator / denominator)

def apply_idf_token_overlap(wordList, y):
    out = []
    for item1 in wordList:
        temp = []
        for item2 in wordList:
            result = np.exp(-idf_token_overlap_matrix(item1, item2))
            temp.append(result)
        out.append(temp)
    model = AgglomerativeClustering(n_clusters=150, affinity="precomputed", linkage="average")
    model.fit(out)
    result = model.labels_
    Out_to_print = {}
    Out_to_print["micro precision"] = micro_precision(result, y)
    Out_to_print["micro recall"] = micro_recall(result, y)
    Out_to_print["macro precision"] = macro_precision(result, y, 150)
    Out_to_print["macro recall"] = macro_recall(result, y, 150)
    Out_to_print["pairwise precision"] = pairwise_precision(result, y)
    Out_to_print["pairwise recall"] = pairwise_recall(result, y)
    return Out_to_print


# In[ ]:

def work_overlap_old(m1, m2):
    entity1 = m1["entity"]
    entity2 = m2["entity"]
    
    payload1 = {
        "name": "requestUrls",
        "timestamp": time.time(),
        "tuple": {
                "id": m1["id"],
                "arg1": entity2,
                "arg2": m1["value"],
                "relation": m1["relation"],
                "arg1Id": m1["f_id"],
                "arg1Name": m1["f_entity"],
                "linkScore": m1["score"],
                "linkAmbiguity": m1["am_score"]
        }
    }
    r1 = requests.post("http://141.212.110.52:8181/service/getUrl", data = json.dumps(payload1))
    url_list1 = json.loads(r1.text)["result"]["url"]
    
    payload2 = {
        "name": "requestUrls",
        "timestamp": time.time(),
        "tuple": {
                "id": m2["id"],
                "arg1": entity2,
                "arg2": m2["value"],
                "relation": m2["relation"],
                "arg1Id": m2["f_id"],
                "arg1Name": m2["f_entity"],
                "linkScore": m2["score"],
                "linkAmbiguity": m2["am_score"]
        }
    }
    
    r2 = requests.post("http://141.212.110.52:8181/service/getUrl", data = json.dumps(payload2))
    url_list2 = json.loads(r2.text)["result"]["url"]
    
    total = 0
    count = 0
    
    if len(set(url_list1).intersection(url_list2)) != 0:
        return 1 
    else:
        for link1 in url_list1:
            for link2 in url_list2:
                payload1 = {
                    "name": "requestTop100",
                    "timestamp": time.time(),
                    "uri": link1
                }
                r1 = requests.post("http://141.212.110.52:8181/service/getTop100", data = json.dumps(payload1))
                dict1 = json.loads(r1.text)["result"]["words"]

                payload2 = {
                    "name": "requestTop100",
                    "timestamp": time.time(),
                    "uri": link2
                }
                r2 = requests.post("http://141.212.110.52:8181/service/getTop100", data = json.dumps(payload2))
                dict2 = json.loads(r2.text)["result"]["words"]
    
                if len(dict1) == 0 or len(dict2) == 0:
                    total += 0
                else:
                    total += jaccard_similarity(dict1, dict2)
                count += 1
    if count == 0:
        return 0
    else:
        return total / count
    
def build_work_overlap_matrix_old(wordList):
    out = []
    for i in range(len(wordList)):
        temp = []
        for j in range(len(wordList)):
            temp.append(-1)
        out.append(temp)

    for i in range(len(wordList)):
        for j in range(len(wordList)):
            if i == j:
                out[i][j] = np.exp(-1)
            if out[i][j] == -1:
                result = work_overlap(wordList[i], wordList[j])
                if result < 0.8:
                    result = 0
                result = np.exp(-result)
                out[i][j] = result
                out[j][i] = result
    return out
    
def apply_work_overlap_old(wordList, y, n_cluster_in, work_matrix):
    model = AgglomerativeClustering(n_clusters=n_cluster_in, affinity="precomputed", linkage="average")
    model.fit(work_matrix)
    result = model.labels_
    Out_to_print = {}
    Out_to_print["micro precision"] = micro_precision(result, y)
    Out_to_print["micro recall"] = micro_recall(result, y)
    Out_to_print["macro precision"] = macro_precision(result, y, n_cluster_in)
    Out_to_print["macro recall"] = macro_recall(result, y, n_cluster_in)
    Out_to_print["pairwise precision"] = pairwise_precision(result, y)
    Out_to_print["pairwise recall"] = pairwise_recall(result, y)
    return Out_to_print

'''