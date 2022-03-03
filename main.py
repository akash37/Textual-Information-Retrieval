import os
import sys
import xml.etree.ElementTree as elementTree
import math
import re
import time
import nltk
from datetime import timedelta
from nltk.corpus import stopwords
from contextlib import redirect_stdout
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
start = time.time()
print("Start Time = " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start))))
experiment = int(sys.argv[1])


'''
Returns the list of files in a particular path
'''


def get_list_of_files(path):
    return os.listdir(path)


'''
This function returns the total text which is to be considered for preprocessing and indexing 
'''


def get_total_text(root, type_of_file):
    if experiment == 1:
        if type_of_file == "collection":
            text = (str(root[1].text) + " " + str(root[2].text)).lower()
        else:
            text = (str(root[1].text) + " " + str(root[2].text)).lower()
    elif experiment == 2:
        if type_of_file == "collection":
            text = (str(root[1].text) + " " + str(root[1].text) + " " + str(root[2].text)).lower()
        else:
            text = (str(root[1].text) + " " + str(root[2].text)).lower()
    elif experiment == 3:
        if type_of_file == "collection":
            text = (str(root[1].text) + " " + str(root[2].text)).lower()
        else:
            text = str(root[1].text).lower()
    elif experiment == 4:
        if type_of_file == "collection":
            text = str(root[1].text).lower()
        else:
            text = str(root[1].text).lower()
    return text


'''
Indexing
This function performs preprocessing and then calculates tf-idf for the list of documents.
It returns dictionary of dictionaries which is the index. It does the indexing of the document.
Returns Document_Length, Term_Count(Inverted Document Index), Index of collections
Document_Length --> It is used during the BM25 similarity score calculations
{
    "doc_id_1": number of words present in the document 1 after preprocessing,
    "doc_id_2": number of words present in the document 2 after preprocessing,
    "doc_id_3": number of words present in the document 3 after preprocessing
    ....
}

Term Count --> This dictionary of dictionaries serves as the inverted document index
{
    "term_1" : {"doc_id_1": number of times term_1 occurs in doc_id_1,
                "doc_id_2": number of times term_1 occurs in doc_id_2,
                ....},
    "term_2" : {"doc_id_1": number of times term_2 occurs in doc_id_1,
                "doc_id_2": number of times term_2 occurs in doc_id_2,
                ....},
}

Index of documents --> This dictionary of dictionaries serves as the index of documents
{
    "doc_id_1" : {"term_1": val1,
                  "term_2": val2,
                  "term_3": val3
                  .... },
    "doc_id_2" : {"term_1": val4,
                  "term_5": val4
                  ....}
}
'''


def create_tf_idf(path, type_of_file):
    collection_files = get_list_of_files(path)
    n = len(collection_files)
    doc_length = {}
    term_freq = {}
    term_count = {}
    idf = {}
    tf_idf = {}
    for i in range(0, n):
        tree = elementTree.parse(os.path.join(path, collection_files[i]))
        root = tree.getroot()
        doc_id = root[0].text
        text = get_total_text(root, type_of_file)
        text = re.sub('[^a-zA-Z0-9]+', ' ', text)
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english') and len(word) > 2]
        total = len(words)
        tf_idf[doc_id] = {}
        for j in range(0, len(words)):
            word = words[j]
            if word in term_count:
                if doc_id in term_count[word]:
                    term_count[word][doc_id] += 1
                    term_freq[word][doc_id] = term_count[word][doc_id] / total
                    idf[word] = math.log(n/len(term_count[word]))
                else:
                    term_count[word][doc_id] = 1
                    term_freq[word][doc_id] = 1 / total
                    idf[word] = math.log(n / len(term_count[word]))
            else:
                term_count[word] = {}
                term_freq[word] = {}
                term_count[word][doc_id] = 1
                term_freq[word][doc_id] = 1 / total
                idf[word] = math.log(n / len(term_count[word]))
            val = 0
            if word in term_freq:
                val = term_freq[word][doc_id] * idf[word]
            tf_idf[doc_id][word] = val
        doc_length.update({doc_id: total})
    return doc_length, term_count, tf_idf


'''
This function calculates the matching for a particular query with list of collection documents for Vector Space Model
'''


def calculate_matching(collections, query):
    matches = {}
    d1 = 0
    d2 = 0
    for document, value in collections.items():
        numerator = 0
        for k1, v2 in collections[document].items():
            d1 += v2 * v2
        for term, val in query.items():
            if term in collections[document]:
                numerator += collections[document][term] * query[term]
            d2 += query[term] * query[term]

        if d1 != 0 and d2 != 0:
            match_value = numerator / (math.sqrt(d1) * math.sqrt(d2))
            matches.update({document: match_value})
            d2 = 0
            d1 = 0
    return dict(sorted(matches.items(), key=lambda item: item[1], reverse=True))


'''
This function is called to get the output. It takes model_name as an input parameter and generates the document
with top 1000 matches for a particular query
'''


def get_output(model_name):
    with open('output_' + model_name + '_experiment_' + str(experiment) + '.txt', 'w') as f:
        with redirect_stdout(f):
            for k, v in index_of_queries.items():
                if model_name == "vsm":
                    matches = calculate_matching(index_of_collections, index_of_queries[k])
                elif model_name == "bm25":
                    matches = calculate_bm_25(index_of_collections, index_of_queries[k], 2, 0.75)
                elif model_name == "query_likelihood":
                    matches = calculate_language_model_score(index_of_collections, index_of_queries[k])
                with redirect_stdout(f):
                    rank = 1
                    for doc_id, match_val in matches.items():
                        print(str(k) + " 0 " + doc_id + " " + str(rank) + " " + str(match_val) + " " + str(experiment))
                        rank += 1
                        if rank == 1000:
                            break


'''
This function is used for BM25, it takes input as index of collections, query, k, b
Dictionary collection_term_count which is calculated from create_tf_idf function 
is used to perform calculations of idf and BM25 score for a particular query.
Returns a dictionary of documents which is sorted in descending order according to score
{
    "doc_id_1" : score with query,
    "doc_id_2" : score with query,
    ...
}
'''


def calculate_bm_25(index, query, k, b):
    bm25_score = {}
    n = len(collection_length)
    for doc_id, terms in index.items():
        total = 0
        for term, count in query.items():
            if term in collection_term_count and doc_id in collection_term_count[term]:
                idf = math.log((n - len(collection_term_count[term])) + 0.5 / (len(collection_term_count[term]) + 0.5))
                val = (collection_term_count[term][doc_id] * (k + 1)) / (collection_term_count[term][doc_id] +
                                         k * (1 - b + (b * (collection_length[doc_id] / average_length))))
                total += idf * val
        if total > 0:
            bm25_score.update({doc_id: total})
    return dict(sorted(bm25_score.items(), key=lambda item: item[1], reverse=True))


'''
This function is used for Query Likelihood Model, it takes input as index of collections, query
Dictionary collection_term_count which is calculated from create_tf_idf function 
is used to calculated the probability of term in a given a document.
Returns a dictionary of documents which is sorted in descending order according to score
{
    "doc_id_1" : score with query,
    "doc_id_2" : score with query,
    ...
}
'''


def calculate_language_model_score(index, query):
    language_model_score = {}
    for doc_id, terms in index.items():
        prob = 1
        for term, count in query.items():
            if term in collection_term_count and doc_id in collection_term_count[term]:
                prob = prob * (collection_term_count[term][doc_id]/collection_length[doc_id])
            else:
                prob = 0
                break
        language_model_score.update({doc_id: prob})
    return dict(sorted(language_model_score.items(), key=lambda item: item[1], reverse=True))


query_path = "topics"
dir_path = "COLLECTION"
collection_length, collection_term_count, index_of_collections = create_tf_idf(dir_path, "collection")
query_length, query_term_count, index_of_queries = create_tf_idf(query_path, "queries")
average_length = sum(collection_length.values()) / len(collection_length) # average length is used in BM25
get_output("vsm")
get_output("bm25")
get_output("query_likelihood")
end = time.time()
total_time_taken = end - start
print("Total Time Taken = " + str(timedelta(seconds=total_time_taken)))
