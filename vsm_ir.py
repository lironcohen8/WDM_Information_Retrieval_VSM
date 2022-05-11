import json
import os
import sys
import string
import math
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import xml.etree.ElementTree as ET
from pathlib import Path

JSON_FILE_NAME = "vsm_inverted_index.json"
QUERY_RESULT_FILE_NAME = "ranked_query_docs.txt"
BM25_K = 1.5
BM25_B = 0.75
records_dict = {}
words_dict = {}

def create_index(dir_path):
    files = Path(dir_path).glob('*.xml')
    for file in files:
        doc = ET.parse(file)
        root = doc.getroot()
        records = root.findall("RECORD")
        for record in records:
            max_freq = 0
            record_num = record.find("RECORDNUM").text.strip()
            records_dict[record_num] = {"words_cnt" : 0, "words" : {}}
            record_words = get_words_from_record(record)
            for word in record_words:
                # stemming, removing pactuations and coverting to lower case
                clean_word = ps.stem(word.translate(str.maketrans('', '', string.punctuation)).lower())  
                if clean_word not in set(stopwords.words('english')):
                    update_words_dict(clean_word, record_num)               
                    records_dict[record_num]["words_cnt"] += 1
                    max_freq = max(max_freq, words_dict[clean_word]["docs"][record_num])
            calc_tf_values(record_num, max_freq)
            print(record_num)
    calc_idf_values()
    save_index_dict_to_json()

def get_words_from_record(record):
    title_words = record.find("TITLE").text.strip().split()
    abstract_words = record.find("ABSTRACT")
    abstract_words = abstract_words.text.strip().split() if abstract_words != None else []
    extract_words = record.find("EXTRACT")
    extract_words = extract_words.text.strip().split() if extract_words != None else []          
    record_words = title_words + abstract_words + extract_words
    return record_words           

def update_words_dict(clean_word, record_num):
    if clean_word not in words_dict.keys():
        words_dict[clean_word] = {"docs" : {}, "idf": 0}
    if record_num not in words_dict[clean_word]["docs"].keys():
        words_dict[clean_word]["docs"][record_num] = 0
    words_dict[clean_word]["docs"][record_num] += 1

def calc_tf_values(record_num, max_freq):
    for word in words_dict.keys():
        # normalize value to get tf for every word in every record
        if record_num in words_dict[word]["docs"].keys():
            tf = words_dict[word]["docs"][record_num] / max_freq
            words_dict[word]["docs"][record_num] = tf
            records_dict[record_num]["words"][word] = tf

def calc_idf_values():
    D = len(words_dict.keys())
    for word in words_dict.keys():
        df = len(words_dict[word]["docs"])
        words_dict[word]["idf"] = math.log2(D / df)

def calc_weight_values():
    for word in words_dict.keys():
        for record_num in words_dict[word]["docs"].keys():
            idf = words_dict[word]["idf"]
            records_dict[record_num]["words"][word] *= idf

def save_index_dict_to_json():
    index_dicts = {"words_dict" : words_dict, "records_dict" : records_dict}
    with open(JSON_FILE_NAME, 'w') as f:
        json.dump(index_dicts, f, indent=4)

def ask_question(ranking, index_path, query):
    words_dict, records_dict = load_index_dict_from_json(index_path)
    query_dict = parse_query(query)
    if ranking == "tfidf":
        sorted_records = calc_tfidf_grades(query_dict)
    else:
        sorted_records = calc_bm25_grades(query_dict)
    save_query_result_to_txt(sorted_records)

def parse_query(query):
    max_freq = 0
    query_dict = {"words_cnt" : 0, "words" : {}}
    query_words = query.strip().split()
    for word in query_words:
        # stemming, removing pactuations and coverting to lower case
        clean_word = ps.stem(word.translate(str.maketrans('', '', string.punctuation)).lower())  
        if clean_word not in set(stopwords.words('english')):
            if clean_word not in query_dict["words"].keys():
                query_dict["words"][clean_word] = 0
            query_dict["words"][clean_word] += 1
            query_dict["words_cnt"] += 1
            max_freq = max(max_freq, query_dict["words"][clean_word])

    # Calculate idf
    D = query_dict["words_cnt"]
    idf = math.log2(D)

    # Calculate tf
    for word in query_dict["words"].keys():
        query_dict["words"][word] = query_dict["words"][word] * idf / max_freq    
    return query_dict

def load_index_dict_from_json(index_path):
    with open(index_path) as f:
        index_dict = json.load(f)
    return index_dict['words_dict'], index_dict['records_dict']

def calc_tfidf_grades(query_dict):
    nomin = 0
    denom_query = 0
    denom_record = 0
    relevant_records = {}

    for record_num in records_dict.keys():
        common_words = query_dict["words"].keys() & records_dict[record_num]["words"].keys()
        if len(common_words > 0):
            for word in common_words:
                query_weight = query_dict["words"][word]
                record_weight = records_dict[record_num]["words"][word]
                nomin += (query_weight * record_weight)
                denom_query += (query_weight * query_weight)
                denom_record += (record_weight * record_weight)
            cos_similarity = nomin / (math.sqrt(denom_query) * math.sqrt(denom_record))
            relevant_records[record_num] = cos_similarity
    
    sorted_records = sorted(relevant_records.items(), key=lambda x: x[1], reverse=True)
    return sorted_records

def calc_bm25_grades(query_dict):
    D = len(words_dict.keys())
    lst = [record["words_cnt"] for record in records_dict.keys()]
    avgdl = sum(lst) / len(lst)
    N = len(records_dict.keys())
    relevant_records = {}

    for record_num in records_dict.keys():
        bm25_grade = calc_bm25_grade_for_record(D, avgdl, N, query_dict, record_num)
        relevant_records[record_num] = bm25_grade

    sorted_records = sorted(relevant_records.items(), key=lambda x: x[1], reverse=True)
    return sorted_records
        
def calc_bm25_grade_for_record(D, avgdl, N, query_dict, record_num):
    bm25_grade = 0
    common_words = query_dict["words"].keys() & records_dict[record_num]["words"].keys()
    if len(common_words > 0):
        for word in common_words:
            n = len(words_dict[word]["docs"].keys())
            idf = math.log(1 + ((N - n + 0.5) / (n + 0.5)))
            tf = words_dict[word]["docs"][record_num]
            nomin = idf * tf * (BM25_K + 1)
            right_denom = BM25_K * (1 - BM25_B + (BM25_B * D / avgdl))
            bm25_grade += (nomin / tf + right_denom)
    return bm25_grade

def save_query_result_to_txt(sorted_records):
    with open(QUERY_RESULT_FILE_NAME, 'w') as f:
        f.write((os.linesep).join(sorted_records))

if __name__ == '__main__':
    if sys.argv[1] == "create_index":
        create_index(sys.argv[2])
    elif sys.argv[1] == "query":
        ask_question(sys.argv[2], sys.argv[3], sys.argv[4])