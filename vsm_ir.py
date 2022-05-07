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
            words_cntr = 0
            record_num = record.find("RECORDNUM").text.strip()
            records_dict[record_num] = {}
            record_words = get_words_from_record(record)
            for word in record_words:
                clean_word, validationValue = update_word_dict(word, record_num)
                words_cntr += validationValue
                if validationValue > 0:
                    max_freq = max(max_freq, words_dict[clean_word]["docs"][record_num])
            calc_tf_values(record_num, max_freq)
            records_dict[record_num] = words_cntr
            print(record_num)
    calc_idf_values()
    pass

def get_words_from_record(record):
    title_words = record.find("TITLE").text.strip().split()
    abstract_words = record.find("ABSTRACT")
    abstract_words = abstract_words.text.strip().split() if abstract_words != None else []
    extract_words = record.find("EXTRACT")
    extract_words = extract_words.text.strip().split() if extract_words != None else []          
    record_words = title_words + abstract_words + extract_words
    return record_words           

def update_word_dict(word, record_num):
    validationValue = 0
    # stemming, removing pactuations and coverting to lower case
    clean_word = ps.stem(word.translate(str.maketrans('', '', string.punctuation)).lower())  
    if clean_word not in set(stopwords.words('english')):
        validationValue = 1
        if clean_word not in words_dict.keys():
            words_dict[clean_word] = {"docs" : {}, "idf": 0}
        if record_num not in words_dict[clean_word]["docs"].keys():
            words_dict[clean_word]["docs"][record_num] = 0
        words_dict[clean_word]["docs"][record_num] += 1
    return clean_word, validationValue
        
def calc_tf_values(record_num, max_freq):
    for word in words_dict.keys():
        # normalize value to get tf for every word in every record
        if record_num in words_dict[word]["docs"].keys():
            words_dict[word]["docs"][record_num] = words_dict[word]["docs"][record_num] / max_freq

def calc_idf_values():
    D = len(words_dict.keys())
    for word in words_dict.keys():
        df = len(words_dict[word]["docs"])
        words_dict[word]["idf"] = math.log2(D / df) 

if __name__ == '__main__':
    if sys.argv[1] == "create_index":
        create_index(sys.argv[2])
    elif sys.argv[1] == "query":
        #ask_question(sys.argv[2])
        pass