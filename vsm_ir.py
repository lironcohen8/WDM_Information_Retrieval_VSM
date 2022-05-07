from contextlib import AbstractAsyncContextManager
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def create_index_dict(dir_path):
    records_dict = {}
    files = Path(dir_path).glob('*.xml')
    for file in files:
        doc = ET.parse(file)
        root = doc.getroot()
        records = root.findall("RECORD")
        for record in records:
            record_num = record.find("RECORDNUM").text.strip()
            records_dict[record_num] = {}
            title_words = record.find("TITLE").text.strip().split()
            abstract_words = record.find("ABSTRACT")
            abstract_words = abstract_words.text.strip().split() if abstract_words != None else []
            extract_words = record.find("EXTRACT")
            extract_words = extract_words.text.strip().split() if extract_words != None else []          
            record_words = title_words + abstract_words + extract_words
            for word in record_words:
                if (word not in records_dict[record_num].keys()):
                    records_dict[record_num][word] = 0
                records_dict[record_num][word] += 1
    pass
            
            


if __name__ == '__main__':
    if sys.argv[1] == "create_index":
        create_index_dict(sys.argv[2])
    elif sys.argv[1] == "query":
        #ask_question(sys.argv[2])
        pass