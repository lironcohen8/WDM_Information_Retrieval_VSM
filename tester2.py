import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET


def calculate_scores(queries_path, index_path):
    queries = list()
    combined_cumulative_gain = 0
    combined_recall = 0
    combined_precision = 0
    combined_f_score = 0
    with open(queries_path) as queries_file:
        tree = ET.parse(queries_file)
    root = tree.getroot()
    for query_el in root.findall('QUERY'):
        scores_el = query_el.find('Records').findall('Item')
        scores_dict = dict()
        for score_el in scores_el:
            doc_num = int(score_el.text)
            score_str = score_el.attrib['score']
            scores_dict[doc_num] = calculate_score_from_str(score_str)
        query = {
            'num': query_el.find('QueryNumber').text,
            'text': query_el.find('QueryText').text
        }
        query['text'] = re.sub('\n', ' ', query['text']).strip()
        query['text'] = re.sub(' +', ' ', query['text'])
        queries.append(query)
        # run query
        os.system(f'python vsm_ir.py query bm25 "{index_path}" "{query["text"]}"')
        results = get_results()
        evaluation = evaluate_results(results, scores_dict)
        combined_cumulative_gain += evaluation['cumulative_gain']
        combined_recall += evaluation['recall']
        combined_precision += evaluation['precision']
        combined_f_score += evaluation['F']
        query['evaluation'] = evaluation
    queries.sort(key=lambda query: query['evaluation']['F'], reverse=True)
    avg_cumulative_gain = combined_cumulative_gain / len(queries)
    avg_recall = combined_recall / len(queries)
    avg_precision = combined_precision / len(queries)
    avg_f_score = combined_f_score / len(queries)
    output_dict = {
        'Average Cumulative Gain': avg_cumulative_gain,
        'Average Recall': avg_recall,
        'Average Precision': avg_precision,
        'Average F Score': avg_f_score,
        'Queries': queries
    }
    with open('combined_queries_scores.json', 'w') as outfile:
        json.dump(output_dict, outfile, indent=4)


def calculate_score_from_str(str):
    total_score = 0
    for c in str:
        score = int(c)
        total_score += score
    return total_score


def evaluate_results(results, query_scores):
    ideal_ranking = sorted(query_scores.items(), key=lambda item: item[1], reverse=True)
    cumulative_gain = 0
    ideal_cumulative_gain = 0.000001
    recall_amount = 0.000001
    for i in range(len(results)):
        doc_num = results[i]
        relevance = query_scores.get(doc_num)
        max_relevance = ideal_ranking[i][1] if i < len(ideal_ranking) else 0
        if relevance is not None:
            recall_amount += 1
        else:
            relevance = 0
        if i == 0:
            cumulative_gain += relevance
            ideal_cumulative_gain += max_relevance
        else:
            cumulative_gain += relevance / math.log(i + 1, 2)
            ideal_cumulative_gain += max_relevance / math.log(i + 1, 2)
    recall = recall_amount / (len(query_scores) + 0.000001)
    precision = recall_amount / (len(results) + 0.000001)
    print("precision " + str(precision))
    print("Recall: " + str(recall))
    F = 2 * precision * recall / (precision + recall)
    cumulative_gain /= ideal_cumulative_gain
    return {
        'cumulative_gain': cumulative_gain,
        'recall': recall,
        'precision': precision,
        'F': F
    }


def get_results():
    with open('ranked_query_docs.txt') as ranked_results_file:
        lines = ranked_results_file.readlines()
        doc_nums = [int(x.strip()) for x in lines]
    return doc_nums


if __name__ == '__main__':
    queries_path = sys.argv[1]
    index_path = sys.argv[2]
    calculate_scores(queries_path, index_path)
