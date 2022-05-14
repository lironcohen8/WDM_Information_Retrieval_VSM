import xml.etree.ElementTree as ET
from vsm_ir import ask_question
import numpy as np
import matplotlib.pyplot as plt


def test(path):
    tree = ET.parse(path)
    root = tree.getroot()

    queries = root.findall("./QUERY")
    out_file_name = "ranked_query_docs.txt"

    for query in queries:
        query_number = query.findall("./QueryNumber")[0].text.strip()
        query_text = query.findall("./QueryText")[0].text
        num_results_expected = int(query.findall("./Results")[0].text.strip())

        items = query.findall("./Records/Item")
        relevant_documents = []

        for item in items:
            score = np.mean(np.array([int(char) for char in item.attrib['score']]))
            relevant_documents.append((item.text, score))

        relevant_documents.sort(key=lambda tup: tup[1])

        relevant_documents = [tup[0] for tup in relevant_documents if tup[1] > 0]

        # ask_question("vsm_inverted_index.json",
        #              query_text,
        #              out_file_name)

        with open(out_file_name, "r") as our_output:
            our_documents = [string.strip() for string in our_output.readlines()]

        recalls = np.array([i * 10**-1 for i in range(0, 11)])
        precisions = []
        j = 0  # index of the recall point

        num_relevant_retrieved = 0
        curr_max_precision = 0
        for k, doc_id in enumerate(our_documents):
            if doc_id in relevant_documents:
                num_relevant_retrieved += 1
                curr_precision = num_relevant_retrieved / (k + 1)
                curr_recall = num_relevant_retrieved / num_results_expected
                if curr_recall > recalls[j + 1]:
                    precisions.append(curr_max_precision)
                    # if len(precisions) == 8:
                    #     print("here")
                    curr_max_precision = 0
                    j += 1
                if curr_recall == 1:
                    precisions.append(curr_max_precision)
                    curr_max_precision = 0
            else:
                curr_precision = num_relevant_retrieved / (k + 1)
            curr_max_precision = max(curr_precision, curr_max_precision)

        precisions = np.array(precisions)
        print([doc for doc in relevant_documents if doc not in our_documents])
        print(f"precisions = {precisions}")
        print(f"recalls = {recalls[:len(precisions)]}")
        plt.plot(recalls[:len(precisions)], precisions)
        plt.title(f"Query #{query_number}")
        plt.show()
        f_scores = [2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]) for i in range(0, len(recalls))]
        print(f"f_scores = {f_scores}")

        # assert our_documents == relevant_documents and len(our_documents) == num_results_expected, f"query_number = {query_number}"


if __name__ == "__main__":
    test("./cfc-xml_corrected/cfquery.xml")