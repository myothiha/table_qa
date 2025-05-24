import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd
from libs.datasets.test_dataset import TestDataset
from libs.util import extract_function4, postprocess4, run_code_simple, run_code_with_timeout

result_df = pd.read_csv("../result/LLAMA3_70B/testresult_zero_shot_with_metadata_with_table_cleaner.csv")
dataset = TestDataset(path="../data/test/", csv_filename="new_test_qa.csv")


for i, row in result_df.iterrows():
    # print(row['question'])
    if "Is the best possible review for room ratings found in more than fifteen reviews" in row['question']:
        print(row['question'])

    llm_response = row['llm_responses']
    df = dataset.load_table(row['dataset'])


    # if "CODE_ERROR" in str(result):
    if "Is the best possible review for room ratings found in more than fifteen reviews" in row['question']:
        print(llm_response)
        extracted_code = extract_function4(llm_response)

        result = run_code_simple(extracted_code, df)

        with open("test.log.txt", "a", encoding="utf-8") as f:
            f.write("====================\n")
            f.write("Start LLM Response\n")
            f.write("====================\n")
            f.write(str(llm_response) + "\n")
            f.write("====================\n")
            f.write("End LLM Response\n")
            f.write("====================\n\n")

            f.write("====================\n")
            f.write("Start Extracted Code\n")
            f.write("====================\n")
            f.write(str(extracted_code) + "\n")
            f.write("====================\n")
            f.write("End Extracted Code\n")
            f.write("====================\n\n")

            f.write("====================\n")
            f.write("Start Simple Code Result\n")
            f.write("====================\n")
            f.write(str(result) + "\n")
            f.write("====================\n")
            f.write("End Simple Code Result\n")
            f.write("====================\n\n")