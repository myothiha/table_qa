import pandas as pd
from dotenv import load_dotenv
import os
from databench_eval import Runner, Evaluator, utils as db_utils

from libs.metadata_manager import TableMetadataManager
from libs.datasets.test_dataset import TestDataset
from libs.llm_loader.llm_wrapper.claude_ai_wrapper import ClaudeAIWrapper
from libs.prompt.prompt_manager import PromptTemplateManager
from libs.question_classifier import QuestionClassifier
from libs.util import postprocess4
from libs.question_answering_module.question_answering_model import QAModel

load_dotenv()

dataset = TestDataset(path="data/test/", csv_filename="new_test_qa.csv")

# Load LLM
model_name = "claude-3-5-sonnet-20240620"
template_name = "few_shots_with_md"
clean_table = True
code_repair = True
llm = ClaudeAIWrapper("claude-3-5-sonnet-20240620")

# Load QA Classifier and Model
questionClassifier = QuestionClassifier(classifier_path="myothiha/fine-tuned-roberta-question-classifier")
questionAnweringModel = QAModel(llm, questionClassifier)

# Output file setup
filename = f"result/{model_name}/testresult_{template_name}"

if clean_table:
    filename += "_tc"

if clean_table:
    filename += "_cr"

csv_filename = f"{filename}.csv"

# Load prompt template manager
manager = PromptTemplateManager(prompt_dir="prompt_templates")

# Write header once
questions_list = []
processed_datasets = []
if os.path.exists(csv_filename):
    current_df = pd.read_csv(csv_filename)
    questions_list = current_df['question'].to_list()
    processed_datasets = current_df['dataset'].to_list()
else:
    pd.DataFrame(columns=["question", "dataset", "predicted_type", "type", "answer", "answer_lite", "llm_answer", "llm_answer_lite", "result", "result_lite", "llm_responses", "llm_responses_lite", ]).to_csv(csv_filename, index=False)

evaluator = Evaluator()

no_of_incorrect_answers = 0
for row in dataset.get_data_list():
    question = row['question']
    dataset_name = row['dataset']
    predicted_type = row['predicted_type']
    type = row['type']
    answer = row['answer']
    answer_lite = row['answer_lite']

    if question in questions_list and dataset_name in processed_datasets:
        continue
    
    df_table = dataset.load_table(dataset_name)
    df_sample_table = dataset.load_sample_table(dataset_name)

    llm_answer, llm_response = questionAnweringModel(df_table, question, template=template_name, table_cleaner=clean_table, code_repair=code_repair, max_new_tokens=500)
    llm_answer_lite, llm_response_lite = questionAnweringModel(df_sample_table, question, template=template_name, table_cleaner=clean_table, code_repair=code_repair, max_new_tokens=500)

    if len(str(llm_answer)) > 1000:
        llm_answer = "answer-is-too-long"

    if len(str(llm_answer_lite)) > 1000:
        llm_answer_lite = "answer-is-too-long"

    is_correct = evaluator.default_compare(str(llm_answer), str(answer), type)

    if not is_correct:
        no_of_incorrect_answers += 1
        print(is_correct, 'Correct Answer:', answer, '. LLM Output.', llm_answer)

    is_sample_answer_correct = evaluator.default_compare(str(llm_answer_lite), str(answer_lite), type)

    # Convert to DataFrame and append to CSV
    row_df = pd.DataFrame([{
        "question": question,
        "dataset": dataset_name,
        "predicted_type": predicted_type,
        "type": type,
        "answer": answer,
        "answer_lite": answer_lite,
        "llm_answer": llm_answer,
        "llm_answer_lite": llm_answer_lite,
        "result": is_correct,
        "result_lite": is_sample_answer_correct,
        "llm_response": llm_response,
        "llm_responses_lite": llm_response_lite,
    }])
    row_df.to_csv(csv_filename, mode='a', header=False, index=False)