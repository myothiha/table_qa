import pandas as pd

from libs.question_classifier import QuestionClassifier
import os
from libs.llm_loader.llm_wrapper.llm_wrapper import LLMWrapper
from libs.util import extract_function4, postprocess4
from libs.error_fix_module.ai_code_repair import AICodeRepair
from libs.metadata_manager import TableMetadataManager
from dotenv import load_dotenv
from libs.prompt.prompt_manager import PromptTemplateManager
prompt = """

"""

load_dotenv()

model_name = "CODESTRAL_22B"
model_path = os.getenv(model_name)
print("Load model from:", model_path)
# llm = LLMWrapper(model_path)
# codeRepair = AICodeRepair(llm)

llm_response = 'def answer(df):\n    \'\'\'Write a python function to find "Which first tier category name has the most descendants? (direct or otherwise)"\'\'\'\n    # Let\'s think step by step.\n    # Input: df, a pandas dataframe.\n    # Output: category\n    # Process:\n    # 1. Require columnss to answer the question: [\'Tier_1\']\n    # 2. Count the number of descendants for each Tier_1 category.\n    # 3. Find the Tier_1 category with the most descendants.\n    # 4. Return the Tier_1 category name.\n    # Write your code here:\n    descendant_counts = df[\'Tier_1\'].value_counts()\n    max_descendants = max(descendant_counts)\n    most_descendants_tier_1 = [tier_1 for tier_1, count in descendant_counts.items() if count == max_descendants]\n    return most_descendants_tier_1[0]\nans = answer(df)'

question = "Which first tier category name has the most descendants?"
ds_id = "069_Taxonomy"
sub_datapath = os.path.join("data/test", ds_id, "all.parquet")
df = pd.read_parquet(sub_datapath)
llm_answer = postprocess4(llm_response, df)
print(llm_answer)

# codeRepair = AICodeRepair(llm)

# attempt = 0
# while "CODE_ERROR" in llm_answer and attempt < 3:
#     llm_answer, llm_response = codeRepair(llm_response, llm_answer, df, question)
#     attempt += 1
#     print(llm_answer)
#     print(llm_response)

# print(llm_answer)