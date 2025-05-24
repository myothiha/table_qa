from transformers import AutoTokenizer, AutoModelForSequenceClassification

from libs.metadata_manager import TableMetadataManager
from libs.prompt.prompt_manager import PromptTemplateManager
from libs.util import postprocess4
from libs.table_cleaner.table_cleaner import TableCleaner
from libs.error_fix_module.ai_code_repair import AICodeRepair

class QAModel:
    def __init__(self, llm, questionClassifier):
        self.llm = llm
        self.questionClassifier = questionClassifier
        self.manager = PromptTemplateManager(prompt_dir="prompt_templates")
        self.codeRepair = AICodeRepair(llm)

    def __call__(self, dataframe, question, template = "zero_shot_with_metadata", table_cleaner = False, code_repair = False, **generate_kwargs):
        print("QA Module:")
        if table_cleaner:
            dataframe = TableCleaner.clean_columns(dataframe)

        metadataManager = TableMetadataManager(df=dataframe)
        metadata = metadataManager.format_metadata_prompt()

        predicted_type = self.questionClassifier(question)[0]
        prompt = self.manager.format(template, metadata=metadata, question=question, column_list=dataframe.columns, return_type=predicted_type)
                
        llm_response = self.llm(prompt, **generate_kwargs)
        
        llm_answer = postprocess4(llm_response, dataframe)

        attempt = 0
        while "CODE_ERROR" in str(llm_answer) and code_repair and attempt < 1:
            llm_answer, llm_response = self.codeRepair(llm_response, llm_answer, dataframe, question)
            attempt += 1

        print("llm Answer:", llm_answer)

        return llm_answer, llm_response

        