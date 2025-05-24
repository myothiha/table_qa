from libs.prompt.prompt_manager import PromptTemplateManager
from libs.util import extract_function4, postprocess4
from libs.metadata_manager import TableMetadataManager

class AICodeRepair:

    def __init__(self, llm):
        self.llm = llm
        
        self.templateManager = PromptTemplateManager(prompt_dir="prompt_templates")

    def __call__(self, code_str, error_message, df, question, prompt_template = "error_fix"):
        self.metaDataManager = TableMetadataManager(df)
        metadata = self.metaDataManager.format_metadata_prompt()
        extracted_code = extract_function4(code_str).replace("answer(", "error_function(")
        prompt = self.templateManager.format(prompt_template, metadata=metadata, question=question, code=extracted_code, error=error_message, return_type=type)

        response = self.llm(prompt, max_new_tokens=400)
        result = postprocess4(response, df)
        
        return result, response
    