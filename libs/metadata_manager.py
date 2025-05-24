import json
from pathlib import Path
import pandas as pd
import pandas.api.types as ptypes

class TableMetadataManager:
    def __init__(self, df):
        self.df = df
        self.columns = self.df.columns
        self.metadata = dict()

        self._load_column_metadata()

    def _load_column_metadata(self):
        num_examples = 5
        for column, dtype in self.df.dtypes.items():
            if dtype == "category":
                num_examples = 10
            
            # Sometime the column contains very long string.
            unique_examples = self.df[column].dropna().unique()
            if len(unique_examples) > 0:
                if len(str(self.df[column].dropna().unique()[0])) > 200:
                    num_examples = 2

            self.metadata[column] = {
                "name": column,
                "type": self._normalize_dtype(dtype),
                "values": list(unique_examples[:num_examples])
            }

    def _normalize_dtype(self, dtype):
        if ptypes.is_integer_dtype(dtype):
            return "int"
        elif ptypes.is_float_dtype(dtype):
            return "float"
        elif ptypes.is_bool_dtype(dtype):
            return "bool"
        elif ptypes.is_datetime64_any_dtype(dtype):
            return "datetime"
        elif ptypes.is_object_dtype(dtype) or ptypes.is_string_dtype(dtype):
            return "str"
        else:
            return "category"
    
    def get_metadata(self):
        return self.metadata
    
    def format_metadata_prompt(self):
        lines = [f"# Metadata:", "# Columns and Types:"]
        for col in self.df.columns:
            col_meta = self.metadata[col]
            col_name = col_meta["name"]
            col_type = col_meta["type"]
            values = col_meta["values"]
            
            line = f"# - {col_name}: {col_type} — e.g., {values}"
            lines.append(line)
            
        return "\n".join(lines)
