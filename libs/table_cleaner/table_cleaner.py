import re

class TableCleaner:

    def __init__(self, dataframe):
        self.df = dataframe

    def clean_columns(df):
        columns = df.columns
        
        # Regex pattern to extract everything before the first special character (<, [, etc.)
        pattern = r"^[^<\[]+"
        special_characters = r'[^A-Za-z0-9\s]'

        # Extract column names before the first special character
        cleaned_columns = [re.match(pattern, col).group(0) for col in columns]
        # Remove all special characters except spaces
        cleaned_columns = [re.sub(special_characters, '', col) for col in cleaned_columns]
        cleaned_columns = [col.replace(' ','_') for col in columns]

        df.columns = cleaned_columns
        return df