import pandas as pd
import os

class TestDataset:
    def __init__(self, path, csv_filename = "test_qa.csv"):
        self.path = path
        self.csv_filename = csv_filename
        self.dataframe = pd.read_csv(os.path.join(self.path, self.csv_filename))
        self.data_list = []
        self._process_rows()

    def _process_rows(self):
        for i, row in self.dataframe.iterrows():
            data = { 
                "question": row['question'],
                "dataset": row['dataset'],
                "predicted_type": row['predicted_type'],
                "type": row['type'],
                "answer": row['answer'],
                "answer_lite": row['answer_lite'],
            }
            self.data_list.append(data)
    
    def get_data_list(self):
        return self.data_list

    def load_table(self, ds_id):
        sub_datapath = os.path.join(self.path, ds_id, "all.parquet")
        df = pd.read_parquet(sub_datapath)
        return df
    
    def load_sample_table(self, ds_id):
        sub_datapath = os.path.join(self.path, ds_id, "sample.parquet")
        df = pd.read_parquet(sub_datapath)
        return df