from torch.utils.data import Dataset
import torch
import numpy as np

# Custom Dataset class for PyTorch
class VQADataset(Dataset):
    """
    Custom PyTorch Dataset for VQA (Visual Question Answering).

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset.
    - text_cols (list): List of column names containing text data.
    - image_cols (list): List of column names containing image data.
    - label_col (str): Column name containing labels (unused in __getitem__ directly if labels are passed separately, but init logic suggests labels might be handled here).
    - mlb (sklearn.preprocessing.MultiLabelBinarizer): MultiLabelBinarizer object.
    - train_columns (list): List of columns from the training set.
    - labels (np.ndarray): Optional, if labels are pre-processed and passed.
    """
    def __init__(self, df, text_cols, image_cols, label_col, mlb, train_columns, labels=None):
        self.text_data = df[text_cols].values.astype(np.float32)
        self.image_data = df[image_cols].values.astype(np.float32)
        self.mlb = mlb
        self.train_columns = train_columns
        
        if labels is not None:
            self.labels = labels.values.astype(np.float32)
        elif label_col in df.columns:
                self.labels = self.mlb.transform(df[label_col]).astype(np.float32)
        else:
             self.labels = None

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text_sample = torch.tensor(self.text_data[idx], dtype=torch.float32)
        image_sample = torch.tensor(self.image_data[idx], dtype=torch.float32)
        
        if hasattr(self, 'labels') and self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return {'text': text_sample, 'image': image_sample, 'labels': label}
        else:
            return {'text': text_sample, 'image': image_sample}
