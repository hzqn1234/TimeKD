import torch
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader

from clm import GenPromptEmb
from tqdm import tqdm

import pandas as pd

class Amex_Dataset:
    def __init__(self,df_series,df_feature,uidxs,df_y=None):
        self.df_series = df_series
        self.df_feature = df_feature
        self.df_y = df_y
        self.uidxs = uidxs

    def __len__(self):
        return (len(self.uidxs))

    def __getitem__(self, index):
        i1,i2,idx = self.uidxs[index]
        series = self.df_series.iloc[i1:i2+1,1:].drop(['year_month'],axis=1).values

        if len(series.shape) == 1:
            series = series.reshape((-1,)+series.shape[-1:])
        # series_ = series.copy()
        # series_[series_!=0] = 1.0 - series_[series_!=0] + 0.001
        # feature = self.df_feature.loc[idx].values[1:]
        # feature_ = feature.copy()
        # feature_[feature_!=0] = 1.0 - feature_[feature_!=0] + 0.001
        if self.df_y is not None:
            label = self.df_y.loc[idx,[label_name]].values
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    # 'FEATURE': np.concatenate([feature,feature_]),
                    'LABEL': label,
                    }
        else:
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    # 'FEATURE': np.concatenate([feature,feature_]),
                    }

    def collate_fn(self, batch):
        """
        Padding to same size.
        """

        batch_size = len(batch)
        batch_series = torch.zeros((batch_size, 13, batch[0]['SERIES'].shape[1]))
        batch_mask = torch.zeros((batch_size, 13))
        # batch_feature = torch.zeros((batch_size, batch[0]['FEATURE'].shape[0]))
        batch_y = torch.zeros((batch_size, 1))

        for i, item in enumerate(batch):
            v = item['SERIES']
            batch_series[i, :v.shape[0], :] = torch.tensor(v).float()
            batch_mask[i,:v.shape[0]] = 1.0
            # v = item['FEATURE'].astype(np.float32)
            # batch_feature[i] = torch.tensor(v).float()
            if self.df_y is not None:
                v = item['LABEL'].astype(np.float32)
                batch_y[i] = torch.tensor(v).float()

        return {'batch_series':batch_series
                ,'batch_mask':batch_mask
                # ,'batch_feature':batch_feature
                ,'batch_y':batch_y}

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--folds", type=int, default=5)

    return parser.parse_args()

def save_train_embeddings(args):
    print(f'save_train_embeddings')

    input_path = '../../000_data/amex/13month_1pct'
    train_series     = pd.read_feather(f'{input_path}/df_nn_series_train.feather')
    train_series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_train.feather').values
    train_y = pd.read_csv(f'{input_path}/train_labels.csv')




    print(df_nn_series_train.head(2))

    return

if __name__ == "__main__":
    args = parse_args()

    t1 = time.time()
    save_train_embeddings(args)
    t2 = time.time()
    print(f"Total time spent on save_train_embeddings: {(t2 - t1)/60:.4f} minutes")


