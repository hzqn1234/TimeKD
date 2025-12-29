import torch
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader

from amex_generate_embedding import GenPromptEmb
from tqdm import tqdm

import pandas as pd
import numpy as np

class Amex_Dataset:
    # def __init__(self,df_series,df_feature,uidxs,df_y=None):
    def __init__(self,df_series,uidxs,df_y=None,label_name = 'target',id_name = 'customer_ID'):
        self.df_series = df_series
        # self.df_feature = df_feature
        self.df_y = df_y
        self.uidxs = uidxs
        self.label_name = label_name
        self.id_name = id_name

    def __len__(self):
        return (len(self.uidxs))

    def __getitem__(self, index):
        i1,i2,idx = self.uidxs[index]
        series = self.df_series.iloc[i1:i2+1,1:].drop(['S_2'],axis=1).values
        time_ref = self.df_series.iloc[i1:i2+1,1:]['S_2']
        # series = self.df_series.iloc[i1:i2+1,1:].drop(['year_month','S_2'],axis=1).values

        if len(series.shape) == 1:
            series = series.reshape((-1,)+series.shape[-1:])
        # series_ = series.copy()
        # series_[series_!=0] = 1.0 - series_[series_!=0] + 0.001
        # feature = self.df_feature.loc[idx].values[1:]
        # feature_ = feature.copy()
        # feature_[feature_!=0] = 1.0 - feature_[feature_!=0] + 0.001
        if self.df_y is not None:
            label = self.df_y.loc[idx,[self.label_name]].values
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    # 'FEATURE': np.concatenate([feature,feature_]),
                    'LABEL': label,
                    'time_ref': time_ref,
                    'idx': idx,
                    }
        else:
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    # 'FEATURE': np.concatenate([feature,feature_]),
                    'time_ref': time_ref,
                    'idx': idx,
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
        batch_time_ref = np.array([sample['time_ref'] for sample in batch])
        batch_idx = np.array([sample['idx'] for sample in batch])

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
                ,'batch_y':batch_y
                ,'batch_time_ref':batch_time_ref
                ,'batch_idx':batch_idx
                }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=223)
    parser.add_argument("--input_len", type=int, default=13)
    parser.add_argument("--output_len", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=6)

    return parser.parse_args()

def save_train_embeddings(args, train_test = 'train'):
    print(f'save_train_embeddings')

    input_path = '../../000_data/amex/13month_10pct'
    series     = pd.read_feather(f'{input_path}/df_nn_series_{train_test}.feather')
    series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_{train_test}.feather').values
    
    if train_test == 'train':
        y = pd.read_csv(f'{input_path}/{train_test}_labels.csv')
        dataset = Amex_Dataset(series,series_idx,y)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True, drop_last=True, collate_fn=dataset.collate_fn,num_workers=args.num_workers)
    elif train_test == 'test':
        dataset = Amex_Dataset(series,series_idx)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=True, drop_last=True, collate_fn=dataset.collate_fn,num_workers=args.num_workers)
    else:
        print(f'train_test: {train_test}')
        exit()

    gen_prompt_emb = GenPromptEmb(
        # data_path=args.data_path,
        model_name=args.model_name,
        num_nodes=args.num_nodes,
        device=args.device,
        input_len=args.input_len,
        output_len=args.output_len,
        d_model=args.d_model,
        l_layer=args.l_layers,
    ).to(args.device)

    emb_path = f"amex_emb/10pct/{train_test}/"
    os.makedirs(emb_path, exist_ok=True)

    # embeddings_list = []

    bar = tqdm(dataloader)
    for data in bar:
        y = data['batch_y'].to(args.device)
        x = data['batch_series'].to(args.device)
        time_ref = data['batch_time_ref']
        idx = data['batch_idx']
        
        # hd_start_date = f'{time_ref[0,0]}'
        # hd_end_date = f'{time_ref[0,-1]}'
        # print(hd_start_date, hd_end_date)

        # print(f'x_shape: {x.shape}')
        # print(f'y_shape: {y.shape}')
        # print(f'time_ref_shape: {time_ref.shape}')
        # exit()

        embeddings = gen_prompt_emb.generate_embeddings(x, y, time_ref)
        # print(f'embeddings_shape: {embeddings.shape}')

        file_path = f"{emb_path}/{idx[0]}.h5"
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('stacked_embeddings', data=embeddings.detach().cpu().numpy())

    return 

if __name__ == "__main__":
    args = parse_args()

    t1 = time.time()
    save_train_embeddings(args, 'train')
    t2 = time.time()
    print(f"Total time spent on save_train_embeddings: {(t2 - t1)/60:.4f} minutes")

    # t3 = time.time()
    # save_train_embeddings(args, 'test')
    # t4 = time.time()
    # print(f"Total time spent on save_test_embeddings: {(t4 - t3)/60:.4f} minutes")
