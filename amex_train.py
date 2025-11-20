import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
from torch.utils.data import DataLoader
# from data_provider.data_loader_emb import Dataset_ETT_minute
from model.TimeKD import Dual
from utils.kd_loss import KDLoss
from utils.metrics import MSE, MAE, metric
import faulthandler
from tqdm import tqdm
import pandas as pd
from utils.tools import StandardScaler
import h5py

faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="Amex", help="data path")
    parser.add_argument("--channel", type=int, default=512, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=13, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=1, help="out_len")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lrate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--feature_w", type=float, default=0.01, help="weight of feature kd loss")
    parser.add_argument("--fcst_w", type=float, default=1, help="weight of forecast loss")
    parser.add_argument("--recon_w", type=float, default=0.5, help="weight of reconstruction loss")
    parser.add_argument("--att_w", type=float, default=0.01, help="weight of attention kd loss")
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument("--epochs", type=int, default=100, help="")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        "--es_patience",
        type=int,
        default=30,
        help="quit if no improvement after this many iterations",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
        help="save path",
    )
    return parser.parse_args()

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

class trainer:
    def __init__(
        self,
        scaler,
        channel,
        num_nodes,
        seq_len,
        pred_len,
        dropout_n,
        d_llm,
        e_layer,
        head,
        lrate,
        wdecay,
        feature_w,
        fcst_w,
        recon_w,
        att_w,
        device,
        epochs
    ):
        self.model = Dual(
            device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len, 
            dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, head=head
        )
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 100), eta_min=1e-8, verbose=True)
        self.MSE = MSE
        self.MAE = MAE
        self.clip = 5
        self.scaler = scaler
        self.device = device

        self.feature_loss = 'smooth_l1'  
        self.fcst_loss = 'smooth_l1'
        self.recon_loss = 'smooth_l1'
        self.att_loss = 'smooth_l1'   
        self.fcst_w = 1
        self.recon_w = 0.5
        self.feature_w = 0.1     
        self.att_w = 0.01
        self.criterion = KDLoss(self.feature_loss, self.fcst_loss, self.recon_loss, self.att_loss,  self.feature_w,  self.fcst_w,  self.recon_w,  self.att_w)

        # print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))
        print(self.model)

    def train(self, x, y, emb):
        self.model.train()
        self.optimizer.zero_grad()
        ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(x, emb)
        loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip) 
        self.optimizer.step() 
        mse = self.MSE(ts_out, y) 
        mae = self.MAE(ts_out, y)
        return loss.item(), mse.item(), mae.item()

    def eval(self, x, y, emb):
        self.model.eval()
        with torch.no_grad():
            ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(x, emb)
            loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
            mse = self.MSE(ts_out, y)
            mae = self.MAE(ts_out, y)
        return loss.item(), mse.item(), mae.item()


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def main_train():
    print(f'Training...')

    args = parse_args()
    input_path = '../../000_data/amex/13month_0.1pct'
    train_series     = pd.read_feather(f'{input_path}/df_nn_series_train.feather')
    train_series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_train.feather').values
    test_series     = pd.read_feather(f'{input_path}/df_nn_series_test.feather')
    test_series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_test.feather').values
    
    train_y = pd.read_csv(f'{input_path}/train_labels.csv')
    train_dataset = Amex_Dataset(train_series,train_series_idx,train_y)
    train_dataloader = DataLoader(train_dataset,batch_size=1,shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn,num_workers=args.num_workers)

    test_dataset = Amex_Dataset(test_series,test_series_idx)
    test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=True, drop_last=True, collate_fn=test_dataset.collate_fn,num_workers=args.num_workers)

    validation_dataloader = train_dataloader ## TODO - split train-validation

    seed_it(args.seed)
    device = torch.device(args.device)
    
    loss = 9999999
    test_log = 999999
    epochs_since_best_mse = 0

    path = os.path.join(args.save, args.data_path, 
                        f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.lrate}_{args.dropout_n}_{args.seed}_{args.att_w}/")
    if not os.path.exists(path):
        os.makedirs(path)
     
    his_loss = []
    val_time = []
    train_time = []
    test_time = []
    print(args)

    engine = trainer(
        scaler=StandardScaler,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        head=args.head,
        lrate=args.lrate,
        wdecay=args.weight_decay,
        feature_w=args.feature_w,
        fcst_w=args.fcst_w,
        recon_w=args.recon_w,
        att_w=args.att_w,
        device=device,
        epochs=args.epochs
        )

    print("Start training...", flush=True)

    for i in range(1, args.epochs + 1):
        t1 = time.time()
        train_loss = []
        train_mse = []
        train_mae = []
        
        for iter, data in enumerate(tqdm(train_dataloader)):
            y = data['batch_y']
            x = data['batch_series']
            idx = data['batch_idx'][0]

            emb_path = f"amex_emb/train/"
            file_path = os.path.join(emb_path, f"{idx}.h5")

            with h5py.File(file_path, 'r') as hf:
                emb_data = hf['stacked_embeddings'][:]
                emb_tensor = torch.from_numpy(emb_data).unsqueeze(0)

            trainx = torch.Tensor(x).to(device).float()
            trainy = torch.Tensor(y).to(device).float()
            emb = torch.Tensor(emb_tensor).to(device).float()

            metrics = engine.train(trainx, trainy, emb)
            train_loss.append(metrics[0])
            train_mse.append(metrics[1])
            train_mae.append(metrics[2])

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # Validation
        val_loss = []
        val_mse = []
        val_mae = []
        s1 = time.time()

        for iter, data in enumerate(validation_dataloader):
            y = data['batch_y']
            x = data['batch_series']
            idx = data['batch_idx'][0]

            emb_path = f"amex_emb/train/"
            file_path = os.path.join(emb_path, f"{idx}.h5")

            with h5py.File(file_path, 'r') as hf:
                emb_data = hf['stacked_embeddings'][:]
                emb_tensor = torch.from_numpy(emb_data).unsqueeze(0)
            
            valx = torch.Tensor(x).to(device).float()
            valy = torch.Tensor(y).to(device).float()
            emb = torch.Tensor(emb_tensor).to(device).float()

            metrics = engine.eval(valx, valy, emb)
            val_loss.append(metrics[0])
            val_mse.append(metrics[1])
            val_mae.append(metrics[2])

        s2 = time.time()
        log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mse = np.mean(train_mse)
        mtrain_mae = np.mean(train_mae)
        mvalid_loss = np.mean(val_loss)
        mvalid_mse = np.mean(val_mse)
        mvalid_mae = np.mean(val_mae)

        his_loss.append(mvalid_loss)
        print("-----------------------")

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train MSE: {:.4f}, Train MAE: {:.4f}"
        print(
            log.format(i, mtrain_loss, mtrain_mse, mtrain_mae),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid MSE: {:.4f}, Valid MAE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_mse, mvalid_mae),
            flush=True,
        )

        if mvalid_loss < loss:
            print("###Update tasks appear###")

            loss = mvalid_loss
            torch.save(engine.model.state_dict(), path + "best_model.pth")
            bestid = i
            epochs_since_best_mse = 0
            print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
            print("epoch: ", i)
        else:
            epochs_since_best_mse += 1
            print(f"No update. epochs_since_best_mse: {epochs_since_best_mse}")

        engine.scheduler.step()

        # if epochs_since_best_mse >= args.es_patience and i >= args.epochs//2: # early stop
        if epochs_since_best_mse >= args.es_patience and i >= min(args.epochs//2, 10): # early stop
            break

    # Output consumption
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))

    # Test
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))
   

# def main_test():
#     engine.model.load_state_dict(torch.load(path + "best_model.pth"), strict=False)
    
#     test_outputs = []
#     test_y = []
    
#     test_start_time = time.time()
#     for iter, data in enumerate(test_dataloader):
#         y = data['batch_y']
#         x = data['batch_series']

#         testx = torch.Tensor(x).to(device).float()
#         testy = torch.Tensor(y).to(device).float()
#         with torch.no_grad():
#             preds = engine.model(testx, None)
#         test_outputs.append(preds[2])
#         test_y.append(testy)


#     test_pre = torch.cat(test_outputs, dim=0)
#     test_real = torch.cat(test_y, dim=0)

#     amse = []
#     amae = []
    
#     for j in range(args.pred_len):
#         pred = test_pre[:, j,].to(device)
#         real = test_real[:, j, ].to(device)
#         errors = metric(pred, real)
#         amse.append(errors[0])
#         amae.append(errors[1])
    

#     test_end_time = time.time()
#     print(f"Test time (total): {test_end_time - test_start_time:.4f} seconds")

#     log = "On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
#     print(log.format(np.mean(amse), np.mean(amae)))
#     print("Average Testing Time: {:.4f} secs".format(np.mean(test_time)))

#     ## output test result to log file
#     test_result_df = pd.DataFrame()
#     test_result_df['lr'] = [args.lrate]
#     test_result_df['test mse'] = [np.mean(amse)]
#     test_result_df['test mae'] = [np.mean(amae)]

#     test_result_df['best epoch'] = [bestid]
#     test_result_df['best valid loss'] = [str(round(his_loss[bestid - 1], 4))]
#     if not os.path.exists('./experiment_log.csv'):
#         test_result_df.to_csv('./experiment_log.csv',index=False)
#     else:
#         test_result_df.to_csv('./experiment_log.csv',index=False,header=None,mode='a') 


if __name__ == "__main__":
    t1 = time.time()
    main_train()
    t2 = time.time()
    print("Total train time spent: {:.4f}".format(t2 - t1))
