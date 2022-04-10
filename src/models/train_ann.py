# %%
import argparse
import os
import sys
import shutil
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import StandardScaler, RobustScaler
import random
import pickle
import time
import datetime
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from tqdm import tqdm
from pathlib import Path

#%%
# Determine the current path and add project folder and src into the syspath 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # Project src directory.
RANDOM_SEED = 200

random.seed(RANDOM_SEED)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    sys.path.append(str(ROOT / 'src'))  # add src to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative from current working directory

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):

    setprint(verbose = args.verbose)

    verboseprint(f"Running with args {args} \on device:{DEVICE}")


    df_cnn = pd.read_csv(args.base_dir / args.project / args.cnn_data)
    df_cnn.rename(columns={'patient_id':'ResponseId'}, inplace=True)

    proj_dir = args.base_dir / args.project 
    X_train,y_train, colnames = get_xy(proj_dir / args.train_data, df_cnn)
    X_val, y_val, _ = get_xy(proj_dir / args.val_data, df_cnn, )
    X_test, y_test, _ = get_xy(proj_dir / args.test_data, df_cnn)

    NUM_FEATURES = X_train.shape[1]
    NUM_CLASSES = y_train.max() +1

    # Scale the values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    verboseprint(f"Columns: {colnames}")
    verboseprint(f"X_train {X_train.shape} y_train:{y_train.shape}, X_val:{X_val.shape} y_val:{y_val.shape}, X_test:{X_test.shape}, y_test:{y_test.shape}")

    train_dataset = CovidDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = CovidDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = CovidDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.val_batch, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.val_batch, shuffle=False)

    print(f"Training batches: {len(train_loader)} Validation batches: {len(val_loader)}")


    model = ANN(num_features=  NUM_FEATURES, num_class=NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    best_acc = 0
    best_epoch = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train(train_loader=train_loader, 
            model = model, criterion=criterion, optimizer=optimizer,
            epoch= epoch)
        acc = validate(val_loader= val_loader, model = model, 
            criterion= criterion, epoch=epoch)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        best_epoch = epoch +1 if is_best else best_epoch

        save_checkpoint(args, 
                epoch, 
                NUM_CLASSES, 
                is_best, best_acc, best_epoch, 
                model, optimizer, 
                sc, colnames)

        epoch_time = datetime.timedelta(seconds = time.time() - epoch_start)
        avg_epoch_time = datetime.timedelta(seconds = (time.time() - start_time)/(epoch + 1))
        last_indicator = '**' if epoch == args.epochs -1 else ''

        print(f"\nEpoch Stats [{(epoch+1):d}/{args.epochs:d}]{last_indicator} \tAccuracy {acc:6.3f} \tBest {is_best} \tTime(s) {epoch_time} \tAvg Time(s) {avg_epoch_time}")

    total_time = datetime.timedelta(seconds = time.time() - start_time)
    print(f"\nCompleted [{(epoch+1):d}]  \tTime {total_time} \tBest Accuracy {best_acc:6.3f} \tBest Epoch {best_epoch:d}")


#%%
def train(train_loader, model, criterion, optimizer, epoch):
    # Average Meter calculates current and average values for each of the meters
    batch_time = AverageMeter('Time', ':6.3f') # Time in seconds it took to perform data load, forward and backward pass
    losses = AverageMeter('Loss',':.3e')
    accuracies = AverageMeter('Acc',':6.3f')
    # Create a Progress object that would automatically print all the meters for a given set of epochs
    num_batches = len(train_loader)
    progress = ProgressMeter(
        num_batches = num_batches, 
        meters = [accuracies, losses, batch_time],
        prefix = "Epoch [{}] Train ".format(epoch+1))

    # Set the model to train mode
    model.train() 

    end = time.time() #We reset this at end of each loop. Hence end instead of start 😊
    for i, (X, targets) in enumerate(train_loader):
        # Tensor has 4 dimensions. [batches, channels, 2x2 matrix]
        # 0th dimension of a tensor is batch_size. 
        # If fewer images are left than batch size, it will contain fewer images
        num_X = X.size(dim = 0)

        # Send images and targets to appropriate device
        # The model should have already been sent to this device
        # Note: Unlike model.to the tensor conversion is NOT in-place, hence needs to be reassigned back
        X = X.to(device = DEVICE)
        targets = targets.to(device = DEVICE)

        #Perform a forward pass and calculate loss and accuracies obtained
        outputs = model(X) 
        loss = criterion(outputs, targets) 
        acc = multi_acc(outputs, targets)

        # Since by default pytorch accumulates gradients
        # Zero out the gradients and perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the metrics. Since we want average per image, pass the number of images instead of 1
        losses.update(loss.item(), num_X)
        accuracies.update(acc, num_X)

        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i+1)

#%%
def validate(val_loader, model, criterion, epoch):
    # This function uses much of the logic from train. Please see train for explanation of functions

    # Average Meter calculates current and average values for each of the meters
    batch_time = AverageMeter('Time', ':6.3f') # Time in seconds it took to perform data load, forward and backward pass
    losses = AverageMeter('Loss',':.3e')
    accuracies = AverageMeter('Accuracy',':6.3f')
    # Create a Progress object that would automatically print all the meters for a given set of epochs
    num_batches = len(val_loader)
    progress = ProgressMeter(
        num_batches = num_batches, 
        meters = [accuracies, losses, batch_time ],
        prefix = "Epoch [{}] Val ".format(epoch+1))
    
    model.eval() # This is same as model.train(mode=False), but more readable

    # Perform operations on tensors WITHOUT building dynamic computation graph
    # see https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
    with torch.no_grad():
        end = time.time()
        for i, (X, targets) in enumerate(val_loader):
            num_X = X.size(dim = 0)

            X = X.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, targets) 
            acc = multi_acc(outputs, targets)

            # No backpropagation in validation mode
            losses.update(loss.item(), num_X)
            accuracies.update(acc, num_X)

            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i+1)


    return accuracies.avg

def save_checkpoint(args, epoch, num_classes, is_best, best_acc, best_epoch, model, optimizer, scaler, colnames):
    filename = args.project + '_ann_checkpoint.pth.tar'
    best_file = args.project + '_ann_model_best.pth.tar'
    if args.model_dir:
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        
        filename = os.path.join(args.model_dir, filename)
        best_file = os.path.join(args.model_dir, best_file)

    model_state = {
        'epoch': epoch+1,
        'num_classes': num_classes,
        'state_dict': model.state_dict(),
        'best_acc' : best_acc,
        'best_epoch' : best_epoch,
        'optimizer' : optimizer.state_dict(),
        'scaler' : pickle.dumps(scaler),
        'colnames': colnames
    }
    torch.save(model_state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)
        print(f"Saved best model to {best_file}")
    else:
        print(f"Saved checkpoint to {filename}")    

#%%
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

#%%
#%%
class CovidDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]        

#%%
class ANN(nn.Module):
    def __init__(self,num_features, num_class, hidden_multiplier=1):
        super(ANN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_features*hidden_multiplier, bias = True), 
            nn.ReLU(),
            nn.Linear(num_features*hidden_multiplier, num_features*hidden_multiplier, bias = True),
            nn.ReLU(),
            nn.Linear(num_features*hidden_multiplier, num_class, bias = True)
        )

    def forward(self, x):
        return self.layers(x)

#%%
def parse_args(known=False):
    parser = argparse.ArgumentParser('ANN Model')
    parser.add_argument('--project', type=str, default='cdetect', help='Name of the project.' )
    parser.add_argument('--train-data', type=str, default='train/data_train.csv', help='File containing Train meta data' )
    parser.add_argument('--val-data', type=str, default='val/data_val.csv', help='File containing Validation meta data' )
    parser.add_argument('--test-data', type=str, default='test/data_test.csv', help='File containing Test meta data' )
    parser.add_argument('--cnn-data', type=str, default='cnn_output.csv', help='File containing output from CNN training run' )
    parser.add_argument('--base-dir',type=str, default=ROOT / 'data/processed', help='Path under which source data dir resides')
    parser.add_argument('--model-dir',type=str, default=ROOT / 'models', help='Path to save models to')

    parser.add_argument('--epochs',
            type=int, metavar='N',  default=5, 
            help='total epochs to run %(default)d')    
    
    parser.add_argument('--lr', '--learning-rate', 
            type=float,  default=1e-2, 
            metavar='LR', 
            help='initial learning rate (default: %(default)f)', dest='lr')
    
    parser.add_argument('--tb', '--train-batch-size', '--train-batch', 
            type=int,
            metavar='TB',  default=2, 
            help='train batch (default: %(default)d)',
            dest='train_batch')

    parser.add_argument('--vb', '--val-batch', '--val-batch-size', 
            type=int,
            metavar='VB',  default=1, 
            help='validation batch (default: %(default)d)',
            dest='val_batch')

    parser.add_argument('-v','--verbose', action='store_true')

    args = parser.parse_known_args()[0] if known else parser.parse_args() #This line is useful when debugging in VSCode, since VSCode sends additional parameters
    return args

#%%
def setprint(verbose=False):
    """ Print to console if verbose is True, else do nothing. 
    """
    global verboseprint 
    verboseprint = print if verbose else lambda *a, **k: None

#%%
# Load Data and Transform
def get_xy(f_name, df_cnn, cough=True, race=False, colnames=None):
    # Loads files and converts cough, race to 1-Hot encoding
    # Combines data with result from CNN dataset
    # If colnames is provided (as a List-Like parameter):
    #    ensures that X has columns initialized to 0, for columns that do not have data
    # e.g. training data may have 6 races. 
    # However due to limited data, validation and test data may not have same number of races
    # passing the colnames from train, ensures that val and test data also have the missing races
    # in the same order
    # If colnames are not provided, colnames are obtained from X and returned
    # e.g usage
    # X_train, y_train, colnames = get_xy(f_name=<train_file_path>, df_cnn=<data_frame from cnn result>)
    # X_val,y_val, _ = get_xy(fname=<val_file_path>, , df_cnn=<data_frame from cnn result>, colnames=colnames)
    df = pd.read_csv(f_name)
    df = df.replace({'d_dx':{"bacterial":0,"covid":1,"normal":2}})

    df_race = df['d_race'].str.get_dummies().add_prefix("d_race_")
    df_race.columns = [c.lower().replace(" ","") for c in df_race.columns]
    df_cough = df['v_cough'].str.get_dummies().add_prefix("v_cough_")
    df_cough.columns = [c.lower().replace(" ","") for c in df_cough.columns]

    dfs = [df, df_race,df_cough]
    df_res = reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True),dfs)

    df_res = df_res.merge(df_cnn, on="ResponseId")

    y_col = 'd_dx'
    cnn_cols = [c for c in df_cnn.columns if c.endswith('_score') ]
    race_cols = [c for c in df_race.columns]
    cough_cols = [c for c in df_cough.columns]
    x_cols = ['v_temperature']
    if cough:
         x_cols += cough_cols 
    x_cols = x_cols + ['s_antipyretic','s_odynophagia','s_dysphagia',
                'd_age','d_gender','d_vacc_status'] + cnn_cols 
    if race:
        x_cols += race_cols 

    X = df_res[x_cols]
    y = df_res[y_col]
    if colnames:
        # If colnames is passed, the caller desires all these column names to be present
        # we may have different column lengths between train, val and test
        # Ensure all columns are present in each dataset
        miss_cols = [c for c in colnames if c not in X.columns]
        if len(miss_cols) > 0:
            X[miss_cols] = 0
        X = X[colnames]
    else:
        colnames = [c for c in X.columns] 
    X = np.array(X)
    y = np.array(y)      
    return X, y, colnames  
#%%
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.num_batches = num_batches
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(
            batch,'**' if batch == self.num_batches else '')]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']{:s}'

#%%
def set_args(**kwargs):
    """ Convenience method for running in interactive session
        Parameters can simply be passed as key=value pairs.
        Unlike run, this would just returnd the args to test individiual functions
    """
    # Get the default values populated for all the arguments

    args = parse_args(True)
    for k, v in kwargs.items():
        setattr(args,k,v)
    return args

#%%
def run(**kwargs):
    """ Convenience method for running in interactive session
        Parameters can simply be passed as key=value pairs.
    """
    # Get the default values populated for all the arguments

    args = set_args(**kwargs)
    main(args)
    return args

#%%
if __name__ == "__main__":
    #Required when running in interactive session. 
    # Should be changed to False before running in batch scripts, otherwise parameters specified with spelling errors may just be ignored
    args = parse_args() 
    main(args)