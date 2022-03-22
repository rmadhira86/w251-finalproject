# 
# Prepare dataset into train, val. Optionally augment data
# Input required:
#    Location of CSV file containing survey responses
#    location of images folder (typically trimmed images)
#    Destination


# Using #%% python magic allows Visual blocks of code to be run in Notebook style in VSS
# Without having to create a Jupyter Notebook. In other IDEs, this may be ignored as simply as comment
#%%
import os
import sys
import shutil
import argparse
import json
import pandas as pd 
from PIL import Image #PIL is installed with pip install Pillow
from pathlib import Path
from tqdm import tqdm 
from sklearn.model_selection import train_test_split

#%%
# Determine the current path and add project folder and src into the syspath 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # Project src directory.

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    sys.path.append(str(ROOT / 'src'))  # add src to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative from current working directory

#%%
def parse_args(known=False):
    parser = argparse.ArgumentParser('Train Val Test Split')
    parser.add_argument('-f','--survey-file',type=str, help='Survey file containing metadata')
    parser.add_argument('-p','--project', type=str, default='cdetect', help='Name of the project.' )
    parser.add_argument('--survey-dir',type=str, default=ROOT / 'data/raw', help='Path to source survey-file from. Ignored if survey-file has path')
    parser.add_argument('--source-dir',type=str, default=ROOT / 'data/interim/cdetect/trim', help='Path to source images from')
    parser.add_argument('--out-dir', type=str, default=ROOT / 'data/processed', help='Path to store processed files' )
    parser.add_argument('--split_size', type=float, nargs=3, default=[0.8,0.1,0.1], help='Proportion of train_val_test to split data into. All three values must be specfied. default:[0.8,0.1,0.1]')
    parser.add_argument('--stratify', action='store_true', help='Whether Train Test splits should be stratified by label default: False')
    parser.add_argument('--clean-run', action='store_true', help='Clear previous output directories and generate fresh clean output')
    parser.add_argument('-v','--verbose', action='store_true')

    args = parser.parse_known_args()[0] if known else parser.parse_args() #This line is useful when debugging in VSCode, since VSCode sends additional parameters
    return args

#%%
def setprint(verbose=False):
    """ Print to console if verbose is True, else do nothing. 
    """
    global verboseprint 
    verboseprint = print if verbose else lambda *a, **k: None

def print_dict(d, indent=3, f=None, display=True):
    d_dumps = json.dumps(d, indent=indent)
    if f:
        with open(Path(f),'w') as fp:
            fp.write(d_dumps)
    if display:
        verboseprint(d_dumps)
#%%
def move_images(df, label_col,key_col, source_dir, out_dir):
    if os.path.exists(out_dir):
        verboseprint(f"Emptying directory: {out_dir}")
        shutil.rmtree(out_dir)
    labels = df[label_col].unique()
    log = {}
    log['keys'] = 0
    log['images'] = 0
    log['ids'] = 0
    for label in labels: 
        verboseprint(f"Processing Label: {label}")
        log[label]= {}

        if type(label) == str: #Ignore NaNs if present
            path_out = Path(out_dir) / label
            verboseprint(f"Creating folder: {path_out}")
            os.makedirs(path_out,exist_ok=True)

            keys = df.loc[df[label_col] == label,key_col].to_list()
            log[label]['keys'] = len(keys)
            log['keys'] += len(keys)
            log[label]['images'] = 0
            log[label]['ids'] = {}
            for key in keys:
                log[label]['ids'][key] = 0
                log['ids'] += 1
                for f in source_dir.rglob(key + '_*'):
                    shutil.copy(src=f,dst=str(path_out))
                    log[label]['images'] += 1
                    log[label]['ids'][key] += 1
                    log['images'] += 1
    return log

#%%
def process_dataset(args,outdir):
    """ Unzip all the input coco files and trim the files. 
    """
    LABEL_COL = 'd_dx'
    KEY_COL = 'ResponseId'

    traindir = outdir / 'train'
    valdir = outdir / 'val'
    testdir = outdir / 'test'

    infile = Path(args.survey_dir / args.survey_file) if os.path.basename(args.survey_file) == args.survey_file else args.survey_file

    verboseprint(f"Reading file: {infile}")
    verboseprint(f"Infile {infile}")
    
    df = pd.read_csv(infile, header=0, skiprows=[1,2])
    verboseprint(df.columns)

    df.to_csv(outdir/'data_original.csv', index=False)

    # EDA Should be done here

    # Basic Pre-processing
    df_na = df[df[LABEL_COL].isna()]
    df = df.dropna(subset=[LABEL_COL])
    df = df[df[LABEL_COL] != 'Other'] # Drop the Other Labels
    df[LABEL_COL] = df[LABEL_COL].str.lower()
    df.loc[(df[LABEL_COL] == 'viral'),LABEL_COL] = 'covid' #Viral is used to classify Phase 1 and 2. For now, combine them

    verboseprint(f'After modifications df now has {df.shape}')

    # We would apply filters, classifications etc here
    # But for now, go ahead and split the dataset.
    train, val, test = args.split_size
    stratify = df[[LABEL_COL]] if args.stratify else None 
    df_train, df_val = train_test_split(df, test_size=val+test, stratify=stratify)
    if test > 0:
        stratify = df_val[[LABEL_COL]] if args.stratify else None 
        df_val, df_test = train_test_split(df_val, test_size=round(test/(val+test),2), stratify=stratify)
    else:
        df_test = pd.DataFrame()
    results = {}
    log_train = move_images(df_train, LABEL_COL, KEY_COL, args.source_dir, traindir)
    log_val = move_images(df_val, LABEL_COL, KEY_COL, args.source_dir, valdir)
    log_test = move_images(df_test, LABEL_COL, KEY_COL, args.source_dir, testdir)
    results['train'] = log_train
    results['val'] = log_val
    results['test'] = log_test
    return results


#%%
def main(args):

    setprint(args.verbose) 

    #PDL (Program Design Language):
    #   Unzip the file into interim folder
    #   Since the train folder from roboflow is not just train, rename to input
    #   Trim them up based on bounding boxes specified in the annotations


    verboseprint(f"Running with args: {args}")
    outdir = Path(args.out_dir) / args.project


    results = process_dataset(args, outdir)
    print_dict(results, f=ROOT / 'reports/train_test_split.json')


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
    args = parse_args(True) 
    main(args)

# Example use for default options
# python3 preprocess.py --coco-file w251_final_project_throat_detector.v1-v1.coco.zip --clean-run --verbose

# %%

## References of code snippet motivations:
# https://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
# https://github.com/ultralytics/yolov5/blob/master/train.py