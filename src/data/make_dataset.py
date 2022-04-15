# 
# Prepare dataset into train, val. Optionally augment data
# Input required:
#    Location of CSV file containing survey responses
#    location of images folder (typically trimmed images)
#    Destination


# Using #%% python magic allows Visual blocks of code to be run in Notebook style in VSS
# Without having to create a Jupyter Notebook. In other IDEs, this may be ignored as simply as comment
#%%
import albumentations as A
import cv2 
import os
import sys
import shutil
import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
# from PIL import Image #PIL is installed with pip install Pillow
from pathlib import Path
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import random

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

#%%
def parse_args(known=False):
    parser = argparse.ArgumentParser('Train Val Test Split')
    parser.add_argument('-f','--survey-file',type=str, help='Survey file containing metadata')
    parser.add_argument('-p','--project', type=str, default='cdetect', help='Name of the project.' )
    parser.add_argument('--survey-dir',type=str, default=ROOT / 'data/interim/cdetect', help='Path to source survey-file from. Ignored if survey-file has path')
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

def print_obj(d, indent=3, f=None, display=True):
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
    log['dir'] = str(out_dir)
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
    df_train, df_val = train_test_split(df, test_size=val+test, stratify=stratify, random_state=RANDOM_SEED)
    if test > 0:
        stratify = df_val[[LABEL_COL]] if args.stratify else None 
        df_val, df_test = train_test_split(df_val, test_size=round(test/(val+test),2), stratify=stratify)
    else:
        df_test = pd.DataFrame()

    results = {}
    log_train = move_images(df_train, LABEL_COL, KEY_COL, args.source_dir, traindir)
    df_train.to_csv(traindir/'data_train.csv', index=False)
    log_val = move_images(df_val, LABEL_COL, KEY_COL, args.source_dir, valdir)
    df_val.to_csv(valdir/'data_val.csv', index=False)
    log_test = move_images(df_test, LABEL_COL, KEY_COL, args.source_dir, testdir)
    df_test.to_csv(testdir/'data_test.csv', index=False)

    results['train'] = log_train
    results['val'] = log_val
    results['test'] = log_test

    return results

#%%
# Wrapper functions since CV2 gives and expects images in BGR and albumentations works with RGB
def get_image(infile):
    verboseprint(f"Reading image {infile}")
    img = cv2.imread(infile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def write_image(outfile, img, max_size=120):
    if len(outfile) > max_size:
        outdir = os.path.dirname(outfile)
        f_name, f_ext = os.path.splitext(os.path.basename(outfile))
        f_basename, f_ext2 = os.path.splitext(f_name)
        f_newext2 = f_ext2[(len(outfile)-max_size):]
        outfile2 = os.path.join(outdir, f_basename + f_newext2 + f_ext) 
        verboseprint(f"Changed file {outfile} to {outfile2}")
        outfile = outfile2   
    cv2.imwrite(outfile, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return outfile
#%%
def augment_images(indir, outdir_compose, outdir_all=None):
    # Augments images with a set of image augmentations using the albumentations library
    # Writes the original + composite image from augmentation pipeline into the outdir_compose folder. 
    # Writes the original + composite image + images for each augmentation into outdir_all folder. 
   
    verboseprint(f"Processing {indir} into {outdir_compose} and {outdir_all}")
    # Specify the augmentations to use to run an augmentation pipeline and create a composite image
    aug_titles = ['Original','Composed']
    augCompose = A.Compose([
            A.SmallestMaxSize(max_size=350),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
            A.RandomCrop(height=256, width=256),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5)])
    
    # Specify the augmentations to individual process and add.
    # This need not be the same as the augCompose argument above. Also, please ensure that always_apply is True
    # Sometimes you may need to apply processing like resizing images before and / or after applying an augmentation
    #   In those cases specify the aug_prefix and aug_suffix augmentations
    aug_prefix = [A.SmallestMaxSize(max_size=350)]
    aug_titles += ['ShiftScaleRotate','RandomCorp','RGBShift','MultNoise','HueSat','RandBrightCont']        
    aug_list = [
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, always_apply=True),
            A.RandomCrop(height=256, width=256, always_apply=True),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, always_apply=True),
            A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, always_apply=True),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, always_apply=True),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), always_apply=True)]
    aug_suffix = []

    if not os.path.isdir(indir):
        raise ValueError(f"indir directory {indir} does not exist")
    
    if outdir_compose and os.path.exists(outdir_compose):
        verboseprint(f"Emptying directory: {outdir_compose}")
        shutil.rmtree(outdir_compose)
    if outdir_all and os.path.exists(outdir_all):
        verboseprint(f"Emptying directory: {outdir_all}")
        shutil.rmtree(outdir_all)
    image_paths = []
    for path, subdirs, files in tqdm(os.walk(indir)):
        for filename in files:
            in_fname = os.path.join(path,filename)
            f_name, f_ext = os.path.splitext(filename)
            outsubdir = path.replace(str(indir),'')
            if f_ext in ['.jpg','.png','.jpeg']:
                img = get_image(in_fname)
                this_img_paths = [in_fname]

                aug_img = augCompose(image = img)

                if outdir_compose:
                    outdir = str(outdir_compose) + outsubdir
                    os.makedirs(outdir,exist_ok=True)
                    verboseprint(f"Outdir: {outdir}")

                    out_file = os.path.join(str(outdir),f_name +f_ext)
                    out_file = write_image(out_file, img)
                    this_img_paths.append(out_file)

                    out_file = os.path.join(str(outdir),f_name + '_' + aug_titles[1] +f_ext)
                    verboseprint(f"Out_file: {out_file}")
                    out_file = write_image(out_file, aug_img['image'])
                    this_img_paths.append(out_file)
    
                if outdir_all:
                    outdir = str(outdir_all) + outsubdir
                    os.makedirs(outdir,exist_ok=True)
                    # Both original and Composed files must also be copied
                    outsubdir = path.replace(str(indir),'')
                    outdir = str(outdir_all) + outsubdir

                    out_file = os.path.join(outdir,f_name +f_ext)
                    out_file = write_image(out_file, img)
                    this_img_paths.append(out_file)

                    out_file = os.path.join(outdir,f_name + '_' + aug_titles[1] +f_ext)
                    out_file = write_image(out_file, aug_img['image'])
                    this_img_paths.append(out_file)

                    for i, aug in enumerate(aug_list):
                        augC = A.Compose(aug_prefix + [aug] + aug_suffix)
                        augi_img = augC(image= img)
                        out_file = os.path.join(outdir,f_name + '_' + aug_titles[i+2] + f_ext)
                        out_file = write_image(out_file, augi_img['image'])
                        this_img_paths.append(out_file)
                image_paths.append(this_img_paths)
            else:
                if outdir_compose:
                    outdir = str(outdir_compose) + outsubdir
                    os.makedirs(outdir,exist_ok=True)
                    shutil.copy(str(in_fname), os.path.join(outdir,filename))
                if outdir_all:
                    outdir = str(outdir_all) + outsubdir
                    os.makedirs(outdir,exist_ok=True)
                    shutil.copy(str(in_fname), os.path.join(outdir,filename))

    return image_paths

# Ideally augment_images should include this code and operate on parameters.
# Would require future refactoring to minimize code duplication.
def augment_images_alt(indir, outdir_compose, outdir_all=None):
    # Augments images with a set of image augmentations using the albumentations library
    # Writes the original + composite image from augmentation pipeline into the outdir_compose folder. 
    # Writes the original + composite image + images for each augmentation into outdir_all folder. 
   
    verboseprint(f"Processing {indir} into {outdir_compose} and {outdir_all}")
    # Specify the augmentations to use to run an augmentation pipeline and create a composite image
    aug_titles = ['Original','Composed']
    # Change rotate_limit to +/- 45 degrees, remove the random crop and increase probability of Hue and BrightnessContrast to 0.8
    augCompose = A.Compose([
            A.SmallestMaxSize(max_size=350),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.25), contrast_limit=(-0.1, 0.25), p=0.8)])
    
    # Specify the augmentations to individual process and add.
    # This need not be the same as the augCompose argument above. Also, please ensure that always_apply is True
    # Sometimes you may need to apply processing like resizing images before and / or after applying an augmentation
    #   In those cases specify the aug_prefix and aug_suffix augmentations
    aug_prefix = [A.SmallestMaxSize(max_size=350)]
    aug_titles += ['ShiftScaleRotate','RandomCorp','RGBShift','MultNoise','HueSat','RandBrightCont']        
    aug_list = [
            A.SmallestMaxSize(max_size=350),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, always_apply=True),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, always_apply=True),
            A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, always_apply=True),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.25), contrast_limit=(-0.1, 0.25), always_apply=True)]
    aug_suffix = []

    if not os.path.isdir(indir):
        raise ValueError(f"indir directory {indir} does not exist")
    
    if outdir_compose and os.path.exists(outdir_compose):
        verboseprint(f"Emptying directory: {outdir_compose}")
        shutil.rmtree(outdir_compose)
    if outdir_all and os.path.exists(outdir_all):
        verboseprint(f"Emptying directory: {outdir_all}")
        shutil.rmtree(outdir_all)
    image_paths = []
    for path, subdirs, files in tqdm(os.walk(indir)):
        for filename in files:
            in_fname = os.path.join(path,filename)
            f_name, f_ext = os.path.splitext(filename)
            outsubdir = path.replace(str(indir),'')
            if f_ext in ['.jpg','.png','.jpeg']:
                img = get_image(in_fname)
                this_img_paths = [in_fname]

                aug_img = augCompose(image = img)

                if outdir_compose:
                    outdir = str(outdir_compose) + outsubdir
                    os.makedirs(outdir,exist_ok=True)
                    verboseprint(f"Outdir: {outdir}")

                    out_file = os.path.join(str(outdir),f_name +f_ext)
                    out_file = write_image(out_file, img)
                    this_img_paths.append(out_file)

                    out_file = os.path.join(str(outdir),f_name + '_' + aug_titles[1] +f_ext)
                    verboseprint(f"Out_file: {out_file}")
                    out_file = write_image(out_file, aug_img['image'])
                    this_img_paths.append(out_file)
    
                if outdir_all:
                    outdir = str(outdir_all) + outsubdir
                    os.makedirs(outdir,exist_ok=True)
                    # Both original and Composed files must also be copied
                    outsubdir = path.replace(str(indir),'')
                    outdir = str(outdir_all) + outsubdir

                    out_file = os.path.join(outdir,f_name +f_ext)
                    out_file = write_image(out_file, img)
                    this_img_paths.append(out_file)

                    out_file = os.path.join(outdir,f_name + '_' + aug_titles[1] +f_ext)
                    out_file = write_image(out_file, aug_img['image'])
                    this_img_paths.append(out_file)

                    for i, aug in enumerate(aug_list):
                        augC = A.Compose(aug_prefix + [aug] + aug_suffix)
                        augi_img = augC(image= img)
                        out_file = os.path.join(outdir,f_name + '_' + aug_titles[i+2] + f_ext)
                        out_file = write_image(out_file, augi_img['image'])
                        this_img_paths.append(out_file)
                image_paths.append(this_img_paths)
            else:
                if outdir_compose:
                    outdir = str(outdir_compose) + outsubdir
                    os.makedirs(outdir,exist_ok=True)
                    shutil.copy(str(in_fname), os.path.join(outdir,filename))
                if outdir_all:
                    outdir = str(outdir_all) + outsubdir
                    os.makedirs(outdir,exist_ok=True)
                    shutil.copy(str(in_fname), os.path.join(outdir,filename))

    return image_paths

def copy_folders(indir, outdir_compose, outdir_all):

    verboseprint(f"Processing {indir} into {outdir_compose} and {outdir_all}")

    if not os.path.isdir(indir):
        raise ValueError(f"indir directory {indir} does not exist")
    
    if outdir_compose and os.path.exists(outdir_compose):
        verboseprint(f"Emptying directory: {outdir_compose}")
        shutil.rmtree(outdir_compose)
    if outdir_all and os.path.exists(outdir_all):
        verboseprint(f"Emptying directory: {outdir_all}")
        shutil.rmtree(outdir_all)
    if outdir_compose:
        shutil.copytree(str(indir), str(outdir_compose))
    if outdir_all:
        shutil.copytree(str(indir), str(outdir_all))

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
    print_obj(results, f=ROOT / 'reports/train_test_split.json')

    # Validation and Test images are typically not augmented using these techniques
    # Augmentations like converting to Tensor will be done in the main pipeline
    outdir_compose = Path(args.out_dir) / f"{args.project}_compose" 
    outdir_all = Path(args.out_dir) / f"{args.project}_all" 
    aug_files = augment_images(indir = outdir / 'train', outdir_compose = outdir_compose / 'train', outdir_all= outdir_all / 'train')
    print_obj(aug_files, f=ROOT / 'reports/augmented_files.json')

    copy_folders(indir = outdir / 'val', 
                outdir_compose = outdir_compose / 'val', 
                outdir_all = outdir_all / 'val')

    copy_folders(indir = outdir / 'test', 
                outdir_compose = outdir_compose / 'test', 
                outdir_all = outdir_all / 'test')

    # Duplicate code for recreating alternate set of augmentations 
    outdir_compose = Path(args.out_dir) / f"{args.project}_compose_2" 
    outdir_all = Path(args.out_dir) / f"{args.project}_all_2" 
    aug_files = augment_images_alt(indir = outdir / 'train', outdir_compose = outdir_compose / 'train', outdir_all= outdir_all / 'train')
    print_obj(aug_files, f=ROOT / 'reports/augmented_files_alt.json')

    copy_folders(indir = outdir / 'val', 
                outdir_compose = outdir_compose / 'val', 
                outdir_all = outdir_all / 'val')

    copy_folders(indir = outdir / 'test', 
                outdir_compose = outdir_compose / 'test', 
                outdir_all = outdir_all / 'test')


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