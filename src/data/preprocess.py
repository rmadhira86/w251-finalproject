# 
# Initial PreProcessor to crop images to size for training
# Input required:
#    Location of Zip file that contains a coco formatted file generated typically from roboflow
#    The zip file should contain _annotations file and all images.


# Happy path, simple runs:
# 1. Download a zip file from roboflow into data/raw as a zip file
# 2. Run Either of the below. Remove to --verbose or -v to remove verbose prints
#    a. python3 preprocess.py --coco-file <filename in raw folder> --verbose
#    b. python3 preprocess.py -f <filename in raw folder> -v
#    c. If running interatively, run all cells till run function, then run
#          run(coco_file='<filename in raw folder>', verbose=True)

# Most other options provide options to override default locations of files and some run options
# Use below command to see the help
#    python3 preprocess.py --help t


# Using #%% python magic allows Visual blocks of code to be run in Notebook style in VSS
# Without having to create a Jupyter Notebook. In other IDEs, this may be ignored as simply as comment
#%%
import os
import sys
import shutil
import argparse
import json
from PIL import Image #PIL is installed with pip install Pillow
from pathlib import Path
from tqdm import tqdm 

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
    parser = argparse.ArgumentParser('Data Preprocessor')
    parser.add_argument('-f','--coco-file',type=str, help='Zipfile containing annotations and images.')
    parser.add_argument('--coco-dir',type=str, default=ROOT / 'data/raw', help='Path to --coco-file. Ignored if path specified in --coco-file')
    parser.add_argument('--inter-dir', type=str, default=ROOT / 'data/interim', help='Path to store interim files' )
    parser.add_argument('--out-dir', type=str, default=ROOT / 'data/processed', help='Path to store interim files' )
    parser.add_argument('--clean-run', action='store_true', help='Clear previous interim and output directories and generate fresh clean output')
    parser.add_argument('-p','--project', type=str, default='cdetect', help='Name of the project.' )
    parser.add_argument('-v','--verbose', action='store_true')

    args = parser.parse_known_args()[0] if known else parser.parse_args() #This line is useful when debugging in VSCode, since VSCode sends additional parameters
    return args

#%%
def trim_images(src, dst, clear_dst=False):
    if not os.path.exists(dst):
        os.mkdir(dst)
    elif clear_dst:
        os.remove(Path(dst)/ '*')

    with open(str(Path(src) / '_annotations.coco.json')) as f:
        annots = json.load(f)
    images = annots['images']
    bboxes = {}
    for a in annots['annotations']:
        file_name = [i['file_name'] for i in images if i['id'] == a['image_id']][0]
        bboxes[file_name] = a['bbox']
    i = 0    
    for f, b in tqdm(bboxes.items(), desc="Trimming Images"):
        file_name = Path(src) / f
        if os.path.isfile(file_name):
            im = Image.open(file_name)
            imcrop = im.crop((b[0],b[1],b[0]+b[2],b[1]+b[3]))
            imcrop.save(Path(dst) / f)
            # Show the first 5 images. Later we can move this as a parameter
            if i < 5:
                im.show()
                imcrop.show()
            i += 1

    # verboseprint(bboxes)
#%%
def setprint(verbose=False):
    global verboseprint 
    verboseprint = print if verbose else lambda *a, **k: None

#%%
def main(args):
    setprint(args.verbose)

    #PDL (Program Design Language):
    #   Unzip the file into interim folder
    #   Since the train folder from roboflow is not just train, rename to input
    #   Trim them up based on bounding boxes specified in the annotations


    verboseprint(f"Running with args: {args}")
    inzip = Path(args.coco_dir / args.coco_file) if os.path.basename(args.coco_file) == args.coco_file else args.coco_file
    interdir = Path(args.inter_dir) / args.project


    if args.clean_run and os.path.exists(interdir):
        verboseprint(f"Emptying directory: {interdir}")
        shutil.rmtree(interdir)
    elif os.path.exists(interdir):
        verboseprint(f"Retaining files from previous runs in: {interdir}")


    verboseprint(f"Unzipping file: {inzip} to {interdir}")
    os.system(f"unzip -o {inzip} -d {interdir}") #Overwrite the files
    inputdir = interdir / 'input'
    traindir = interdir / 'train'
    if os.path.exists(inputdir):
        for f in traindir.rglob('*'):
            dst = str(inputdir / os.path.basename(f))
            shutil.move(src=f,dst=dst)
    else:
        shutil.move(src=str(interdir / 'train'), dst=str(interdir / 'input'))

    # Now trim the images
    trim_images(src=interdir / 'input', dst=interdir / 'trim', clear_dst=args.clean_run)

#%%
def run(**kwargs):
    """ Convenience method for running in interactive session
        Parameters can simply be passed as key=value pairs.
    """
    # Get the default values populated for all the arguments

    args = parse_args(True)
    for k, v in kwargs.items():
        setattr(args,k,v)
    main(args)
    return args

#%%
if __name__ == "__main__":
    #Required when running in interactive session. 
    # Should be changed to False before running in batch scripts, otherwise parameters specified with spelling errors may just be ignored
    args = parse_args(True) 
    main(args)


# %%

## References of code snippet motivations:
# https://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
# https://github.com/ultralytics/yolov5/blob/master/train.py