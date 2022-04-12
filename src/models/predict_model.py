#%%
import sys, os
from pathlib import Path

# Add the (Y)et(A)nother(E)fficient(D)et(P)y(T)orch folder to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yaedpyt'))
from yaedpyt.backbone import EfficientDetBackbone
from yaedpyt.efficientdet.utils import BBoxTransform, ClipBoxes
from yaedpyt.utils.utils import preprocess, invert_affine, postprocess
from train_ann import ANN

import argparse
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms 

#%%
# Determine the current path and add project folder and src into the syspath 
# Inspiration from YOLOV5 code
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # Project src directory.

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    sys.path.append(str(ROOT / 'src'))  # add src to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative from current working directory

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def range_temp(arg):
    """ Type function for argparse - to restrict temperature range """
    MIN_VAL, MAX_VAL = 92.0, 110.0
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError(f"Temperature must be between {MIN_VAL} and {MAX_VAL}")
    return f

def parse_args(known=False):
    parser = argparse.ArgumentParser('Covid CNN Inference')
    parser.add_argument('-d','--image-dir',type=str, help='Location of Image file', default=ROOT / "data/external")
    parser.add_argument('-g','--no_graphs',action='store_true', help='Hide images from display')
    parser.add_argument('--image_size', type=int, default=512, help='Image Size to use (default: %(default)d x %(default)d)' )
    parser.add_argument('-v','--verbose', action='store_true')

    args = parser.parse_known_args()[0] if known else parser.parse_args() #This line is useful when debugging in VSCode, since VSCode sends additional parameters
    return args

def load_detect_model(device):
    throat_model_file = str(ROOT / "models/cdetect_throat_detection.pth")
    model_det = EfficientDetBackbone(compound_coef=0, num_classes=1,
                                # replace this part with your project's anchor config
                                ratios=[(1.0, 1.0), (1.3, 0.8), (1.9, 0.5)],
                                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    model_det.load_state_dict(torch.load(throat_model_file, map_location=device))

    return model_det    

def load_cnn_model(model_file, device, num_classes=3):

    ARCH = 'vgg16'
    LR = 0.0001 #Do we need this, if loading from saved model?

    # Now start with CNN Detection
    model_cnn = models.__dict__[ARCH](pretrained=False)
    #Change the model's output layer to 3 classes (covid/normal/bacterial)
    model_cnn.classifier[6] = nn.Linear(model_cnn.classifier[6].in_features, num_classes)
#    optimizer = torch.optim.SGD(model_cnn.parameters(), lr=LR)
    
    checkpoint = torch.load(model_file,map_location=device)
    model_cnn.load_state_dict(checkpoint['state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer'])

    return model_cnn

def load_ann_model(device, num_classes=3):
    ann_model_file = str(ROOT / "models/cdetect_ann_model_best.pth.tar")

    checkpoint = torch.load(ann_model_file,map_location=DEVICE)
    colnames = checkpoint['colnames']
    model_ann = ANN(len(colnames), num_classes)
#    optimizer = torch.optim.Adam(model_ann.parameters(), lr=LR)

    model_ann.load_state_dict(checkpoint['state_dict'])

    df_ann = pd.DataFrame(0,index=np.arange(1),columns=colnames)
    #Read back the StandardScaler used during training
    sc = checkpoint['scaler']

#    optimizer.load_state_dict(checkpoint['optimizer'])
    return model_ann, df_ann, sc    

def detect_throat(image_path, image_size, model):
    orig_img, framed_imgs, framed_metas = preprocess(image_path, max_size=image_size)
    img = cv2.cvtColor(orig_img[0], cv2.COLOR_BGR2RGB)
    # We will have only 1 image in framed_imgs and original_imgs
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    # the image as NumPy array has shape [height, width, 3], 
    # when you permute the dimensions you swap the height and width dimension, creating a tensor with shape [batch_size, channels, width, height].
    # copied for reference from https://stackoverflow.com/questions/62482336/classification-with-pretrained-pytorch-vgg16-model-and-its-classes
    x = x.to(torch.float32).permute(0, 3, 1, 2) # batch_size x channels x height x width
    model.requires_grad_(False)
    model.eval()
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold=0.5, iou_threshold=0.5)

    out = invert_affine(framed_metas, out)
    roi = out[0]['rois'][0]
    (x1, y1, x2, y2) = roi.astype(int)
    throat = img[y1:y2, x1:x2]
    return img, throat

def predict_cnn(throat, model, image_size):
    imagenet_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
    imagenet_std_RGB = [0.229, 0.224, 0.225]
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean_RGB, imagenet_std_RGB),
    ])


    img1 = img_transform(throat)
    model.eval()
    pred = model(img1.unsqueeze(0))
    return pred

def getColor(pred):
    if pred == "covid":
        annot_color = (255,0,0)
    elif pred == "bacterial":
        annot_color = (255,255,0)
    else:
        annot_color = (0,255,0)
    return annot_color

# %%
def main(args):
    abbr_dict = {
                'gender':{'m':'Male','f':'Female'},
                'cough': {'n':'None', 'd':'Dry', 'w':'Wet'},
                's_antipyretic': {'n':'No','y':'Yes'},
                's_odynophagia': {'n':'None', 'l':'Low', 'm':'Medium','h':'High'},
                's_dysphagia': {'n':'None', 'l':'Low', 'm':'Medium','h':'High'}}
    class2ids = {
                'dx':{"bacterial":0,"covid":1,"normal":2},
                'gender':{'m':0,'f':1},
                's_antipyretic': {"n":0,"y":1},
                's_odynophagia': {'n':0, 'l':1, 'm':2,'h':3},
                's_dysphagia' : {'n':0, 'l':1, 'm':2,'h':3}}

    ids2class = {kc:{v:k for k,v in class2ids[kc].items()} for kc in class2ids.keys()}

    imglist = os.listdir(args.image_dir)
    imglist = [f for f in imglist if f.endswith('.jpg') and not "inference" in f]

    print("Loading model .det.", end=".")
    model_det =  load_detect_model(device=DEVICE)
    print(".cnn.", end=".")
    model_cnn = load_cnn_model(model_file = str(ROOT / "models/cdetect_cnn_baseline.pth.tar"), 
                    device=DEVICE) 
    print(".gan.", end=".")
    model_gan = load_cnn_model(model_file = str(ROOT / "models/cdetect_cnn_gan.pth.tar"), 
                    device=DEVICE) 
    print(".ann.", end=".")
    model_ann, df_ann, scaler = load_ann_model(device=DEVICE)
    print("DONE")

    show_graphs = not args.no_graphs 
    if show_graphs:
        cv2.namedWindow('inputs', cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('original',cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('result',cv2.WINDOW_KEEPRATIO)

    # Get the list of files in the folder
    imglist = [f for f in os.listdir(args.image_dir) if f.endswith(".jpg") and not 'inference' in f]
    imgrange = range(1,len(imglist)+1)
    imgdict = dict(zip(imgrange, imglist))

    inputs = {}
    while(True):
        age, gender, vacc, temp = input("Age Gender [m|f] Vaccines [0-4] Temperature (F): ").split()
        cough = input("Cough: [n]none, [d]ry or [w]et: ").lower()
        antipyretic = input("Taking fever medicines [y]es or [n]o: ").lower()
        odynophagia = input("Pain Swallowing [n]one [l]ow [m]edium [h]igh: ").lower()
        dysphagia = input("Difficulty Swallowing [n]one [l]ow [m]edium [h]igh: ").lower()
        image_file_id = input(f"Select image {imgdict}: ")
        image_file_id = int(image_file_id)

        age = int(age)
        inputs['Age'] = age
        inputs['Gender'] = abbr_dict['gender'][gender]
        vacc = int(vacc)
        inputs['Vaccine doses'] = vacc 
        temp = float(temp)
        inputs['Temperature'] = temp
        inputs['Cough'] = abbr_dict['cough'][cough]
        inputs['Antipyretic'] = abbr_dict['s_antipyretic'][antipyretic] 
        inputs['Odynophagia'] = abbr_dict['s_odynophagia'][odynophagia]
        inputs['Dysphagia'] = abbr_dict['s_dysphagia'][dysphagia]
        inputs['Image File'] = imgdict[image_file_id] 

        if show_graphs:
            bg_img = np.ones(shape=(args.image_size, args.image_size,3), dtype=np.int16)
            cv2.putText(bg_img, json.dumps(inputs),fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 255, 0),thickness=3)
            cv2.imshow('inputs',bg_img)
            cv2.moveWindow("inputs", 10, 20)
    
        img, throat = detect_throat(os.path.join(args.image_dir,imgdict[image_file_id]), args.image_size, model_det)
        # Detect and trim throat from the uploaded image
        if show_graphs:
            cv2.imshow('`original',img)
            cv2.resizeWindow('original', 300, 300)
            cv2.moveWindow("original", 400, 20)

            cv2.imshow('result',throat)
            cv2.resizeWindow('result', 300, 300)
            cv2.moveWindow("result", 700, 20)
    
        pred_cnn = predict_cnn(throat, model_cnn, args.image_size)
        pred_cnn_text = f"CNN:{ids2class['dx'][pred_cnn.detach().cpu().numpy()[0].argmax()]}"
        if show_graphs:
            cv2.putText(throat,pred_cnn_text , (10,10), cv2.FONT_HERSHEY_SIMPLEX, 3, getColor(pred_cnn),2)
            cv2.imshow('result',throat)
        else:
            print(pred_cnn_text)


        dx_cols = [k + '_score' for k in class2ids['dx'].keys()]
        df_ann[dx_cols] = pred_cnn.tolist()[0]
        df_ann['d_age'] = int(age)
        df_ann['d_vacc_status'] = int(vacc)
        df_ann['d_gender'] = class2ids['gender'][gender]
        df_ann['v_cough_' + inputs['Cough'].lower()] = 1
        df_ann['v_temperature'] = float(temp)
        df_ann['s_antipyretic'] = class2ids['s_antipyretic'][antipyretic]
        df_ann['s_odynophagia'] = class2ids['s_odynophagia'][odynophagia]
        df_ann['s_dysphagia'] = class2ids['s_dysphagia'][dysphagia]

        x = scaler.transform(np.array(df_ann))

        pred_ann = model_ann(torch.from_numpy(x).float())
        pred_ann_text = f"ANN:{ids2class['dx'][pred_ann.detach().cpu().numpy()[0].argmax()]}"
        if show_graphs:
            cv2.putText(throat,pred_ann_text , (10,20), cv2.FONT_HERSHEY_SIMPLEX, 3, getColor(pred_cnn),2)
            cv2.imshow('result',throat)
        else:
            print(pred_ann_text)                
        
        if show_graphs:
            k = cv2.waitKey(0)
            if k == ord('q'):
                break
        else:
            cont = input("[c]ontinue or [q]uit: ").lower()
            if cont == 'q':
                break
    if show_graphs:                
        cv2.destroyAllWindows()
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
