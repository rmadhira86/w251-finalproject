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
import pickle

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
    parser.add_argument('-f','--image-path',type=str, help='Location of Image file', required=True)
    parser.add_argument('--age','--patient-age',type=int,help="Patient Age", dest='age', required=True)
    parser.add_argument('--gender',type=str.lower,choices=['male','female'],help="Gender", required=True)
    parser.add_argument('--temp','--temperature',type=range_temp,help='Temperature in Fahrenheit', required=True, dest='temperature')
    parser.add_argument('--vacc-status','--vacc',type=int,help='Number of vaccination does',choices=range(5), dest='vacc_status')
    parser.add_argument('--cough',type=str.lower, choices=['none','dry','wet'], default='none',help="Cough (default: %(default)s)")
    parser.add_argument('--fever-medicines','--antipyretic','--fm',type=str.lower,choices=['no','yes'],default='no', 
        help="Taking fever medicines i.e. antipyretic (default: %(default)s)",
        dest="s_antipyretic")
    parser.add_argument('--swallow-difficulty','--dysphagia','--sd',type=str.lower,
        choices=['none','low','medium','high'], default='none',
        help="Difficulty in swallowing i.e. dysphagia (default: %(default)s)",
        dest="s_dysphagia")
    parser.add_argument('--swallow-pain','--odynophagia','--sp',type=str.lower,
        choices=['none','low','medium','high'], default='none',
        help="Pain while swallowing i.e. odynophagia (default: %(default)s)",
        dest="s_odynophagia")
    parser.add_argument('--image_size', type=int, default=512, help='Image Size to use (default: %(default)d x %(default)d)' )
    parser.add_argument('-v','--verbose', action='store_true')

    args = parser.parse_known_args()[0] if known else parser.parse_args() #This line is useful when debugging in VSCode, since VSCode sends additional parameters
    return args


# %%
def main(args):
    img_path = args.image_path
    IMG_SIZE = args.image_size

    throat_model_file = str(ROOT / "models/cdetect_throat_detection.pth")

    ARCH = 'vgg16'
    LR = 0.0001 #Do we need this, if loading from saved model?
    NUM_CLASSES = 3
    cnn_model_file = str(ROOT / "models/cdetect_cnn_baseline.pth.tar")

    ann_model_file = str(ROOT / "models/cdetect_ann_model_best.pth.tar")
    class2ids = {
                'dx':{"bacterial":0,"covid":1,"normal":2},
                'gender':{'male':0,'female':1},
                's_antipyretic': {"no":0,"yes":1},
                's_odynophagia': {'none':0, 'low':1, 'medium':2,'high':3},
                's_dysphagia' : {'none':0, 'low':1, 'medium':2,'high':3}}

    ids2class = {kc:{v:k for k,v in class2ids[kc].items()} for kc in class2ids.keys()}


    # age=35
    # gender='male'
    # temperature = 98.4
    # cough = 'none'
    # antipyretic = 'no'
    # odynophagia = 'none'
    # dysphagia = 'none'
    # vacc_status = 3

    # Detect and trim throat from the uploaded image
    original_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=IMG_SIZE)
    # We will have only 1 image in framed_imgs and original_imgs
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    # the image as NumPy array has shape [height, width, 3], 
    # when you permute the dimensions you swap the height and width dimension, creating a tensor with shape [batch_size, channels, width, height].
    # copied for reference from https://stackoverflow.com/questions/62482336/classification-with-pretrained-pytorch-vgg16-model-and-its-classes
    x = x.to(torch.float32).permute(0, 3, 1, 2) # batch_size x channels x height x width
    #x = x.to(torch.float32)
    model = EfficientDetBackbone(compound_coef=0, num_classes=1,
                                # replace this part with your project's anchor config
                                ratios=[(1.0, 1.0), (1.3, 0.8), (1.9, 0.5)],
                                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    model.load_state_dict(torch.load(throat_model_file, map_location=DEVICE))
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
    # For production usage, we would have logged this error with the image
    # So we could learn about the issues faced by model.
    # rois = Region Of InterestS ??
    if len(out[0]['rois']) == 0:
        raise ValueError("Throat not detected in image. Please try again")
    if len(out[0]['rois']) > 1:
        raise ValueError("Too many throats detected. Incorrect image provided.")

    img = cv2.cvtColor(original_imgs[0],cv2.COLOR_BGR2RGB)
    roi = out[0]['rois'][0]
    plt.figure(figsize=(20,20))
    (x1, y1, x2, y2) = roi.astype(int)
    throat = img[y1:y2, x1:x2]

    # Now start with CNN Detection
    model = models.__dict__[ARCH](pretrained=False)
    #Change the model's output layer to 3 classes (covid/normal/bacterial)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    checkpoint = torch.load(cnn_model_file,map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    imagenet_mean_RGB = [0.47889522, 0.47227842, 0.43047404]
    imagenet_std_RGB = [0.229, 0.224, 0.225]
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean_RGB, imagenet_std_RGB),
    ])


    img1 = img_transform(throat)
    model.eval()
    pred = model(img1.unsqueeze(0))
    pred_cnn = ids2class['dx'][pred.detach().cpu().numpy()[0].argmax()]
    print(f"CNN: {pred_cnn}")
    # Now the last stretch - add learning from ANN
    checkpoint = torch.load(ann_model_file,map_location=DEVICE)
    # Create a dataframe with 1 row of all zeros

    df = pd.DataFrame(0,index=np.arange(1),columns=checkpoint['colnames'])
    dx_cols = [k + '_score' for k in class2ids['dx'].keys()]
    df[dx_cols] = pred.tolist()[0]
    df['d_age'] = args.age
    df['d_vacc_status'] = args.vacc_status
    df['d_gender'] = class2ids['gender'][args.gender]
    df['v_cough_' + args.cough] = 1
    df['v_temperature'] = args.temperature
    df['s_antipyretic'] = class2ids['s_antipyretic'][args.s_antipyretic]
    df['s_odynophagia'] = class2ids['s_odynophagia'][args.s_odynophagia]
    df['s_dysphagia'] = class2ids['s_dysphagia'][args.s_dysphagia]

    #Read back the StandardScaler used during training
    sc = pickle.loads(checkpoint['scaler'])
    x = sc.transform(np.array(df))


    model = ANN(df.shape[1], checkpoint['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    pred = model(torch.from_numpy(x).float())
    pred_ann = ids2class['dx'][pred.detach().cpu().numpy()[0].argmax()]
    print(f"with ANN: {pred_ann}")

    annot_color = (255,255,0)
    if (pred_ann != "normal" and pred_cnn != "normal"):
        annot_color = (255,0,0)
    elif (pred_ann == "normal" and pred_cnn == "normal"):
        annot_color = (0,255,0)
    cv2.rectangle(img, (x1,y1), (x2,y2), annot_color,2)
    cv2.putText(img, f'CNN:{pred_cnn} with ANN:{pred_ann}',
                (x1, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 3,
                annot_color, 2)
    cv2.imwrite(args.image_path + '.inference.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Create a custom Window
    cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('result',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    cv2.resizeWindow('result',300,300)   
    key = cv2.waitKey(0) 
    
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
