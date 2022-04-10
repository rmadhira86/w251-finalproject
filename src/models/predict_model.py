#%%
import sys, os
# Add the (Y)et(A)nother(E)fficient(D)et(P)y(T)orch folder to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yaedpyt'))
from yaedpyt.backbone import EfficientDetBackbone
from yaedpyt.efficientdet.utils import BBoxTransform, ClipBoxes
from yaedpyt.utils.utils import preprocess, invert_affine, postprocess

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms 
# %%
img_path = "../../data/external/live_demo_test.jpeg"
IMG_SIZE = 512

throat_model_file = "../../models/cdetect_throat_detection.pth"

ARCH = 'vgg16'
LR = 0.0001 #Do we need this, if loading from saved model?
NUM_CLASSES = 3
cnn_model_file = "../../models/cdetect_cnn_baseline.pth.tar"

#%%
# Detect and trim throat from the uploaded image
original_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=IMG_SIZE)
# We will have only 1 image in framed_imgs and original_imgs
x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

# the image as NumPy array has shape [height, width, 3], 
# when you permute the dimensions you swap the height and width dimension, creating a tensor with shape [batch_size, channels, width, height].
# copied for reference from https://stackoverflow.com/questions/62482336/classification-with-pretrained-pytorch-vgg16-model-and-its-classes
x = x.to(torch.float32).permute(0, 3, 1, 2) # batch_size x channels x height x width
#x = x.to(torch.float32)
# %%
model = EfficientDetBackbone(compound_coef=0, num_classes=1,
                             # replace this part with your project's anchor config
                             ratios=[(1.0, 1.0), (1.3, 0.8), (1.9, 0.5)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
model.load_state_dict(torch.load(throat_model_file, map_location=torch.device('cpu')))
model.requires_grad_(False)
model.eval()
# %%
with torch.no_grad():
    features, regression, classification, anchors = model(x)
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold=0.5, iou_threshold=0.5)

out = invert_affine(framed_metas, out)
# %%
# For production usage, we would have logged this error with the image
# So we could learn about the issues faced by model.
if len(out[0]['rois']) == 0:
    raise ValueError("Throat not detected in image. Please try again")
if len(out[0]['rois']) > 1:
    raise ValueError("Too many throats detected. Incorrect image provided.")

img = cv2.cvtColor(original_imgs[0],cv2.COLOR_BGR2RGB)
roi = out[0]['rois'][0]
plt.figure(figsize=(20,20))
(x1, y1, x2, y2) = roi.astype(int)
cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,0),2)
throat = img[y1:y2, x1:x2]
plt.imshow(img)
# %%
plt.imshow(throat)
# %%
# Now start with CNN Detection
model = models.__dict__[ARCH](pretrained=False)
#Change the model's output layer to 3 classes (covid/normal/bacterial)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
checkpoint = torch.load(cnn_model_file,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

#%%
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
pred.argmax()
# %%
