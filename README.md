# w251-finalproject
## Authors (in random order): Joe Villsenor, Ryan Mitchell, Manohar Madhira, Napoleon Paxton

### Introduction
Early detection of highly contagious diseases, such as COVID-19 is key to controlling their spread and minimizing the effects on those that are infected. In-person tests put the tester and others in the vicinity at risk of infection, and over-the-counter tests are not always readily available due to supply chain issues or perhaps lack of state or federal funding. In this study, we examine the utility of COVID-19 detection through throat image analysis.  

For the paper and results obtained please see:  
[Whitepaper](paper_and_docs/COVID_Detection_through_Image_Analysis.pdf)  
[Presentation](paper_and_docs/Deep%20Learning%20based%20Covid-19%20Identification.pptx)  

Due to the large scope of the project and limited timelines, many of the phases for the project were done in parallel. As a result some code is duplicated in notebooks and scripts.

To duplicate this process, clone this project. For access to dataset collected please contact authors.

### Data Collection & Exploration
Data was collected using the Qualtrics survey from a network of doctors. The raw data has some HEIC images and some jpegs. 

These activities need to be completed only once, and can only be run by people with access to Qualtrics survey results and Roboflow projects.

1. Download the Survey results and convert the HEIC images to JPEG using [convert_heic_images_to_jpg_local](notebooks/convert_heic_images_to_jpg_local.ipynb) notebook.

2. Upload the images to Roboflow and annotate all the throat regions in an image. Download images from Roboflow in coco formatted file generated typically from roboflow. The zip file should contain _annotations file and all images.

3. Run [EDA](notebooks/eda.ipynb) to identify data issues and create final dataset. This notebook requires the original dataset downloaded from Qualtric Survey.

### Data Preparation
After step is completed, we need to trim the images to contain only throat. This data is then used for training CNN and GauGAN models.  

4. Trim the images to contain only throat. 
```
python3 src\data\preprocess.py --coco-file <filename in raw folder>
```
See `python3 src\data\preprocess.py --help` for more options to change defaults.

5. After steps 3 and 4 are complete, run the below for creating train/val/test split. Patients and not images are split into the three groups. All images belonging to patient, will be in the same group. The script below will also augment images. For each the dataset (`project` in our sample we call the project `cdetect`), the script will generate three project folders with train, val and test folders in each. Train, val, and test can then be used to train the CNN and GauGAN models. The folders created are:
    a.  `project` (e.g. `cdetect`). This folder contains train\val\test splits for the trimmed images  
    b.  `project_compose` (e.g. `cdetect_compose`). This folder contains original image + the image received after augmentation.
    c.  `project_all` (e.g. `cdetect_all`). Since an `albumentations` compose pipeline generates only one image with a random combination of the augmentations, this folder saves image for each of the augmentation options.

    Train val and test folders also contain the metadata for each patient that could be used for processing an ANN or regression.
```
python3 src\data\make_dataset.py -f <file containing survey metadata> --stratify
```
    Since input data is skewed `stratify` is required to ensure the each class has representation in train/val/test 

### GauGAN image generation

6.  After step 4, run the notebook [GauGANGeneration](notebooks/GauGANGeneration.ipynb) to generate images to get more data for training. Copy the images manually to the cloud for training and image generation. Due to the long time for training GANs and the project time constraints, we adjusted images and splits in the cloud.

### CNN - Convolutional Neural Network model

7.  After step 5 , run [train_cnn_orig](notebooks/train_cnn_orig.ipynb) followed by [load_cnn_and_output_preds_orig](notebooks/load_cnn_and_output_preds_orig.ipynb) to validate the preformance of CNN models. Repeat this steps for each of the projects you are interested in (e.g. cdetect, cdetect_compose, cdetect_all)

8. After step 5 and 6 and complete, copy the GAN images to the train folder only and retrain CNN using [train_cnn_orig_plus_gan](notebooks/train_cnn_orig_plus_gan.ipynb) notebook. Validate using [load_cnn_and_output_preds_orig_plus_gan](notebooks/load_cnn_and_output_preds_orig_plus_gan.ipynb) notebook.

### ANN - Artificial Neural Network

9.  After steps 7 and 8 are complete, copy the predictions file for the best model and train the ANN model to classify patient. Run script [train_ann](src/models/train_ann.py) to train the model.

```
python3 src\models\train_ann.py --cnn-data <file wth CNN results> --project <cdetect|cdetect_compose etc.>
```

10. Anytime after step 5, run the [train_ann_only](src/models/train_ann_only.py) model to get results based on metadata only.

```
python3 src\models\train_ann_only.py --project <cdetect|cdetect_compose etc.>
```

### Inference
11. Run inference on Jetson or any other machine using the Docker image `rmadhira/l4t-ml-covid-19:latest`. This image modules necessary to the inference using Yet-Another-EfficientDet-Pytorch, in addition to the l4t-ml image.  

Upload test images to a folder on your machine and load the docker image.

Allow docker image to use the default monitor and start docker.
```
export DISPLAY=:0
docker run -it --rm --runtime nvidia --network host -e DISPLAY=$DISPLAY -v /home/rmadhira86/w251/w251-finalproject:/w251-finalproject rmadhira/l4t-ml-c~ovid-19
```


```
python3 /w251-finalproject/src/models/predict_model.py -d <path to images uploaded>
```


