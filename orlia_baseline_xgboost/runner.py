import numpy as np
import cv2
import pandas as pd
import os
import xgboost as xgb
from keras.applications.resnet import ResNet50, preprocess_input
from tensorflow import keras
from PIL import Image
from numpy import asarray



TABULARTEST = 'KORL_avatar_test_X.csv'

test_df = pd.read_csv(TABULARTEST)


# Load XGBoost
model_xgb_1 = xgb.XGBRegressor()
model_xgb_1.load_model("cell1_data/layer1.json")

model_xgb_2 = xgb.XGBRegressor()
model_xgb_2.load_model("cell2_data/layer2.json")

model_xgb_3 = xgb.XGBRegressor()
model_xgb_3.load_model("cell3_data/layer3.json")

model_xgb_4 = xgb.XGBRegressor()
model_xgb_4.load_model("cell4_data/layer4.json")

model_xgb_5 = xgb.XGBRegressor()
model_xgb_5.load_model("cell5_data/layer5.json")

model_xgb_6 = xgb.XGBRegressor()
model_xgb_6.load_model("cell6_data/layer6.json")

model_xgb_7 = xgb.XGBRegressor()
model_xgb_7.load_model("segment_tissue_data/layer7.json")

# Load Resnet 
model_resnet_1 = keras.models.load_model("cell1_data/resnet1")

model_resnet_2 = keras.models.load_model("cell2_data/resnet2")

model_resnet_3 = keras.models.load_model("cell3_data/resnet3")

model_resnet_4 = keras.models.load_model("cell4_data/resnet4")

model_resnet_5 = keras.models.load_model("cell5_data/resnet5")

model_resnet_6 = keras.models.load_model("cell6_data/resnet6")

model_resnet_7 = keras.models.load_model("segment_tissue_data/resnet_st")

outputs = []
# Get result from layer 1, then etc...

IMAGELOCATION_ARR = [r'./segment_tissue_data'] # , r'./cell2_data', r'./cell3_data', r'./cell4_data', r'./cell5_data', r'./cell6_data', r'./segment_tissue_data']


RESNET_MODEL_ARR = [model_resnet_1, model_resnet_2, model_resnet_3, model_resnet_4, model_resnet_5, model_resnet_6, model_resnet_7]

XGB_MODEL_ARR = [model_xgb_1, model_xgb_2, model_xgb_3, model_xgb_4, model_xgb_5, model_xgb_6, model_xgb_7]

RESIZE_ARR = [False, False, False, False, False, False, True]

count = 6
for IMAGELOCATION in IMAGELOCATION_ARR:
    print(IMAGELOCATION)
    
    path = IMAGELOCATION
    list_of_files = []
    YDead = []
    l = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            list_of_files.append(os.path.join(root,file))

    # Get images from layer...
    for name in list_of_files:
        # load the image and convert into 
        # numpy array
        try:
            img = Image.open(name)
            
            if (RESIZE_ARR[count]):
            	newsize = (150, 150)
            	img = img.resize(newsize)

            nameid = (name.split('/')[3].split('_')[0])
            print(nameid)
            numpydata = asarray(img)


            # Check if a data entry exists or matches
            YDead.append(((test_df.loc[test_df['Patient_ID'] == nameid])['Deces']).item())

            l.append(numpydata)
        except Exception as e:
            pass

    arr = np.array(l)
    X = arr # images
    
    print('Getting images')
    print(X.shape)
    
    X_224 = np.array([cv2.resize(xi, (224, 224)) for xi in X])
    X_224 = preprocess_input(X_224.astype('float'))
    
    resnet_model = RESNET_MODEL_ARR[count]
    resnet_features = resnet_model.predict(X_224, verbose=1)
    resnet_features.shape
    
    xgb_model = XGB_MODEL_ARR[count]
    prediction = xgb_model.predict(resnet_features)
    outputs.append([prediction])
    
    count = count + 1
    print('Layer ', count, ' is done!!')
    
print(outputs)
