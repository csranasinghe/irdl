import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import seaborn as sns
sns.set_style('darkgrid')

from tensorflow.keras.models import Model, load_model
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199
print('All modules have been imported')


module_dir = os.path.dirname(__file__)  # get current directory

def make_dataframes(sdir,csvpath):
    birds_df=pd.read_csv(csvpath)
    groups=birds_df.groupby('data set') 
    train_df=groups.get_group('train')
    classes=sorted(train_df['labels'].unique())
    class_count=len(classes)
    return classes, class_count
sdir= os.path.join(module_dir, 'modules')
csvpath=os.path.join(module_dir, 'modules/birds.csv')
classes, class_count=make_dataframes(sdir,csvpath)  
confidence_thr = 0.5
CLASSES = [ "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def applySSD(image):
    blob=None
    (h, w) = image.shape[0] , image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_thr:

            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100) 
            cropped_image = image[startY-15:endY, startX:endX]

    return cropped_image

def ave_predictor(sdir, classes,img_shape, model_path, averaged=True, verbose=True, scale=1): 
    model=load_model(model_path)
    good_image_count=0
    image_list=[]
    if verbose:
        print (' Model is being loaded- this will take about 10 seconds')
    try:
        img=sdir
        img=cv2.resize(img,(img_shape[0], img_shape[1]))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img=img/scale
        good_image_count +=1                       
        image_list.append(img)        
    except:
        print("can't find the image")
    image_array=np.array(image_list)
    preds=model.predict(image_array)    
    psum=[]
    for i in range (class_count): 
        psum.append(0)    
    for p in preds: 
        for i in range (class_count):
            psum[i]=psum[i] + p[i] 
    index=np.argmax(psum)        
    klass=classes[index]
    prob=psum[index]/len(preds) * 100        
 
    return klass, prob, None
print("[INFO] loading model...")
shared_dir = os.path.join(module_dir, 'modules/MobileNet-SSD/')
net = cv2.dnn.readNetFromCaffe(shared_dir+ 'deploy.prototxt' , shared_dir+ 'mobilenet_iter_73000.caffemodel')

def main(file_name):
    vc = cv2.imread(file_name)
    frame = applySSD(vc) 
    model_save_loc =os.path.join(module_dir, 'modules/BIRDS-450-(200 X 200)-99.28.h5')
    img_size=(200,200) 
    klass, probability,_=ave_predictor(frame,classes,img_size,  model_save_loc, averaged=True, verbose=True)
    return klass

# print(main("/media/test_ci0H6RS.jpeg"))