
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")


import librosa.display


import numpy as np
import pandas as pd
import os
module_dir = os.path.dirname(__file__)  # get current directory

import librosa

import seaborn as sns
sns.set()

train_df= pd.read_csv(os.path.join(module_dir, 'models/train.csv'))

ebird_to_id = {}
id_to_ebird = {}
ebird_to_id["nocall"] = 0
id_to_ebird[0] = "nocall"
for idx, unique_ebird_code in enumerate(train_df.ebird_code.unique()):
    ebird_to_id[unique_ebird_code] = str(idx+1)
    id_to_ebird[idx+1] = str(unique_ebird_code)
sequence_length=50

from tensorflow import keras
model = keras.models.load_model(os.path.join(module_dir, 'models/best_model.h5'))

def predict_submission(audio_file_path):        
    loaded_audio_sample = []
    previous_filename = ""
    data_point_per_second = 10
    sample_length = 5*data_point_per_second
    wave_data = []
    wave_rate = None
       
    filename = audio_file_path
    wave_data, wave_rate = librosa.load(filename)
    sample = wave_data[0::int(wave_rate/data_point_per_second)]
    song_sample = np.array(sample[0:sample_length])
    input_data = np.reshape(np.asarray([song_sample]),(1,sequence_length)).astype(np.float32)
    prediction = model.predict(np.array([input_data]))
    predicted_bird = id_to_ebird[np.argmax(prediction)]
    print(predicted_bird)
    return predicted_bird

# audio_file_path = "D:\\BirdML\\sound\\example_test_audio\\BLKFR-10-CPL_20190611_093000.pt540.mp3"
# # example_df = pd.read_csv("D:/BirdML/sound/submission.csv")

# # if os.path.exists(audio_file_path):
# value = predict_submission(audio_file_path)
# value
