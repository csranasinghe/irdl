import pickle
import pandas as pd
import os
module_dir = os.path.dirname(__file__)  # get current directory


def main(list_value):
    filename = os.path.join(module_dir, 'models/final_model.pkl')
    loaded_model=pickle.load(open(filename, 'rb'))
    data=pd.read_excel(os.path.join(module_dir, 'models/Birds.xlsx'))
    data.head()
    beak = data['Beak shape'].str.split(', ', n = 1, expand = True)
    data['Beak shape'] = beak[0]
    eye = data['Eye colour'].str.split(', ', n = 1, expand = True)
    data['Eye colour'] = eye[0]
    wing = data['Wings colour'].str.split(', ', n = 1, expand = True)
    data['Wings colour'] = wing[0]
    bird = data['Bird colour'].str.split(', ', n = 1, expand = True)
    data['Bird colour'] = bird[0]
    data.isna().sum() # no of missing values column wise
    x=data.iloc[:,:5]
    list_row = list_value
    x.loc[len(x)] = list_row
    x = x.astype({"Beak shape":'category',"Eye colour":'category',"Wings colour":'category',"Location":'category',"Bird colour":'category'})
    x_test_new=pd.get_dummies(x,drop_first=True)
    return(loaded_model.predict(x_test_new.iloc[-1:]))
