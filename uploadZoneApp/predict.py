import pickle
import pandas as pd

def main(list_value):
    filename = "/home/aloka/Documents/Bird ML/Server/uploadZoneApp/models/final_model.pkl"
    loaded_model=pickle.load(open(filename, 'rb'))
    data=pd.read_excel('/home/aloka/Documents/Bird ML/classifcation_detail/Birds2.xlsx')
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
