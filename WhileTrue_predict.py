import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import pickle
import sys
import tqdm
from geopy.distance import great_circle as GD
from sklearn.metrics import f1_score

def f1(y_true, y_pred , **kwargs):
    return f1_score(y_true,(y_pred > 0.65).astype(int))

def create_gr_feats(data):
    data['time']=pd.to_datetime(data['time'],  format='%H:%M:%S')
    data['course'] = pd.to_numeric(data['course'], errors='coerce').fillna(0)
    data = data.sort_values(by=['record', 'time'])
    data['velocity'] = pd.to_numeric(data['velocity'], errors='coerce')
    data['velocity'] = 0.514444444444*data['velocity']
    for i in tqdm.tqdm(data['record'].unique()):
        tmp = data[data['record']==i]
        speed = [0]
        distance = [0]
        vector = [[0,0]]
        for j in range(1, len(tmp)):
            start = (tmp.iloc[j-1]['latitude'], tmp.iloc[j-1]['longitude'])
            end = (tmp.iloc[j]['latitude'], tmp.iloc[j]['longitude'])
            time = (tmp.iloc[j]['time']-tmp.iloc[j-1]['time']).total_seconds()
            s = GD(start, end).m
            distance.append(distance[-1]+s/1000)
            new_vec = (tmp.iloc[j]['latitude'] - tmp.iloc[j-1]['latitude'],tmp.iloc[j]['longitude'] - tmp.iloc[j-1]['longitude'])
            vector.append([vector[-1][0]+new_vec[0], vector[-1][1]+new_vec[1]])
            speed.append(s/(time+1))
        data.loc[data['record']==i, 'speed'] = speed
        data.loc[data['record']==i, 'distance'] = distance
        data.loc[data['record']==i, ['x','y']] = vector
    data = data[data['speed']<11]
    return data



def get_predict(argv):
    with open('model.pkl', 'rb') as fp:
        model = pickle.load(fp)
    filename = argv[1]
    data = pd.read_csv(filename)
    test_data = create_gr_feats(data)
    test_pred = model.predict(test_data)
    test_data['pred'] = test_pred.data[:,0]
    tmp = test_data.groupby('record').mean()
    tmp.to_csv(filename[:-4]+'.txt', columns=['pred'], sep='\t')



    

if __name__ == '__main__':
    print(sys.argv)
    get_predict(sys.argv)



