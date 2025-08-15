from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os 
import pickle

test_path = "C:/Project/DACON_Project/Electricity_prediction/data/origin/test.csv"
train_path = "C:/Project/DACON_Project/Electricity_prediction/data/processed"
building_path = "C:/Project/DACON_Project/Electricity_prediction/data/origin/building_info.csv"
models = "C:/Project/DACON_Project/Electricity_prediction/models/ml"
test_dict = {}
answer_list = []
change_name = ['hotel', 'commercial', 'hospital', 'school', 'etc', 'apart', 'research', 'store', 'idc','public']
ko2en_dict = {
 '건물번호': 'b_num',
 '일시': 'date',
 '기온(°C)': 'tmp',
 '강수량(mm)': 'rain',
 '풍속(m/s)': 'wind',
 '습도(%)': 'hum',
 '일조(hr)': 'sunshine',
 '일사(MJ/m2)': 'solar',
 '전력소비량(kWh)': 'power_consumption',
 '건물유형': 'b_type',
 '연면적(m2)': 'total_area',
 '냉방면적(m2)': 'cooling_area',
 '태양광용량(kW)': 'solar_capacity',
 'ESS저장용량(kWh)': 'ess_capacity',
 'PCS용량(kW)': 'pcs_capacity',
}

def rename_dataframe_columns(df, mapping_dict):
    return df.rename(columns=mapping_dict).copy()

def minmax_scale(df: pd.DataFrame, exclude_cols):
    '''MinMax Scalering 적용'''
    target_cols = [i for i in df.columns if i not in exclude_cols]

    scaler = MinMaxScaler()
    df[target_cols] = scaler.fit_transform(df[target_cols])

    return df 

test = pd.read_csv(test_path, encoding='utf-8')
building = pd.read_csv(building_path, encoding='utf-8')

test_df = rename_dataframe_columns(test, ko2en_dict)
test_df['datetime'] = pd.to_datetime(test['일시'], format='%Y%m%d %H')

test_df['weekday'] = test_df['datetime'].dt.weekday 
test_df['time'] = test_df['datetime'].dt.hour

building_info_df = rename_dataframe_columns(building, ko2en_dict)
merge_df = pd.merge(test_df, building_info_df, on='b_num', how='left')


for i, tp in enumerate(merge_df['b_type'].unique()):
    name = change_name[i]
    test_dict[name] = merge_df[merge_df['b_type'] == tp].reset_index(drop=True)
    path = f"{train_path}/{name}_train.csv"
    train_df = pd.read_csv(path, encoding='utf-8')
    cols = list(train_df.columns)

    if "weekday" in cols:
        test_dict[name]['weekday'] = test_dict[name]['weekday'].apply(lambda x: 0 if x < 5 else 1)
    x_test = test_dict[name][cols]

    exclude_cols = ['weekday'] if 'weekday' in cols else []

    x_test = test_dict[name][cols].copy()
    x_test = minmax_scale(x_test, exclude_cols)

    if [f for f in os.listdir(models) if name in f]:
        model_path = os.path.join(models,[f for f in os.listdir(models) if name in f][0])
    else:
        print(f"There isn't the {name} model")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(x_test)
    answer_list.append(y_pred)
    print(f"{name} 예측 완료, 결과 샘플: {y_pred[:5]}")

submission = pd.read_csv('C:/Project/DACON_Project/Electricity_prediction/data/origin/sample_submission.csv')
submission['answer'] = answer_list
submission.to_csv(f'C:/Project/DACON_Project/Electricity_prediction/result/baseline_submission_ML.csv', index=False)
