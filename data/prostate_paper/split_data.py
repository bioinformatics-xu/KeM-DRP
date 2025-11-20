from os.path import join, dirname
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# current_dir = dirname(__file__)
current_dir = r''
input_dir = 'raw_data'
output_dir = 'splits'

input_dir = join(current_dir, input_dir)
output_dir = join(current_dir, output_dir)


def get_response():
    # filename = '41588_2018_78_MOESM5_ESM.xlsx'
    # df = pd.read_excel(join(input_dir, filename), sheet_name='Supplementary_Table3.txt', skiprows=2)
    filename = r''  # CSV 文件的文件名
    df = pd.read_csv(join(input_dir, filename),encoding='gbk')  # 使用 read_csv 而不是 read_excel
    print(df.columns)
    response = pd.DataFrame()
    print (df.head())
    response['id'] = df['id']
    response['response'] = df['response']
    response['ca'] = df['ca']
    # response['response'] = response['response'].replace('Metastasis', 1)
    # response['response'] = response['response'].replace('Primary', 0)
    response = response.drop_duplicates()
    print (response.head())
    print (response.shape)
    return response


response = get_response()
print (response.head())
print (response.info())
all_ids = response
ids = response['id'].values
y = response['response'].values
ca = response['ca'].values
ids_train, ids_test, y_train, y_test = train_test_split(ids, y, test_size=0.1, stratify=ca, random_state=26)
ids_train, ids_validate, y_train, y_validate = train_test_split(ids_train, y_train, test_size=len(y_test),
                                                                stratify=y_train, random_state=26)

test_set = pd.DataFrame({'id': ids_test, 'response': y_test})
train_set = pd.DataFrame({'id': ids_train, 'response': y_train})
validate_set = pd.DataFrame({'id': ids_validate, 'response': y_validate})

print (test_set.response.value_counts() / float(test_set.response.value_counts().sum()))
print (train_set.response.value_counts() / float(train_set.response.value_counts().sum()))
print (validate_set.response.value_counts() / float(validate_set.response.value_counts().sum()))

test_set.to_csv(join(output_dir, 'test_set.csv'))
validate_set.to_csv(join(output_dir, 'validation_set.csv'))
train_set.to_csv(join(output_dir, 'training_set.csv'))

total_number_samples = train_set.shape[0]
number_patients = np.geomspace(30, total_number_samples, 5)
number_patients = [int(s) for s in number_patients][::-1]
print (number_patients)

for i, n, in enumerate(number_patients):
    if i == 0:
        filename = join(output_dir, 'training_set_0.csv')
        train_set.to_csv(filename)
        continue
    number_samples = n
    print (i, number_samples)
    print (ids_train.shape, y_train.shape)
    ids_train, ids_validate, y_train, y_validate = train_test_split(ids_train, y_train, train_size=n,
                                                                    stratify=y_train, random_state=26)
    print (ids_train.shape, y_train.shape)
    train_set = pd.DataFrame({'id': ids_train, 'response': y_train})
    filename = join(output_dir, 'training_set_{}.csv'.format(i))
    train_set.to_csv(filename)
