from dataset_processing import prepare_dataset, class_balancing
from model_fit import lgb_fit_predict
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]

# Загрузим датасеты
print('Загрузка датасетов...')
train = pd.read_csv('../dataset/dataset_raw_train.csv', sep=';')
test = pd.read_csv('../dataset/dataset_raw_test.csv', sep=';')

# Обработаем наши датасеты
prepare_dataset(dataset=train, inter_list=INTER_LIST, dataset_type='train')
prepare_dataset(dataset=test, inter_list=INTER_LIST, dataset_type='test')

# Загрузим обработанные датасеты
train_new = pd.read_csv('../dataset/dataset_train.csv', sep=';')
test_new = pd.read_csv('../dataset/dataset_test.csv', sep=';')

# Балансировка классов
X_train_balanced, y_train_balanced, X_test, y_test = class_balancing(train_new)

# Обучение модели
model = lgb_fit_predict(X_train_balanced, y_train_balanced, X_test, y_test)

# Предсказание на тестовом датасете
user_id = test_new['user_id']
test_new = test_new.drop(['user_id'], axis=1)

test_mm = MinMaxScaler().fit_transform(test_new)

predict_churned_users = model.predict(test_mm)
submissions = pd.concat([user_id, pd.Series(predict_churned_users)], axis=1)
submissions = submissions.rename(columns={0: 'is_churned'})
submissions.to_csv('../result/Yuriy_Ryabinin_churned_users.csv', sep=';', index=None)

print('Prediction is successfully prepared and saved to /result')
