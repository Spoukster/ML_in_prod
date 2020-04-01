from .create_dataset import build_dataset_raw
from .dataset_processing import prepare_dataset, class_balancing
from .model_fit import xgb_fit_predict
from .validation import plot_confusion_matrix, plot_PR_curve, plot_ROC_curve
from sklearn.preprocessing import MinMaxScaler

CHURNED_START_DATE = '2019-09-01'
CHURNED_END_DATE = '2019-10-01'

INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]

# Создадим тренировочный датасет
build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                  churned_end_date=CHURNED_END_DATE,
                  inter_list=INTER_LIST,
                  raw_data_path='train/',
                  dataset_path='dataset/',
                  mode='train')

# Создадим тестовый датасет
build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                  churned_end_date=CHURNED_END_DATE,
                  inter_list=INTER_LIST,
                  raw_data_path='test/',
                  dataset_path='dataset/',
                  mode='test')

# Загрузим датасеты
train = pd.read_csv('dataset/dataset_raw_train.csv', sep=';')
test = pd.read_csv('dataset/dataset_raw_test.csv', sep=';')

# Обработаем наши датасеты
prepare_dataset(dataset=train, dataset_type='train')
prepare_dataset(dataset=test, dataset_type='test')

# Загрузим обработанные датасеты
train_new = pd.read_csv('dataset/dataset_train.csv', sep=';')
test_new = pd.read_csv('dataset/dataset_test.csv', sep=';')

# Балансировка классов
X_train_balanced, y_train_balanced, X_test, y_test = class_balancing(train_new)

# Обучение модели
model = xgb_fit_predict(X_train_balanced, y_train_balanced, X_test, y_test)
predict_test = model.predict(X_test)
predict_test_probas = model.predict_proba(X_test)[:, 1]

# Оценка качества модели
plot_confusion_matrix(y_test.values, predict_test, classes=['churn', 'active'])
plot_PR_curve(y_test.values, predict_test, predict_test_probas)
plot_ROC_curve(classifier=model,
               X=X_test,
               y=y_test.values,
               n_folds=3)
plt.show()

# Предсказание на тестовом датасете
predict_churned_users = model.predict(test_new)
submissions = pd.concat([test_new['Id'], pd.Series(predict_churned_users)], axis=1)
submissions = submissions.rename(columns={0: 'is_churned'})
submissions.to_csv('result/churned_users.csv', index=None)
