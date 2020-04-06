import lightgbm as lgb
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score

warnings.filterwarnings("ignore")


# Функция вычисляющая метрики качества модели
def evaluation(y_true, y_pred, y_prob):
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    ll = log_loss(y_true=y_true, y_pred=y_prob)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_prob)
    print('Метрики модели:')
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1: {}'.format(f1))
    print('Log Loss: {}'.format(ll))
    print('ROC AUC: {}'.format(roc_auc))
    return precision, recall, f1, ll, roc_auc


# Функция обучения модели и ее предсказания
def lgb_fit_predict(X_train, y_train, X_test, y_test):
    clf = lgb.LGBMClassifier(max_depth=4,
                             n_estimators=550,
                             learning_rate=0.01,
                             n_jobs=-1,
                             subsample=1.,
                             colsample_bytree=0.5,
                             min_child_weight=3,
                             reg_alpha=0.,
                             reg_lambda=0.,
                             seed=42)
    print('Обучение модели...')
    clf.fit(X_train, y_train, eval_metric='aucpr', verbose=10)
    predict_proba_test = clf.predict_proba(X_test)
    predict_test = clf.predict(X_test)
    precision_test, recall_test, f1_test, log_loss_test, roc_auc_test = \
        evaluation(y_test, predict_test, predict_proba_test[:, 1])
    return clf
