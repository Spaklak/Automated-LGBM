import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from lightgbm import LGBMClassifier
from ydata_profiling import ProfileReport
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from joblib import dump
import pickle

class ModelPipeline:
    def __init__(self, path, **kwargs):
        self.dataset = pd.read_csv(path, **kwargs)
        self.model = None
        self.grid_model = None
        self.selector = None
        self.shap_values = None
        self.delete_psi = []

        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
    
    def get_target_distribution(self, target_col):
        """Распределение целевой переменной в процентах"""
        return self.dataset[target_col].value_counts(normalize=True).sort_index() * 100
    
    def misiing_data_summary(self):
        """Предоставляет сводку пропущенных данных по каждому столбцу"""
        return self.dataset.isna().sum()
    
    def total_missing_data(self):
        """Возвращает общее количество пропущенных значений в наборе данных"""
        return self.dataset.isna().sum().sum()
    
    def plot_correlation_matrix(self, method='spearman', numeric_only=True, linewidths=0.5, cmap='viridis', **kwargs):
        """Строит матрицу корреляций для числовых столбцов с использованием заданного метода корреляции"""
        plt.figure(figsize=(14,14))
        sns.heatmap(data=self.dataset.corr(method=method, numeric_only=numeric_only), linewidths=linewidths, cmap=cmap, **kwargs)
        plt.show()
    
    def plot_numeric_distribution(self, column, kde=True, alpha=0.5, color='g', figsize=(14, 20), **kwargs):
        """Строит график распределения для указанного числового столбца"""
        plt.figure(figsize=figsize)
        sns.histplot(data=self.dataset[column], kde=kde,alpha=alpha, color=color, **kwargs)
        plt.show()
    
    def smart_report_html(self, file_name):
        """Создает отчет ydata с полной информации о данных"""
        profile = ProfileReport(self.dataset, title='Smart Report')
        profile.to_file(file_name)

    def remove_dublicates(self, **kwargs):
        """Удаляет дубликаты из набора данных"""
        self.dataset = self.dataset.drop_duplicates(**kwargs)

    def fill_missing_with_mean(self, column, accuracy=0):
        """Заполняет пропуски в указанном столбце средним значением, округленным до указанной точности"""
        self.dataset[column] = self.dataset[column].fillna(round(self.dataset[column].mean(), accuracy))
    
    def plot_multiple_numeric_distributions(self, numeric_columns, nrows=1, ncols=1, color='g', alpha= 0.5, kde=True, figsize=(14, 20),  **kwargs):
        """Строит графики распределений для нескольких числовых столбцов"""
        plt.figure(figsize=figsize)
        for i, column in enumerate(numeric_columns):
            plt.subplot(nrows,ncols, i+1)
            sns.histplot(data=self.dataset, x=column, color=color, alpha=alpha, kde=kde, **kwargs)
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_distributions(self, columns, nrows=1, ncols=1, color='g', alpha= 0.5, figsize=(14, 20), **kwargs):
        """Строит распределение категориальных признаков для указанных столбцов"""
        plt.figure(figsize=figsize)
        for i, column in enumerate(columns):
            plt.subplot(nrows,ncols, i+1)
            sns.countplot(data=self.dataset, x=column, color=color, alpha=alpha, **kwargs)
        plt.tight_layout()
        plt.show()

    def split_train_val_test(self, target_col, train_size = 0.3, val_size = 0.1):
        """Разделяет набор данных на обучающую, валидационную и тестовую выборки"""
        self.X = self.dataset.drop(target_col, axis=1)
        self.y = self.dataset[target_col]

        self.X_train, X_val_test, self.y_train, y_val_test = train_test_split(self.X, self.y, test_size=train_size, random_state=42, stratify=self.y)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_val_test, y_val_test, test_size=val_size, random_state=42, stratify=y_val_test)

    def fit_lgbm(self, **kwargs):
        """Обучает классификатор LightGBM на обучающем наборе данных"""
        self.model = LGBMClassifier(**kwargs)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Оценивает модель на обучающей, валидационной и тестовой выборках, строя ROC-кривую и матрицу ошибок"""
        self._plot_model_metrics(self.X_train, self.y_train, "Training Set")
        self._plot_model_metrics(self.X_val, self.y_val, "Validation Set")
        self._plot_model_metrics(self.X_test, self.y_test, "Test Set")

    def _plot_model_metrics(self, X, y, title):
        y_pred = self.model.predict(X)
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        plt.title(f'ROC Curve - {title}')
        plt.show()

        plt.figure()
        ConfusionMatrixDisplay(confusion_matrix(y, y_pred)).plot()
        plt.title(f'Confusion Matrix - {title}')
        plt.show()

    def shap_analysis(self, max_display=20):
        """Выполняет анализ SHAP для определения важности признаков и отображает результаты"""
        explainer = shap.Explainer(self.model)
        self.shap_values = explainer(self.X_val)
        shap.plots.bar(self.shap_values, max_display=max_display)
        shap.plots.beeswarm(self.shap_values, max_display=max_display)
    
    def remove_low_shap_features(self, threshold=0.2):
        """Удаляет признаки с SHAP-важностью ниже указанного порога"""
        importance_dict = dict(zip(self.shap_values.feature_names, np.abs(self.shap_values.values).mean(axis=0)))
        importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        importance_dict = {k: np.round(importance_dict[k], 2) for k in list(importance_dict)}
        list_deleted = []
        for i in importance_dict.keys():
            if importance_dict.get(i) < threshold:
                list_deleted.append(i)
        self.X_test = self.X_test.drop(list_deleted, axis=1)
        self.X_val = self.X_val.drop(list_deleted, axis=1)
        self.X_train = self.X_train.drop(list_deleted, axis=1)
    
    def plot_feature_selection_rfecv(self, step=1, cv=5, scoring='recall', **kwargs):
        """Строит результаты выбора признаков с помощью RFECV на основе показателей кросс-валидации"""
        self.selector = RFECV(self.model, step=step, cv=cv, scoring=scoring, **kwargs)
        self.selector = self.selector.fit(self.X_train, self.y_train)
        plt.figure(figsize=(12,6))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(self.selector.cv_results_['mean_test_score']) + 1), self.selector.cv_results_['mean_test_score'])
        plt.show()
    
    def remove_features_rfe(self, step=1, n_features_to_select=6, **kwargs):
        """Удаление признаков с помощью RFE."""
        self.selector = RFE(self.model, n_features_to_select=n_features_to_select, step=step, **kwargs)
        self.selector = self.selector.fit(self.X_train, self.y_train)
        list_deleted = []
        for i in range(len(self.selector.support_)):
            if not self.selector.support_[i]:
                list_deleted.append(self.X_train.columns[i])
        self.X_test = self.X_test.drop(list_deleted, axis=1)
        self.X_val = self.X_val.drop(list_deleted, axis=1)
        self.X_train = self.X_train.drop(list_deleted, axis=1)
    
    def PSI(self):
        """Удаляет признаки с помощью рекурсивного исключения признаков (RFE)"""
        columns = self.X_train.columns
        for column in columns:
            self._PSI_factor_analysis(self.X_train, self.X_val, column)
        if len(self.delete_psi) > 0:
            self.X_test = self.X_test.drop(self.delete_psi, axis=1)
            self.X_val = self.X_val.drop(self.delete_psi, axis=1)
            self.X_train = self.X_train.drop(self.delete_psi, axis=1)
    
    def _continuous2interval(self, df, df_target, percent_interval=0.1):
        special_target = []
        interval_target = []
        begin = False
        temp_percent = 0
        for index, row in (df[df_target].value_counts(normalize=True)).reset_index().sort_values(by='proportion').iterrows():
            if row[df_target] >= percent_interval:
                special_target.append(row['proportion'])
            else:
                temp_percent += row[df_target]
                if begin == False:
                    begin = row['proportion']
                if temp_percent >= percent_interval:
                    interval_target.append([begin, row['proportion']])
                    begin = False
                    temp_percent = 0
        if begin != False:
            interval_target.append([begin, np.inf])
        return interval_target, special_target

    def _PSI_factor_analysis(self, dev, val, column):
        intervals = [-np.inf] + [i[0] for i in self._continuous2interval(dev, column)[0]] + [np.inf]
        dev_temp = pd.cut(dev[column], intervals).value_counts(sort=False, normalize=True)
        val_temp = pd.cut(val[column], intervals).value_counts(sort=False, normalize=True)
        print('PSI for {}'.format(column))
        probability = sum(((dev_temp - val_temp)*np.log(dev_temp / val_temp)).replace([np.inf, -np.inf], 0))
        print('PSI:', probability < 0.2)
        if probability >= 0.2:
            self.delete_psi.append(column)
    
    def grid_fit(self, param_grid, scoring='recall', cv=None, **kwargs):
        """Проводит поиск оптимальных гиперпараметров с помощью кросс-валидации и `GridSearchCV`"""
        self.grid_model = GridSearchCV(self.model, param_grid=param_grid, cv=cv, scoring=scoring, **kwargs)
        self.grid_model.fit(self.X_train, self.y_train)

    def evaluate_grid_search_model(self):
        """Оценивает настроенную модель по результатам `GridSearchCV` на обучающей, валидационной и тестовой выборках"""
        self._plot_grid_model_metrics(self.X_train, self.y_train, "Training Set")
        self._plot_grid_model_metrics(self.X_val, self.y_val, "Validation Set")
        self._plot_grid_model_metrics(self.X_test, self.y_test, "Test Set")

    def _plot_grid_model_metrics(self, X, y, title):
        y_pred = self.grid_model.predict(X)
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        plt.title(f'ROC Curve - {title}')
        plt.show()

        plt.figure()
        ConfusionMatrixDisplay(confusion_matrix(y, y_pred)).plot()
        plt.title(f'Confusion Matrix - {title}')
        plt.show()

    def save_grid_model_joblib(self, name='gird_model.joblib', **kwargs):
        """Сохраняет настроенную модель в файл с использованием `joblib`"""
        dump(self.grid_search_model, name, **kwargs)
    

    def save_grid_model_pickle(self, name='grid_model.pkl', **kwargs):
        """Сохраняет настроенную модель в файл с использованием `pickle`"""
        with open(name, 'wb') as f:
            pickle.dump(self.grid_search_model, f, **kwargs)