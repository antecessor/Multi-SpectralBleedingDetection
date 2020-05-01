

from scipy.spatial import distance, distance_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import class_weight, shuffle, resample
from stability_selection import StabilitySelection, plot_stability_path



train_original = pd.read_csv("DataUsed/method23_real2.csv")
test_original = pd.read_csv("DataUsed/method23_real2_valid.csv")
df = train_original

# df.insert(3, "num2", num2)
targetIndex = -1
# df = df.iloc[pd.isna(df.iloc[:, targetIndex]).values == False, :]
# df = df.drop(columns=["Num1"])

vars = df.columns[range(len(df.columns) - 1)]
df = df.values
X = df[:, range(0, df.shape[1] - 1)]
Y = df[:, targetIndex]

base_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l2'))
])

selector = StabilitySelection(base_estimator=base_estimator, lambda_name='model__C',
                              lambda_grid=np.logspace(-5, -1, 50)).fit(X, Y)
fig, ax = plot_stability_path(selector)
fig.show()

selected_variables = selector.get_support(indices=True)
selected_scores = selector.stability_scores_.mean(axis=1)


selectedFeatures = pd.DataFrame({"selectedVars": vars[selected_variables], "score": selected_scores[selected_variables]},index=vars[selected_variables])
selectedFeatures.plot(kind='barh')
selectedFeatures.to_excel("stabilityFeatureSelection.xlsx")
# print(selector.get_support(indices=True))

# X = X[:, selected_variables]
class1Data = X[Y == 1, :]
class2Data = X[Y == 0, :]
class1Target = Y[Y == 1]
class2Target = Y[Y == 0]

pipelines = []
res = []
split = 3
kf = KFold(n_splits=split)
fold = 1

trainIndexC1 = []
trainIndexC2 = []
testIndexC1 = []
testIndexC2 = []
targetTrainIndexC1 = []
targetTrainIndexC2 = []
targetTestIndexC1 = []
targetTestIndexC2 = []
for train_index, test_index in kf.split(class1Data):
    trainIndexC1.append(train_index)
    testIndexC1.append(test_index)
for train_index, test_index in kf.split(class2Data):
    trainIndexC2.append(train_index)
    testIndexC2.append(test_index)
for train_index, test_index in kf.split(class1Target):
    targetTrainIndexC1.append(train_index)
    targetTestIndexC1.append(test_index)
for train_index, test_index in kf.split(class2Target):
    targetTrainIndexC2.append(train_index)
    targetTestIndexC2.append(test_index)


def spScore(y_true, y_pred):
    aucValue = roc_auc_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 100 * (2 * aucValue - rec)


for index in range(len(targetTestIndexC2)):
    c1DataTrain, c1DataTest = class1Data[trainIndexC1[index], :], class1Data[testIndexC1[index], :]
    c2DataTrain, c2DataTest = class2Data[trainIndexC2[index], :], class2Data[testIndexC2[index], :]
    c1TargetTrain, c1TargetTest = class1Target[targetTrainIndexC1[index]], class1Target[targetTestIndexC1[index]]
    c2TargetTrain, c2TargetTest = class2Target[targetTrainIndexC2[index]], class2Target[targetTestIndexC2[index]]
    minorClassSize = c2DataTrain.shape[0]

    for i in range(int(c1DataTrain.shape[0] / minorClassSize)):
        X = np.append(class2Data, c1DataTrain[range(i * minorClassSize, (i + 1) * minorClassSize), :], axis=0)
        X = np.append(X, c2DataTrain, axis=0)
        CMS = np.append(class2Target, c1TargetTrain[range(i * minorClassSize, (i + 1) * minorClassSize)], axis=0)
        CMS = np.append(CMS, c2TargetTrain, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, CMS, test_size=0.05, stratify=CMS)

        # smt = SMOTETomek()

        # tree_param = {'bootstrap': [True, False],
        #               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        #               'max_features': ['auto', 'sqrt'],
        #               'min_samples_leaf': [1, 2, 4],
        #               'min_samples_split': [2, 5, 10],
        #               'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600]}
        # grid = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=tree_param, n_iter=10, verbose=2, n_jobs=-1, scoring=make_scorer(roc_auc_score))

        tree_param = {'criterion': ['gini', 'entropy'], 'max_depth': [5, 9, 20, 30, 40, 50, 70, 90, 120, 150]}
        grid = GridSearchCV(DecisionTreeClassifier(), param_grid=tree_param, scoring=make_scorer(f1_score))
        pipeline = pl.make_pipeline(grid)
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(CMS),
                                                          CMS)
        accuracy = []
        recall = []
        fscore = []
        auc = []

        # sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
        # for train_index, test_index in sss.split(X, CMS):

        # df_ = resample(X_all, n_samples=500, replace=False, stratify=y_train)
        # y_ = np.round(df_[:, -1])
        # df = df.select_dtypes(include=['float32', 'float64', 'int'])
        # X_ = df_[:, 0:df_.shape[1] - 1:1]
        # X_, y_ = ros.fit_sample(X_train, y_train)
        # X_, y_ = rus.fit_sample(X_, y_)
        X_, y_ = X_train, y_train
        # X_, y_ = smt.fit_resample(X_train, y_train)
        # X_, y_ = resample(X_, y_,stratify=y_,n_samples=1000)
        # weights = np.zeros([1, len(y_)])
        # weights[0, y_ == 0] = class_weights[0]
        # weights[0, y_ == 1] = class_weights[1]
        pipeline.fit(X_, y_)
        pipelines.append(pipeline)
        y_pred = pipeline.predict(X_test)

        # acc = accuracy_score(y_pred, y_test)
        # rec = recall_score(y_pred, y_test)
        # f1Score = f1_score(y_pred, y_test)
        # aucValue = roc_auc_score(y_pred, y_test)
        # accuracy.append(acc)
        # recall.append(rec)
        # fscore.append(f1Score)
        # auc.append(aucValue)
        #
        # print("Acc: {}".format(acc))
        # print("recal: {}".format(rec))
        # print("f1Score:{}".format(f1Score))
        # print("AUC : {}".format(aucValue))

    Xtrain = np.append(c2DataTrain, c1DataTrain, axis=0)
    CMSTrain = np.append(c2TargetTrain, c1TargetTrain, axis=0)
    X = np.append(c2DataTest, c1DataTest, axis=0)
    CMS = np.append(c2TargetTest, c1TargetTest, axis=0)
    y_pred_train_all = np.zeros([CMSTrain.shape[0], len(pipelines)])
    y_pred_test_all = np.zeros([CMS.shape[0], len(pipelines)])
    for i, pipelineItem in enumerate(pipelines):
        y_pred_train_all[:, i] = pipelineItem.predict(Xtrain)
        y_pred_test_all[:, i] = pipelineItem.predict(X)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(CMSTrain),
                                                      CMSTrain)
    weights = np.zeros([1, len(CMSTrain)])
    weights[0, CMSTrain == 0] = class_weights[0]
    weights[0, CMSTrain == 1] = class_weights[1]
    # param_grid = dict(scale_pos_weight=[1, 10, 25, 50, 75, 99, 100, 1000, 10000])
    tree_param = {'criterion': ['gini', 'entropy'], 'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
    # tree_param = {'bootstrap': [True, False],
    #               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #               'max_features': ['auto', 'sqrt'],
    #               'min_samples_leaf': [1, 2, 4],
    #               'min_samples_split': [2, 5, 10],
    #               'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600]}
    # grid = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=tree_param, n_iter=20, verbose=2, n_jobs=-1, scoring=make_scorer(roc_auc_score))

    grid = GridSearchCV(DecisionTreeClassifier(), param_grid=tree_param, scoring=make_scorer(f1_score))
    # grid = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, n_jobs=-1, scoring=make_scorer(roc_auc_score))

    grid.fit(np.append(y_pred_train_all, Xtrain, axis=1), CMSTrain)
    y_pred_test = grid.predict(np.append(y_pred_test_all, X, axis=1))
    # y_pred_test = y_pred_test_all[:,0]
    acc = accuracy_score(y_pred_test, CMS)
    rec = recall_score(y_pred_test, CMS)
    f1Score = f1_score(y_pred_test, CMS)
    aucValue = roc_auc_score(y_pred_test, CMS)

    # accuracy.append(acc)
    # recall.append(rec)
    # fscore.append(f1Score)
    # auc.append(aucValue)
    res.append(pd.DataFrame({"target": CMS, "prediction": y_pred_test}))

    if fold == split:
        with pd.ExcelWriter('resultFolds.xlsx') as writer:
            for kk in range(len(res)):
                res[kk].to_excel(writer, sheet_name='Fold{}'.format(kk))

    fold = fold + 1
    print("Test ==============================")
    print("Test Acc: {}".format(acc))
    print("Test recal: {}".format(rec))
    print("Test f1Score:{}".format(f1Score))
    print("Test AUC : {}".format(aucValue))
    print("************************************************")
    continue
