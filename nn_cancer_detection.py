from sys import argv
from itertools import cycle
import numpy as np
np.random.seed(3)
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate,\
                                    StratifiedKFold
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc,\
                            precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.disriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,\
                             BaggingClassifier
from sklearn.naive_bayes import GaussianNB

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from tensorflow import set_random_seed
set_random_seed(2)



def main(score_on_test=False):
    print("loading data set.")
    path = ("http://archive.ics.uci.edu/ml/machine-learning-databases/"
            "breast-cancer-wisconsin/breast-cancer-wisconsin.data")
    df = load_data(path)
    df = shuffle(df, random_state=1)
    X_train, Y_train, X_test, Y_test = make_train_test_splits(df)
    
    print("creating classifiers.")
    clfs = np.array([
            ["SVC", SVC(C=1, probability=True, random_state=1)],
            ["MLP", MLPClassifier(alpha=1, max_iter=300, random_state=1)],
            ["KNeighbors", KNeighborsClassifier(3)],
            ["QuadDiscAnalysis", QuadraticDiscriminantAnalysis()],
            ["DecisionTree", DecisionTreeClassifier(max_depth=5)],
            ["RandomForest", RandomForestClassifier(max_depth=5,
                                                    n_estimators=10,
                                                    max_features=1)],
            ["AdaBoost", AdaBoostClassifier()],
            ["GaussianProcess", GaussianProcessClassifier(1.0 * RBF(1.0),
                                                          random_state=1)],
            ["GaussianNB", GaussianNB()]
        ])
    
    print("scoring classifiers.")
    clf_scores_df = score_classifiers(clfs, (X_train, Y_train,
                                             X_test, Y_test))
    visualise_scores(clf_scores_df, "classifiers_scores")
    
    print("creating ensemble classifiers.")
    eclfs = []
    for clf in clfs:
        eclfs.append(create_bagging_clf(clf))
    eclfs = np.array(eclfs)
    print("scoring ensemble classifiers.")
    eclf_scores_df = score_classifiers(eclfs, (X_train, Y_train,
                                               X_test, Y_test))
    visualise_scores(eclf_scores_df, "ensemble_scores")
    
    print("scoring NN model.")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    cv_acc_scores = []
    cv_f1_scores = []
    for train, test in kfold.split(X_train, Y_train):
        model = nn_model()
        model.fit(X_train[train], one_hot(Y_train[train]), epochs=200,
                  verbose=0)
        cv_scores = model.evaluate(X_train[test], one_hot(Y_train[test]),
                                   verbose=0)
        cv_acc_scores.append(cv_scores[1])
        cv_f1_scores.append(cv_scores[2])
        
    
    nn_scores_df = pd.DataFrame([["NN model",
                                  np.mean(cv_f1_scores),
                                  np.mean(cv_acc_scores)
                                  ]], columns=['name', 'f1', 'accuracy'])
    best_clfs_scores_df = pd.concat([clf_scores_df, eclf_scores_df,
                                     nn_scores_df]).nlargest(3, 'f1')
    visualise_scores(best_clfs_scores_df, "cv_best_results")
    
    if score_on_test:
        print("scoring top models on test data.")
        test_scores = []
        trained_models = []
        models_scores = []
        for index, row in best_clfs_scores_df.iterrows():
            if row["name"] == "NN model":
                model = nn_model()
                model.fit(X_train, one_hot(Y_train), epochs=200, verbose=0)
                test_score = model.evaluate(X_test, one_hot(Y_test),
                                            verbose=0)
                test_acc_score = test_score[1]
                test_f1_score = test_score[2]
                trained_models.append([row["name"], model])

                predictions = model.predict(X_test)
                predictions = predictions[:, 1]
                models_scores.append([Y_test, predictions, row["name"]])
                
            else:
                all_clfs = np.concatenate([clfs, eclfs])
                clf = np.extract(all_clfs[:, 0] == row["name"],
                                 all_clfs[:, 1])[0]
                clf.fit(X_train, Y_train)
                y_pred = clf.predict(X_test)
                test_acc_score = accuracy_score(Y_test, y_pred)
                test_f1_score = f1_score(Y_test, y_pred)
                trained_models.append([row["name"], clf])

                if hasattr(clf, "decision_function"):
                    predictions = clf.decision_function(X_test)
                    models_scores.append([Y_test, predictions, row["name"]])
                else:
                    predictions = clf.predict_proba(X_test)
                    predictions = predictions[:, 1]
                    models_scores.append([Y_test, predictions, row["name"]])
                
            test_scores.append([row["name"], test_acc_score, test_f1_score])
        roc(models_scores)
        roc(models_scores, zoomed=True)
        precision_recall_curv(models_scores)
        precision_recall_curv(models_scores, zoomed=True)

        test_scores_df = pd.DataFrame(test_scores, columns=['name',
                                                            'test accuracy',
                                                            'test f1'])
        scores_df = best_clfs_scores_df.merge(test_scores_df, on='name')
        visualise_scores(scores_df, "top_model_scores")
        
        print("visualise decision bounds.")
        for (clf_name, clf) in trained_models:
            pca = PCA(n_components=2)
            X, Y = make_samples(df)
            X_2d = pca.fit_transform(X)
            #Y = one_hot(Y)####
            
            xx, yy = np.mgrid[
                    X_2d[:, 0].min() - .5 : X_2d[:, 0].max() + .5 : 0.2,
                    X_2d[:, 1].min() - .5 : X_2d[:, 1].max() + .5 : 0.2
                    ]
            grid = np.c_[xx.ravel(), yy.ravel()]
            if isinstance(clf, Sequential):
                predictions = clf.predict(pca.inverse_transform(grid))
                predictions = predictions[:, 1]
            elif hasattr(clf, "decision_function"):
                predictions = clf.decision_function(
                        pca.inverse_transform(grid))
            else:
                predictions = clf.predict_proba(pca.inverse_transform(grid))
                predictions = predictions[:, 1]
                
            probs = predictions.reshape(xx.shape)
            
            f, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                                  vmin=0, vmax=1)
            ax_c = f.colorbar(contour)
            ax_c.set_ticks([0, .25, 0.5, .75, 1])
            ax_c.ax.set_yticklabels(['ben', 'ben', "-", "mal", "mal"])
            X_train_2d = pca.transform(X_train)
            ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=Y_train,
                   edgecolors='k', marker="o", cmap="RdBu")
            
            X_test_2d = pca.transform(X_test)
            ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=Y_test,
                   edgecolors='k', marker="v", cmap="RdBu")
            
            ax.set(aspect="equal",
                   xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()),
                   xlabel="$X_1$", ylabel="$X_2$")
            ax.set_title(clf_name)
            plt.savefig("{}_decision_bound.png".format(clf_name))
            
    
    
def visualise_scores(scores_df, img_name):
    scores_df = pd.melt(scores_df, id_vars=['name']).sort_values(['variable',
                                                                  'value'])
    g = sns.factorplot(x='name', y='value', hue='variable', data=scores_df,
                       kind="bar", palette="muted", size=5, aspect=1.5)
    g.set(ylim=(0.88, 1))
    g.set_xticklabels(rotation=-70)
    
    #https://stackoverflow.com/a/39798852
    ax=g.ax
    for p in ax.patches:
        ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2.,
                                          p.get_height()),
             ha='center', va='center', fontsize=11, color='gray', rotation=90,
             xytext=(0, 20),
             textcoords='offset points')
    g.savefig("{}.png".format(img_name))
  
    
def score_classifiers(clfs, data):
    (X_train, Y_train, X_test, Y_test) = data
    scores = []
    scoring = ['f1', 'accuracy']
    for (clf_name, clf) in clfs:
        cv_scores = cross_validate(clf, X_train, Y_train, cv=5,
                                   scoring=scoring)
        clf_scores = [clf_name,
                      cv_scores['test_f1'].mean(),
                      cv_scores['test_accuracy'].mean()
                      ]
        scores.append(clf_scores)
    return pd.DataFrame(scores, columns=['name', 'f1', 'accuracy'])

def load_data(path):
    return pd.read_csv(path, index_col=0, na_values='?').fillna(0)

def make_train_test_splits(df):
    train, test = train_test_split(df, test_size=0.3)
    
    X_train, Y_train = make_samples(train)
    X_test, Y_test = make_samples(test)
    
    return X_train, Y_train, X_test, Y_test
    
def make_samples(df, normalize=True):
    X = df[df.columns[0:-1]].as_matrix().astype(float)
    Y = df[df.columns[-1]].as_matrix().astype(int)
    if normalize:
        X = X/10.0
    Y = np.array(list(map(lambda c: 0 if c==2 else 1, Y)))
    return X, Y

def create_bagging_clf(clf):
    eclf = BaggingClassifier(clf[1], n_estimators=10,
                                max_samples=0.7, max_features=3)
    return ["{}_esbl".format(clf[0]), eclf]

def nn_model():
    model = Sequential()
    model.add(Dense(100, input_shape=(9,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(2, activation='softmax'))
    # Dense(1, activation='sigmoid') were causing nan on f1
            
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                 epsilon=None, decay=0.001, amsgrad=False)
            
    model.compile(loss='categorical_hinge',
      optimizer=adam,
      metrics=['acc', keras_f1_score])
    
    return model


def one_hot(Y):
    Y = Y.reshape((-1, 1))
    Y = np.apply_along_axis(lambda c: [1, 0] if c==0 else [0, 1], 1, Y)
    return Y


def keras_f1_score(y_true, y_pred):

    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    if c3 == 0:
        return 0
    precision = c1 / c2
    recall = c1 / c3

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def roc(models_scores, zoomed=False, img_name="roc"):
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for (y_test, y_score, model_name), color in zip(models_scores, colors):
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color,
             label='ROC curve of {0} (area = {1:0.3f})'
             ''.format(model_name, roc_auc))
    
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    if zoomed:
        plt.xlim([-0.05, 0.4])
        plt.ylim([0.6, 1.05])
        img_name +=' (zoomed)'
    else:
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of top 3 models{}'
              .format(' (zoomed)' if zoomed else ''))
    plt.legend(loc="lower right")
    plt.savefig("{}.png".format(img_name))

    
def precision_recall_curv(models_scores, zoomed=False,
                          img_name="precision_recall"):
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    
    for (y_test, y_score, model_name), color in zip(models_scores, colors):
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        
        average_precision = average_precision_score(y_test, y_score)


        plt.step(recall, precision, color=color, alpha=1,
                 where='post', label='{0} (Average precision = {1:0.3f})'
             ''.format(model_name, average_precision))
        
        plt.fill_between(recall, precision, step='post', alpha=0.01,
                         color=color)
        
        rand_clf = np.count_nonzero(y_test)/y_test.size
        
        plt.plot([0, 1], [rand_clf, rand_clf], color='navy', linestyle='--')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    if zoomed:
        plt.xlim([0.4, 1.05])
        plt.ylim([0.4, 1.05])
        img_name +=' (zoomed)'
    else:
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
    
    plt.title('Precision-Recall curves of top 3 models {}'.format(' (zoomed)'
              if zoomed else ''))
    
    plt.legend(loc="lower left")
    plt.savefig("{}.png".format(img_name))



if __name__ == '__main__':
    score_on_test = False
    if '--score-on-test' in argv:
        score_on_test = True
    main(score_on_test)


