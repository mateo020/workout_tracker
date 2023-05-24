import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
from rf_model import ClassificationRf
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


df = pd.read_pickle("../../data/interim/03_data_features.pkl")
# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "category", "set"], axis=1)

x = df_train.drop("label", axis=1)
y= df_train["label"]

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25,random_state=42 )


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

basic_features = ["acc_x", "acc_y", "acc_z", "gye_x", "gyr_y", "gyr_z", ]
square_features = ["acc_r","gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
times_features = [f for f in df_train.columns if "_temp_" in f]
freq_featues = [f for f in df_train.columns if (("_freq" in f) or ("_pse" in f))]
cluster_featues = ["cluster"]

print("basic featues:", len(basic_features))
print("square featues:", len(square_features))
print("PCA features: ", len(pca_features))
print("Times featues:", len(times_features))
print("freqeuncy featues:", len(freq_featues))
print("cluster features:", len(cluster_featues))

features_set_1 = list(set(basic_features))
features_set_2 =list(set(basic_features + square_features + pca_features))
features_set_3 = list(set(features_set_2 + times_features))
features_set_4 = list(set(features_set_3 + freq_featues + cluster_featues))
# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()

max_features = 10
selected_featues, ordered_features, ordered_scores = learner.forward_selection(max_features, X_train=X_train, y_train=y_train)


plt.figure(figsize=(10,5))
plt.plot(np.arange(1,max_features+1,1),ordered_scores)
plt.xlabel("number of featues")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1,max_features + 1, 1))
plt.show()



# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [
    features_set_1,
    features_set_2,
    features_set_3,
    features_set_4,
    selected_featues
]

feature_names = [
    "features_set_1",
    "features_set_2",
    "features_set_3",
    "features_set_4",
    "selected_featues"
]

selected_featues = [
'acc_y_freq_0.0_Hz_ws_14',
 'gyr_r_freq_0.0_Hz_ws_14',
 'duration',
 'acc_z_freq_0.0_Hz_ws_14',
 'acc_r_max_freq',
 'acc_x_freq_1.786_Hz_ws_14',
 'acc_x_freq_2.143_Hz_ws_14',
 'acc_r_freq_0.357_Hz_ws_14',
 'acc_y',
 'acc_x_max_freq'
    
]

iterations = 1
score_df =pd.DataFrame()



for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])
    
    


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

score_df.sort_values(by="accuracy", ascending=False)
plt.figure(figsize=(10,10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("model")
plt.ylabel("accuracy")
plt.ylim(0.7,1)
plt.legend(loc="lower right")
plt.show()

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_train_prob_y,
    
) = learner.random_forest(
    X_train[features_set_4], y_train, X_test[features_set_4], gridsearch=True
)
accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)


# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()



# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

participant_df = df.drop(["set", "category"], axis=1)
X_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]


X_train = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)
y_train = participant_df[participant_df["participant"] == "A"]["label"]


X_train = X_train.drop(["participant"], axis=1)
X_test = X_test.drop(["participant"], axis=1)



# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_train_prob_y,
    
) = learner.random_forest(
    X_train[features_set_4], y_train, X_test[features_set_4], gridsearch=True
)
accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)


# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

# --------------------------------------------------------------
# export model
# --------------------------------------------------------------
ref_col = x.columns
target = "label"

learner2 = ClassificationRf()

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_train_prob_y,
    model
    
) = learner2.random_forest(
    X_train[features_set_4], y_train, X_test[features_set_4], gridsearch=True
)


joblib.dump(value=[learner,ref_col,target], filename="../../models/model.pkl")
#-----------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import copy


def random_forest(
      
        train_X,
        train_y,
        test_X,
        n_estimators=10,
        min_samples_leaf=5,
        criterion="gini",
        print_model_details=False,
        gridsearch=True,
    ):

        if gridsearch:
            tuned_parameters = [
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "n_estimators": [10, 50, 100],
                    "criterion": ["gini", "entropy"],
                }
            ]
            rf = GridSearchCV(
                RandomForestClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
            )

        # Fit the model

        rf.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        pred_prob_training_y = rf.predict_proba(train_X)
        pred_prob_test_y = rf.predict_proba(test_X)
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

        if print_model_details:
            ordered_indices = [
                i[0]
                for i in sorted(
                    enumerate(rf.feature_importances_), key=lambda x: x[1], reverse=True
                )
            ]
            print("Feature importance random forest:")
            for i in range(0, len(rf.feature_importances_)):
                print(
                    train_X.columns[ordered_indices[i]],
                )
                print(
                    " & ",
                )
                print(rf.feature_importances_[ordered_indices[i]])

        return (
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
            rf
        )

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_train_prob_y,
    model
    
) = random_forest(
    X_train[features_set_4], y_train, X_test[features_set_4], gridsearch=True
)

ref_cols = features_set_4


joblib.dump(value=[model, ref_cols, target], filename="../../models/model.pkl")
prediction = model.predict(X_test[features_set_4]) 


