from classes.FeatureSet import *
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pickle
# SKLearn classes
from sklearn.ensemble import *
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import KFold

#Load training data
fs_train = FeatureSet()
fs_train.load("../Data/Original Training Data/Arabidopsis thaliana/mirbase_arabidopsis_thaliana_combined_training_set.csv")
fs_train.add_instances("../Data/Original Training Data/Arabidopsis lyrata/mirbase_arabidopsis_lyrata_combined_training_set.csv")
fs_train.libsvm_scale()
fs_train.export("../Data/miPie mirBase Plant Classifier/mirbase_plant_training_set.csv")

X_train = np.array([[y for y in x.split(',')[:-1]] for x in open("../Data/miPie mirBase Plant Classifier/mirbase_plant_training_set.csv", 'r').readlines()[1:]]).astype(np.float64)
y_train = np.array([[x.split(',')[-1].strip('\n').strip('"')] for x in open("../Data/miPie mirBase Plant Classifier/mirbase_plant_training_set.csv", 'r').readlines()[1:]])

#Load test data
fs_test = FeatureSet()
fs_test.load("../Data/Original Training Data/Glycine max/mirbase_glycine_max_combined_training_set.csv")
fs_test.libsvm_scale()
fs_test.export("../Data/miPie mirBase Plant Classifier/mirbase_glycine_max_test_set.csv")

X_test = np.array([[y for y in x.split(',')[:-1]] for x in open("../Data/miPie mirBase Plant Classifier/mirbase_glycine_max_test_set.csv", 'r').readlines()[1:]]).astype(np.float64)
y_test = np.array([x.split(',')[-1].strip('\n').strip('"') for x in open("../Data/miPie mirBase Plant Classifier/mirbase_glycine_max_test_set.csv", 'r').readlines()[1:]])

miRNA_IDs = [x.split(',')[0] for x in open("../Data/Original Training Data/Glycine max/mirbase_glycine_max_combined_training_set.csv", 'r').readlines()] #names of miRNAs

#Initalize classifier
rf = RandomForestClassifier(n_estimators = 500)

#Initalize SMOTE algorithm
smoter = SMOTE(sampling_strategy='minority',random_state=42)

##############################
## TESTING ON SEPARATE DATASET
##############################

#Upsample training data
X_train_upsample, y_train_upsample = smoter.fit_resample(X_train,y_train)

#Fit the model on the upsampled training data
model = rf.fit(X_train_upsample, y_train_upsample)

# Score the model on the (non-upsampled) test data
model_proba = model.predict_proba(X_test)
precision, recall, _ = precision_recall_curve(y_test, model_proba[:,0],pos_label='miRNA')

#Plot PR curve for separate test set
f, axes = plt.subplots(1, 1, figsize=(10,5))
lab = 'AUC=%.4f' % (auc(recall, precision))
axes.step(recall, precision, label=lab, lw=2, color='black')
axes.set_xlabel('Recall')
axes.set_ylabel('Precision')
axes.set_title('Precision-Recall Curve for Test on Glycine Max')
axes.legend(loc='lower left', fontsize='small')
f.tight_layout()
f.savefig("../Data/miPie mirBase Plant Classifier/result.png")
f.show()

pickle.dump(model, open("../Data/miPie mirBase Plant Classifier/miPie_mirbase_plant_classifier.pkl", "wb" ))

##Output predictions
predictions = np.hstack((model_proba, np.atleast_2d(model.predict(X_test)).T, np.atleast_2d(y_test).T)) #Append actual class onto array
allPredictions = np.hstack((np.empty((predictions.shape[0], 1),dtype=float), predictions)) #Add column for miRNA name

for i in range(len(allPredictions)):
    allPredictions[i][0] = miRNA_IDs[i] #Append miRNA name data onto array

allPredictions = sorted(allPredictions, key = lambda x : x[2]) #Sort results in order of confidence for positive class

with open("../Data/miPie mirBase Plant Classifier/miPie_mirbase_glycine_max_pred.csv", 'w') as out:
    out.write('miRNA ID' + "," + 'miRNA Probability' + ',' + 'Pseudo Probability' + ',' + 'Model Prediction' + ',' + 'Label' + '\n')
    for p in allPredictions:
        out.write(str(p[0]) + "," + str(p[1])+ ',' + str(p[2]) + ',' + str(p[3]) + ',' + str(p[4]) + '\n')

###########################
## 10-FOLD CROSS-VALIDATION
###########################

scores, y_real, y_proba = [], [], []
kf = KFold(n_splits=10, random_state=42, shuffle=True)

f, axes = plt.subplots(1, 1, figsize=(10,5))

for i, (train_fold_index, val_fold_index) in enumerate(kf.split(X_train, y_train)):
    
    # Get the training data
    X_train_fold, y_train_fold = X_train[train_fold_index], y_train[train_fold_index]
    # Get the validation data
    X_val_fold, y_val_fold = X_train[val_fold_index], y_train[val_fold_index]

    # Upsample ONLY the data in the training section
    X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,y_train_fold)
    # Fit the model on the upsampled training data
    model = rf.fit(X_train_fold_upsample, y_train_fold_upsample)

    # Score the model on the (non-upsampled) validation data
    model_proba = model.predict_proba(X_val_fold)
    precision, recall, _ = precision_recall_curve(y_val_fold, model_proba[:,0],pos_label='miRNA')
    lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    axes.step(recall, precision, label=lab)
    y_real.append(y_val_fold)
    y_proba.append(model_proba[:,0])

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba,pos_label='miRNA')
lab = 'Overall AUC=%.4f' % (auc(recall, precision))
axes.step(recall, precision, label=lab, lw=2, color='black')
axes.set_title('Precision-Recall Curve for 10-Fold Cross-Validation')
axes.set_xlabel('Recall')
axes.set_ylabel('Precision')
axes.legend(loc='lower left', fontsize='small')

f.tight_layout()
f.savefig("../Data/miPie mirBase Plant Classifier/Cross-val PRC.png")
f.show()
