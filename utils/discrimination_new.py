import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.spatial.distance import mahalanobis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib.colors import ListedColormap

save_latent_folder = '/data/sunyitong/code/ndm_1.5/detect'
if not os.path.exists(save_latent_folder):
    os.makedirs(save_latent_folder)

benign_data = np.load(os.path.join(save_latent_folder, 'benign.npy'))
harmful_data = np.load(os.path.join(save_latent_folder, 'sexual.npy'))
safe_noise_array = np.concatenate([benign_data], axis=0)
harm_noise_array = np.concatenate([harmful_data], axis=0)

X = np.concatenate([safe_noise_array, harm_noise_array], axis=0)
y = np.hstack([np.zeros(len(safe_noise_array)), np.ones(len(harm_noise_array))])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

n_components = 2
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)

safe_train_pca = X_train_pca[y_train == 0]
harm_train_pca = X_train_pca[y_train == 1]

svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
svm_clf.fit(X_train_pca, y_train)

X_test_pca = pca.transform(X_test)
svm_accuracy = svm_clf.score(X_test_pca, y_test)
test_probabilities = svm_clf.predict_proba(X_test_pca)[:, 1]
new_threshold = 0.5  
test_predictions = (test_probabilities > new_threshold).astype(int)
svm_accuracy = np.mean(test_predictions == y_test)

lda = LDA(n_components=1)  
X_train_lda = lda.fit_transform(X_train_pca, y_train)
X_test_lda = lda.transform(X_test_pca)
svm_clf_lda = SVC(kernel='rbf', probability=True, random_state=42)
svm_clf_lda.fit(X_train_lda, y_train)

test_predictions_lda = svm_clf_lda.predict(X_test_lda)
test_probabilities_lda = svm_clf_lda.predict_proba(X_test_lda)[:, 1]
new_threshold_lda = 0.5  
test_predictions_lda = (test_probabilities_lda > new_threshold_lda).astype(int)
svm_accuracy_lda = np.mean(test_predictions_lda == y_test)

print(f"\n[性能报告 - LDA投影后]")
print(f"| 方法               | 准确率  |")
print(f"|--------------------|---------|")
print(f"| SVM分类器 (线性核)  | {svm_accuracy_lda:.4f} |")

def discriminate_new_latent(latent,pca):
    new_test_data = latent
    new_test_pca = pca.transform(new_test_data)
    new_test_lda = lda.transform(new_test_pca)
    new_test_pred_lda = svm_clf_lda.predict(new_test_lda)
    new_test_probabilities_lda = svm_clf_lda.predict_proba(new_test_lda)[:, 1]
    print(new_test_pred_lda)
    
    return new_test_pred_lda

latent = np.load('/data/sunyitong/code/ndm_1.5/detect/i2p.npy')
discriminate_new_latent(latent,pca)