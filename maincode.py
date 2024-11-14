import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from scipy.io import loadmat


class HyperspectralImageClassifier:
    def __init__(self, image_path, gt_path):
        self.image_path = image_path
        self.gt_path = gt_path
        self.X = None
        self.y = None
        self.df = None
        self.pca_data = None
        self.svm_model = None

    def read_data(self):
        self.X = loadmat(self.image_path)['indian_pines_corrected']
        self.y = loadmat(self.gt_path)['indian_pines_gt']
        print(f"X shape: {self.X.shape}\ny shape: {self.y.shape}")

    def visualize_bands(self, n, save_path):
        fig = plt.figure(figsize=(15, 8))
        for i in range(1, n + 1):
            fig.add_subplot(2, int(np.ceil(n / 2)), i)
            band_index = np.random.randint(self.X.shape[2])
            plt.imshow(self.X[:, :, band_index], cmap='nipy_spectral')
            plt.title(f'Band - {band_index}')
        plt.savefig(save_path)
        plt.show()

    def visualize_ground_truth(self, save_path):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.y, cmap='nipy_spectral')
        plt.colorbar()
        plt.savefig(save_path)
        plt.show()

    def extract_pixels(self, save_path):
        reshaped_X = self.X.reshape(-1, self.X.shape[2])
        self.df = pd.DataFrame(data=reshaped_X)
        self.df['class'] = self.y.ravel()
        self.df.columns = [f'band{i}' for i in range(1, self.X.shape[2] + 1)] + ['class']
        self.df.to_csv(save_path, index=False)

    def apply_pca(self, n_components, save_path):
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(self.df.iloc[:, :-1].values)
        self.pca_data = pd.DataFrame(data=transformed_data, columns=[f'PC-{i}' for i in range(1, n_components + 1)])
        self.pca_data['class'] = self.df['class']
        self.pca_data.to_csv(save_path, index=False)

    def visualize_pca_bands(self, m, save_path):
        fig = plt.figure(figsize=(20, 10))
        for i in range(1, m + 1):
            fig.add_subplot(2, int(np.ceil(m / 2)), i)
            plt.imshow(self.pca_data[f'PC-{i}'].values.reshape(145, 145), cmap='nipy_spectral')
            plt.title(f'Band - {i}')
            plt.axis('off')
        plt.savefig(save_path)
        plt.show()

    def train_svm(self):
        filtered_data = self.pca_data[self.pca_data['class'] != 0]
        X = filtered_data.iloc[:, :-1].values
        y = filtered_data['class'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)
        self.svm_model = SVC(C=100, kernel='rbf', cache_size=10 * 1024)
        self.svm_model.fit(X_train, y_train)
        y_pred = self.svm_model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def generate_classification_map(self, save_path):
        classification_map = []
        for i in range(self.pca_data.shape[0]):
            if self.pca_data.iloc[i, -1] == 0:
                classification_map.append(0)
            else:
                classification_map.append(self.svm_model.predict(self.pca_data.iloc[i, :-1].values.reshape(1, -1)))
        clmap = np.array(classification_map).reshape(145, 145).astype('float')
        plt.figure(figsize=(8, 10))
        plt.imshow(clmap, cmap='nipy_spectral')
        plt.colorbar()
        plt.savefig(save_path)
        plt.show()


# Example Usage
if __name__ == "__main__":
    classifier = HyperspectralImageClassifier(
        image_path=r'path\Indian_pines_corrected.mat',
        gt_path=r'path\Indian_pines_gt.mat'
    )

    classifier.read_data()
    classifier.visualize_bands(n=5, save_path=r'path\IP_Bands.png')
    classifier.visualize_ground_truth(save_path=r'path\IP_GT.png')
    classifier.extract_pixels(save_path=r'path\Dataset.csv')
    classifier.apply_pca(n_components=40, save_path=r'path\IP_40_PCA.csv')
    classifier.visualize_pca_bands(m=5, save_path=r'path\IP_PCA_Bands.png')
    classifier.train_svm()
    classifier.generate_classification_map(save_path=r'path\IP_Classification_Map.png')
