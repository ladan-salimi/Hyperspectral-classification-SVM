import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sn
from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

#Read IndianPins Dataset
def read_HSI():
    X= loadmat(r'C:\Users\ladan\Desktop\clustring\Indian_pines_corrected.mat')['indian_pines_corrected']
    y = loadmat(r'C:\Users\ladan\Desktop\clustring\Indian_pines_gt.mat')['indian_pines_gt']
    print(f"X shape: {X.shape}\ny shape: {y.shape}")
    return X, y
X, y = read_HSI()

#Visulaization of the N randomly selected bands over 200 is: 
n=int(input("enter the number of considered bands to show:"))
fig = plt.figure(figsize = (15, 8))
for i in range(1, n+1):
    k=int(n/2)
    fig.add_subplot(2,k, i)
    q = np.random.randint(X.shape[2])
    plt.imshow(X[:,:,q], cmap='nipy_spectral')
    plt.title(f'Band - {q}')
plt.savefig(r'C:\Users\ladan\Desktop\clustring\IP_Bands.png'IP_Bands.png')
            
#Visualize the Ground Truth
plt.figure(figsize=(10, 8))
plt.imshow(y, cmap='nipy_spectral')
plt.colorbar()
plt.savefig(r'C:\Users\ladan\Desktop\clustring\IP_GT.png')
plt.show()
            
# Function for Convert the .mat dataset into csv type
def extract_pixels(X, y):
    q = X.reshape(-1, X.shape[2])
    df = pd.DataFrame(data = q)
    df = pd.concat([df, pd.DataFrame(data = y.ravel())], axis=1)
    df.columns= [f'band{i}' for i in range(1, 1+X.shape[2])]+['class']
    df.to_csv(r'C:\Users\ladan\Desktop\clustring\Dataset.csv')
    return df
df = extract_pixels(X, y)
df.head()
            
#Statistics values for each cell
df.iloc[:, :-1].describe()

#Dimentionality Reduction:PCA
pca = PCA(n_components = 40)
dt = pca.fit_transform(df.iloc[:, :-1].values)#applying the PCA to all data
q = pd.concat([pd.DataFrame(data = dt), pd.DataFrame(data = y.ravel())], axis = 1)
q.columns = [f'PC-{i}' for i in range(1,41)]+['class']
q.head()
q.to_csv(r'C:\Users\ladan\Desktop\clustring\IP_40_PCA.csv', index=False)
            
#Display the bands after PCA:The first m principal components 
fig = plt.figure(figsize = (20, 10))
m=int(input("enter the number of m first bands after PCA bands to show:"))
for i in range(1, m+1):
    l=int(m/2)
    fig.add_subplot(2,l,i)
    plt.imshow(q.loc[:, f'PC-{i}'].values.reshape(145, 145), cmap='nipy_spectral')
    plt.axis('off')
    plt.title(f'Band - {i}')
plt.savefig(r'C:\Users\ladan\Desktop\clustring\IP_PCA_Bands.png')
            
#SVM Classifier
x = q[q['class'] != 0]
X = x.iloc[:, :-1].values
y = x.loc[:, 'class'].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11, stratify=y)
svm = SVC(C = 100, kernel = 'rbf', cache_size = 10*1024)
svm.fit(X_train, y_train)
ypred = svm.predict(X_test)
print(classification_report(y_test, ypred))

#Finally, the classification Map is shown below:
l=[]#creat a list
for i in range(q.shape[0]):
    if q.iloc[i, -1] == 0:
        l.append(0)
    else:
        l.append(svm.predict(q.iloc[i, :-1].values.reshape(1, -1))) 
clmap = np.array(l).reshape(145, 145).astype('float')
plt.figure(figsize=(10, 8))
plt.imshow(clmap, cmap='nipy_spectral')
plt.colorbar()




