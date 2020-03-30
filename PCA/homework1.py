#Milan Kresovic - Erasmus student s266915 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time

"""Choose image count:
0 - the whole data set
n - (no less than 60) from each category n/4 samples"""
imgCnt = 100
if imgCnt == 0:
  imgCntMat = len(glob.glob('images/*/*.jpg'))
else:
  imgCntMat = imgCnt
Data = np.zeros((imgCntMat, 154587))
Class = np.zeros((imgCntMat, 1))

"""Reading the dataset"""
def readImg(Data, Class, imgCnt = 0):
    i = 0
    if (imgCnt == 0):
        folder = ["dog", "guitar", "house", "person"]
        folderNum = 0
        while 1:
            if (folderNum == 4):
                break
            print("Next: " + folder[folderNum] + "\n")
            
            for path in glob.glob("images/" + folder[folderNum] + "/*.jpg"):
                Data[i] = np.asarray(Image.open(path)).ravel()
                Class[i] = folderNum
                i = i + 1
            
            folderNum = folderNum + 1
    else:
        folder = ["dog", "guitar", "house", "person"]
        folderNum = 0
        while 1:
            if (folderNum == 4):
                break

            for path in glob.glob("images/" + folder[folderNum] + "/*.jpg"):
                Data[i] = np.asarray(Image.open(path)).ravel()
                Class[i] = folderNum
                i = i + 1
                if (i % (imgCnt / 4) == 0):
                    folderNum = folderNum + 1
                    break
"""Applying PCA"""
def gettingComponents(DataStd):
    pca = PCA()
    DAC = pca.fit_transform(DataStd) #getting all the principal components
    D60C = DAC.copy()[:, :60]
    D6C = DAC.copy()[:, :6]
    D2C = DAC.copy()[:, :2]
    D6LastC = DAC.copy()[:, -6:]
    return DAC, D60C, D6C, D2C, D6LastC, pca

"""Reconstructing the data with using some of the principal components"""
def reconstruction(DAC, D60C, D6C, D2C, D6LastC, Data, DataStd):
    DAC_R = PCA().fit(DataStd).inverse_transform(DAC)
    DAC_R = dataRstd(DAC_R, Data)
    
    D60C_R = PCA(60).fit(DataStd).inverse_transform(D60C)
    D60C_R = dataRstd(D60C_R, Data)

    D6C_R = PCA(6).fit(DataStd).inverse_transform(D6C)
    D6C_R = dataRstd(D6C_R, Data)

    D2C_R = PCA(2).fit(DataStd).inverse_transform(D2C)
    D2C_R = dataRstd(D2C_R, Data)

    D6LastC_R = PCA(6).fit(DataStd).inverse_transform(D6LastC)
    D6LastC_R = dataRstd(D6LastC_R, Data)
    return DAC_R, D60C_R, D6C_R, D2C_R, D6LastC_R

"""Standardization of the data"""
def dataStandardization(Data):
    DataStd = (Data - np.mean(Data))/np.std(Data)
    return DataStd

"""Restandardization of the data"""
def dataRstd(R, Data):
    R = (R * np.std(Data)) + np.mean(Data)
    return R

def unraveling(R, imgNo):
    img = np.reshape(R[imgNo], (227,227,3)).astype(int)
    return img

"""Plotting reconstructed images"""
def plotImg(DAC_R, D60C_R, D6C_R, D2C_R, D6LastC_R, imgNo):
    plt.figure(figsize = (10, 10))
    
    imgAll = unraveling(DAC_R, imgNo)
    plt.subplot(1, 5, 1)
    plt.imshow(imgAll)
    plt.title("All Components")
    
    img60 = unraveling(D60C_R, imgNo)
    plt.subplot(1, 5, 2)
    plt.imshow(img60)
    plt.title("60 Components")
    
    img6 = unraveling(D6C_R, imgNo)
    plt.subplot(1, 5, 3)
    plt.imshow(img6)
    plt.title("6 Components")
    
    img2 = unraveling(D2C_R, imgNo)
    plt.subplot(1, 5, 4)
    plt.imshow(img2)
    plt.title("2 Components")
    
    img6Last = unraveling(D6LastC_R, imgNo)
    plt.subplot(1, 5, 5)
    plt.imshow(img6Last)
    plt.title("Last 6 Components")
    
    plt.tight_layout()
    plt.draw()

"""Plotting principal components for different classes"""
def plotClasses(DAC):
    plt.figure(figsize = (10, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(DAC[:, 0], DAC[:, 1], c = Class.ravel())
    plt.title("1st and 2nd component")

    plt.subplot(2, 2, 2)
    plt.scatter(DAC[:, 2], DAC[:, 3], c = Class.ravel())
    plt.title("3rd and 4th component")

    plt.subplot(2, 2, 3)
    plt.scatter(DAC[:, 9], DAC[:, 10], c = Class.ravel())
    plt.title("10th and 11th component")

    plt.tight_layout()
    plt.draw()

"""Plotting cumulative explained variance ratio """
def choosingNumOfComponents(pca):
    plt.figure(figsize = (7, 7))
    
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of PC')
    plt.ylabel('cumulative explained variance')
    plt.grid()
    
    plt.tight_layout()
    plt.draw()

"""Naive Bayes classifier"""
def NBC(datas, gnbModels):
    for data in datas:
        DataTrain, DataTest, ClassTrain, ClassTest = train_test_split(data, Class, test_size = 0.2)

        startTime = time.time()
        gnb = GaussianNB()
        gnb.fit(DataTrain, ClassTrain.ravel())
        ClassPred = gnb.predict(DataTest)
        endTime = time.time()

        gnbModels.append(gnb)
        print("\n-----------------------\n\tAccuracy: {}\n\tTime needed: {}\n".format(metrics.accuracy_score(ClassTest, ClassPred), endTime - startTime))

"""Create a meshgrid by giving x and y data and h as the stepsize"""
def makeMeshgrid(x, y, h=.5):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

"""Plot the decision boundaries by giving axes, model classifier and created meshgrid"""
def plotContours(ax, gnbModel, xx, yy, **params):
    Z = gnbModel.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

"""Plotting the decision boundaries of the classifier"""
def plottingPrediction(data, Class, gnbModel, title1, title2):
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Set-up grid for plotting.
    X0, X1 = data[:, 0], data[:, 1]
    xx, yy = makeMeshgrid(X0, X1)

    plotContours(ax, gnbModel, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=Class.ravel(), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel(title2 + " PC")
    ax.set_xlabel(title1 + " PC")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title1 + " and " + title2 + " principal component")

readImg(Data, Class, imgCnt)

DataStd = dataStandardization(Data)
print("Standardization of the data\n")
print(DataStd.shape)

print("Getting matrices with different number of components")
DAC, D60C, D6C, D2C, D6LastC, pca = gettingComponents(DataStd)
print("Shapes of all the matrices with different number of components:\n\tFirst 60 {}\n\tFirst 6 {}\n\tFirst 2 {}\n\tLast 6 {}\n\tAll {}\n".format(D60C.shape, D6C.shape, D2C.shape, D6LastC.shape, DAC.shape))

print("Reconstructing images")
DAC_R, D60C_R, D6C_R, D2C_R, D6LastC_R = reconstruction(DAC, D60C, D6C, D2C, D6LastC, Data, DataStd)
print("Reconstructed images\n")

print("Plotting reconstruction")
imgNo = 1
plotImg(DAC_R, D60C_R, D6C_R, D2C_R, D6LastC_R, imgNo)

print("Plotting visualization of the dataset\n")
plotClasses(DAC)

print("Choosing the number of the components\n")
choosingNumOfComponents(pca)

datas = (D2C, DAC[:, 2:4], DataStd)
gnbModels = []
NBC(datas, gnbModels)

print("Plotting decision boundaries for model\n")
plottingPrediction(datas[0], Class, gnbModels[0], "1st", "2nd")
plt.show()
