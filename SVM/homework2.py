#Milan Kresovic - Erasmus student s266915
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import operator
from matplotlib import pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings("ignore")

"""
Choosing plots from which part of the homework should be shown
    0 - not showing
    1 - showing
"""
partOne = 1
partTwo = 1
partThree = 1
partFour = 1

"""
Number of folds
"""
K = 5

"""Splitting the data into sets"""
def splitData(Data, Class, trainProp, testProp, valProp = 0):
    sumProp = trainProp + valProp + testProp
    if valProp == 0:
        DataTrain, DataTest, ClassTrain, ClassTest = train_test_split(Data, Class, test_size = testProp/sumProp)
        return DataTrain, DataTest, ClassTrain, ClassTest
    else:
        DataTrain, DataTest, ClassTrain, ClassTest = train_test_split(Data, Class, test_size = testProp/sumProp)
        DataTrain, DataVal, ClassTrain, ClassVal = train_test_split(DataTrain, ClassTrain, test_size = valProp/sumProp)
        return DataTrain, DataVal, DataTest, ClassTrain, ClassVal, ClassTest

"""Make a list of different values for parameters in range minExp-maxExp"""
def listOfParameters(minExp, maxExp):
    listParam = [10**exp for exp in range(minExp, maxExp + 1)]
    return listParam

"""Applying SVM with list of parameter C onto the dataset"""
def CListWithSVM(C, kernelStr, DataTrain, DataTest, ClassTrain, ClassTest, Gamma = None):
    models = []
    accuracyList = []

    if not Gamma:
        Gamma = 'auto'
    
    for cParam in C:
        model = svm.SVC(C = cParam, gamma = Gamma, kernel = kernelStr)
        model.fit(DataTrain, ClassTrain.ravel())
        accuracy = model.score(DataTest, ClassTest)
        accuracyList.append(accuracy)
        models.append(model)
        #print("------------------\n\tAccuracy: {}\n\tC param: {}\n".format(accuracy, cParam))

    return models, accuracyList

"""Applying SVM with list of parameter C and Gamma onto the dataset"""
def GammaListWithSVM(C, Gamma, kernelStr, DataTrain, DataTest, ClassTrain, ClassTest):
    models = []
    accuracyList = []

    for gammaParam in Gamma:
        modelsC, accuracyListC = CListWithSVM(C, kernelStr, DataTrain, DataTest, ClassTrain, ClassTest, gammaParam)
        models.append(modelsC)
        accuracyList.append(accuracyListC)

    return models, accuracyList

"""Create a meshgrid by giving x and y data and h as the stepsize"""
def makeMeshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

"""Plot the decision boundaries by giving axes, model classifier and created meshgrid"""
def plotContours(ax, model, xx, yy, **params):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

"""Plotting decision boundaries for models"""
def plottingModels(Data, Class, models, title = None):
    # Set-up grid for plotting.
    X0, X1 = Data[:, 0], Data[:, 1]
    xx, yy = makeMeshgrid(X0, X1)
    
    if (len(models) != 1):
        titles = ('C=10^-3', 'C=10^-2', 'C=10^-1', 'C=10^0', 'C=10^1', 'C=10^2', 'C=10^3')
        fig, axes = plt.subplots(nrows = 3, ncols = 3)
        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        plt.tight_layout()
        i = 0
        
        for ax, model, title in zip(axes.flatten(), models, titles):
            if i == 6:
                ax = axes.flatten()[7]
            plotContours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=Class, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
            ax.set_ylabel("Sepal width")
            ax.set_xlabel("Sepal length")
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)
            if i == 6:
                break
            i = i + 1
        fig.delaxes(axes.flatten()[6])
        fig.delaxes(axes.flatten()[8])
            
    else:
        fig, ax = plt.subplots()
        #plt.tight_layout()
        
        plotContours(ax, models[0], xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=Class.ravel(), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_ylabel("Sepal width")
        ax.set_xlabel("Sepal length")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title) 

"""Plotting chart with value of accuracy for different parameters C"""
def plotAccuracy(accuracyList, title = None):
    fig, ax = plt.subplots()

    titles = ('10^-3', '10^-2', '10^-1', '10^0', '10^1', '10^2', '10^3')
    position = np.arange(len(titles))
    
    rects = ax.bar(position, accuracyList, align = 'center')
    autolabel(rects, ax)
    plt.ylim(0, 1)
    
    plt.xticks(position, titles)
    plt.ylabel("Accuracy")
    plt.xlabel("C parameters")
    plt.title(title)
    plt.draw()

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%.3f' % height,
                ha='center', va='bottom')

"""Taken from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html"""
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.set_ylabel("C")
    ax.set_xlabel("Gamma")
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

"""Plotting heatmap of values of accuracy for different parameters C and Gamma"""
def plotAccuracyHeatmap(accuracyList, cList, gammaList):
    fig, ax = plt.subplots()
    cListStr = [str(x) for x in cList]
    gammaListStr = [str(x) for x in gammaList]
    accuracyList = np.flip(np.array(np.transpose(np.matrix(accuracyList))), 0)
    cListStr = np.flip(cListStr, 0)
        
    im, cbar = heatmap(accuracyList, cListStr, gammaListStr, ax = ax, cmap = "GnBu")
    texts = annotate_heatmap(im)
    fig.tight_layout()

"""Find the parameter C and Gamma that give the best accuracy"""
def getBestCandGamma(accuracyList, cList, gammaList):
    accuracyMatrix = np.matrix(accuracyList)
    
    #transposing matrix to find best accuracy with smalles C parameter
    accuracyMatrix = np.transpose(accuracyMatrix)
    ind = np.unravel_index(np.argmax(accuracyMatrix, axis=None), accuracyMatrix.shape)
    bestAcc = accuracyMatrix[ind]
    bestC = cList[ind[0]]
    bestGamma = gammaList[ind[1]]
    return bestAcc, bestC, bestGamma

"""Finding average of all the matrices of accuracy values"""
def mean2D(accuracyListFolds, K, cList, gammaList):
    meanList = np.zeros(accuracyListFolds[0].shape)
    for k in range(K):
        meanList += accuracyListFolds[k]

    meanList = meanList / K
    return meanList

"""Applying K-fold cross-validation"""
def applyKFoldCV(DataFolds, ClassFolds, cList, gammaList, K):
    accuracyListFolds = []
    for foldId in range(K):
        DataVal = DataFolds[foldId]
        ClassVal = ClassFolds[foldId]
        
        DataTrainK = np.array([x for i,x in enumerate(DataFolds) if i != foldId])
        ClassTrainK = np.array([x for i,x in enumerate(ClassFolds) if i != foldId])

        DataTrainK = DataTrainK.reshape((-1, 2))
        ClassTrainK = ClassTrainK.reshape((-1, 2))
        
        models, accuracyList = GammaListWithSVM(cList, gammaList, kernelStr, DataTrainK, DataVal, ClassTrainK, ClassVal)
        accuracyListFolds.append(accuracyList)
    
    accuracyListFolds = np.array(accuracyListFolds)
    meanList = mean2D(accuracyListFolds, K, cList, gammaList)
    return meanList

print("Loading Iris dataset")
irisData = datasets.load_iris()

print("Reading first two features (spatal length, spatal width)")
Data = irisData.data[:, 0:2]
Class = irisData.target
print("Data and Class are loaded")

print("Splitting the data")
trainProp = 5
valProp = 2
testProp = 3
DataTrain, DataVal, DataTest, ClassTrain, ClassVal, ClassTest = splitData(Data, Class, trainProp, testProp, valProp)

print("Making list of parameters")
C = listOfParameters(-3, 3)
print(C)


"""
------------------------------PART ONE--------------------------------------
-------- Training linear SVM for different values of parameter C -----------
----------------------------------------------------------------------------
"""
print("\n-----------PART ONE----------\nLinear SVM for different values of C\n----------------------------\n\n")

print("1. Applying SVM onto the data\n")
kernelStr = "linear"
models, accuracyList = CListWithSVM(C, kernelStr, DataTrain, DataVal, ClassTrain, ClassVal)

if partOne:
    print("1.1 Plotting accuracy for different C parameters\n")
    plotAccuracy(accuracyList, "Part One")

print("2. Getting C parameter with the highest accuracy")
index, bestAcc = max(enumerate(accuracyList), key=operator.itemgetter(1))
bestC = C[index]
print("-----------------------\n\tHighest accuracy: %.2f" % bestAcc + "\n\tC parameter: {}\n".format(bestC))

if partOne:
    print("3. Plotting models decision boundaries\n")
    plottingModels(Data, Class, models, "Part One")

print("4. Applying SVM onto test data\n")
models, accuracyList = CListWithSVM([bestC], kernelStr, DataTrain, DataTest, ClassTrain, ClassTest)

if partOne:
    print("5. Plotting model decision boundaries for the best C parameter on test data\n")
    plottingModels(Data, Class, models, "Best C = {}".format(bestC) + "\nAccuracy = %.2f" % bestAcc)

"""
------------------------------PART TWO--------------------------------------
--- Training SVM with RBF kernel for different values of parameter C -------
----------------------------------------------------------------------------
"""
print("\n-----------PART TWO----------\n\tSVM with RBF kernel\n-----------------------------\n\n")

print("1. Applying SVM with RBF kernel onto the data\n")
kernelStr = "rbf"
models, accuracyList = CListWithSVM(C, kernelStr, DataTrain, DataVal, ClassTrain, ClassVal)

if partTwo:
    print("1.1 Plotting accuracy for different C parameters\n")
    plotAccuracy(accuracyList, "Part Two")

print("2. Getting C parameter with the highest accuracy")
index, bestAcc = max(enumerate(accuracyList), key=operator.itemgetter(1))
bestC = C[index]
print("-----------------------\n\tHighest accuracy: %.2f" % bestAcc + "\n\tC parameter: {}\n".format(bestC))

if partTwo:
    print("3. Plotting models decision boundaries\n")
    plottingModels(Data, Class, models, "Part Two")

print("4. Applying SVM onto test data\n")
models, accuracyList = CListWithSVM([bestC], kernelStr, DataTrain, DataTest, ClassTrain, ClassTest)

if partTwo:
    print("5. Plotting model decision boundaries for the best C parameter on test data\n")
    plottingModels(Data, Class, models, "Best C = {}".format(bestC) + "\nAccuracy = %.2f" % bestAcc)

"""
------------------------------PART THREE--------------------------------------
------------ Training SVM with RBF kernel, tuning C and Gamma ----------------
------------------------------------------------------------------------------
"""

print("\n-----------PART THREE----------\nTraining SVM with RBF kernel, tuning C and Gamma\n-----------------------------\n\n")

print("1. Applying SVM with RBF kernel onto the data with different values of C and Gamma\n")
C = listOfParameters(-3, 3)
Gamma = listOfParameters(-3, 3)
kernelStr = "rbf"
models, accuracyList = GammaListWithSVM(C, Gamma, kernelStr, DataTrain, DataVal, ClassTrain, ClassVal)

if partThree:
    print("1.1 Plotting accuracy for different C parameters and Gamma parameters\n")
    plotAccuracyHeatmap(accuracyList, C, Gamma)

print("2. Getting C and Gamma parameter with the highest accuracy (Choosing smaller C)")
bestAcc, bestC, bestGamma = getBestCandGamma(accuracyList, C, Gamma)

print("-----------------------\n\tHighest accuracy: %.2f" % bestAcc + "\n\tC parameter: {}\n\tGamma parameter: {}\n".format(bestC, bestGamma))

print("3. Applying SVM onto test data\n")
models, accuracyList = GammaListWithSVM([bestC], [bestGamma], kernelStr, DataTrain, DataTest, ClassTrain, ClassTest)

if partThree:
    print("4. Plotting model decision boundaries for the best C and Gamma parameter on test data\n")
    plottingModels(Data, Class, models[0], "Best C = {}, Best Gamma = {}".format(bestC, bestGamma) + "\nAccuracy = %.2f" % bestAcc)

"""
------------------------------PART FOUR--------------------------------------
Training SVM with RBF kernel, tuning C and Gamma with K-fold Cross Validation
-----------------------------------------------------------------------------
"""
print("\n-----------PART FOUR----------\nTraining SVM with RBF kernel, tuning C and Gamma with K-fold Cross Validation\n-----------------------------\n\n")

print("1. Merging training and validation sets")
DataTrain = np.concatenate([DataTrain, DataVal], axis = 0)
ClassTrain = np.concatenate([ClassTrain, ClassVal], axis = 0)
print("\tDataTrain size: {}\n\tClassTrain size: {}\n".format(DataTrain.shape, ClassTrain.shape))

print("2. Making 5 folds out of the data and class\n")
DataFolds = np.array_split(DataTrain, K)
ClassFolds = np.array_split(ClassTrain, K)

print("3. Performing 5-fold validation\n")
meanList = applyKFoldCV(DataFolds, ClassFolds, C, Gamma, K)

if partFour:
    print("3.1 Plotting mean accuracy for different C parameters and Gamma parameters\n")
    plotAccuracyHeatmap(meanList, C, Gamma)

print("4. Getting C and Gamma parameter with the highest mean accuracy (Choosing smaller C)")
bestAcc, bestC, bestGamma = getBestCandGamma(meanList, C, Gamma)

print("-----------------------\n\tHighest mean accuracy: %.2f" % bestAcc + "\n\tC parameter: {}\n\tGamma parameter: {}\n".format(bestC, bestGamma))

print("5. Applying SVM onto test data\n")
models, accuracyList = GammaListWithSVM([bestC], [bestGamma], kernelStr, DataTrain, DataTest, ClassTrain, ClassTest)

if partFour:
    print("6. Plotting model decision boundaries for the best C and Gamma parameter determined using K-fold cross-validation on test data\n")
    plottingModels(Data, Class, models[0], "Best C = {}, Best Gamma = {}".format(bestC, bestGamma) + "\nAccuracy = %.2f" % bestAcc)

plt.show()
