# MLAI course projects

Projects done as part of MLAI course.

## PRINCIPAL COMPONENT ANALYSIS

For this project the dataset of 400 images of four different classes was used. Each image was of the size 227x227 and in the RGB format, so the total number of features in each image was 227x227x3. The data was loaded and PCA was applied on it. The result of PCA were principal components in order of significance, which were used to discuss how well is the image reconstructed using different number of principal components. Dataset was also projected onto the graph for different pairs of PC and further discussed for which PC is the variance the biggest.

* Link to the dataset: [*link*](http://www.google.com)

## SUPPORT VECTOR MACHINES

Dataset used for this project was Iris dataset from sklearn library. Dataset was split into training, validation and testing sets. This project includes:
- Finding optimal value of *parameter C* for *linear SVM* for this dataset
- Finding optimal values of *parameter C* and *Gamma parameter* for *SVM with RBF kernel* for this dataset
- Usage of *K-Fold cross-validation* for finding optimal values

## CONVOLUTIONAL NEURAL NETWORKS

For purpose of this project, the training and testing was done on the CIFAR 100 dataset. The first part is training fully connected neural network (FCNN), that is, network in which each neuron is connected to every neuron in the previous layer.

The second part is training of our CNN with different number of convolutional filters, with or without batch normalization (normalizing the inputs of each layer), but also trying wider fully connected layer or/and dropout on the fully connected layer. Beside this, different schemes of data augmentation (random horizontal flipping and random crop) were also tried.

At the end, state-of-the-art pre-trained model called *ResNet18* was trained with the best data augmentation to compare the results with our CNN.
