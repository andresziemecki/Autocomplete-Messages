import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

def LoadData(train_path, test_path, seed):
    
    # Load the data
    train = pd.read_csv("/home/andres/Documents/DataScience/DataSets/DigitRecognizer/train.csv")
    test = pd.read_csv("/home/andres/Documents/DataScience/DataSets/DigitRecognizer/test.csv")

    # There is a column with label "label" which is the number that represent all those pixels
    Y_train = train["label"]

    # Drop 'label' column so in X_train we'll have only the pixels values
    X_train = train.drop(labels = ["label"],axis = 1) 

    # free some space
    del train 
    # Anyway this will be deleted at the end of the function

    # We can use seaborn for statistical data visualization for Y_train
    # g = sns.countplot(Y_train)

    # See if we have similar counts for every numbers
    # Y_train.value_counts()

    # Check for null and missing values
    # X_train.isnull().any().describe()
    # test.isnull().any().describe()

    # There is no missing values in the train and test dataset. So we can safely go ahead.

    # NORMALIZATION

    # We perform a grayscale normalization to reduce the effect of illumination's differences.
    X_train = X_train / 255.0
    test = test / 255.0

    # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
    X_train = X_train.values.reshape(-1,28,28,1)
    test = test.values.reshape(-1,28,28,1)
    # Keras requires an extra dimension in the end which correspond to channels. 
    # MNIST images are gray scaled so it use only one channel. For RGB images, there is 3 channels.

    # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    Y_train = to_categorical(Y_train, num_classes = 10)

    # Split training and valdiation set
    # Set the random seed
    random_seed = seed


    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

    # I choosed to split the train set in two parts : a small fraction (10%) became the validation set which the model
    # is evaluated and the rest (90%) is used to train the model.

    # Since we have 42 000 training images of balanced labels (see 2.1 Load data), a random split of the train set
    #  doesn't cause some labels to be over represented in the validation set. Be carefull with some unbalanced dataset a simple random split could cause inaccurate evaluation during the validation.

    # To avoid that, you could use stratify = True option in train_test_split function 
    # (Only for >=0.17 sklearn versions).


    # Some examples:
    # g = plt.imshow(X_train[0][:,:,0])

    return X_train, X_val, Y_train, Y_val, test