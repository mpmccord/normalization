import os
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Gets the file path given a file name in the format myFile.filetype
# Note: this assumes that the file is in the same directory as main.py
def getPath(myfile):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, myfile)
    return file_path

# Creates a numpy array given a filename
# This assumes that there is one header line to skip and that
# the file is a csv or otherwise separated by commas
def arr(myFile):
    arr = np.genfromtxt(getPath(myFile), skip_header=1, delimiter=',')
    ones = np.ones( (arr.shape[0],arr.shape[1]) )
    A = np.hstack( (arr, ones) )
    return arr

# Gets the first row of the file and returns a list of the names of the columns
# Assumes that the first row is separated by commas
def getLabels(filename):
    myFile = open(getPath(filename), "r")
    labels = myFile.readline()
    labels = labels.split(",")
    myFile.close()
    return labels
# Note: this requires
def pairplot(arr, labels, X):
    df = pd.DataFrame(arr, columns=labels)
    sns.pairplot(df, x_vars=labels, y_vars=X)
    plt.show()

def scatterplot(arr, c1, c2, labels):
    plt.xlabel(labels[c1])
    plt.ylabel(labels[c2])
    sns.scatterplot(arr[c1], arr[c2])
    plt.show()
def scaleTranslate(arr, transMatrix, scaleMatrix):
    size = arr.shape[1]
    T = np.eye(size)
    T[0:-1, -1] = -transMatrix[0,0:-1]

    S = np.eye(size)
    S[0:-1,0:-1] = S[0:-1,0:-1] * (1/scaleMatrix[0,0:-1])
    N = S @ T
    return N
def normRange(arr):
     mins = np.min(arr, axis=0).reshape( (1, arr.shape[1]) )
     maxs = np.max(arr, axis=0).reshape( (1, arr.shape[1]) )
     ranges = ( maxs - mins ).reshape( (1, arr.shape[1]) )
     return arr @ scaleTranslate(arr, mins, ranges)
def normZScore(arr):
    mu = np.mean(arr, axis=0).reshape(1, arr.shape[1])
    sigma = np.std(arr, axis=0).reshape(1, arr.shape[1])
    return arr @ scaleTranslate(arr, mu, sigma)
def plot_cov(arr):
    cov_matrix = np.cov(arr, rowvar=False)
    plt.imshow(cov_matrix)
    plt.show()
def showStats(arr):
    print("Standard Deviation: ")
    std = np.std(arr, axis=0)
    print(std)
    plt.plot(std, '--or')
    print("Covariance: ")
    cov = np.cov(arr, rowvar=False)
    plt.plot(cov, '--ob')
    print("Mean")
    means = np.mean(arr, axis=0)
    plt.plot(means, '--om')
    
"""
# ((int) y coordinate and the labels of the columns
def scatterplot(arr, X, Y, labels):
    sns.scatterplot(arr[X][:], arr[Y][:])
    plt.xlabel(labels[X])
    plt.ylabel(labels[Y])
    plt.show()
    """
# Sample of my code given the forestfires.csv file
def graphForestFires():
    # Setting up the format for the files
    fileName = "forestfires.csv"
    labels = getLabels(fileName)
    fires = arr("forestfires.csv")
    area = labels[12]

    # Showing stats for original dataset
    showStats(fires)

    # Showing the original dataset and its covariance 
    pairplot(fires, labels, area)
    
    # plt.title("Covariance")
    # plot_cov(fires)
    
    # Now showing the dataset normalized by range
    normFires = normRange(fires)
    pairplot(normFires, labels, area)
    plot_cov(normFires)
    showStats(normFires)
    
    # Now normalizing by z-score
    normFires = normZScore(fires)
    pairplot(normFires, labels, area)
    plot_cov(normFires)
    showStats(normFires)

def main():
    graphForestFires()
if __name__ == '__main__':
    print(main())