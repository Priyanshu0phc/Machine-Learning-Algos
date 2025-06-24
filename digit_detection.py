#importing Libraries required.
import numpy as np
import struct
import matplotlib.pyplot as plt


# Function to load MNIST images
def load_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))  # Read header
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return images

# Function to load MNIST labels
def load_labels(filename):
    with open(filename, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))  # Read header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load train data
train_images = load_images("train-images.idx3-ubyte")
train_labels = load_labels("train-labels.idx1-ubyte")

# Load test data
test_images = load_images("t10k-images.idx3-ubyte")
test_labels = load_labels("t10k-labels.idx1-ubyte")
# Select only labels 0, 1, and 2
train_filter = np.isin(train_labels, [0, 1, 2])
test_filter = np.isin(test_labels, [0, 1, 2])

# Apply filter to images and labels
train_images_filtered = train_images[train_filter]
train_labels_filtered = train_labels[train_filter]

test_images_filtered = test_images[test_filter]
test_labels_filtered = test_labels[test_filter]

# After revoming all non 0,1 and 2 images..

def sample_class_data(images, labels, class_label, sample_size=100):
    class_indices = np.where(labels == class_label)[0]  # Get indices for class
    sampled_indices = np.random.choice(class_indices, sample_size, replace=False)  # Randomly select
    return images[sampled_indices], labels[sampled_indices]

# Sample 100 images per class for train
train_images_final = []
train_labels_final = []
for digit in [0, 1, 2]:
    imgs, lbls = sample_class_data(train_images_filtered, train_labels_filtered, digit, 100)
    train_images_final.append(imgs)
    train_labels_final.append(lbls)

# Sample 100 images per class for test
test_images_final = []
test_labels_final = []
for digit in [0, 1, 2]:
    imgs, lbls = sample_class_data(test_images_filtered, test_labels_filtered, digit, 100)
    test_images_final.append(imgs)
    test_labels_final.append(lbls)

# Convert lists to numpy arrays
train_images_final = np.vstack(train_images_final)
train_labels_final = np.hstack(train_labels_final)
test_images_final = np.vstack(test_images_final)
test_labels_final = np.hstack(test_labels_final)

#Normalizing..
train_images_final = train_images_final / 255.0
test_images_final = test_images_final / 255.0

print("Data Loaded and preprocessed sucessfully !")
print("No Of Training data points, selected for each class: ",100)
print("Total number of Traing images, selected across all classes; ",len(test_images_final))




#Step-2
# Maximum Likelihood Estimation

def compute_mean(images):
    num_samples,num_features=images.shape 
    mean_vector =np.zeros(num_features)
    for i in range(num_samples):
        mean_vector += images[i]  #Sum of all image vectors

    mean_vector /= num_samples  #Divide by number of samples to get the mean
    return mean_vector

def compute_covariance(images, mean_vector):
    num_samples, num_features =images.shape
    centered_data=images-mean_vector #Centered data
    covariance_matrix =(centered_data.T @ centered_data)/num_samples-1 #Applied formula

    return covariance_matrix

# Function to compute MLE estimates (mean and covariance) for each class
def compute_mle(images, labels, classes=[0, 1, 2]):
    class_means = {} #Dictionaries to store, means of each class
    class_covariances = {} #Dictionary to store, covariance matrix of each class

    for digit in classes:
        # Select images belonging to the class
        class_images = images[labels == digit]

        # Compute mean vector
        mean_vector = compute_mean(class_images)
        # Compute covariance matrix
        covariance_matrix = compute_covariance(class_images, mean_vector)

        class_means[digit] = mean_vector
        class_covariances[digit] = covariance_matrix

    return class_means, class_covariances

# Compute MLE estimates for the training set
train_means, train_covariances = compute_mle(train_images_final, train_labels_final)
#MLE Parameters, computed..




#Step-3
##PCA

X = train_images_final.T  # Transpose so that each column is an image (784 x 300)
#Compute mean across all 300 training images
mu = compute_mean(train_images_final)  
# Center the data
Xc = X - mu[:, np.newaxis]  # Shape: (784, 300)

# Compute the covariance matrix using matrix multiplication
S = (Xc @ Xc.T)/(X.shape[1]-1)  # Shape: (784, 784)

#getting eigenvector, eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(S)  # Solves S v = λ v
#Sorting eigenValues, and corresponding eigenvectors..
sorted_indices = np.argsort(eigenvalues)[::-1]  # indices that would sort the eigenvalues in Descending order
eigenvalues = eigenvalues[sorted_indices] #Sorting eigenvalues
eigenvectors = eigenvectors[sorted_indices] #Shuffling eigenvector accordingly


#Deciding how many principal components to take, to retain 95% variance
# making Cumulative variance 
cumulative_variance = np.cumsum(eigenvalues)/np.sum(eigenvalues)
num_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1  # Find first index(One based Indexing) > 95%
Up = eigenvectors[:, :num_components_95] #Take only required components
print("Number of Principal Components selected for 95% Variance: ",num_components_95)
#Reducing dimentions

Y = np.dot(Up.T, Xc) #As stated in assingment
#Transformed, centered data matrix

#applying transformation to first test image.. 
x_test_c=test_images_final[0]-mu
y_test=np.dot(Up.T,x_test_c) # Example transformation




# Step-4: Fisher's Discriminant Analysis (FDA)
def compute_accuracy(predictions, labels):
    correct = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            correct += 1
    ans=(correct / len(labels))*100
    return round(ans,2)

def compute_scatter_matrices(images, labels, classes=[0, 1, 2]):
    overall_mean = compute_mean(images)
    S_B = np.zeros((images.shape[1], images.shape[1]))  # Between-class scatter matrix
    S_W = np.zeros((images.shape[1], images.shape[1]))  # Within-class scatter matrix
    
    for c in classes:
        class_images = images[labels == c]
        mean_c = compute_mean(class_images)
        S_B += len(class_images) * np.outer(mean_c - overall_mean, mean_c - overall_mean)
        S_W += compute_covariance(class_images, mean_c) * (len(class_images) - 1)
    
    return S_B, S_W

S_B, S_W = compute_scatter_matrices(train_images_final, train_labels_final)

eigenvalues_fda, eigenvectors_fda = np.linalg.eigh(np.linalg.pinv(S_W) @ S_B)
sorted_indices_fda = np.argsort(eigenvalues_fda)[::-1]
W_fda = eigenvectors_fda[sorted_indices_fda[:2]].T  # Select top 2 eigenvectors for visualization


# Project training and test data onto FDA components
train_images_fda = (train_images_final)@ W_fda
test_images_fda = (test_images_final)@ W_fda


# Defining LDA, and QDA functions..
def lda_train(images, labels, classes=[0, 1, 2]):
    means = {}
    priors = {}
    covariance=np.zeros((images.shape[1], images.shape[1]))
    
    for c in classes:
        class_images = images[labels == c]
        means[c] = compute_mean(class_images)
        priors[c] = len(class_images) / len(labels)
        covariance += compute_covariance(class_images, means[c]) * (len(class_images) - 1)
    
    covariance /= len(labels) - len(classes)
    return means, priors, covariance

def lda_predict(images, means, priors, covariance, classes=[0, 1, 2]):
    inv_cov = np.linalg.inv(covariance)
    predictions = []
    for x in images:
        scores = {c: -0.5 * (x - means[c]).T @ inv_cov @ (x - means[c]) + np.log(priors[c]) for c in classes}
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions)

means_lda, priors_lda, cov_lda = lda_train(train_images_fda, train_labels_final)
pred_train_lda = lda_predict(train_images_fda, means_lda, priors_lda, cov_lda)
pred_test_lda = lda_predict(test_images_fda, means_lda, priors_lda, cov_lda)



def qda_train(images, labels, classes=[0, 1, 2], reg_param=1e-4):
    means = {}
    priors = {}
    covariances = {}
    for c in classes:
        # Select images for class c
        class_images = images[labels == c]
        # Compute mean for class c
        means[c] = compute_mean(class_images)
        # Compute prior for class c
        priors[c] = len(class_images) / len(labels)
        # Compute covariance and add regularization for numerical stability
        covariances[c] = compute_covariance(class_images, means[c]) + np.eye(class_images.shape[1]) * reg_param
        
    return means, priors, covariances


def qda_predict(images, means, priors, covariances, classes=[0, 1, 2], min_det=1e-10):
    predictions = []
    for x in images:
        scores = {}
        for c in classes:
            cov_inv = np.linalg.inv(covariances[c])
            det_cov = np.linalg.det(covariances[c])
            det_cov = max(det_cov, min_det)
            diff = x - means[c]
            # Compute the QDA discriminant function:
            # g_c(x) = -0.5 * log(|Sigma_c|) - 0.5*(x-mu_c)^T Sigma_c^{-1} (x-mu_c) + log(prior)
            scores[c] = -0.5 * np.log(det_cov) - 0.5 * diff.T @ cov_inv @ diff + np.log(priors[c])
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions)


# Apply FDA on train and test sets, compute classification accuracy for LDA and QDA
means_fda_lda, priors_fda_lda, cov_fda_lda = lda_train(train_images_fda, train_labels_final)
pred_train_fda_lda = lda_predict(train_images_fda, means_fda_lda, priors_fda_lda, cov_fda_lda)
pred_test_fda_lda = lda_predict(test_images_fda, means_fda_lda, priors_fda_lda, cov_fda_lda)
print("LDA on FDA Train Accuracy:", compute_accuracy(pred_train_fda_lda,train_labels_final))
print("LDA on FDA Test Accuracy:", compute_accuracy(pred_test_fda_lda, test_labels_final))

means_fda_qda, priors_fda_qda, covs_fda_qda = qda_train(train_images_fda, train_labels_final)
pred_train_fda_qda = qda_predict(train_images_fda, means_fda_qda, priors_fda_qda, covs_fda_qda)
pred_test_fda_qda = qda_predict(test_images_fda, means_fda_qda, priors_fda_qda, covs_fda_qda)
print("QDA on FDA Train Accuracy:", compute_accuracy(pred_train_fda_qda, train_labels_final))
print("QDA on FDA Test Accuracy:", compute_accuracy(pred_test_fda_qda, test_labels_final))

# Step-5: PCA followed by LDA
num_components_90 = np.where(cumulative_variance >= 0.90)[0][0] + 1
Up_90 = eigenvectors[:, :num_components_90]
train_images_pca_90 = train_images_final @ Up_90
test_images_pca_90 = test_images_final @ Up_90
means_pca_lda, priors_pca_lda, cov_pca_lda = lda_train(train_images_pca_90, train_labels_final)
pred_train_pca_lda = lda_predict(train_images_pca_90, means_pca_lda, priors_pca_lda, cov_pca_lda)
pred_test_pca_lda = lda_predict(test_images_pca_90, means_pca_lda, priors_pca_lda, cov_pca_lda)
print("LDA on PCA (90%) Train Accuracy:", compute_accuracy(pred_train_pca_lda, train_labels_final))
print("LDA on PCA (90%) Test Accuracy:", compute_accuracy(pred_test_pca_lda, test_labels_final))



# Train LDA on PCA-transformed data (95% variance)
train_images_pca_95 = train_images_final @ Up   # Result: (300, num_components_95)
test_images_pca_95  = test_images_final  @ Up   # (300, num_components_95)

# Now, train LDA on the PCA-transformed training data.
means_pca_lda_95, priors_pca_lda_95, cov_pca_lda_95 = lda_train(train_images_pca_95, train_labels_final)
pred_train_pca_lda_95 = lda_predict(train_images_pca_95, means_pca_lda_95, priors_pca_lda_95, cov_pca_lda_95)
pred_test_pca_lda_95  = lda_predict(test_images_pca_95, means_pca_lda_95, priors_pca_lda_95, cov_pca_lda_95)

print("LDA on PCA (95%) Train Accuracy:", compute_accuracy(pred_train_pca_lda_95, train_labels_final))
print("LDA on PCA (95%) Test Accuracy:",  compute_accuracy(pred_test_pca_lda_95, test_labels_final))



# Use only first two principal components
Up_2 = eigenvectors[:,:2]
train_images_pca_2 = train_images_final @ Up_2
test_images_pca_2 = test_images_final @ Up_2
means_pca_2_lda, priors_pca_2_lda, cov_pca_2_lda = lda_train(train_images_pca_2, train_labels_final)
pred_train_pca_2_lda = lda_predict(train_images_pca_2, means_pca_2_lda, priors_pca_2_lda, cov_pca_2_lda)
pred_test_pca_2_lda = lda_predict(test_images_pca_2, means_pca_2_lda, priors_pca_2_lda, cov_pca_2_lda)
print("LDA on First Two PCA Components Train Accuracy:", compute_accuracy(pred_train_pca_2_lda, train_labels_final))
print("LDA on First Two PCA Components Test Accuracy:", compute_accuracy(pred_test_pca_2_lda, test_labels_final))


# Plotting 
plt.figure(figsize=(8, 6))
markers = ['o', 's', '^']
colors = ['r', 'g', 'b']
for cls, marker, color in zip([0, 1, 2], markers, colors):
    idx = np.where(train_labels_final == cls)
    plt.scatter(train_images_fda[idx, 0], train_images_fda[idx, 1], marker=marker, color=color, label=f'Class {cls}')
plt.title('FDA Transformed Training Data')
plt.xlabel('FDA Component 1')
plt.ylabel('FDA Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Visualization for PCA (using only first two components)
plt.figure(figsize=(8, 6))
for cls, marker, color in zip([0, 1, 2], markers, colors):
    idx = np.where(train_labels_final == cls)
    plt.scatter(train_images_pca_2[idx, 0], train_images_pca_2[idx, 1], marker=marker, color=color, label=f'Class {cls}')
plt.title('PCA Transformed Training Data (2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
