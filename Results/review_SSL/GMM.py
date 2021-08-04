import sys
import os
import numpy as np
from Utils.data_loader import load_joint_data_as_matrix

class SSGMM:
    """
    Implements the expectation-maximisation (EM) algorithm for the
    Gaussian mixture model (GMM). The algorithm is based on the
    pseudo-code described in the book by C. Bishop "Pattern Recognition
    and Machine Learning", chapter 9.
    """

    def __init__(self,
                 n_components,
                 means=None,
                 covariances=None,
                 mixing_probs=None,
                 epsilon=1e-6,
                 callback=None):
        """
        Arguments:
        n_components -- number of mixtures (components) to fit
        means -- (optional) initial array of mean vectors (numpy array of numpy arrays)
        covariances -- (optional) initial array of covariance matrices (numpy array of numpy arrays)
        mixing_probs -- (optional) initial vector (numpy array) of mixing probabilities
        epsilon -- (optional) convergence criterion
        """
        self.n_components = n_components
        self.means = means
        self.covariances = covariances
        self.mixing_probs = mixing_probs
        self.epsilon = epsilon
        self.__callback = callback

    def fit(self, features, y):
        """
        Fits a GMM into a set of feature data.
        Arguments:
        features -- input features data set
        """
        # Initialise
        n, _ = features.shape
        norm_densities = np.empty((n, self.n_components), np.float)
        responsibilities = np.empty((n, self.n_components), np.float)
        old_log_likelihood = 0
        self._initialise_parameters(features)

        it = 0
        while True:
            it += 1
            print("Iteration",it)
            # Compute normal densities
            for i in np.arange(n):
                x = features[i]

                for j in np.arange(self.n_components):
                    norm_densities[i][j] = self.multivariate_normal_pdf(x, self.means[j], self.covariances[j])

            # Estimate log likelihood
            log_vector = np.log(np.array([np.dot(self.mixing_probs.T, norm_densities[i]) for i in np.arange(n)]))
            log_likelihood = np.dot(log_vector.T, np.ones(n))

            self.call_back()

            # Check for convergence
            if np.absolute(log_likelihood - old_log_likelihood) < self.epsilon or i > 10:
                break

            # E-step: evaluate responsibilities
            for i in np.arange(n):
                x = features[i]
                denominator = np.dot(self.mixing_probs.T, norm_densities[i])
                if y[i] == -1:
                    for j in np.arange(self.n_components):
                        responsibilities[i][j] = self.mixing_probs[j] * norm_densities[i][j] / denominator
                else:
                    responsibilities[i][:] = 0
                    responsibilities[i][y[i]] = 1

            # M-step: re-estimate the parameters
            for i in np.arange(self.n_components):
                responsibility = (responsibilities.T)[i]

                # Common denominator
                denominator = np.dot(responsibility.T, np.ones(n))

                # Update mean
                self.means[i] = np.dot(responsibility.T, features) / denominator

                # Update covariance
                difference = features - np.tile(self.means[i], (n, 1))
                self.covariances[i] = np.dot(np.multiply(responsibility.reshape(n, 1), difference).T,
                                             difference) / denominator

                # Update mixing probabilities
                self.mixing_probs[i] = denominator / n

            old_log_likelihood = log_likelihood

    def cluster(self, features):
        """
        Returns a numpy array containing partitioned feature data. The
        distance measure used to compute the distance between a feature point
        and a Gaussian distribution is Mahanalobis distance.
        """
        # Initialise
        n, _ = features.shape
        partition = np.empty(n, np.int)
        distances = np.empty(self.n_components, np.float)
        cov_inverses = [np.linalg.inv(cov) for cov in self.covariances]

        # Assign each feature point to a Gaussian distribution
        for i in np.arange(n):
            x = features[i]

            # Compute Mahanalobis distances from each mixture
            for j in np.arange(self.n_components):
                distances[j] = np.dot(np.dot((x - self.means[j]).T, cov_inverses[j]), x - self.means[j])

            # Find index of the minimum distance, and assign to a cluster
            partition[i] = np.argmin(distances)

        return partition

    def call_back(self):
        if self.__callback:
            dct = {
                'mixing_probs': self.mixing_probs,
                'means': self.means,
                'covariances': self.covariances
            }
            self.__callback(dct)

    def multivariate_normal_pdf(self, x, mean, covariance):
        """
        Returns normal density value for an n-dimensional random
        vector x.
        """
        centered = x - mean
        cov_inverse = np.linalg.inv(covariance)
        cov_det = np.linalg.det(covariance)
        exponent = np.dot(np.dot(centered.T, cov_inverse), centered)
        return np.exp(-0.5 * exponent) / np.sqrt(cov_det * np.power(2 * np.pi, self.n_components))

    def _initialise_parameters(self, features):
        """
        Initialises parameters: means, covariances, and mixing probabilities
        if undefined.
        Arguments:
        features -- input features data set
        """
        if not self.covariances:
            n, m = features.shape

            # Shuffle features set
            indices = np.arange(n)
            np.random.shuffle(np.arange(n))
            features_shuffled = np.array([features[i] for i in indices])

            # Split into n_components subarrays
            divs = int(np.floor(n / self.n_components))
            features_split = [features_shuffled[i:i + divs] for i in range(0, n, divs)]

            if not self.covariances:
                covariances = []
                for i in np.arange(self.n_components):
                    covariances.append(np.cov(features_split[i].T))
                self.covariances = np.array(covariances)

        if not self.mixing_probs:
            self.mixing_probs = np.repeat(1 / self.n_components, self.n_components)

if __name__ == "__main__":
    num_samples = 100
    dataset = "MNIST"
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_joint_data_as_matrix(num_samples,dataset=dataset)

    y =np.random.randint(0,10,100)
    X = np.random.randn(100,784)  + y[:,np.newaxis]
    y[:50] = -1

    means = np.zeros((10,784))
    print((y[:,np.newaxis] == 0).shape)
    print(X.shape)
    for i in range(10):
        means[i] = np.mean(X[np.repeat(y[:,np.newaxis],784,axis=1) == i],axis=0)


    model = SSGMM(n_components=10,means=means)
    model.fit(X,y)
    u = "debug"



