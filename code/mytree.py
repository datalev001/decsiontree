###hard decsion tree 
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
import pandas as pd

def gini_impurity(y):
    m = len(y)
    return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

def best_split(X, y):
    best_gini = 1.0
    best_idx, best_thr = None, None
    m, n = X.shape

    for idx in range(n):
        thresholds = np.unique(X[:, idx])
        for thr in thresholds:
            left_mask = X[:, idx] < thr
            right_mask = ~left_mask
            if sum(left_mask) == 0 or sum(right_mask) == 0:
                continue

            gini = (sum(left_mask) * gini_impurity(y[left_mask]) +
                    sum(right_mask) * gini_impurity(y[right_mask])) / m

            if gini < best_gini:
                best_gini = gini
                best_idx = idx
                best_thr = thr

    return best_idx, best_thr

class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        # Check for NaN or infinite values in the data
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Input data contains NaN values.")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError("Input data contains infinite values.")

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=gini_impurity(y),
            num_samples=X.shape[0],
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define cross-validation method
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize classifier
clf = DecisionTreeClassifier(max_depth=3)

# Perform cross-validation with cross_val_score
cv_scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')

# Print cross-validation scores
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")

# Manual cross-validation with debugging
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Train indices: {train_index}")
    print(f"Test indices: {test_index}")
    print(f"Accuracy: {accuracy}")






###########
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, entropy

class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, depth=10, min_samples_split=2, num_cut_points=100, split_way='entropy'):
        self.depth = depth
        self.min_samples_split = min_samples_split
        self.num_cut_points = num_cut_points  # Number of potential cut points
        self.split_way = split_way  # Criterion for splitting
        self.tree = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Ensure compatibility with scikit-learn
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=self.depth)
        return self

    def _build_tree(self, X, y, depth):
        if depth == 0 or len(X) < self.min_samples_split:
            return self._leaf_value(y)
        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return self._leaf_value(y)
        left_idx = X[:, feat_idx] < threshold
        right_idx = ~left_idx
        left_tree = self._build_tree(X[left_idx], y[left_idx], depth-1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth-1)
        return (feat_idx, threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        best_score = -np.inf
        split_idx, split_thresh = None, None
        for feat_idx in range(X.shape[1]):
            thresholds = np.linspace(np.min(X[:, feat_idx]), np.max(X[:, feat_idx]), self.num_cut_points)
            for threshold in thresholds:
                left_idx = X[:, feat_idx] < threshold
                right_idx = ~left_idx
                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue
                
                if self.split_way == 'KS':
                    # Calculate KS statistic using original target values
                    ks_stat, _ = ks_2samp(y[left_idx], y[right_idx])
                    score = ks_stat
                elif self.split_way == 'entropy':
                    # Calculate information gain
                    left_entropy = entropy(np.bincount(y[left_idx], minlength=self.n_classes_) + 1e-10)
                    right_entropy = entropy(np.bincount(y[right_idx], minlength=self.n_classes_) + 1e-10)
                    total_entropy = (sum(left_idx) * left_entropy + sum(right_idx) * right_entropy) / len(y)
                    info_gain = entropy(np.bincount(y, minlength=self.n_classes_)) - total_entropy
                    score = info_gain
                elif self.split_way == 'both':
                    # Calculate KS statistic using original target values
                    ks_stat, _ = ks_2samp(y[left_idx], y[right_idx])
                    # Calculate information gain
                    left_entropy = entropy(np.bincount(y[left_idx], minlength=self.n_classes_) + 1e-10)
                    right_entropy = entropy(np.bincount(y[right_idx], minlength=self.n_classes_) + 1e-10)
                    total_entropy = (sum(left_idx) * left_entropy + sum(right_idx) * right_entropy) / len(y)
                    info_gain = entropy(np.bincount(y, minlength=self.n_classes_)) - total_entropy
                    # Normalize KS and entropy scores
                    score = (ks_stat + info_gain) / 2  # Adjust this line as necessary for normalization
                else:
                    raise ValueError(f"Unknown split_way: {self.split_way}")
                
                if score > best_score:
                    best_score = score
                    split_idx = feat_idx
                    split_thresh = threshold
                
        return split_idx, split_thresh

    def _leaf_value(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _predict(self, inputs, tree):
        if not isinstance(tree, tuple):
            return tree
        feat_idx, threshold, left_tree, right_tree = tree
        if inputs[feat_idx] < threshold:
            return self._predict(inputs, left_tree)
        else:
            return self._predict(inputs, right_tree)

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the custom decision tree using different split ways
split_ways = ['KS', 'entropy', 'both']
for split_way in split_ways:
    custom_tree = DecisionTree(depth=12, num_cut_points=300, split_way=split_way)
    custom_tree.fit(X_train, y_train)
    
    # Perform cross-validation
    tree_scores = cross_val_score(custom_tree, X, y, cv=3)
    tree_mean_accuracy = tree_scores.mean()
    print(f"Custom Decision Tree Mean Accuracy with split_way={split_way} (CV): {tree_mean_accuracy}")


###################
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, entropy

class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, depth=10, min_samples_split=2, num_cut_points=100, split_way='entropy', soft=False, B=5):
        self.depth = depth
        self.min_samples_split = min_samples_split
        self.num_cut_points = num_cut_points  # Number of potential cut points
        self.split_way = split_way  # Criterion for splitting
        self.soft = soft  # Soft tree option
        self.B = B  # Parameter for logistic function used in soft tree
        self.tree = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Ensure compatibility with scikit-learn
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=self.depth)
        return self

    def _build_tree(self, X, y, depth):
        if depth == 0 or len(X) < self.min_samples_split:
            return self._leaf_value(y)
        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return self._leaf_value(y)
        left_idx = X[:, feat_idx] < threshold
        right_idx = ~left_idx
        left_tree = self._build_tree(X[left_idx], y[left_idx], depth-1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth-1)
        return (feat_idx, threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        best_score = -np.inf
        split_idx, split_thresh = None, None
        for feat_idx in range(X.shape[1]):
            thresholds = np.linspace(np.min(X[:, feat_idx]), np.max(X[:, feat_idx]), self.num_cut_points)
            for threshold in thresholds:
                left_idx = X[:, feat_idx] < threshold
                right_idx = ~left_idx
                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue
                
                if self.split_way == 'KS':
                    # Calculate KS statistic using original target values
                    ks_stat, _ = ks_2samp(y[left_idx], y[right_idx])
                    score = ks_stat
                elif self.split_way == 'entropy':
                    # Calculate information gain
                    left_entropy = entropy(np.bincount(y[left_idx], minlength=self.n_classes_) + 1e-10)
                    right_entropy = entropy(np.bincount(y[right_idx], minlength=self.n_classes_) + 1e-10)
                    total_entropy = (sum(left_idx) * left_entropy + sum(right_idx) * right_entropy) / len(y)
                    info_gain = entropy(np.bincount(y, minlength=self.n_classes_)) - total_entropy
                    score = info_gain
                elif self.split_way == 'both':
                    # Calculate KS statistic using original target values
                    ks_stat, _ = ks_2samp(y[left_idx], y[right_idx])
                    # Calculate information gain
                    left_entropy = entropy(np.bincount(y[left_idx], minlength=self.n_classes_) + 1e-10)
                    right_entropy = entropy(np.bincount(y[right_idx], minlength=self.n_classes_) + 1e-10)
                    total_entropy = (sum(left_idx) * left_entropy + sum(right_idx) * right_entropy) / len(y)
                    info_gain = entropy(np.bincount(y, minlength=self.n_classes_)) - total_entropy
                    # Normalize KS and entropy scores
                    score = (ks_stat + info_gain) / 2  # Adjust this line as necessary for normalization
                else:
                    raise ValueError(f"Unknown split_way: {self.split_way}")
                
                if score > best_score:
                    best_score = score
                    split_idx = feat_idx
                    split_thresh = threshold
                
        return split_idx, split_thresh

    def _leaf_value(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        if self.soft:
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            return np.array([self._predict_class(inputs, self.tree) for inputs in X])

    def predict_proba(self, X):
        return np.array([self._predict_proba(inputs, self.tree) for inputs in X])

    def _predict_class(self, inputs, tree):
        if not isinstance(tree, tuple):
            return tree
        feat_idx, threshold, left_tree, right_tree = tree
        
        if inputs[feat_idx] < threshold:
            return self._predict_class(inputs, left_tree)
        else:
            return self._predict_class(inputs, right_tree)

    def _predict_proba(self, inputs, tree):
        if not isinstance(tree, tuple):
            return np.eye(self.n_classes_)[tree]
        feat_idx, threshold, left_tree, right_tree = tree
        
        prob = 1 / (1 + np.exp((inputs[feat_idx] - threshold) * self.B))  # Adjusted direction
        left_prob = self._predict_proba(inputs, left_tree)
        right_prob = self._predict_proba(inputs, right_tree)
        
        # Ensure the weighted sum of probabilities is correctly normalized
        combined_prob = prob * left_prob + (1 - prob) * right_prob
        
        # Debugging: print the probabilities at each step
        #print(f"Input: {inputs}")
        #print(f"Threshold: {threshold}, Probability: {prob}")
        #print(f"Left Prob: {left_prob}, Right Prob: {right_prob}")
        #print(f"Combined Prob before normalization: {combined_prob}")
        
        # Normalize the probabilities to ensure they sum to 1
        combined_prob /= combined_prob.sum()
        
        # Debugging: print the normalized probabilities
        #print(f"Combined Prob after normalization: {combined_prob}")
        
        return combined_prob

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the custom decision tree using different split ways and soft option
split_ways = ['KS', 'entropy', 'both']
for split_way in split_ways:
    for soft in [True, False]:
        print(f"Training with split_way={split_way}, soft={soft}")
        custom_tree = DecisionTree(depth=12, num_cut_points=300, split_way=split_way, soft=soft, B=5)
        custom_tree.fit(X_train, y_train)
        y_pred = custom_tree.predict(X_test)
        print(f"Predictions: {y_pred}")
        print(f"True Labels: {y_test}")
        print(f"Accuracy with split_way={split_way}, soft={soft}: {accuracy_score(y_test, y_pred)}")

        # Perform cross-validation
        tree_scores = cross_val_score(custom_tree, X, y, cv=3)
        tree_mean_accuracy = tree_scores.mean()
        print(f"Custom Decision Tree Mean Accuracy with split_way={split_way}, soft={soft} (CV): {tree_mean_accuracy}")


#####extended decision tree######
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, entropy

class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, depth=10, min_samples_split=2, num_cut_points=100, split_way='entropy', soft=False, B=5):
        self.depth = depth
        self.min_samples_split = min_samples_split
        self.num_cut_points = num_cut_points  # Number of potential cut points
        self.split_way = split_way  # Criterion for splitting
        self.soft = soft  # Soft tree option
        self.B = B  # Parameter for logistic function used in soft tree
        self.tree = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # Ensure compatibility with scikit-learn
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=self.depth)
        return self

    def _build_tree(self, X, y, depth):
        if depth == 0 or len(X) < self.min_samples_split:
            return self._leaf_value(y)
        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return self._leaf_value(y)
        left_idx = X[:, feat_idx] < threshold
        right_idx = ~left_idx
        left_tree = self._build_tree(X[left_idx], y[left_idx], depth-1)
        right_tree = self._build_tree(X[right_idx], y[right_idx], depth-1)
        return (feat_idx, threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        best_score = -np.inf
        split_idx, split_thresh = None, None
        for feat_idx in range(X.shape[1]):
            thresholds = np.linspace(np.min(X[:, feat_idx]), np.max(X[:, feat_idx]), self.num_cut_points)
            for threshold in thresholds:
                left_idx = X[:, feat_idx] < threshold
                right_idx = ~left_idx
                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue
                
                if self.split_way == 'KS':
                    # Calculate KS statistic using original target values
                    ks_stat, _ = ks_2samp(y[left_idx], y[right_idx])
                    score = ks_stat
                elif self.split_way == 'entropy':
                    # Calculate information gain
                    left_entropy = entropy(np.bincount(y[left_idx], minlength=self.n_classes_) + 1e-10)
                    right_entropy = entropy(np.bincount(y[right_idx], minlength=self.n_classes_) + 1e-10)
                    total_entropy = (sum(left_idx) * left_entropy + sum(right_idx) * right_entropy) / len(y)
                    info_gain = entropy(np.bincount(y, minlength=self.n_classes_)) - total_entropy
                    score = info_gain
                elif self.split_way == 'both':
                    # Calculate KS statistic using original target values
                    ks_stat, _ = ks_2samp(y[left_idx], y[right_idx])
                    # Calculate information gain
                    left_entropy = entropy(np.bincount(y[left_idx], minlength=self.n_classes_) + 1e-10)
                    right_entropy = entropy(np.bincount(y[right_idx], minlength=self.n_classes_) + 1e-10)
                    total_entropy = (sum(left_idx) * left_entropy + sum(right_idx) * right_entropy) / len(y)
                    info_gain = entropy(np.bincount(y, minlength=self.n_classes_)) - total_entropy
                    # Normalize KS and entropy scores
                    score = (ks_stat + info_gain) / 2  # Adjust this line as necessary for normalization
                else:
                    raise ValueError(f"Unknown split_way: {self.split_way}")
                
                if score > best_score:
                    best_score = score
                    split_idx = feat_idx
                    split_thresh = threshold
                
        return split_idx, split_thresh

    def _leaf_value(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        if self.soft:
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            return np.array([self._predict_class(inputs, self.tree) for inputs in X])

    def predict_proba(self, X):
        return np.array([self._predict_proba(inputs, self.tree) for inputs in X])

    def _predict_class(self, inputs, tree):
        if not isinstance(tree, tuple):
            return tree
        feat_idx, threshold, left_tree, right_tree = tree
        
        if inputs[feat_idx] < threshold:
            return self._predict_class(inputs, left_tree)
        else:
            return self._predict_class(inputs, right_tree)

    def _predict_proba(self, inputs, tree):
        if not isinstance(tree, tuple):
            return np.eye(self.n_classes_)[tree]
        feat_idx, threshold, left_tree, right_tree = tree
        
        prob = 1 / (1 + np.exp((inputs[feat_idx] - threshold) * self.B))  # Adjusted direction
        left_prob = self._predict_proba(inputs, left_tree)
        right_prob = self._predict_proba(inputs, right_tree)
        
        # Ensure the weighted sum of probabilities is correctly normalized
        combined_prob = prob * left_prob + (1 - prob) * right_prob
       
        # Normalize the probabilities to ensure they sum to 1
        combined_prob /= combined_prob.sum()
        
        return combined_prob

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train and evaluate the custom decision tree using different split ways and soft option
split_ways = ['KS', 'entropy', 'both']
for split_way in split_ways:
    for soft in [True, False]:
        custom_tree = DecisionTree(depth=12, num_cut_points=300, split_way=split_way, soft=soft, B=5)
        custom_tree.fit(X_train, y_train)
       
        # Perform cross-validation
        tree_scores = cross_val_score(custom_tree, X, y, cv=3)
        tree_mean_accuracy = tree_scores.mean()
        print(f"Custom Decision Tree Mean Accuracy with split_way={split_way}, soft={soft} (CV): {tree_mean_accuracy}")
