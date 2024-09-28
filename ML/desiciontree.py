import numpy as np
from collections import Counter
from typing import Tuple, List, Dict, Optional

class TreeNode:
    def __init__(self, data: np.ndarray, feature_idx: int, threshold: float, 
                 class_probs: np.ndarray, info_gain: float) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.class_probs = class_probs
        self.info_gain = info_gain
        self.feature_importance = self.data.shape[0] * self.info_gain
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None

    def __str__(self) -> str:
        if self.left or self.right:
            return (f"NODE | Info Gain = {self.info_gain:.4f} | "
                    f"Split: If X[{self.feature_idx}] < {self.threshold}, go left; else, go right.")
        else:
            label_counts = Counter(self.data[:, -1])
            output = ", ".join(f"{label}->{count}" for label, count in label_counts.items())
            return f"LEAF | Label Counts = {output} | Probabilities = {self.class_probs}"

class DecisionTree:
    def __init__(self, max_depth=4, min_samples_leaf=1, min_info_gain=0.0, 
                 feature_split_strategy=None) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_info_gain = min_info_gain
        self.feature_split_strategy = feature_split_strategy

    def _entropy(self, probabilities: List[float]) -> float:
        return -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    def _class_probabilities(self, labels: np.ndarray) -> List[float]:
        total_count = len(labels)
        return [count / total_count for count in Counter(labels).values()]

    def _data_entropy(self, labels: np.ndarray) -> float:
        return self._entropy(self._class_probabilities(labels))
    
    def _partition_entropy(self, subsets: List[np.ndarray]) -> float:
        total_count = sum(len(subset) for subset in subsets)
        return sum(self._data_entropy(subset) * (len(subset) / total_count) for subset in subsets)
    
    def _split(self, data: np.ndarray, feature_idx: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        mask = data[:, feature_idx] < threshold
        return data[mask], data[~mask]
    
    def _select_features(self, data: np.ndarray) -> np.ndarray:
        feature_indices = np.arange(data.shape[1] - 1)
        if self.feature_split_strategy == "sqrt":
            return np.random.choice(feature_indices, int(np.sqrt(len(feature_indices))), replace=False)
        elif self.feature_split_strategy == "log":
            return np.random.choice(feature_indices, int(np.log2(len(feature_indices))), replace=False)
        return feature_indices
        
    def _find_best_split(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
        best_entropy = float('inf')
        feature_indices = self._select_features(data)
        best_split = None

        for idx in feature_indices:
            thresholds = np.percentile(data[:, idx], np.arange(25, 100, 25))
            for threshold in thresholds:
                left_subset, right_subset = self._split(data, idx, threshold)
                entropy = self._partition_entropy([left_subset[:, -1], right_subset[:, -1]])
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_split = (left_subset, right_subset, idx, threshold)

        return best_split + (best_entropy,)
    
    def _calculate_class_probs(self, data: np.ndarray) -> np.ndarray:
        labels = data[:, -1].astype(int)
        total = len(labels)
        probs = np.zeros(len(self.labels_in_train), dtype=float)
        for i, label in enumerate(self.labels_in_train):
            count = np.sum(labels == i)
            probs[i] = count / total if count > 0 else 0
        return probs

    def _build_tree(self, data: np.ndarray, depth: int) -> TreeNode:
        if depth >= self.max_depth:
            return None
        
        split_data = self._find_best_split(data)
        if split_data[-1] < self.min_info_gain or \
           any(group.shape[0] < self.min_samples_leaf for group in split_data[:2]):
            return TreeNode(data, -1, -1, self._calculate_class_probs(data), 0)

        left_data, right_data, idx, threshold, _ = split_data
        node = TreeNode(data, idx, threshold, self._calculate_class_probs(data), split_data[-1])
        node.left = self._build_tree(left_data, depth + 1)
        node.right = self._build_tree(right_data, depth + 1)
        return node
    
    def _predict_single(self, sample: np.ndarray) -> np.ndarray:
        node = self.tree
        while node:
            if sample[node.feature_idx] < node.threshold:
                if node.left is None:
                    return node.class_probs
                node = node.left
            else:
                if node.right is None:
                    return node.class_probs
                node = node.right
        return node.class_probs

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.labels_in_train = np.unique(y)
        train_data = np.hstack((X, y.reshape(-1, 1)))
        self.tree = self._build_tree(train_data, 0)
        self.feature_importances = {i: 0 for i in range(X.shape[1])}
        self._calculate_feature_importance(self.tree)
        total_importance = sum(self.feature_importances.values())
        self.feature_importances = {k: v / total_importance for k, v in self.feature_importances.items()}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self._predict_single, 1, X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
    
    def _print_tree_recursively(self, node: TreeNode, level=0) -> None:
        if node:
            self._print_tree_recursively(node.left, level + 1)
            print('    ' * level + '-> ' + str(node))
            self._print_tree_recursively(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_tree_recursively(self.tree)

    def _calculate_feature_importance(self, node: TreeNode) -> None:
        if node:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)
