
import numpy as np
import faiss
from sklearn.metrics import roc_curve, auc

class FacialVerifier:
    def __init__(self, known_embeddings, optimal_threshold=None):
        self.known_embeddings = np.array(known_embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(self.known_embeddings.shape[1])
        self.index.add(self.known_embeddings)
        self.optimal_threshold = optimal_threshold if optimal_threshold else 1

    def verify(self, embeddings):
        # returns binary list of whether the embeddings are verified
        distances = self._compute_distances(embeddings)
        return np.array(distances) <= self.optimal_threshold
        
    def test_thresholds(self, unknown_embeddings, true_labels, name, thresholds):
        distances = self._compute_distances(unknown_embeddings)
        true_labels = np.array([1 if label == name else 0 for label in true_labels])

        fpr_list = []
        tpr_list = []
        accuracies = []
        fscores = []

        for threshold in thresholds:
            predicted_labels = np.array(distances) <= threshold

            tp = np.sum((predicted_labels == 1) & (true_labels == 1))
            fp = np.sum((predicted_labels == 1) & (true_labels == 0))
            tn = np.sum((predicted_labels == 0) & (true_labels == 0))
            fn = np.sum((predicted_labels == 0) & (true_labels == 1))

            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)

            fpr_list.append(fpr)
            tpr_list.append(tpr)

            accuracy = (tp + tn) / len(unknown_embeddings)
            fscore = self.calculate_fscore(tp, fp, tn, fn)
            accuracies.append(accuracy)
            fscores.append(fscore)
            print(f"Threshold: {threshold:.2f}, Accuracy: {accuracy:.2f}, FPR: {fpr:.2f}, TPR: {tpr:.2f} FSCORE: {fscore:.2f}")
        # optimal threshold is the one that maximizes the fscore
        self.optimal_threshold = thresholds[np.argmax(fscores)]
        return fpr_list, tpr_list, accuracies, fscores, thresholds

    # def _compute_distances(self, unknown_embeddings):
    #     unknown_embeddings = np.array(unknown_embeddings).astype('float32')
    #     distances, _ = self.index.search(unknown_embeddings, 1)
    #     return distances.flatten()
    # takes weighted average distance between k nearest neighbors
    def _compute_distances(self, unknown_embeddings, k=5):
        k = min(k, len(self.known_embeddings))
        unknown_embeddings = np.array(unknown_embeddings).astype('float32')
        distances, indices = self.index.search(unknown_embeddings, k)
        weights = 1.0 / (distances + 1e-6)  # Add a small constant to avoid division by zero
        weighted_distances = np.sum(distances * weights, axis=1) / np.sum(weights, axis=1)
        return weighted_distances

    def _compute_accuracy(self, distances, true_labels, threshold):
        predicted_labels = np.array(distances) <= threshold
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy

    def compute_auc(self, fpr_list, tpr_list):
        auc_value = np.trapz(tpr_list, fpr_list)
        return auc_value
    
    def calculate_fscore(self, tp, fp, tn, fn):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * (precision * recall) / (precision + recall)
        return fscore