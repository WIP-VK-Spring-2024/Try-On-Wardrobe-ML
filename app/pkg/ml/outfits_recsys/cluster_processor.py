from collections import Counter

import numpy as np
from scipy.special import softmax
from sklearn.cluster import KMeans

from app.pkg.ml.outfits_recsys.cluster_optimizer import ClustersOptimizer
from app.pkg.settings import settings
from app.pkg.logger import get_logger

logger = get_logger(__name__)

class ClustersProcessor:
    def __init__(self, save_plots=True, pre_name_save='naming'):
        self.outfits_embeddings = []
        self.clusters_amount = 1
        self.clusters_optimizer = ClustersOptimizer(save_plots=save_plots,
                                                    pre_name_save=pre_name_save)
        self.clusterizer = None
        self.cluster_stats = None # {cluster_id:dots_amount}
        self.embs_center_distances = None # [[dist_to_center1, dist_to_center2,...],
                                          #  [same for second embedding]  ]
        self.outfits = None
        self.user_ids = None
        self.outfit_ids =  None

    def fit(self, outfits):
        self.setup_embeddings(outfits)
        self.setup_clusters_amount()
        self.setup_clusterizer()
        return self

    def setup_embeddings(self, outfits):
        embs = [outfit['tensor'] for outfit in outfits]
        user_ids =[outfit['user_id'] for outfit in outfits]
        outfit_ids =[outfit['outfit_id'] for outfit in outfits]
        
        self.outfits = np.array(outfits)
        self.user_ids =  np.array(user_ids)
        self.outfit_ids =  np.array(outfit_ids)
        
        self.outfits_embeddings =  np.array(embs)

    def setup_clusters_amount(self, k=None):
        if k is None:
            assert len(self.outfits_embeddings) > 2
            self.clusters_amount = round(self
                                    .clusters_optimizer
                                    .calculate_optimal_k(
                                        self.outfits_embeddings)
                                    )
        else:
            self.clusters_amount = k


    def setup_clusterizer(self):       
        self.clusterizer = KMeans(n_clusters=self.clusters_amount,
                                  random_state=42,
                                  n_init="auto")
        self.clusterizer.fit(self.outfits_embeddings)
        self.embs_center_distances = self.clusterizer.transform(self.outfits_embeddings)
        
        self.cluster_stats = Counter(self.clusterizer.labels_)

    
    def get_clusters_centers(self):
        assert self.clusterizer is not None
        return self.clusterizer.cluster_centers_

    def get_clusters_labels(self):
        assert self.clusterizer is not None
        return self.clusterizer.labels_
        
    def get_cluster_indexes_per_embeddings(self,
                                           embeddings: np.ndarray):
        """
        Gets clusters indexes for input embedding vectors
        """
        assert self.clusterizer is not None
        return self.clusterizer.predict(embeddings)

    def sample_from_center(self, center_index, sample_amount=1, user_id=None):
        label_mask = self.clusterizer.labels_ == center_index

        if user_id is not None:
            user_mask = self.user_ids != user_id
        else:
            logger.warn("UUID for sampling wasn't transfered")
            user_mask = np.ones(label_mask.shape,)


        outfits_mask = (label_mask) & (user_mask)

        # selecting outfit ids
        selected_ids = self.outfit_ids[outfits_mask]

        # embs_of_center = self.outfits_embeddings[outfits_mask]

        dist_of_center_embs = self.embs_center_distances[outfits_mask, center_index]
        embs_prob = softmax(dist_of_center_embs)
        sample_indexes = np.random.choice(range(len(embs_prob)), sample_amount, p=embs_prob)
        return selected_ids[sample_indexes]

    def get_user_embs(self, user_id):
        user_mask = self.user_ids == user_id
        # selected_outfits_ids = self.outfit_ids[user_mask]
        return self.outfits[user_mask]
