import os
import io
import pickle
from typing import Union, List, Dict
from copy import deepcopy

import numpy as np
import torch

from app.pkg.ml.outfits_recsys.cluster_processor import ClustersProcessor
from app.pkg.ml.buffer_converters import BytesConverter
from app.pkg.settings import settings
from app.pkg.logger import get_logger

logger = get_logger(__name__)

class CrossUsersOutfitRecSys:
    def __init__(self):
        self.global_embeddings = []
        self.bytes_converter = BytesConverter() 
        self.weights_path = settings.ML.WEIGHTS_PATH
        self.model_path = os.path.join(self.weights_path, "cu_recsys.pkl")
        if not os.path.isfile(self.model_path):
            self.global_cluster_processor = ClustersProcessor(pre_name_save='global')
            logger.info("Global ClustersProcessor is inited")
        else:
            with open(self.model_path, 'rb') as f:
                self.global_cluster_processor = pickle.load(f)
                logger.info("Global ClustersProcessor is loaded from file")

    def update_global_outfits_from_bytes(self,
                                         outfits,
                                         cloth_id_to_bytes):
        """
        Adds outfit tensor field
        
        Args:
            outfits - list:
                [
                    {"user_id":int/str,
                     "outfit_id":int/str,
                     "clothes": [cloth_id, ...]
                     }, ...
                ]
            cloth_id_to_bytes - dict[cloth_id] -> io.BytesIO (cloth tensor)
        """
        outfits = self.outfits_from_cloth_bytes(
            outfits=outfits,
            cloth_id_to_bytes=cloth_id_to_bytes
            )
        self.update_global_outfits(outfits)


    def update_global_outfits_list_format(self,
                                         outfits,
                                         cloth_id_to_list):
        """
        Adds outfit tensor field
        
        Args:
            outfits - list:
                [
                    {"user_id":int/str,
                     "outfit_id":int/str,
                     "clothes": [cloth_id, ...]
                     }, ...
                ]
            cloth_id_to_list - dict[cloth_id] -> list (cloth tensor)
        """
        outfits = self.outfits_from_cloth_list(
            outfits=outfits,
            cloth_id_to_list=cloth_id_to_list
            )
        self.update_global_outfits(outfits)


    def outfits_from_cloth_list(self, outfits, cloth_id_to_list):
        """
        Adds outfit tensor field
        
        Args:
            outfits - list:
                [
                    {"user_id":int/str,
                     "outfit_id":int/str,
                     "clothes": [cloth_id, ...]
                     }, ...
                ]
            cloth_id_to_bytes - dict[cloth_id] -> io.BytesIO (cloth tensor)
        """
        cloth_id_to_tensor = {}
        for cloth_id, cloth_list in cloth_id_to_list.items():
            tensor = torch.tensor(cloth_list)
            cloth_id_to_tensor[cloth_id] = tensor 

        outfits = self.get_outfits_vector(outfits, cloth_id_to_tensor)
        return outfits

    def get_outfits_vector(self, outfits, cloth_id_to_tensor):
        for outfit in outfits:
            clothes_ids = outfit['clothes']
            clothes_tensors = [cloth_id_to_tensor[cloth_id].reshape(1,-1)
                              for cloth_id in clothes_ids]
            clothes_tensors = torch.concatenate(clothes_tensors)
            outfit['tensor'] = clothes_tensors.mean(0)
        return outfits



    def update_global_outfits(self, outfits):
        # if from_bytes:
        #     outfits = self.outfits_from_bytes(outfits)
        self.global_cluster_processor = ClustersProcessor(pre_name_save='global')
        self.setup_global_outfits(outfits)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.global_cluster_processor, f)

    def setup_global_outfits(self, outfits):
        if len(outfits) == 0:
            raise ValueError("Amount of outfits must be >0")
        self.global_cluster_processor.fit(outfits)



    # def recommend_from_bytes(self,
    #               user_outfits: List[Dict[str, Union[io.BytesIO, str]]],
    #               samples=10):
    #     """
    #     Recommends outfits (ids)
    #     Args:
    #         user_outfits - list, containing dicts with keys "user_id", "outfit_id", "tensor"
    #     Returns:
    #         list of outfit ids
    #     """
    #     converted_user_outfits = self.outfits_from_bytes(user_outfits)
    #     return self.recommend(converted_user_outfits, samples)


    def outfits_from_cloth_bytes(self, outfits, cloth_id_to_bytes):
        """
        Adds outfit tensor field
        
        Args:
            outfits - list:
                [
                    {"user_id":int/str,
                     "outfit_id":int/str,
                     "clothes": [cloth_id, ...]
                     }, ...
                ]
            cloth_id_to_bytes - dict[cloth_id] -> io.BytesIO (cloth tensor)
        """
        cloth_id_to_tensor = {}
        for cloth_id, cloth_bytes in cloth_id_to_bytes.items():
            tensor = (self
                      .bytes_converter
                      .bytes_to_torch(cloth_bytes)
                     )
            cloth_id_to_tensor[cloth_id] = tensor 

        outfits = self.get_outfits_vector(outfits, cloth_id_to_tensor)
        return outfits

    # def outfit_from_bytes(self, outfit):
    #     """
    #     Makes outfit's tensor from bytes tensors in outfit dict
    #     Requires has io.BytesIO key "tensor" in clothes
    #     """
    #     tensors = [(self
    #                 .bytes_converter
    #                 .bytes_to_torch(cloth['tensor'])
    #                 .unsqueeze(0)
    #                 )
    #                for cloth in outfit["clothes"]]
    #     tensors = torch.concatenate(tensors)
    #     outfit['tensor'] = tensors.mean(0)
    #     return outfit

    def recommend(self,
                  user_id,
                  samples=10):
        """
        Recomends outfits similar to user's

        Args:
            user_id - id(or uuid) of user to recommend
            samples - amount of outfits to recommend 

        Returns:
          list of outfit_id's with len=samples
        """
        global_cluster_indexes = np.arange(0,
                                    self.global_cluster_processor.clusters_amount,
                                    step=1)


        global_centers_stats = np.array(
             [
              self.global_cluster_processor.cluster_stats[cl_index]
              for cl_index in global_cluster_indexes
            ]
        )
        global_centers_stats_normalized = global_centers_stats/global_centers_stats.sum(0)


        user_outfits = self.global_cluster_processor.get_user_embs(user_id)

        if len(user_outfits) == 0:
            logger.info(f"User: {user_id} has no outfits")
            global_probs_list = [global_centers_stats_normalized for _ in range(samples)]

        else:
            local_cluster_processor = ClustersProcessor(pre_name_save='local')
            local_cluster_processor.fit(user_outfits)

            local_cluster_indexes = np.arange(0,
                                        local_cluster_processor.clusters_amount,
                                        step=1)

            

            local_centers_stats = np.array(
                [
                local_cluster_processor.cluster_stats[cl_index]
                for cl_index in local_cluster_indexes
                ]
            )
                        # gl_centers = self.global_cluster_processor.clusterizer.cluster_centers_
            local_centers = local_cluster_processor.clusterizer.cluster_centers_

            # get distances matrix with format f(local, global)
            # f(l,g) = dist(l,g)*dots_amount_normalized(g)       
            prob_local_global = self.global_cluster_processor.clusterizer.transform(local_centers)       
            
            prob_local_global = prob_local_global * global_centers_stats_normalized           

            # normalize for local prob (sum of global probs = 1)
            prob_local_global = (prob_local_global.T / prob_local_global.sum(1)).T

            # sampling local centers
            local_centers_probs = local_centers_stats/local_centers_stats.sum(0)        
            sampled_local_clusters_centers = np.random.choice(local_cluster_indexes,
                                        size=samples,
                                        p=local_centers_probs)

            global_probs_list = [prob_local_global[cl_local] for cl_local in sampled_local_clusters_centers]


        outfits_id = []
        for global_probs in global_probs_list:
            selected_gl_center = np.random.choice(
                global_cluster_indexes, size=1, p=global_probs
            )[0]
            outfit_id = self.global_cluster_processor.sample_from_center(
                center_index=selected_gl_center,
                sample_amount=1,
                user_id=user_id,
            )[0]
            outfits_id.append(outfit_id)
        return outfits_id

        
