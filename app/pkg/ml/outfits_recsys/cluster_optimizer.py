import os


import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from app.pkg.settings import settings
from app.pkg.logger import get_logger

logger = get_logger(__name__)

class ApproximBase:

    @staticmethod
    def interpolate(x, *params):
        pass

    @staticmethod
    def get_optimal_x(*params):
        """
        Returns x where derivative is -1

        In specific cases derivative can be
        equaled to another value
        """
        pass

class ExpApproximation(ApproximBase):

    @staticmethod
    def interpolate(x, a, b, c):
        return a*np.exp(-b*x) + c

    @staticmethod
    def get_optimal_x(a, b, c):
        return np.log(b*a)/(b)

class HyperbolaApproximation(ApproximBase):

    @staticmethod
    def interpolate(x, a, b):
        return a/x + b

    @staticmethod
    def get_optimal_x(a, b,):
        return a**(1/2)



class ApproxFunctions:
    """
    Functions to approximate dependency (k, inertia_)
    """
    exponential = ExpApproximation()


class ClustersOptimizer:
    """
    Optimizes cluster parameters such as clusters amount
    """
    def __init__(self,
                 approxim: ApproximBase = ApproxFunctions.exponential,
                 save_plots: bool = True,
                 pre_name_save="pre_name"
                 ):
        """
        approxim - (k, inertia_) dependency approximator
        save_plots - bool - is needed to save plots due to work
        """
        self.approximation = approxim
        self.approx_func = self.approximation.interpolate

        self.pre_name_save = pre_name_save
        self.save_plots = save_plots
        self.save_meta_path = settings.ML.META_SAVE
        if not os.path.exists(self.save_meta_path):
            os.mkdir(self.save_meta_path)



    def calculate_optimal_k(self, embeddings):
        """
        k is clusters amount.

        Calculating optimal clusters amount is based on
        approximation of plot(k_range, inertia) and finding
        it's derivative==-1
        """
        wcss = []
        len_embs = len(embeddings)
        max_possible_k = round(len_embs**(1/2))
        k_range = range(1, max_possible_k + 1)
        logger.debug(f"For calculating optimal_k got {len_embs} embeddings. Also {max_possible_k=}")

        for i in k_range:
            kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
            kmeans.fit(embeddings)
            wcss.append(kmeans.inertia_)


        wcss_postprocessed = np.array(wcss)**(1/2.85)

        calculated_params = False
        try:
            params, param_cov = curve_fit(self.approx_func,
                                        k_range,
                                        wcss_postprocessed,
                                        )
            

            optimal_k = self.approximation.get_optimal_x(*params)
            if optimal_k > max_possible_k:
                optimal_k = max_possible_k
            calculated_params = True 

        except RuntimeError:
            optimal_k = max_possible_k
            logger.warn(f"Cant calculate optimal_k. Set up to {optimal_k}")
            calculated_params = False
        except TypeError as e:
            optimal_k = max_possible_k
            logger.warn(f"Error while calculating optimal k. Set up to {optimal_k}. Possible reason: little data. Error: {e}")
            calculated_params = False

        if np.isnan(optimal_k):
            logger.warn("Calculated optimal_k is None")
            optimal_k = max_possible_k

        if self.save_plots and calculated_params:
            self.plot_orig_inertia(k_range, wcss)

            inertia_approx = self.approx_func(k_range, *params)

            k_range_np = np.array(k_range)
            tangent = -1*(k_range_np - optimal_k)\
                + self.approx_func(optimal_k, *params)

            self.plot_approx_inertia(k_range,
                                       wcss_postprocessed,
                                       optimal_k,
                                       inertia_approx,
                                       tangent,
                                       )
            logger.info(f"Plotted inertia graphics for {len(embeddings)} embeddings. "\
                        f"See '{self.save_meta_path}' path")
        
        return optimal_k

    def plot_orig_inertia(self,
                          k_range,
                          inertia):
        """
        Plots graphic with original (k, inertia) dependency
        """
        save_path = os.path.join(
            self.save_meta_path,
            f"{self.pre_name_save}_orig_inertia_plot.jpg"               
            )
        plt.plot(k_range, inertia)
        plt.grid()
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS(inertia)')
        plt.savefig(save_path)
        plt.close()

    def plot_approx_inertia(self,
                              k_range,
                              inertia_mod,
                              optimal_k,
                              inertia_approx,
                              tangent,
                              ):
        """
        Plots graphic with postprocessed (k, inertia) dependency,
        it's optimal k tangent, and inertia approximation
        """
        save_path = os.path.join(
            self.save_meta_path,
            f"{self.pre_name_save}_approx_inertia_plot.jpg"               
        )

        # plot approximation
        plt.plot(k_range, inertia_approx, label='approximation')


        # tangent
        plt.plot(k_range, tangent, label='tangent')

        # plot modified original inertia
        plt.plot(k_range, inertia_mod, label='original')
        plt.grid()
        plt.title('Elbow Method. '\
                  f'Found optimal k={round(optimal_k,2)}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('inertia ^ (1/2.8)')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
