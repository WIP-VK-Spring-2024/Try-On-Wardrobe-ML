import numpy as np


from app.pkg.ml.try_on.preprocessing.cut_sam_pipeline.sam_points_strategies import PointsFormingSamStrategies

class KeyPointsFormer:
    """
    Generates keypoints from image mask
    """
    def __init__(self):
        pass

    def __call__(self,
                 mask,
                 keypoint_strategy: PointsFormingSamStrategies = PointsFormingSamStrategies.strategy_0):
        # mask is h,w tensor
        h, w = mask.shape
        indices = np.where(mask)

        form_strategy = keypoint_strategy
        # Определить диапазон допустимых координат по ширине и высоте
        min_i, max_i = np.min(indices[0]), np.max(indices[0])
        min_j, max_j = np.min(indices[1]), np.max(indices[1])

        target_points = form_strategy(min_i, min_j, max_i, max_j)

        nearest_points = [
            self.find_nearest_point_in_mask(mask, target_point)
            for target_point in target_points
        ]

        input_points = [[[point[1], point[0]] for point in nearest_points]]
        return input_points

    def find_nearest_point_in_mask(self, mask, point):
        """
        Finds nearest point to input, but belonging to mask
        """
        indices = np.where(mask)

        target_i = point[0]
        target_j = point[1]

        # Найти ближайшую точку маски к целевым координатам
        distances = np.sqrt((indices[0] - target_i)**2 + (indices[1] - target_j)**2)
        nearest_index = np.argmin(distances)
        nearest_point = (indices[0][nearest_index], indices[1][nearest_index])

        return nearest_point
