
import sys
import os
# from _pickle import dump
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

class GT_Observaion(object):
    def __init__(self):
        return
    def get_landmarkdf(self):
        filepath = '../data/map_data.txt'
        names = ['x','y','landmark_id']
        df = pd.read_csv(filepath, sep=None, header=None, names=names,engine='python')
#         print(df.describe())
        return df
    def get_gtdf(self):
        filepath = '../data/gt_data.txt'
        names = ['x','y','theta']
        df = pd.read_csv(filepath, sep=None, header=None, names=names,engine='python')
#         print(df.describe())
        return df
    def get_observation(self, step_id):
        filepath = '../data/observation/observations_{:06d}.txt'.format(step_id)
        names = ['x','y']
        df = pd.read_csv(filepath, sep=None, header=None, names=names,engine='python')
        print("observation {}, expected length {}".format(filepath, len(df)))
        df['distance'] = np.sqrt(df['x'].values **2 + df['y'].values **2)
        print(df)
        return df
    def __disp_data(self,landmark_df, sorted_landmark, gt_record):
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        ax.scatter(landmark_df['x'], landmark_df['y'], color='green')
        for i, txt in enumerate(landmark_df['landmark_id']):
            ax.annotate(str(txt), (landmark_df['x'][i],landmark_df['y'][i]))
            
        for i, landmark in sorted_landmark.iterrows():
            x1 = landmark['x']
            y1 = landmark['y']
            x2 = gt_record['x']
            y2 = gt_record['y']
#             new_x, newy_y = self._transform_coor(gt_record, landmark)
#             print('relative x, y: {},{}'.format(new_x, newy_y))
            ax.plot([x1,x2],[y1,y2])
            ax.annotate(str(i+1), ((x1+x2)/2,(y1 +y2)/2))
        
        return
    def _transform_coor(self, particle, landmark):
        cos_theta = math.cos(particle.theta - math.pi / 2);
        sin_theta = math.sin(particle.theta - math.pi / 2);
        transformed_landmark_x = -(landmark.x - particle.x) * sin_theta + (landmark.y - particle.y) * cos_theta;
        transformed_landmark_y = -(landmark.x - particle.x) * cos_theta - (landmark.y - particle.y) * sin_theta;
        
        return transformed_landmark_x,transformed_landmark_y
    def derive_observation(self, step_id):
        landmark_df = self.get_landmarkdf()
        particle = self.get_gtdf().iloc[step_id-1] 
        print("ground truth vehicle postion {}".format(particle))
        distances = []
        for _, landmark in landmark_df.iterrows():
            dist = math.sqrt((particle['x'] - landmark['x'])**2 + (particle['y'] - landmark['y']) **2 )
            distances.append(dist)
        
        landmark_df['distance'] = distances
        
        sorted_landmark = landmark_df.sort_values('distance')
        sorted_landmark = sorted_landmark[sorted_landmark['distance']<=50]
        sorted_landmark = sorted_landmark.reset_index()
        
        
        cos_theta = math.cos(particle.theta - math.pi / 2);
        sin_theta = math.sin(particle.theta - math.pi / 2);
        new_x = -(sorted_landmark['x'] - particle.x) * sin_theta + (sorted_landmark['y'] - particle.y) * cos_theta;
        new_y = -(sorted_landmark['x'] - particle.x) * cos_theta - (sorted_landmark['y'] - particle.y) * sin_theta;
        
        sorted_landmark['x'] = new_x;
        sorted_landmark['y'] = new_y;
        
        
        
#         sorted_landmark['x'] = sorted_landmark['x']-particle['x']
#         sorted_landmark['y'] = sorted_landmark['y']-particle['y']
        print("predicted observation, length {}".format(len(sorted_landmark)))
        print(sorted_landmark)
        self.__disp_data(landmark_df, sorted_landmark, particle)
        
        return
   
    def run(self):
#         df = self.get_gtdf()
        step_id = 1980
        self.get_observation(step_id)
        self.derive_observation(step_id)
        
        plt.show()
        return




if __name__ == "__main__":   
    obj= GT_Observaion()
    obj.run()
  