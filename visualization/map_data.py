
import sys
import os
# from _pickle import dump
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

class MapData(object):
    def __init__(self):
        return
    def get_df(self):
        filepath = '../data/map_data.txt'
        names = ['x','y','landmark_id']
        df = pd.read_csv(filepath, sep=None, header=None, names=names,engine='python')
        print(df.describe())
        return df
    def disp_landmarks(self,df):
        fig, ax = plt.subplots()
        ax.scatter(df['x'], df['y'], color='green')
        
        for i, txt in enumerate(df['landmark_id']):
            ax.annotate(str(txt), (df['x'][i],df['y'][i]))
        
        gt_df = self.get_gtdf()
        ax.scatter(gt_df['x'], gt_df['y'])
        ax.set_title('ground truth and landmarks')
        return
    def get_gtdf(self):
        filepath = '../data/gt_data.txt'
        names = ['x','y','yaw']
        df = pd.read_csv(filepath, sep=None, header=None, names=names,engine='python')
        print(df.describe())
        return df
    def run(self):
        df = self.get_df()
        self.disp_landmarks(df)
        plt.show()
        return




if __name__ == "__main__":   
    obj= MapData()
    obj.run()
  