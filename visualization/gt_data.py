
import sys
import os
# from _pickle import dump
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

class GTData(object):
    def __init__(self):
        return
    def get_gtdf(self):
        filepath = '../data/gt_data.txt'
        names = ['x','y','yaw']
        df = pd.read_csv(filepath, sep=None, header=None, names=names,engine='python')
        print(df.describe())
        return df
    def disp_gt(self,df):
        fig, ax = plt.subplots()
        ax.scatter(df['x'], df['y'])
        ax.set_title('ground truth')
       
        return
    def run(self):
        df = self.get_gtdf()
        self.disp_gt(df)
        plt.show()
        return




if __name__ == "__main__":   
    obj= GTData()
    obj.run()
  