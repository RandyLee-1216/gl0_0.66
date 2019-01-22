import scipy
from scipy import linalg
import pandas as pd
import numpy as np
import argparse, os, glob, time
import matplotlib as mpl
mpl.use('Agg')

class Base(object):
    def __init__(self, args):
        self.dataset   = args.dataset
        self.date      = args.date
        self.test_data = args.test_data
        self.train_dir = os.path.join('results', self.dataset, self.date)
        self.test_dir  = os.path.join(self.train_dir, 'test')
        fid_stats_dir  = './fid_stats'
        if not os.path.exists(fid_stats_dir):
            os.makedirs(fid_stats_dir)

##################################################################################################
class FID(Base):
    def __init__(self, args):
        super(FID, self).__init__(args)
        #self.train_

    def run(self):
        print("==================loading data====================")

        # read train latent codes from csv
        train_csv = glob.glob(self.train_dir+'/*.csv')
        print('train_csv : %s' % train_csv)
        df_train = pd.read_csv(train_csv[0], header=None, delimiter=' ')
        # because the number are so many, we sample it
        self.train_latent = df_train.values

        test_csv_ok = glob.glob(self.train_dir+'/*.csv')
        test_csv_ng = glob.glob(self.test_dir+'/'+self.test_data+'*csv')
        print('test_csv_ok : %s' % test_csv_ok)
        print('test_csv_ng : %s' % test_csv_ng)
        df_test_ok = pd.read_csv(test_csv_ok[0], header=None, delimiter=' ')
        df_test_ng = pd.read_csv(test_csv_ng[0], header=None, delimiter=' ')
        self.test_latent_ok = df_test_ok.values
        self.test_latent_ng = df_test_ng.values
        
        self.train_mu       = np.mean(self.train_latent, axis=1)
        self.train_sigma    = np.var(self.train_latent, axis=1)
        self.test_ok_mu     = np.mean(self.test_latent_ok, axis=1)
        self.test_ok_sigma  = np.var(self.test_latent_ok, axis=1)
        self.test_ng_mu     = np.mean(self.test_latent_ng, axis=1)
        self.test_ng_sigma  = np.var(self.test_latent_ng, axis=1)

        self.mu = np.mean(self.train_mu)
        count_value_ok = 0
        count_value_ng = 0
        fid_value = 2
        
        for i in range(len(self.test_ng_mu)):
            diff_fid = self.test_ng_mu[i] - self.mu
            value_fid = diff_fid**2 + self.test_ng_sigma[i]
            if fid_value > value_fid:
                fid_value = value_fid
        print('FID value : %f' % fid_value)
        
        for k in range(len(self.test_ok_mu)):
            diff_ok = self.test_ok_mu[k] - self.mu
            value_ok = diff_ok**2 + self.test_ok_sigma[k]
            if value_ok > fid_value:
                count_value_ok += 1
                print('Overkill value : %f' % value_ok)

        print("+---------------+----------------+")
        print("|OK:     %6d | overkill:%6d|" %((len(self.test_ok_mu)-count_value_ok), count_value_ok))
        print("|---------------+----------------|")
        print("|leakage:%6d | NG:      %6d|" %(count_value_ng, (len(self.test_ng_mu)-count_value_ng)))
        print("+---------------+----------------+")

##########################################################################
def parse_args():
    desc = "FID"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-dataset',  type=str, required=True)
    parser.add_argument('-date',     type=str, required=True)
    parser.add_argument('-test_data',type=str)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    if args is None:
        exit()

    print("[*] Running Separate Frechet distance statistics.")
    fid = FID(args)
    fid.run()
