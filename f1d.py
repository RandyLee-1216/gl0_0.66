import pandas as pd
import numpy as np
import argparse
import os, glob
import matplotlib as mpl
mpl.use('Agg')

class Base(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.date = args.date
        self.test_data = args.test_data
        self.train_dir = os.path.join('results', self.dataset, self.date)
        self.test_dir = os.path.join(self.train_dir, 'test')
        fid_stats_dir = './fid_stats'
        if not os.path.exists(fid_stats_dir):
            os.makedirs(fid_stats_dir)

class FIDPREDICTSINGLE(Base):
    def __init__(self, args):
        super(FIDPREDICTSINGLE, self).__init__(args)
        #self.train_

    def loadData(self):
        print("--loading data")

        # read train latent codes from csv
        train_csv = glob.glob(self.train_dir+'/*.csv')
        print(train_csv)
        df_train = pd.read_csv(train_csv[0],
            header=None, delimiter=' ')
        # because the number are so many, we sample it
        self.train_latent = df_train.values

        if self.test_data == self.dataset:
            test_csv = glob.glob(self.train_dir+'/*.csv')
        else:
            test_csv = glob.glob(self.test_dir+'/'+self.test_data+'*csv')
        print(test_csv)
        df_test = pd.read_csv(test_csv[0],
            header=None, delimiter=' ')
        self.test_latent = df_test.values

    def computeStatistics(self):
        self.train_mu = np.mean(self.train_latent, axis=1)
        self.train_sigma = np.var(self.train_latent, axis=1)

        self.test_mu = np.mean(self.test_latent, axis=1)
        self.test_sigma = np.var(self.test_latent, axis=1)

    def run(self):
        self.loadData()
        self.computeStatistics()

        self.mu = np.mean(self.train_mu)
        count = 0
        for i in range(len(self.test_mu)):
            diff = self.test_mu[i] - self.mu   
            value = diff**2 + self.test_sigma[i]
            print(value)
            if value > 0.00998:
                count += 1
        print("There are %d defective images out of %d samples." %(count, len(self.test_mu))) 

class FIDSEP(Base):
    def __init__(self, args):
        super(FIDSEP, self).__init__(args)
        # create fid stats folder
        fid_stats_dir = './fid_stats'
        if not os.path.exists(fid_stats_dir):
            os.makedirs(fid_stats_dir)
        self.train_outname = os.path.join('fid_stats',self.dataset+'_'+self.date+'.npz')

        if self.dataset == self.test_data:
            self.test_outname = self.train_outname
        else:
            self.test_outname = os.path.join('fid_stats',self.test_data+'_'+self.date+'_test.npz')

    def check_fid_stats(self):
        if not glob.glob(self.train_outname): 
            raise Exception("You need to run FID first")
        else:
            print("Train statistics found!")
        print("Load test latent vectors")
        self.test_latent = self.load_data()
        
    def load_data(self):
        test_csv = glob.glob(self.test_dir+'/'+self.test_data+'*csv')
        print(test_csv)
        df_test = pd.read_csv(test_csv[0],
            header=None, delimiter=' ')
        latent = df_test.as_matrix()
        return latent
        
    def calculate_statistics(self, i):
        vec = self.test_latent[i:i+1].transpose()
        mu = np.mean(vec, axis=0)
        #sigma = np.var(self.test_latent[i*stride:i*stride+stride], rowvar=True)
        
        #sigma = np.cov(vec, rowvar=True)
        print(vec.shape)
        tmp = np.ones((1, len(mu)))
        I = np.identity(len(mu))
        sigma = np.abs(vec * tmp * I)
        return mu, sigma

    def calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        #print("calculate fid")
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        #sigma1 = tmp * sigma1
        #sigma2 = sigma2 * tmp
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"
        
        diff = mu1 - mu2
        
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            warnings.warn(("fid calculation produces singular product; adding {} to diagonal of conv estimates").format(eps))
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
            # numerical errpr might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag,0,atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        return diff.dot(diff)+np.trace(sigma1)+np.trace(sigma2)-2*tr_covmean
        
        #print(diff)
        #return diff.dot(diff)# + (sigma1-sigma2)**2
    def fit(self):
        start_time = time.time()

        # first check fid stats
        self.check_fid_stats()

        #self.fid_recon = []
        f_train = np.load(self.train_outname)
        mu_train, sigma_train = f_train['mu'][:], f_train['sigma'][:]
        f_train.close()
        print(sigma_train)
        print(len(self.test_latent[:])//2)
        fp_count = 0
        fn_count = 0
        for i in range(len(self.test_latent[:])//2):
            mu_test, sigma_test = self.calculate_statistics(i)
        
            #print(mu_train.shape, sigma_train.shape)
            #print(mu_test.shape, sigma_test.shape)
        
            fid_value = self.calculate_fid(mu_train, sigma_train,
                mu_test, sigma_test)
            #self.fid_recon.append(fid_value)
            print("FID value is %s" %(fid_value))
            sigma_max = np.max(sigma_train)
            if (np.mean(mu_train) + sigma_max) < fid_value:
                fn_count = fn_count + 1
            '''
            if fid_value < 0.9:
                fp_count = fp_count + 1
            if fid_value > 1.1:
                fn_count = fn_count + 1
            '''
        #print(len(self.fid_recon))
        print(fp_count)
        print(fn_count)
        elapsed_time = time.time() - start_time
        #print("FID value is %s, consuming %s s" %(fid_value,elapsed_time))

        #self.recon()
        return fid_value


class FID(Base):
    def __init__(self, args):
        super(FID, self).__init__(args)

        # create fid stats folder
        fid_stats_dir = './fid_stats'
        if not os.path.exists(fid_stats_dir):
            os.makedirs(fid_stats_dir)
        self.train_outname = os.path.join('fid_stats',self.dataset+'_'+self.date+'.npz')
        #self.train_outname = os.path.join('fid_stats', self.dataset+'_'+self.date+'_test.npz')

        if self.dataset == self.test_data:
            self.test_outname = self.train_outname
        else:
            self.test_outname = os.path.join('fid_stats',self.test_data+'_'+self.date+'_test.npz')

    def check_fid_stats(self):
        if not glob.glob(self.train_outname): 
            print("No train statistics found... creating")
            self.train_latent = self.load_data(stage='train')
            self.calculate_statistics(stage='train')
        else:
            print("Train statistics found!")
        if not glob.glob(self.test_outname):
            print("No test statistics found... creating")
            self.test_latent = self.load_data(stage='test')
            self.calculate_statistics(stage='test')
        else:
            print("Test statistics found!")

    def load_data(self, stage):
        print("load data")
        latent = []
        if stage == 'train':
            # read train latent codes from csv
            train_csv = glob.glob(self.train_dir+'/*.csv')
            df_train = pd.read_csv(train_csv[0],
                header=None, delimiter=' ')
            # because the number are so many, we sample it
            latent = df_train.as_matrix()

        elif stage == 'test':  
            test_csv = glob.glob(self.test_dir+'/'+self.test_data+'*csv')
            print(test_csv)
            df_test = pd.read_csv(test_csv[0],
                header=None, delimiter=' ')
            
            latent = df_test.values
        else:
            raise Exception("Unknown stage for load data!")
        return latent

    def calculate_statistics(self, stage):
        print("calculate statistics")
        if stage == 'train':
            mu = np.mean(self.train_latent, axis=0)
            sigma = np.cov(self.train_latent, rowvar=False)
            np.savez_compressed(self.train_outname, mu=mu, sigma=sigma)
        elif stage == 'test':
            mu = np.mean(self.test_latent, axis=0)
            sigma = np.cov(self.test_latent, rowvar=False)
            np.savez_compressed(self.test_outname, mu=mu, sigma=sigma)
        else:
            raise Exception("Unknown stage for calculate statistics!")

    def calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        print("calculate fid")
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"
        
        diff = mu1 - mu2
        print(np.sum(np.absolute(diff)))
        print("mean mu diff: ", np.mean(diff))
        print("diff.dotdiff: ", diff.dot(diff))
        
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            warnings.warn(("fid calculation produces singular product; adding {} to diagonal of conv estimates").format(eps))
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
            # numerical errpr might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag,0,atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        print("trace(sigma1): ", np.trace(sigma1))
        print("trace(sigma2): ", np.trace(sigma2))
        print("covmean: ", tr_covmean)
        
        return diff.dot(diff)+np.trace(sigma1)+np.trace(sigma2)-2*tr_covmean

    def run(self):
        start_time = time.time()

        # first check fid stats
        self.check_fid_stats()

        f_train = np.load(self.train_outname)
        mu_train, sigma_train = f_train['mu'][:], f_train['sigma'][:]
        f_train.close()
        f_test = np.load(self.test_outname)
        mu_test, sigma_test = f_test['mu'][:], f_test['sigma'][:]
        f_test.close()
        print(mu_train.shape, sigma_train.shape)
        print(mu_test.shape, sigma_test.shape)
        
        fid_value = self.calculate_fid(mu_train, sigma_train,
            mu_test, sigma_test)
            
        elapsed_time = time.time() - start_time
        print("FID value is %s, consuming %s s" %(fid_value,elapsed_time))
        return fid_value

def parse_args():
    desc = "FID"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-dataset', type=str, default='solar', help='The name of dataset', required=True)
    parser.add_argument('-date', type=str, help='Date of this experiment', required=True)
    parser.add_argument('-test_data', type=str, help='Test which?')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    if args is None:
        exit()
        
    print("[*] Running Frechet distance statistics.")
    fid = FID(args)
    fid.run()

    print("[*] Running Separate Frechet distance statistics.")
    fid = FIDPREDICTSINGLE(args)
    fid.run()
