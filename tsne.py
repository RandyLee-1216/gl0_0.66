from __future__ import division, print_function
# from sklearn.manifold import TSNE
import sklearn
import scipy, scipy.misc
from scipy.misc import imread, imresize
from scipy import linalg
import pandas as pd
import math, csv, argparse
import os, glob, cv2
import matplotlib as mpl
mpl.use('Agg')
from PIL import Image

def read_csv(args):
    train_dir = os.path.join('results', args.dataset, args.date)
    print("train_dir: ", train_dir)
    test_dir = os.path.join(train_dir, 'test')
    #test_dir = train_dir
    print("test_dir: ", test_dir)

    train_csv = glob.glob(train_dir+'/*.csv')
    print(train_csv)
    df_train = pd.read_csv(train_csv[0],
        header=None, delimiter=' ')
    #test_csv = glob.glob(test_dir+'/*csv')
    test_csv = glob.glob(test_dir+'/'+args.test_data+'*csv')
    print(test_csv)
    df_test = pd.read_csv(test_csv[0],
        header=None, delimiter=' ')

    if args.td2 is not None:
        test_csv_2 = glob.glob(test_dir+'/'+args.td2+'*csv')
        print(test_csv_2)
        df_test_2 = pd.read_csv(test_csv_2[0],
            header = None, delimiter=' ')
        return df_train, df_test, df_test_2

    return df_train, df_test

def load_data_image(path, size, stage):
    img = []
    filenames = list(map(lambda x: os.path.join(path,x), os.listdir(path)))
    samples = np.random.randint(len(filenames), size=size)
    if stage == 'train':

        for i in range(len(samples)):
            img_tmp = cv2.imread(filenames[i], 0)
            img.append(img_tmp.flatten())
            #print(img_tmp.flatten().shape)
        target = np.chararray(len(samples))
        target[:] = 'b'
    else:
        for i in range(len(samples)):
            img_tmp = cv2.imread(filenames[i], 0)
            img.append(img_tmp.flatten())
        target = np.chararray(len(samples))
        target[:] = 'r'
    data = np.array(img).squeeze()
    return data, target
        

def load_data(df, size, stage):
    latent = []
    samples = np.random.randint(len(df[:]), size=size)
    for i in range(len(samples)):
        latent_tmp = np.vstack(df.iloc[samples[i]]).transpose()
        latent.append(latent_tmp)
    target = np.chararray(len(samples))

    if stage == 'train':
        target[:] = 'b'
    elif stage == 'test':
        target[:] = 'r'
    elif stage == 'two':
        target[:] = 'g'
    data = np.array(latent).squeeze()
    
    return data, target

def wrap(*args):
    ''' wrapping data in sklearn format'''
    #for i in range(3):
    #data = np.concatenate([args[0], args[2], args[4]])
    #target = np.concatenate([args[1], args[3], args[5]])
    data = np.concatenate([args[0], args[2]])
    target = np.concatenate([args[1], args[3]])
    return Bunch(data=data, target=target, DESCR='latent')

def wrap_train(*args):
    ''' wrapping data in sklearn format'''
    #data = np.concatenate([args[0], args[2]])
    #target = np.concatenate([args[1], args[3]])
    data = args[0]
    target = args[1]
    return Bunch(data=data, target=target, DESCR='latent')

def parse_args():
    desc = "TSNE"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-dataset', type=str, default='solar', help='The name of dataset', required=True)
    parser.add_argument('-date', type=str, help='Date of this experiment', required=True)
    parser.add_argument('-test_data', type=str, help='Test which?', required=True)
    parser.add_argument('-td2', type=str, help='Second test data')
    parser.add_argument('-space', type=str, help='Image space or latnet space', required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    if args is None:
        exit()
    
    if args.space == 'latent':
        print("Performing t-SNE on latent space")
        train_dir = os.path.join('results', args.dataset, args.date)
        save_path = os.path.join(train_dir, args.test_data+'_tsne.png')

        df_tr, df_te = read_csv(args)
        
        train_data, train_target = load_data(df_tr, 300, stage='train')
        test_data, test_target = load_data(df_te, 30, stage='test')
        
        #data = wrap_train(train_data, train_target)
        data = wrap(train_data, train_target, test_data, test_target)
        #print(data.target)
    elif args.space == 'image':
        print("Performing t-SNE on image space")
        train_dir = os.path.join('../data', args.dataset, 'ok')
        test_dir = os.path.join('../data', args.dataset, args.test_data)
        save_path = os.path.join('results', args.dataset, args.date, args.test_data+'_imgspace.png')
        
        train_data, train_target = load_data_image(train_dir, 300, stage='train')
        test_data, test_target = load_data_image(test_dir, 30, stage='test')
        
        data = wrap(train_data, train_target, test_data, test_target)
    elif args.space == 'two':
        print("Performing t-SNE on latent space with two defects")
        train_dir = os.path.join('results', args.dataset, args.date)
        save_path = os.path.join(train_dir, 'two_tsne.png')

        df_tr, df_te, df_te_2 = read_csv(args)

        train_data, train_target = load_data(df_tr, 300, stage='train')
        test_data, test_target = load_data(df_te, 30, stage='test')
        test_data_2, test_target_2 = load_data(df_te_2, 30, stage='two')

        data = wrap(train_data, train_target, test_data, test_target, test_data_2, test_target_2)
    
    for i in range(10):
        print("Start TSNE")
        X_tsne = TSNE(n_components=2).fit_transform(data.data)
        print("TSNE prediction finished")
        _,c = X_tsne.shape
    
        # plotting
        plt.figure()
        plt.scatter(X_tsne[0:300,0], X_tsne[0:300,1], c='b',
            marker='o', label="t-SNE Visualization")
        plt.scatter(X_tsne[300:330,0], X_tsne[300:330,1], c='r',
            marker='x', label="t-SNE Visualization")
        if args.space == 'two':
            plt.scatter(X_tsne[330:360,0], X_tsne[330:360,1], c='g',
                marker='x', label="t-SNE Visualization")
        save_path = os.path.join(train_dir, 'two_tsne'+str(i)+'.png')
        plt.savefig(save_path, dpi=120)
        plt.show()
        print('TSNE visualization completed!')

if __name__=="__main__":
    args = parse_args()
    if args is None:
        exit()
    
    main()
