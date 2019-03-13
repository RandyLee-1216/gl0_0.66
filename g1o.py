import os, glob, random
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
from PIL import Image
import scipy
from scipy import linalg
from tqdm import tqdm
import torch, glob
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import utils
from utils import colors
#-------------------------------------------------------------------------------
use_cuda = True
def maybe_cuda(tensor):
    return tensor.cuda() if use_cuda else tensor

#-------------------------------------------------------------------------------
def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w), 
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)

#-------------------------------------------------------------------------------
def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.size()
    img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)

#-------------------------------------------------------------------------------
def laplacian_pyramid(img, kernel, max_levels=7):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr

#-------------------------------------------------------------------------------
class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        
    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.size()[1] != input.size()[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma, 
                n_channels=input.size()[1], cuda=input.is_cuda
            )
        pyr_input  = laplacian_pyramid( input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

#-------------------------------------------------------------------------------
class IndexedDataset(Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return (img, label, idx)
    
#-------------------------------------------------------------------------------
class Generator_128(nn.Module):
    def __init__(self, code_dim, n_filter=64, out_channels=3):
        super(Generator_128, self).__init__()
        self.code_dim = code_dim
        nf = n_filter
        self.dcnn = nn.Sequential(
            # nn.ConvTranspose2d(input_dim, output_dim, kernal_size, stride, padding, bias)
            nn.ConvTranspose2d(code_dim,nf * 8, 4, 1, 0, bias=False), # 2X2 -> 4X4
            nn.BatchNorm2d(nf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 8,  nf * 4, 4, 2, 1, bias=False), # 4x4 -> 8X8
            nn.BatchNorm2d(nf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4,  nf * 2, 4, 2, 1, bias=False), # 8x8 -> 16X16
            nn.BatchNorm2d(nf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2,  nf    , 4, 2, 1, bias=False), # 16x16 -> 32X32
            nn.BatchNorm2d(nf    ), nn.ReLU(True),
            #nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=False), # 32x32 -> 64X64 # original
            nn.ConvTranspose2d(nf,     nf // 2, 4, 2, 1, bias=False), # 32X32 -> 64X64 # change filter number
            # add one layer to increase size to 128X128
            nn.BatchNorm2d(nf//2), nn.ReLU(True),
            nn.ConvTranspose2d(nf//2, out_channels, 4, 2, 1, bias=False), # 64X64 -> 128X128
            
            nn.Tanh(), # original
        )

    def forward(self, code):
        return self.dcnn(code.view(code.size(0), self.code_dim, 1, 1))

#-------------------------------------------------------------------------------
def train(
        date,
        dataset='DAGM_8',
        image_output_prefix='glo',
        code_dim=100,
        epochs=25,
        use_cuda=True,
        batch_size=128,
        lr_g=.1,
        lr_z=.1,
        max_num_samples=100000,
        init='pca',
        n_pca=(64 * 64 * 3 * 2),
        loss='lap_l1',
):
    print(colors.BLUE+"================start training================"+colors.ENDL)
    save_dir = 'results/'+dataset+'/'+date
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load datasets for training and validation set
    if dataset == 'DAGM_10':
        train_loader = utils.load(data_dir='../data/DAGM_10/mix',
            batch_size=batch_size, img_size=128, convert='L')
        val_loader = utils.load(data_dir='../data/DAGM_10/mix',
            batch_size=8*8, img_size=128, convert='L')
    elif dataset == 'DAGM_8':
        train_loader = utils.load(data_dir='../data/DAGM_8/mix',
            batch_size=batch_size, img_size=128, convert='L')
        val_loader = utils.load(data_dir='../data/DAGM_8/mix',
            batch_size=8*8, img_size=128, convert='L')
    else:
        raise Exception("No such dataset!!")

    # we don't really have a validation set here, but for visualization let us 
    # just take the first couple images from the dataset

    # initialize representation space:
    if init == 'pca':
        from sklearn.decomposition import PCA

        # first, take a subset of train set to fit the PCA
        X_pca = np.vstack([
            X.cpu().numpy().reshape(len(X), -1)
            for i, (X, _, _)
             in zip(tqdm(range(n_pca // train_loader.batch_size), 'collect data for PCA'), 
                    train_loader)
        ])
        print("perform PCA...")
        pca = PCA(n_components=code_dim)
        pca.fit(X_pca)
        # then, initialize latent vectors to the pca projections of the complete dataset
        Z = np.empty((len(train_loader.dataset), code_dim))
        print(Z.shape)
        for X, _, idx in tqdm(train_loader, 'pca projection'):
            idx = idx.numpy()
            Z[idx] = pca.transform(X.cpu().numpy().reshape(len(X), -1))

    elif init == 'random':
        Z = np.random.randn(len(train_loader.dataset), code_dim)
    
    else:
        raise Exception("-i : choices=[pca random]")

    # project the latent vectors into a unit ball
    Z = utils.project_l2_ball(Z)

    # we use only 1 output channel
    g = maybe_cuda(Generator_128(code_dim, out_channels=1))
        
    loss_fn = LapLoss(max_levels=3) if loss == 'lap_l1' else nn.MSELoss()
    zi = maybe_cuda(torch.zeros((batch_size, code_dim)))
    zi = Variable(zi, requires_grad=True)
    optimizer = SGD([
        {'params': g.parameters(), 'lr': lr_g}, 
        {'params': zi, 'lr': lr_z}
    ])

    Xi_val, _, idx_val = next(iter(val_loader))

#    utils.imsave(save_dir+'/target.png',
#           make_grid(Xi_val.cpu(),nrow=8,normalize=True,range=(0,1)).numpy().transpose(1,2,0))

#    overall_loss = []
    
    # =================================================================================================
    # =========================================First part==============================================
    # =================================================================================================
    # do droping in one-third of all epochs
    epochs_d = round(epochs/3)
    
    # do droping 3 times
    cycle_d = 3
    for cyc in range(cycle_d):
        
        utils.imsave(save_dir+'/target.png',
           make_grid(Xi_val.cpu(),nrow=8,normalize=True,range=(0,1)).numpy().transpose(1,2,0))

        overall_loss = []

        for epoch in range(epochs_d):
            losses= []
            progress = tqdm(total=len(train_loader)-1, desc='epoch % 2d' %(epoch+1))
            
            for i, (Xi, yi, idx) in enumerate(train_loader):
                if i == train_loader.dataset.__len__() // batch_size:
                    break
                Xi = Variable(maybe_cuda(Xi))
                zi.data = maybe_cuda(torch.FloatTensor(Z[idx.numpy()]))

                optimizer.zero_grad()
                rec = g(zi)
                loss = loss_fn(rec, Xi)
                loss.backward()
                optimizer.step()

                Z[idx.numpy()] = utils.project_l2_ball(zi.data.cpu().numpy())

                losses.append(loss.item())
                progress.set_postfix({'loss': np.mean(losses[-100:])})
                progress.update()
        
            overall_loss.append(np.mean(losses[:]))
            progress.close()
            
            # visualize reconstructions
            rec = g(Variable(maybe_cuda(torch.FloatTensor(Z[idx_val.numpy()]))))
            if (epoch % 10)==0 :
                utils.imsave(save_dir+'/%s_epoch_%03d.png' % (image_output_prefix, epoch+1), 
                       make_grid(rec.data.cpu(),nrow=8,normalize=True,range=(0,1)).numpy().transpose(1,2,0))
        
            utils.loss_plot(overall_loss, save_dir+'/train_loss.png')
            print("save loss plot")    
            
        # save generator model
        torch.save(g.state_dict(), os.path.join(save_dir,image_output_prefix+'_rec_epoch_'+str(epochs)+'.pth'))
        print("generator model saved")    
        
        print("saving optimized latent code")
        with open(save_dir+'/train_latent_code.csv', 'w') as f:
            np.savetxt(f, Z, delimiter=' ')
        
        print(colors.BLUE+"training finish!"+colors.ENDL)
        
        # Start test and drop
        train_dir = save_dir
        train_csv = glob.glob(train_dir+'/*.csv')
        print('train_csv : %s' % train_csv)
        df_train = pd.read_csv(train_csv[0], header=None, delimiter=' ')
        train_latent = df_train.values
        
        train_mu    = np.mean(train_latent, axis=1)
        train_sigma = np.var(train_latent, axis=1)
        
        mu = np.mean(train_mu)
        count_value = 0
        ok_dir = os.listdir(os.path.join('../data', dataset, 'ok'))
        
        print(colors.BLUE+"====================Droping Finish==================="+colors.ENDL)
    
    # =================================================================================================
    # ========================================Second part==============================================
    # =================================================================================================
    
    
    
def test(
        date,
        test_data, 
        dataset='DAGM_8',
        image_output_prefix='glo',
        code_dim=100, 
        epochs=25,
        use_cuda=True,
        batch_size=128,
        lr_g=0.0,
        lr_z=.1,
        max_num_samples=100000,
        init='pca',
        n_pca=(64 * 64 * 3 * 2),
        loss='lap_l1',
):
    print(colors.BLUE+"[*] start testing"+colors.ENDL)
    
    if test_data == 'DAGM_10':
        test_loader = utils.load(data_dir='../data/DAGM_10/ng_original',
            batch_size=batch_size, img_size=128, convert='L')
    elif test_data == 'DAGM_8':
        test_loader = utils.load(data_dir='../data/DAGM_8/ng',
            batch_size=batch_size, img_size=128, convert='L')
    elif test_data == 'IC':
        test_loader = utils.load(data_dir='../data/IC/ng',
            batch_size=batch_size, img_size=128, convert='L')
    elif test_data == 'IC1':
        test_loader = utils.load(data_dir='../data/IC1/ng',
            batch_size=batch_size, img_size=128, convert='L')
    elif test_data == 'IC2':
        test_loader = utils.load(data_dir='../data/IC2/ng',
            batch_size=batch_size, img_size=128, convert='L')
    elif test_data == 'glue':
        test_loader = utils.load(data_dir='../data/glue/ng',
            batch_size=batch_size, img_size=128, convert='L')
    elif test_data == 'dark':
        test_loader = utils.load(data_dir='/home/itri/ddataa/LHE_dark/train/NG',
            batch_size=batch_size, img_size=128, convert='L')
    else:
        raise Exception(colors.FAIL+"No such dataset!!"+colors.ENDL)

    save_dir = 'results/'+dataset+'/'+date
    test_save_dir = save_dir+'/test'
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)

    # initialize representation space:
    if init == 'pca':
        from sklearn.decomposition import PCA

        # first, take a subset of train set to fit the PCA
        X_pca = np.vstack([
            X.cpu().numpy().reshape(len(X), -1)
            for i, (X, _, _)
             in zip(tqdm(range(n_pca // test_loader.batch_size), 'collect data for PCA'),
                    test_loader)
        ])
        print("performing PCA...")
        pca = PCA(n_components=code_dim)
        pca.fit(X_pca)
        # then, initialize latent vectors to the pca projections of the complete dataset
        Z = np.empty((len(test_loader.dataset), code_dim))
        print(Z.shape)
        for X, _, idx in tqdm(test_loader, 'pca projection'):
            idx = idx.numpy()
            Z[idx] = pca.transform(X.cpu().numpy().reshape(len(X), -1))

    elif init == 'random':
        Z = np.random.randn(len(test_loader.dataset), code_dim)

    # because we want to see the difference, so we don't project to unit ball
    # but it will not show the convex property, so we still project it to unit ball
    Z = utils.project_l2_ball(Z)
    g = maybe_cuda(Generator_128(code_dim, out_channels=1))
    
    print(save_dir)
    pretrained_file = glob.glob(save_dir+'/*.pth')
    if not pretrained_file:
        raise Exception(colors.FAIL+"No pre-trained file found!"+colors.ENDL)  
    g.load_state_dict(torch.load(pretrained_file[0]))
    print("load pre-trained weights success")
    

    loss_fn = LapLoss(max_levels=3) if loss == 'lap_l1' else nn.MSELoss()
    #loss_fn = nn.L1Loss().cuda()
    #loss_fn = nn.MSELoss().cuda()
    zi = maybe_cuda(torch.zeros((batch_size, code_dim)))
    zi = Variable(zi, requires_grad=True)
    optimizer = Adam([{'params': zi, 'lr': lr_z}])
    #optimizer = SGD([{'params': zi, 'lr': lr_z}])
    # fix the parameters of the generator
    for param in g.parameters():
        param.requires_grad = False

    for i, (Xi, yi, idx) in enumerate(test_loader):
        if i == (test_loader.dataset.__len__() // batch_size):
            print(len(idx))
            break
           
        losses = []
        progress = tqdm(total=epochs-1, desc='batch iter % 4d' % (i+1))
        Xi = Variable(maybe_cuda(Xi))
        zi.data = maybe_cuda(torch.FloatTensor(Z[idx.numpy()]))
        epoch_start_time = time.time()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(g(zi), Xi)
            loss.backward()
            optimizer.step()
            # we don't project back to unit ball in test stage
            #Z[idx.numpy()] = zi.data.cpu().numpy()
            Z[idx.numpy()] = utils.project_l2_ball(zi.data.cpu().numpy())
            #print(Z[idx.numpy()])
            losses.append(loss.item())
            progress.set_postfix({'loss': np.mean(losses[-100:])})
            progress.update()
         
        progress.close()

    print("saving optimized latent code")
    with open(test_save_dir+'/'+test_data+'_test_latent_code.csv', 'w') as f:
        np.savetxt(f, Z, delimiter=' ')
    print(colors.BLUE+"====================Testing Finish==================="+colors.ENDL)
