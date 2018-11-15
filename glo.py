from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch, glob
from torch import nn
import torch.nn.functional as fnn
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torchvision.datasets import LSUN
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import argparse, os, glob, time
import utils, random
from utils import colors

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
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)

#-------------------------------------------------------------------------------
def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)

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
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

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
class Generator(nn.Module):
    def __init__(self, code_dim, n_filter=64, out_channels=3):
        super(Generator, self).__init__()
        self.code_dim = code_dim
        nf = n_filter
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(code_dim, nf * 8, 4, 1, 0, bias=False), # 1x1 -> 4X4
            nn.BatchNorm2d(nf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False), # 4x4 -> 8X8
            nn.BatchNorm2d(nf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False), # 8x8 -> 16X16
            nn.BatchNorm2d(nf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf    , 4, 2, 1, bias=False), # 16x16 -> 32X32
            nn.BatchNorm2d(nf), nn.ReLU(True),
            nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=False), # 32x32 -> 64X64 # original
            #nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=False), # 32X32 -> 64X64 # change filter number
            # add one layer to increase size to 128X128
            #nn.BatchNorm2d(nf//2), nn.ReLU(True),
            #nn.ConvTranspose2d(nf//2, out_channels, 4, 2, 1, bias=False), # 64X64 -> 128X128
            
            nn.Tanh(),
        )

    def forward(self, code):
        return self.dcnn(code.view(code.size(0), self.code_dim, 1, 1))

#-------------------------------------------------------------------------------
class Generator_128(nn.Module):
    def __init__(self, code_dim, n_filter=64, out_channels=3):
        super(Generator_128, self).__init__()
        self.code_dim = code_dim
        nf = n_filter
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(code_dim, nf * 8, 4, 1, 0, bias=False), # 1X1 -> 4X4
            nn.BatchNorm2d(nf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False), # 4x4 -> 8X8
            nn.BatchNorm2d(nf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False), # 8x8 -> 16X16
            nn.BatchNorm2d(nf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf    , 4, 2, 1, bias=False), # 16x16 -> 32X32
            nn.BatchNorm2d(nf), nn.ReLU(True),
            #nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=False), # 32x32 -> 64X64 # original
            nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=False), # 32X32 -> 64X64 # change filter number
            # add one layer to increase size to 128X128
            nn.BatchNorm2d(nf//2), nn.ReLU(True),
            nn.ConvTranspose2d(nf//2, out_channels, 4, 2, 1, bias=False), # 64X64 -> 128X128
            
            nn.Tanh(), # original
        )

    def forward(self, code):
        return self.dcnn(code.view(code.size(0), self.code_dim, 1, 1))
        
#-------------------------------------------------------------------------------
class Generator_150(nn.Module):
    def __init__(self, code_dim, n_filter=64, out_channels=3):
        super(Generator_150, self).__init__()
        self.code_dim = code_dim
        nf = n_filter
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(code_dim, nf * 8, 4, 1, 0, bias=False), # 1X1 -> 4X4
            nn.BatchNorm2d(nf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False), # 4x4 -> 8X8
            nn.BatchNorm2d(nf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False), # 8x8 -> 16X16
            nn.BatchNorm2d(nf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf    , 8, 2, 1, bias=False), # 16x16 -> 36X36
            nn.BatchNorm2d(nf), nn.ReLU(True),
            #nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=False), # 32x32 -> 64X64 # original
            nn.ConvTranspose2d(nf, nf // 2, 4, 2, 0, bias=False), # 36X36 -> 74X74 # change filter number
            # add one layer to increase size to 150x150
            nn.BatchNorm2d(nf//2), nn.ReLU(True),
            nn.ConvTranspose2d(nf//2, out_channels, 4, 2, 0, bias=False), # 74x74 -> 150X150
            
            nn.Tanh(), # original
        )

    def forward(self, code):
        return self.dcnn(code.view(code.size(0), self.code_dim, 1, 1))

#-------------------------------------------------------------------------------
def interpolation(
        date,
        dataset='solar',
        image_output_prefix='glo',
        code_dim=100,
        epochs=25,
        use_cuda=True,
        batch_size=128,
):
    print(colors.BLUE+"[*] start interpolating"+colors.ENDL)
    save_dir = 'results/'+dataset+'/'+date
    test_save_dir = save_dir+'/test'
    ip_save_dir = save_dir+'/ip'
    if not os.path.exists(ip_save_dir):
        os.makedirs(ip_save_dir)
        
    if dataset == 'wood':
        csv_name = glob.glob(test_save_dir+'/*csv')
    else:
        csv_name = glob.glob(test_save_dir+'/gas_leak*csv')
        #csv_name = glob.glob(save_dir+'/*csv')
    df = pd.read_csv(csv_name[0], header=None, delimiter=' ')
    print("csv file of shape = "+str(df.shape))
    
    def maybe_cuda(tensor):
        return tensor.cuda() if use_cuda else tensor
    
    g = maybe_cuda(Generator(code_dim))
    pretrained_file = glob.glob(save_dir+'/*.pth')
    if pretrained_file is not None:
        g.load_state_dict(torch.load(pretrained_file[0]))
        print("load pre-trained weights success")
        for name, p in g.named_parameters():
            print(name)
            #print(p)
    else:
        raise Exception("No pre-trained file found!")
    
    # arbitrarily select 2 vectors
    random_num = random.sample(range(0,df.shape[0]), 2)
    print(random_num)
    z1 = np.vstack(np.array(df.iloc[random_num[0]])).transpose()
    z2 = np.vstack(np.array(df.iloc[random_num[1]])).transpose()
    z1 = maybe_cuda(torch.FloatTensor(z1))
    z2 = maybe_cuda(torch.FloatTensor(z2))
    
    # perform interpolation
    z = utils.ip_z(z1, z2, batch_size=64)
    z = Variable(z).cuda()
    rec = g(z)
    
    # visualize reconstruction results
    utils.imsave(ip_save_dir+'/rec.png',    
                 make_grid(rec[:64].data.cpu(),nrow=8,normalize=True,range=(0,1)).numpy().transpose(1,2,0))
    print(colors.BLUE+"[*] interpolation finish!"+colors.ENDL)

#-------------------------------------------------------------------------------
def test(
        date,
        test_data, 
        dataset='solar',
        image_output_prefix='glo',
        code_dim=100, 
        epochs=25,
        use_cuda=True,
        batch_size=128,
        lr_g=0.0,
        lr_z=10.,
        max_num_samples=100000,
        init='pca',
        n_pca=(64 * 64 * 3 * 2),
        loss='lap_l1',
):
    print(colors.BLUE+"[*] start testing"+colors.ENDL)
    
    if test_data == 'solar':
        test_loader = utils.load(
            data_dir='../data/solar/test', 
            batch_size=batch_size, img_size=64)
    elif test_data == 'gas_leak_dirt':
        test_loader, num_h, num_v = utils.load_full_chip(
            data_dir='../data/gas_leak_dirt/full_chip',
            dataname=dataset, batch_size=batch_size, img_size=64)
    elif test_data == 'flower_chip':
        test_loader = num_h, num_v = utils.load_full_chip(
            data_dir='../data/flower_chip/OK/full_chip',
            dataname=dataset, batch_size=batch_size, img_size=512)
    elif test_data == 'wood':
        test_loader = utils.load(
            data_dir='../data/wood/ok', 
            batch_size=batch_size, img_size=64, convert='L')
    elif dataset == 'lens':
        test_loader = utils.load(
            #data_dir='../contact_lens/line/cut/ng',
            data_dir='../contact_lens/line/cut/ng2',
            batch_size=batch_size, img_size=128, convert='L')
    else:
        raise Exception(colors.FAIL+"No such dataset!!"+colors.ENDL)

    save_dir = 'results/'+dataset+'/'+date
    test_save_dir = save_dir+'/test'
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)

    def maybe_cuda(tensor):
        return tensor.cuda() if use_cuda else tensor

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
    
    if dataset == 'solar':
        g = maybe_cuda(Generator(code_dim))
    elif dataset == 'wood':
        g = maybe_cuda(Generator(code_dim, out_channels=1))
    else:
        g = maybe_cuda(Generator_150(code_dim, out_channels=1))
    
    print(save_dir)
    pretrained_file = glob.glob(save_dir+'/*.pth')
    if not pretrained_file:
        raise Exception(colors.FAIL+"No pre-trained file found!"+colors.ENDL)  
    g.load_state_dict(torch.load(pretrained_file[0]))
    print("load pre-trained weights success")
    '''
    rec = g(Variable(maybe_cuda(torch.FloatTensor(Z[:64]))))
    utils.imsave(test_save_dir+'/%s_rec_epoch_%03d.png' % (image_output_prefix, 1), 
               make_grid(rec.data.cpu(),nrow=8,normalize=True,range=(0,1)).numpy().transpose(1,2,0))
    '''
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
    '''
    if dataset == 'solar' and test_data != 'solar':
        # create a graph for reconstruction
        rec_img = np.zeros((num_h*64,num_v*64,3))
        diff_img = np.zeros((num_h*64,num_v*64,3))
    '''
    for i, (Xi, yi, idx) in enumerate(test_loader):
        if i == (test_loader.dataset.__len__() // batch_size):
            print(len(idx))
            #break
           
        losses = []
        progress = tqdm(total=epochs-1, desc='batch iter % 3d' % (i+1))
        Xi = Variable(maybe_cuda(Xi))
        zi.data = maybe_cuda(torch.FloatTensor(Z[idx.numpy()]))
        epoch_start_time = time.time()
        
        '''
        # initialize rec with true data
        rec = Xi
        '''
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            rec = g(zi)
            #print(rec.size())
            loss = loss_fn(rec, Xi)
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
    print(colors.BLUE+"[*] test stage finish!"+colors.ENDL)

#-------------------------------------------------------------------------------
def train(
        date,
        dataset='solar',
        image_output_prefix='glo',
        code_dim=100, 
        epochs=25,
        use_cuda=True,
        batch_size=128,
        lr_g=1.,
        lr_z=10.,
        max_num_samples=100000,
        init='pca',
        n_pca=(64 * 64 * 3 * 2),
        loss='lap_l1',
):
    print(colors.BLUE+"[*] start training"+colors.ENDL)
    save_dir = 'results/'+dataset+'/'+date
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def maybe_cuda(tensor):
        return tensor.cuda() if use_cuda else tensor

    # load datasets for training and validation set
    if dataset == 'solar':
        train_loader = utils.load(
            data_dir='../data/solar/train',
            batch_size=batch_size)
        val_loader = utils.load(
            data_dir='../data/solar/train',
            batch_size=8*8)
    elif dataset == 'wood':
        train_loader = utils.load(
            data_dir='../data/wood/ok', 
            batch_size=batch_size, convert='L')
        val_loader = utils.load(
            data_dir='../data/wood/ok',
            batch_size=8*8, convert='L')
    elif dataset == 'DAGM_10':
        train_loader = utils.load(
            data_dir='../data/DAGM_10/ok',
            batch_size=batch_size, img_size=128, convert='L')
        val_loader = utils.load(
            data_dir='../data/DAGM_10/ok',
            batch_size=8*8, img_size=128, convert='L')
    elif dataset == 'flower_chip':
        train_loader = utils.load_multi(data_dir='../data/flower_chip/OK/train', batch_size=batch_size)
        val_loader = utils.load_multi(data_dir='../data/flower_chip/OK/train', batch_size=8*8)
    elif dataset == 'benq':
        train_loader = utils.load(
            data_dir='/home/itri/kevin/data/benq',
            batch_size=batch_size, img_size=128, convert='L')
        val_loader = utils.load(
            data_dir='/home/itri/kevin/data/benq',
            batch_size=8*8, img_size=128, convert='L')
    elif dataset == 'wood_mixed':
        train_loader = utils.load(
            data_dir='../data/wood/mixed',
            batch_size=batch_size, convert='L')
        val_loader = utils.load(
            data_dir='../data/wood/mixed',
            batch_size=8*8, convert='L')
    elif dataset == 'dark':
        train_loader = utils.load(
            data_dir='../0713/0713/OK_eqal',
            batch_size=batch_size, img_size=150, convert='L')
        val_loader = utils.load(
            data_dir='../0713/0713/OK_eqal',
            batch_size=8*8, img_size=150, convert='L')
    elif dataset == 'lens':
        train_loader = utils.load(
            data_dir='../contact_lens/line/cut/ok',
            batch_size=batch_size, img_size=128, convert='L')
        val_loader = utils.load(
            data_dir='../contact_lens/line/cut/ok',
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

    # project the latent vectors into a unit ball
    Z = utils.project_l2_ball(Z)

    # we use only 3 channels for solar panel and 1 channel otherwise
    if dataset == 'solar':
        g = maybe_cuda(Generator(code_dim))
    elif dataset == 'wood':
        g = maybe_cuda(Generator(code_dim, out_channels=1)) 
    elif dataset == 'wood_mixed':
        g = maybe_cuda(Generator(code_dim, out_channels=1))
    else:
        g = maybe_cuda(Generator_128(code_dim, out_channels=1))
        
    loss_fn = LapLoss(max_levels=3) if loss == 'lap_l1' else nn.MSELoss()
    zi = maybe_cuda(torch.zeros((batch_size, code_dim)))
    zi = Variable(zi, requires_grad=True)
    optimizer = SGD([
        {'params': g.parameters(), 'lr': lr_g}, 
        {'params': zi, 'lr': lr_z}
    ])

    Xi_val, _, idx_val = next(iter(val_loader))

    utils.imsave(save_dir+'/target.png',
           make_grid(Xi_val.cpu(),nrow=8,normalize=True,range=(0,1)).numpy().transpose(1,2,0))

    overall_loss = []
    for epoch in range(epochs):
        epoch_start_time = time.time()
        losses = []
        progress = tqdm(total=len(train_loader)-1, desc='epoch % 3d' %(epoch+1))

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
        utils.imsave(save_dir+'/%s_rec_epoch_%03d.png' % (image_output_prefix, epoch+1), 
               make_grid(rec.data.cpu(),nrow=8,normalize=True,range=(0,1)).numpy().transpose(1,2,0))
        print("avg epoch time", time.time() - epoch_start_time)
    
    utils.loss_plot(overall_loss, save_dir+'/train_loss.png')
    print("save loss plot")

    # save generator model
    torch.save(g.state_dict(), os.path.join(save_dir,image_output_prefix+'_rec_epoch_'+str(epochs)+'.pth'))
    print("generator model saved")

    print("saving optimized latent code")
    with open(save_dir+'/train_latent_code.csv', 'w') as f:
        np.savetxt(f, Z, delimiter=' ')
        
    print(colors.BLUE+"training finish!"+colors.ENDL)
