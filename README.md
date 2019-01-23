# glo_0.1
1. Training :                            python main.py -date=DATE -dataset=DATASET -s=train -e=epoch
2. Testing :                             python main.py -date=DATE -dataset=DATASET -s=test  -e=epoch    -test_data=TEST_DATA
3. Interpolation between latent vectors :python main.py -date=DATE -dataset=DATASET -s=ip    -e=epoch
4. Frechet Inception Distance :          python fid.py  -date=DATE -dataset=DATASET -test_data=TEST_DATA
5. Visualizing latent space :            python tsne.py -date=DATE -dataset=DATASET -test_data=TEST_DATA -space=latent
5. Visualizing image space :             python tsne.py -date=DATE -dataset=DATASET -test_data=TEST_DATA -space=image
