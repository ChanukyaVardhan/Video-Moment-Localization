mkdir data

## Download data for charades dataset
wget http://cvlab.postech.ac.kr/research/LGI/charades_data.tar.gz
tar zxvf charades_data.tar.gz
mv charades data
rm charades_data.tar.gz

## Download data for activitynet captions dataset
wget https://www.dropbox.com/sh/dszrtb85nua2jqe/AABpxSfzGoFs1j6k5LpE2f46a/ActivityNet/activitynet_v1-3.part-00
wget https://www.dropbox.com/sh/dszrtb85nua2jqe/AACx8ycE-7zwuPQa7j6eSFeQa/ActivityNet/activitynet_v1-3.part-01
wget https://www.dropbox.com/sh/dszrtb85nua2jqe/AAD_hhuUE8a6zwwN8TTtTUNVa/ActivityNet/activitynet_v1-3.part-02
wget https://www.dropbox.com/sh/dszrtb85nua2jqe/AABmpioAzCRTR6j0Eu1R4pLta/ActivityNet/activitynet_v1-3.part-03
wget https://www.dropbox.com/sh/dszrtb85nua2jqe/AAArYY7XdWlgjfI_D_S4nXWfa/ActivityNet/activitynet_v1-3.part-04
wget https://www.dropbox.com/sh/dszrtb85nua2jqe/AADjnmEPcewHfzBjgunZ2GYba/ActivityNet/activitynet_v1-3.part-05
cat activitynet_v1-3.part-* > temp.zip
rm activitynet_v1-3.part-*
unzip temp.zip
rm temp.zip
wget https://raw.githubusercontent.com/microsoft/VideoX/master/2D-TAN/data/ActivityNet/train.json
wget https://raw.githubusercontent.com/microsoft/VideoX/master/2D-TAN/data/ActivityNet/val.json
wget https://raw.githubusercontent.com/microsoft/VideoX/master/2D-TAN/data/ActivityNet/test.json
mkdir data/activitynet
mv train.json val.json test.json sub_activitynet_v1-3.c3d.hdf5 data/activitynet

## Download data for TACoS dataset
wget https://www.dropbox.com/sh/dszrtb85nua2jqe/AACNL0hqugZb0JY7Mmn9IOIQa/TACoS/tall_c3d_features.hdf5
wget https://raw.githubusercontent.com/microsoft/VideoX/master/2D-TAN/data/TACoS/train.json
wget https://raw.githubusercontent.com/microsoft/VideoX/master/2D-TAN/data/TACoS/val.json
wget https://raw.githubusercontent.com/microsoft/VideoX/master/2D-TAN/data/TACoS/test.json
mkdir data/tacos
mv train.json val.json test.json tall_c3d_features.hdf5 data/tacos
