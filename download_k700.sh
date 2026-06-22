#!bin/bash
wget -c https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_001.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_002.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_003.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_004.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_005.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_006.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_007.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_008.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_009.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/train/k700_train_010.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/val/k700_val_001.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/val/k700_val_003.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/val/k700_val_005.tar.gz
wget -c https://s3.amazonaws.com/kinetics/700_2020/val/k700_val_009.tar.gz
for archive in k700_train_*.tar.gz; do
    tar xzfv "$archive" -C videos/train/
done
for archive in k700_val_*.tar.gz; do
    tar xzfv "$archive" -C videos/infer/
done
