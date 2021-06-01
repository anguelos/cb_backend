### Install packages in ubuntu for deployment and remote development
```bash
sudo apt-get install parallel byobu mc lynx python3-pip pssh
pip3 install --user gpustat  # run it as a gpu htop with: watch -n 0.5 -c gpustat -cp --color
```

### Install packages in ubuntu for deployment and remote development
```bash
pip3 install --upgrade pip # opencv-python requires this
pip3 install -r requirements.txt 
```

### If you have no root:
```bash
wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py --user
```

### Installing MKL:
This is not required for fast numpy indexing as the intel-scipy provides an MKL multithreaded backend.   

based on https://github.com/eddelbuettel/mkl4deb
```bash
cd /tmp
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update
sudo apt-get install intel-mkl-64bit-2018.2-046

sudo update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 50
sudo update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3 libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 50
sudo update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so  liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 50
sudo update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu  /opt/intel/mkl/lib/intel64/libmkl_rt.so 50

sudo sh -c 'echo "/opt/intel/lib/intel64"     >  /etc/ld.so.conf.d/mkl.conf'
sudo sh -c 'echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf'
sudo ldconfig

sudo sh -c 'echo "MKL_THREADING_LAYER=GNU" >> /etc/environment' # ~/.bash_profile and other locations works as well
```

### Install train/evaluation data and models
```bash
mkdir -p models
mkdir -p ./data/compiled_fake_db/
cd ./data/

wget -c rr.visioner.ca/assets/cbws/fake_db.tar.bz2
wget -c rr.visioner.ca/assets/cbws/demo_db.tar.bz2

tar -xpvjf fake_db.tar.bz2
(mkdir demo_db; cd demo_db; tar -xpvjf demo_db.tar.bz2)

mkdir -p fake_db_overlaid_all/chronicle
cp -Rp ./fake_db/*/chronicle/* ./fake_db_overlaid_all/chronicle
mkdir -p fake_db_overlaid_test/chronicle
cp -Rp ./fake_db/chudenice_2/chronicle/* ./fake_db/plasy/chronicle/* ./fake_db_overlaid_test/chronicle

cd ../models
wget rr.visioner.ca/assets/cbws/phocnet_0x0.pt
wget rr.visioner.ca/assets/cbws/srunet.pt
wget rr.visioner.ca/assets/cbws/box_iou.pt
cd ..

```

### Filesystems
```bash
mkdir -p /data/storage/overlay/cb_91/fake_db
mount -t overlay overlay -o lowerdir=/data/storage/new_root/fake_db,workdir=/data/storage/overlay/cb_91/fake_db /data/storage/union_fake_db

```