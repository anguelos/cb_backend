### Install packages in ubuntu for deployment and remote development
```bash
sudo apt-get install parallel byobu mc lynx python3-pip 
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
