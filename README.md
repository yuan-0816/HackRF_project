# HackRF project
---
## Updating Firmware
1. Download the last release [HackRF Releases](https://github.com/greatscottgadgets/hackrf/releases).
2. connect HackRF to your computer via USB.
3. Open terminal and navigate to the folder where you downloaded the repostory.
4. Run the following command to update the firmware:
```
cd firmware-bin
hackrf_spiflash -w hackrf_one_usb.bin
``` 

## Installation HackRF Software form source
1. Open terminal and navigate to the folder where you downloaded the repostory.
2. Run the following command to install the software:
```
cd hackrf/host
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig
```
3. Verify the installation by running:
```
hackrf_info
```
---
## Third-Party Installation
### gps-sdr-sim
1. Download the gps-sdr-sim from GitHub:
```
# if repo not in third_party folder, run:
cd third_party
git clone https://github.com/osqzss/gps-sdr-sim.git
``` 
2. Build gps-sdr-sim:
```
cd gps-sdr-sim
gcc gpssim.c -lm -O3 -o gps-sdr-sim
```
3. Verify the installation by running:
```
./gps-sdr-sim -h
```




---
## requirements
### Ubuntu
```
sudo apt update
sudo apt install -y \
    gnuradio \
    gr-osmosdr \
    hackrf \
```
### python dependencies
```
pip install --upgrade pip

# 數值與繪圖
pip install numpy scipy matplotlib

# MAVLink 解析
pip install pymavlink

# 如果想用 Python 操作 HackRF（非必需，但有用）
pip install hackrf
```
