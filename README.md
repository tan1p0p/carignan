# Carignan: Calcium real-time imaging for graphical network analysis of neurons
Real-time calucium imaging software based on [CaImAn](https://caiman.readthedocs.io/en/master/).
- frame-by-frame OnACID with simple GUI
    - works on Windows, Mac or Linux
    - false-positive neuron detector based on [CNMF-E Reviewer](https://github.com/jf-lab/cnmfe-reviewer)
        - Issue: automl model cannot use in Windows
- real-time detection of synchronous firing of neurons
- laser handling module to interfere with mouse neurons

system overview: https://docs.google.com/presentation/d/1JQzA2FbH6-Qm684_ZjaZ5lzTnqpJ5fwVmMV1iSqYTr8

## Setup
**please set Python to version 3.7 before runnning below**
```bash
python -V # >> 3.7.0

# Mac / Linux
source setup.sh

# Windows
setup.ps1
```
- Download data.zip from [here](https://drive.google.com/file/d/1DZVDDY6LErDou6d9tBWW139qIyP7aYQm) and unzip.
- Download .npy files to data/cnmfe-reviewer/ from [here](https://drive.google.com/drive/folders/1pGGwUzSI7Hm6gBrilP1SIm0C5bnX7MSO).

<!-- **Optional:** Sample full video data is [here](https://drive.google.com/drive/folders/19JVMEmVVxG6AtkfQFEvBlNtHI4BpuHR0). -->

## Run
```bash
# from miniscope
python main.py -s

# from video file
python main.py -f
```
