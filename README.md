# online-calcium-imaging
Real-time calucium imaging software based on [CaImAn](https://caiman.readthedocs.io/en/master/).
- frame-by-frame OnACID with simple GUI
    - works on both Windows and Mac
    - false-positive neuron detector based on [CNMF-E Reviewer](https://github.com/jf-lab/cnmfe-reviewer)
- real-time detection of synchronous firing of neurons
- laser handling module to interfere with mouse neurons

system overview: https://docs.google.com/presentation/d/1JQzA2FbH6-Qm684_ZjaZ5lzTnqpJ5fwVmMV1iSqYTr8/edit?usp=sharing

## Setup
**please set Python to version 3.7 before runnning below**
```bash
python -V # >> 3.7.0

# Mac / Linux
source setup.sh
```
**then download `data/` from [here](https://drive.google.com/file/d/1DZVDDY6LErDou6d9tBWW139qIyP7aYQm/view?usp=sharing) and unzip.**

Sample full video data is [here](https://drive.google.com/drive/folders/19JVMEmVVxG6AtkfQFEvBlNtHI4BpuHR0?usp=sharing).

## Run
```bash
# from miniscope
python main.py -s

# from video file
python main.py -f
```