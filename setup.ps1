# install CaImAn by using anaconda
git clone https://github.com/flatironinstitute/CaImAn
cd CaImAn
conda env create -f environment.yml -n caiman
conda activate caiman
pip install .
mv caiman .caiman # need to change dir name for fix caiman import path in caimanmanager.py
python caimanmanager.py install --force
cd ../
rm -r -fo CaImAn

# install cnmfe-reviewer from source
git clone https://github.com/jf-lab/cnmfe-reviewer.git
cd cnmfe-reviewer
pip install .
mkdir ../data/cnmfe-reviewer/
mv data/* ../data/cnmfe-reviewer/
cd ../
rm -r -fo cnmfe-reviewer
