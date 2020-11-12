# install CaImAn by using anaconda
git clone https://github.com/flatironinstitute/CaImAn
cd CaImAn
conda env create -f environment.yml -n caiman
conda actiavte caiman
pip install .
mv caiman .caiman # need to change dir name for fix caiman import path in caimanmanager.py
python caimanmanager.py install
cd ../
rm CaImAn/
