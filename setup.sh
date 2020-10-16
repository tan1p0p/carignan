# setup python environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install -r requirements.txt

# install CaImAn from source
git clone https://github.com/flatironinstitute/CaImAn
cd CaImAn
pip install .
mv caiman .caiman # need to change dir name for fix caiman import path in caimanmanager.py
python caimanmanager.py install
cd ../
rm -rf CaImAn/

