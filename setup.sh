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

# install auto-sklearn
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
pip install auto-sklearn

# install cnmfe-reviewer from source
git clone git@github.com:jf-lab/cnmfe-reviewer.git
cd cnmfe-reviewer
pip install .
mv data/ ../data/cnmfe-reviewer
cd ../
rm -rf cnmfe-reviewer/ 
