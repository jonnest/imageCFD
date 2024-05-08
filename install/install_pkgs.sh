# all python packages
env_name="processing"

conda create -n $env_name python=3.9 -y
conda activate $env_name
conda install conda-forge::pyvista
conda install conda-forge::scikit-image
# pip install --upgrade --force-reinstall scikit-learn
conda install conda-forge::h5py
conda install conda-forge::scipy
pip install pyacvd
conda install conda-forge::tqdm
conda install conda-forge::skan
conda install conda-forge::trimesh
conda install conda-forge::rtree
conda install pyembree
pip install pyembree
# conda install conda-forge::pyembree
pip install pymeshfix


### put all in bash
conda activate;
env_name="proc2.0";
yes | conda create -n $env_name python=3.9;
yes | conda activate $env_name;
yes | conda install conda-forge::pyvista;
yes | pip install pymeshlab;
yes | conda install conda-forge::h5py;
yes | pip install pyacvd;
yes | conda install conda-forge::scikit-image;
yes | conda install conda-forge::scipy;
yes | conda install conda-forge::tqdm;
yes | conda install conda-forge::skan;
yes | conda install conda-forge::trimesh;
yes | conda install conda-forge::rtree;
yes | pip install pyembree;
yes | pip install pymeshfix;