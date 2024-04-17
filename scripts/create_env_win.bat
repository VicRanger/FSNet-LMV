conda create --name study-py311-cu121 python=3.11 --yes
conda activate study-py311-cu121
conda update -n base -c defaults conda --yes
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia --yes
conda install matplotlib --yes
conda install -c conda-forge tqdm --yes
conda install -c anaconda scikit-image --yes
conda install -c conda-forge tensorboard --yes
conda install -c anaconda pillow --yes
conda install snakeviz --yes
conda install lxml --yes
pip install opencv-python==4.6.0.66
pip install onnx
pip install onnxsim
pip install pytorch-msssim
pip install lpips
pip install yacs
@REM choose your cuda version (cupy-cuda12x or cupy-cuda11x)
pip install cupy-cuda12x
@REM optional: install ipykernel
@REM pip install ipykernel
@REM conda install -n study-py311-cu117 ipykernel
@REM python -m ipykernel install --user --name study-py311-cu117 --display-name "study-py311-cu117"