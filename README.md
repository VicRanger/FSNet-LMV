# Adaptive Recurrent Frame Prediction with Learnable Motion Vectors [SIGGRAPH ASIA 2023 Conference Paper]

This is offical repository for our paper, **Adaptive Recurrent Frame Prediction with Learnable Motion Vectors**<br>
**Authors:** Zhizhen Wu, Chenyu Zuo, Yuchi Huo, Yazhen Yuan, Yinfan Peng, Guiyang Pu, Rui Wang and Hujun Bao.<br>
in SIGGRAPH Asia 2023 Conference Proceedings

> **Abstract:** The utilization of dedicated ray tracing graphics cards has revolutionized the production of stunning visual effects in real-time rendering. However, the demand for high frame rates and high resolutions remains a challenge. The pixel warping approach is a crucial technique for increasing frame rate and resolution by exploiting the spatio-temporal coherence. To this end, existing superresolution and frame prediction methods rely heavily on motion vectors from rendering engine pipelines to track object movements. This work builds upon state-of-the-art heuristic approaches by exploring a novel adaptive recurrent frame prediction framework that integrates learnable motion vectors. Our framework supports the prediction of transparency, particles, and texture animations, with improved motion vectors that capture shading, reflections, and occlusions, in addition to geometry movements. In addition, we introduce a feature streaming neural network, dubbed FSNet, that allows for the adaptive prediction of one or multiple sequential frames. Extensive experiments against state-of-the-art methods demonstrate that FSNet can operate at lower latency with significant visual enhancements and can upscale frame rates by at least two times. This approach offers a flexible pipeline to improve the rendering frame rates of various graphics applications and devices.

# Setup

1. Make a directory for workspace and clone the repository: 
```
mkdir fsnet-lmv; cd fsnet-lmv
git clone https://github.com/VicRanger/FSNet-LMV code
cd code
```
2. Install conda env: `scripts/create_env.sh` in Linux or `scripts/create_env.bat` in Windows

# Dataset Generation

## compress raw files into npz files
```
python src/test/test_export_buffer.py --config config/export/export_st.yaml
```

# Training
```
python src/test/test_trainer.py --train  --test --config config/shadenet_v5d4.yaml
```

#  Inference
(TBD)
# Resource
## A sample of dataset for inference.
(TBD) 

# Citation

Thank you for being interested in our paper. <br>
If you find our paper helpful or use our work in your research, please cite:
```
@inproceedings{10.1145/3610548.3618211,
author = {Wu, Zhizhen and Zuo, Chenyu and Huo, Yuchi and Yuan, Yazhen and Peng, Yifan and Pu, Guiyang and Wang, Rui and Bao, Hujun},
title = {Adaptive Recurrent Frame Prediction with Learnable Motion Vectors},
year = {2023},
isbn = {9798400703157},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3610548.3618211},
doi = {10.1145/3610548.3618211},
booktitle = {SIGGRAPH Asia 2023 Conference Papers},
articleno = {10},
numpages = {11},
keywords = {Frame Extrapolation, Real-time Rendering, Spatial-temporal},
location = {, Sydney, NSW, Australia, },
series = {SA '23}
}
```


# Contact
If you have any questions or suggestions about this repo, please feel free to contact me (jsnwu99@gmail.com).