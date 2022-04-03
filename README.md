![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# [CVPR 2022] Eigenlanes: Data-Driven Lane Descriptors for Structurally Diverse Lanes
### Dongkwon Jin, Wonhui Park, Seong-Gyun Jeong, Heeyeon Kwon, and Chang-Su Kim

<img src="https://github.com/dongkwonjin/Eigenlanes/blob/main/Overview.png" alt="overview" width="90%" height="90%" border="10" />

Official implementation for **"Eigenlanes: Data-Driven Lane Descriptors for Structurally Diverse Lanes"** 
[[paper]](https://arxiv.org/abs/2203.15302) [[supp]](https://drive.google.com/file/d/1nRqSsf2bBDAA_s5XZ_BuKPyuHEr3OHJt/view?usp=sharing) [[video]](https://www.youtube.com/watch?v=XhEj3o3iihQ).

We construct a new dataset called **"SDLane"**. SDLane is available at [here](https://www.42dot.ai/akit/dataset). But, only test set is provided due to privacy issues. All dataset will be provided soon.

### Video
<a href="https://www.youtube.com/watch?v=XhEj3o3iihQ" target="_blank"><img src="https://img.youtube.com/vi/XhEj3o3iihQ/0.jpg" alt="Video" width="30%" height="30%" border="10"/></a>

### Related work
We wil also present another paper, **"Eigencontours: Novel Contour Descriptors Based on Low-Rank Approximation"**, accepted to CVPR 2022 (oral) [[paper]](https://arxiv.org/abs/2203.15259).
Congratulations to my eigenbrother!


### Requirements
- PyTorch >= 1.6
- CUDA >= 10.0
- CuDNN >= 7.6.5
- python >= 3.6

### Installation
1. Download repository. We call this directory as `ROOT`:
```
$ git clone https://github.com/dongkwonjin/Roadlane-Eigenlanes.git
```

2. Download [pre-trained model](https://drive.google.com/file/d/1rXck6jMQzsqIn_r3oOjLrPlUyYp8bXjN/view?usp=sharing) parameters and [preprocessed data](https://drive.google.com/file/d/1a7VtFmuLWBx1TaW7zkQg78wOjAxeWmJ9/view?usp=sharing) in `ROOT`:
```
$ cd ROOT
$ unzip pretrained.zip
$ unzip preprocessing.zip
```
4. Create conda environment:
```
$ conda create -n eigenlanes python=3.7 anaconda
$ conda activate eigenlanes
```
4. Install dependencies:
```
$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
$ pip install -r requirements.txt
```
 
    
### Directory structure
    .                           # ROOT
    ├── Preprocessing           # directory for data preprocessing
    │   ├── culane              # dataset name (culane, tusimple)
    |   |   ├── P00             # preprocessing step 1
    |   |   |   ├── code
    |   |   ├── P01             # preprocessing step 2
    |   |   |   ├── code
    |   │   └── ...
    │   └── ...                 # etc.
    ├── Modeling                # directory for modeling
    │   ├── culane              # dataset name (culane, tusimple)
    |   |   ├── code
    │   ├── tusimple           
    |   |   ├── code
    │   └── ...                 # etc.
    ├── pretrained              # pretrained model parameters 
    │   ├── culane              
    │   ├── tusimple            
    │   └── ...                 # etc.
    ├── preprocessed            # preprocessed data
    │   ├── culane              # dataset name (culane, tusimple)
    |   |   ├── P03             
    |   |   |   ├── output
    |   |   ├── P04             
    |   |   |   ├── output
    |   │   └── ...
    │   └── ...
    .

### Evaluation (for CULane)
To test on CULane, you need to install official CULane evaluation tools. The official metric implementation is available [here](https://github.com/XingangPan/SCNN/tree/master/tools/lane_evaluation). Please downloads the tools into `ROOT/Modeling/culane/code/evaluation/culane/`. The tools require OpenCV C++. Please follow [here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) to install OpenCV C++. Then, you compile the evaluation tools. We recommend to see an [installation guideline](https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/INSTALL.md)
```
$ cd ROOT/Modeling/culane/code/evaluation/culane/
$ make
```

### Train
1. Set the dataset you want to train (`DATASET_NAME`)
2. Parse your dataset path into the `-dataset_dir` argument.
3. Edit `config.py` if you want to control the training process in detail
```
$ cd ROOT/Modeling/DATASET_NAME/code/
$ python main.py --run_mode train --pre_dir ROOT/preprocessed/DATASET_NAME/ --dataset_dir /where/is/your/dataset/path/ 
```

### Test
1. Set the dataset you want to test (`DATASET_NAME`)
2. Parse your dataset path into the `-dataset_dir` argument.
3. If you want to get the performances of our work,
```
$ cd ROOT/Modeling/DATASET_NAME/code/
$ python main.py --run_mode test_paper --pre_dir ROOT/preprocessed/DATASET_NAME/ --paper_weight_dir ROOT/pretrained/DATASET_NAME/ --dataset_dir /where/is/your/dataset/path/
```
4. If you want to evaluate a model you trained,
```
$ cd ROOT/Modeling/DATASET_NAME/code/
$ python main.py --run_mode test --pre_dir ROOT/preprocessed/DATASET_NAME/ --dataset_dir /where/is/your/dataset/path/
```

### Preprocessing
Data preprocessing is divided into five steps, which are P00, P01, P02, P03, and P04. Below we describe each step in detail.
1. In P00, the type of ground-truth lanes in a dataset is converted to pickle format.
2. In P01, each lane in a training set is represented by 2D points sampled uniformly in the vertical direction.
3. In P02, lane matrix is constructed and SVD is performed. Then, each lane is transformed to its coefficient vector.
4. In P03, clustering is performed to obtain lane candidates.
5. In P04, training labels are generated to train the SI module in the proposed SIIC-Net.

If you want to get the preproessed data, please run the preprocessing codes in order. Also, you can download the preprocessed data.
```
$ cd ROOT/Preprocessing/DATASET_NAME/PXX_each_preprocessing_step/code/
$ python main.py --dataset_dir /where/is/your/dataset/path/
```

### Reference
```
@Inproceedings{
    Jin2022eigenlanes,
    title={Eigenlanes: Data-Driven Lane Descriptors for Structurally Diverse Lanes},
    author={Jin, Dongkwon and Park, Wonhui and Jeong, Seong-Gyun and Kwon, Heeyeon and Kim, Chang-Su},
    booktitle={CVPR},
    year={2022}
}
```
