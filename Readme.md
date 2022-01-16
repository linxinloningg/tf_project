###  TensorFlow保姆级别教入门，不会就把我头打爆。

### 前言：

仓库地址：https://github.com/linxinloningg/tf_project

数据集地址：https://aistudio.baidu.com/aistudio/datasetdetail/125619

### 玩转TensorFlow：

* **环境配置**：TensorFlow的环境配置比较多但又感觉很少人讲清楚，我安装的是GPU版的TensorFlow：

  * 首先要有conda，创建一个安装TensorFlow的虚拟环境，

    ```
    conda create -n tf_train python==3.7.3
    conda activate tf_train
    ```

    名字随意（我就叫tf_train）,python版本得3.7.3

  * 然后进入该虚拟环境，敲下

    ```bash
    conda install cudatoolkit=10.1
    conda install cudnn==7.6.5
    pip install tensorflow-gpu==2.3.0
    ```

    显而易见，TensorFlow GPU版的安装要搭配上对应cudatoolkit和cudnn一起使用，所以安装各个版本的TensorFlow最好先去官网查看一下相对应的配置。

  * 测试一下

    ```python
    import tensorflow as tf
    print(tf.test.is_gpu_available())
    ```

    如果输出True，即开发环境已经配置好了

* **下载源码**：TensorFlow是一个训练框架，所以官网貌似并没有提供什么源码供用户去训练，但要想自己实现也并不难，这里提供一下我的源码，并接下我将逐步讲解这一份源码的使用，仓库：https://github.com/linxinloningg/tf_project

  ![](readme.assets/image-20220113173442395.png)

  打卡仓库链接将会看到项目的主要文件代码：

  * data目录下的data.json

    ```json
    {
      "train": "data_set/train",
      "val": "data_set/val",
      "name": [
        "O",
        "R"
      ],
      "height": 224,
      "width": 224,
      "training": {
        "verbose": 1,
        "shuffle": "False",
        "validation_freq": 1
      },
      "early_stop": {
        "monitor": "val_accuracy",
        "min_delta": 0.01,
        "patience": 5,
        "mode": "max"
      }
    }
    ```

    * 设置train 训练集数据目录路径
    * 设置val 验证集数据目录路径
    * 设置 name 分类的名字
    * 设置图片大小height、width
    * 如果想要启动early_stop，可以设置相关参数
    * 还有一些训练过程model.fit 函数的超参

  * tools目录下的data_split.py 文件，这是一个将数据集按文件夹目录划分

    可以搭配百度爬虫https://github.com/linxinloningg/lightweight_spider/tree/main/baidu_pic

    获取不同类别的照片数据，然后按比例划分得到自己的数据集

    ```bash
    --scr "需要划分的文件夹路径"
    --target "新的数据保存的文件夹路径"
    --scale	"划分比例"
    ```

  * utils一些将来可能用到的文件

  * **detect.py**：目标检测代码文件

    python detect.py --data data.json配置文件 --model 训练好的模型文件 --source 检测文件

    ```python
    '--data', default='data/data.json', help="The path of data.json"
    '--model', help="The model for test"
    '--source', help="The source of test"
    ```

  * **train_cnn.py** :cnn模型训练代码文件

    python train_cnn.py --data data.json配置文件 --epochs 训练轮数 --batch_size 一次传入文件数量 --early_stop 1/0 是否启用early_stop 

    将会在models目录下生成模型h5文件

    ```python
    '--data', default='data/data.json', help="The path of data.json"
    '--epochs', default=1, help="The epochs of train"
    '--batch_size', default=16, help="The batch-size of train"
    '--early_stop', default=0, help="early stop for train or not"
    ```

  * **train_mobilenet.py**:mobilenet模型训练代码文件

    同上,将会在models目录下生成模型h5文件

  * **val.py**：验证模型代码文件

    python train_mobilenet.py --data data.json配置文件 --model 训练好的模型文件 --test 验证集文件路径/可以用val文件路径  --batch_size 一次传入文件数量

    将会在results目录下生成val.jpg

  * **release.py**：释放显存代码文件

* **准备数据集**：数据集主要分为train和val，可以在data.json中指定train和val的路径，train和val下都存在着不同类型的分类图片文件夹，我建议在代码文件夹下的data_set下准备数据集

  data_set
  └─ score
         ├─ train
         │    ├─ 分类一# 下面放分类一训练集图片
         │    ├─ 分类二# 下面放分类二训练集图片
         │    └─等等 
         └─ val
                ├─ 分类一# 下面放分类一测试集标签
                ├─ 分类二# 下面放分类二测试集标签
                ├─ 等等

* **模型训练**

  * 在终端调用不用的模型训练文件即可,可能在将会来添加更多的模型训练代码文件

    ```bash
    python train_cnn.py --data data.json配置文件 --epochs 训练轮数 --batch_size 一次传入文件数量 --early_stop 1/0 是否启用early_stop 
    ```

* 模型使用，可能在将会来添加更多方式的代码检测文件

  ```bash
   # 检测摄像头
   	暂无
   # 检测图片文件
    python detect.py --data data.json --model cnn.h5 --source file.jpg  # image 
   # 检测视频文件
      暂无  
   # 检测一个目录下的文件
   	暂无
   # 检测网络视频
     	暂无
   # 检测流媒体
     	暂无    
  ```

  

