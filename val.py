# -*- coding: utf-8 -*-


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json


def parse_args():
    """
    :return:进行参数的解析
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/data.json', help="The path of data.json")
    parser.add_argument('--model', help="The model for test")
    parser.add_argument('--test', default='data_set/val', help="The path of test")
    parser.add_argument('--batch_size', default=16, help="The batch-size of test")
    opt = parser.parse_args()
    return opt


# 数据加载，分别从训练的数据集的文件夹和测试的文件夹中加载训练集和验证集
def data_load(test_data_dir, img_height, img_width, batch_size):
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = test_ds.class_names
    # 返回处理之后的测试集和类名
    return test_ds, class_names


# 测试模型准确率
def val(configs, model_path, test_data_dir, batch_size):
    # todo 加载数据, 修改为你自己的数据集的路径
    test_ds, class_names = data_load(test_data_dir, configs['height'], configs['width'], batch_size)
    # todo 加载模型，修改为你的模型名称
    model = tf.keras.models.load_model(model_path)

    # 测试
    loss, accuracy = model.evaluate(test_ds)
    # 输出结果
    print('test accuracy :', accuracy)

    # 对模型分开进行推理
    test_real_labels = []
    test_pre_labels = []
    for test_batch_images, test_batch_labels in test_ds:
        test_batch_labels = test_batch_labels.numpy()
        test_batch_pres = model.predict(test_batch_images)

        test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
        test_batch_pres_max = np.argmax(test_batch_pres, axis=1)

        # 将推理对应的标签取出
        for i in test_batch_labels_max:
            test_real_labels.append(i)

        for i in test_batch_pres_max:
            test_pre_labels.append(i)

    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

    print(heat_maps)
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)

    heat_maps_float = heat_maps / heat_maps_sum
    print(heat_maps_float)

    show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                  save_name="results/{}".format(model_path.split('/')[-1].replace('.h5', '.jpg')))


def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
    # 这里是创建一个画布
    fig, ax = plt.subplots()
    im = ax.imshow(harvest, cmap="YlGnBu")

    # 这里是修改标签

    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))

    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值

    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)


if __name__ == '__main__':
    args = parse_args()
    val(json.load(open(args.data, 'r')), args.model, args.test, int(args.batch_size))
