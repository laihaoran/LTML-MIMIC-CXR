import pandas as pd
import numpy as np
import os
import pickle as pkl
import cv2
import shutil
from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
import mmcv
from mllt.datasets import build_dataset
from mllt.apis import init_dist
from mllt.datasets import build_dataloader
from mllt.models import build_classifier
from mllt.core.evaluation.eval_tools import lists_to_arrays, eval_acc, eval_F1, eval_bacc, eval_recall, eval_precision, eval_SE, eval_auc
from mllt.core.evaluation.mean_ap import eval_map
from mllt.models.losses import accuracy
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def read_pkl():
    pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/VOCdevkit/longtail2012/class_freq.pkl'
    with open(pklpath, 'rb') as f:
        data1 = pkl.load(f)
    # cprob = condition_prob(np.asarray(data1['gt_labels']))
    # class_freq, neg_class_freq = count_freq(np.asarray(data1['gt_labels']))
    # freq = split_head_middle_tail(class_freq)
    pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/VOCdevkit/longtail2012/class_split.pkl'
    with open(pklpath, 'rb') as f:
        data2 = pkl.load(f)
    pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/VOCdevkit/terse_gt_2012.pkl'
    with open(pklpath, 'rb') as f:
        data3 = pkl.load(f)

    # row_combine, class_freq, neg_class_freq, freq, cprob = write_pkl()

    # print(cprob)
    # print(data1['condition_prob'])


def read_my_pkl():
    pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_freq.pkl'
    with open(pklpath, 'rb') as f:
        data1 = pkl.load(f)
    # cprob = condition_prob(np.asarray(data1['gt_labels']))
    # class_freq, neg_class_freq = count_freq(np.asarray(data1['gt_labels']))
    # freq = split_head_middle_tail(class_freq)
    pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_split.pkl'
    with open(pklpath, 'rb') as f:
        data2 = pkl.load(f)
    pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/VOCdevkit/terse_gt_2012.pkl'
    with open(pklpath, 'rb') as f:
        data3 = pkl.load(f)

    # row_combine, class_freq, neg_class_freq, freq, cprob = write_pkl()

    # print(cprob)
    # print(data1['condition_prob'])


def reobtain_train():

    with open('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_name.pkl', 'rb') as f:
         data = pkl.load(f)
    # with open('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_freq.pkl', 'rb') as f:
    #      data1 = pkl.load(f)
    
    spath = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimi-label_finall.csv'
    df = pd.read_csv(spath)
    class_name = []
    table = df.head()
    # count = [0 for i, n in enumerate(table)]
    cou = 2
    combine_clo = []
    for i, t in enumerate(table):
        if i > cou:
            class_name.append(t)
            clo = df[t]
            clo = np.asarray(clo)
            combine_clo.append(np.expand_dims(np.where(clo == 1, 1, 0), axis=1))
    combine_clo = np.concatenate(combine_clo, axis=1)


    img_ids1 = mmcv.list_from_file('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/test.txt')
    img_ids2 = mmcv.list_from_file('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/val.txt')

    img_all = df['location'].tolist()
    train = list(set(img_all) - set(img_ids1) - set(img_ids2))

    gt_label = []
    for n in train:
        ind = img_all.index(n)
        gt_label.append(combine_clo[ind])

    with open('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_freq.pkl', 'rb') as f:
         data = pkl.load(f)
    
    data1 = {}
    data1['gt_labels'] = gt_label
    data1['class_freq'] = data['class_freq']
    data1['neg_class_freq'] = data['neg_class_freq']
    data1['condition_prob'] = data['condition_prob']

    with open('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_freq1.pkl', 'wb') as f:
        pkl.dump(data1 ,f)

    with open('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/img_id.txt', 'w') as f:
        for n in train:
            f.writelines(n + '\n')



def write_pkl():
    os.makedirs('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/', exist_ok=True)

    spath = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimi-label_finall.csv'
    df = pd.read_csv(spath)
    class_name = []
    table = df.head()
    # count = [0 for i, n in enumerate(table)]
    cou = 3
    combine_clo = []
    for i, t in enumerate(table):
        if i > cou:
            class_name.append(t)
            clo = df[t]
            clo = np.asarray(clo)
            combine_clo.append(np.expand_dims(np.where(clo == 1, 1, 0), axis=1))
    combine_clo = np.concatenate(combine_clo, axis=1)

    pkl_save_path = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic_pkl'
    os.makedirs(pkl_save_path, exist_ok=True)

    row_ = []
    for i in range(combine_clo.shape[0]):
        row_.append(combine_clo[i,:])
    for n, m in zip(df['location'].tolist(), row_):
        with open(os.path.join(pkl_save_path, n + '.pkl'), 'wb') as f:
            pkl.dump(m, f)

    ## split dataset from classification
    # clf = IterativeStratification(n_splits=5)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X=np.expand_dims(df['location'].values, axis=1) ,  y=combine_clo, test_size=0.2)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X=X_train, y=y_train, test_size=0.1)

    row_combine = []
    for i in range(y_train.shape[0]):
        row_combine.append(y_train[i,:])

    cprob = condition_prob(combine_clo)
    class_freq, neg_class_freq = count_freq(combine_clo)
    freq = split_head_middle_tail(class_freq, thred=[1000, 10000])

    data1 = {}
    data1['gt_labels'] = row_combine
    data1['class_freq'] = class_freq
    data1['neg_class_freq'] = neg_class_freq
    data1['condition_prob'] = cprob

    data2 = freq

    data1_pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_freq.pkl'
    class_pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_name.pkl'
    data2_pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_split.pkl'
    data3_img_idpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/img_id.txt'
    data4_img_idpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/val.txt'
    data5_img_idpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/test.txt'
    data6_pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/val.pkl'
    data7_pklpath = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/test.pkl'

    with open(data1_pklpath, 'wb') as f:
        pkl.dump(data1 ,f)

    with open(class_pklpath, 'wb') as f:
        pkl.dump(class_name ,f)

    with open(data1_pklpath, 'wb') as f:
        pkl.dump(data1 ,f)

    with open(data2_pklpath, 'wb') as f:
        pkl.dump(data2 ,f)

    with open(data3_img_idpath, 'w') as f:
        for n in np.squeeze(X_train).tolist():
            f.writelines(n + '\n')
    
    with open(data4_img_idpath, 'w') as f:
        for n in np.squeeze(X_val).tolist():
            f.writelines(n + '\n')
    
    with open(data5_img_idpath, 'w') as f:
        for n in np.squeeze(X_test).tolist():
            f.writelines(n + '\n')

    with open(data6_pklpath, 'wb') as f:
        pkl.dump(y_val ,f)
    
    with open(data7_pklpath, 'wb') as f:
        pkl.dump(y_test ,f)

    # return row_combine, class_freq, neg_class_freq, freq, cprob

def count_freq(gt_labels):
    class_freq = np.zeros((gt_labels.shape[1]), dtype=np.int64)
    neg_class_freq = np.zeros((gt_labels.shape[1]), dtype=np.int64)
    total = gt_labels.shape[0]
    for i in range(gt_labels.shape[1]):
        n =  len(np.where((gt_labels[:, i] == 1))[0])
        class_freq[i] = n
        neg_class_freq[i] = total - n
    return class_freq, neg_class_freq


def split_head_middle_tail(class_freq, thred = [20, 100]):
    re_freq = np.sort(class_freq)
    index = np.argsort(class_freq)
    # freq = ['tail', 'middle', 'head']
    freq = {}

    ind = np.where(re_freq < thred[0])
    freq['tail'] = set(index[ind])

    ind = np.where( (thred[0] <= re_freq) & ( re_freq < thred[1]))
    freq['middle'] = set(index[ind])

    ind = np.where( thred[1] <= re_freq)
    freq['head'] = set(index[ind])
    return freq


def condition_prob(gt_labels):
    cprob = np.zeros((gt_labels.shape[1], gt_labels.shape[1]))
    for i in range(gt_labels.shape[1]):
        for j in range(gt_labels.shape[1]):
            nij =  len(np.where((gt_labels[:, i] == 1) & (gt_labels[:, j] == 1))[0])
            nj =  len(np.where((gt_labels[:, j] == 1))[0])
            cprob[j, i] = nij / nj
    return cprob

def read_result():
    path = '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_voc_resnet50_pfc_DB/gt_and_results_e8.pkl'
    with open(path, 'rb') as f:
        data = pkl.load(f)
    path = '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_voc_resnet50_pfc_DB/gt_and_results_e8.pkl'
    with open(path, 'rb') as f:
        data = pkl.load(f)

def mimic_image_file_cmp():

    path1 = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic-cxr-2.0.0-metadata.csv'
    path2 = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic_label.csv'
    imgpath = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic'
    imgpath1 = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall'
    df = pd.read_csv(path1)
    head = df['ViewCodeSequence_CodeMeaning'].tolist()
    dicom_id = df['dicom_id'].tolist()
    study_id = df['study_id'].tolist()
    unique_head = list(set(head))

    # df2 = pd.read_csv(path2)
    # study = df2['study'].tolist()
    # print(len(list(study)))
    # print(len(list(set(study))))

    # for d in unique_head:
    #     print(d)
    ###
    # lateral                           侧面            侧面
    # nan                                          未知 侧面  正面
    # left lateral                      左侧            侧面
    # Erect                             直立            侧面
    # antero-posterior                  前侧-后侧       正面
    # left anterior oblique             左前斜          正面
    # postero-anterior                  后前侧          正面
    # Recumbent                         卧位            正面
    ###
    # for d, h in zip(dicom_id, head) :
    #     if h in unique_head:
    #         shutil.copy(os.path.join(imgpath, d + '.jpg'), imgpath1)
    #         os.rename(os.path.join(imgpath1, d + '.jpg'), os.path.join(imgpath1, str(h) + '.jpg'))
            # img = cv2.imread(os.path.join(imgpath, d + '.jpg'), cv2.IMREAD_GRAYSCALE)
            # if img is not None:
            #     cv2.imshow('imshow',img)
            #     cv2.waitKey(0)
                # print(d)
            # unique_head.pop(unique_head.index(h))

    CM = ['lateral', 'left lateral', 'Erect']  #0
    ZM = ['antero-posterior', 'left anterior oblique', 'postero-anterior', 'Recumbent']  # 1
    new_dicom_id = []
    new_study_id = []
    oriter = []

    for d, s, h in zip(dicom_id, study_id, head):
        # if h in CM:
        #     new_dicom_id.append(d)
        #     new_study_id.append(s)
        #     oriter.append(0)
        # elif h in ZM:
        #     new_dicom_id.append(d)
        #     new_study_id.append(s)
        #     oriter.append(1)

        if h in ZM and s not in new_study_id:
            new_dicom_id.append(d)
            new_study_id.append(s)
            oriter.append(h)

    pd.DataFrame(data={'study': new_study_id, 'dicom_id': new_dicom_id, 'ViewCodeSequence_CodeMeaning': oriter}).to_csv(os.path.join(imgpath1, 'mimic-file-name.csv'), index=False)

def map_study_name():
    path1 = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic-file-name.csv'
    path2 = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic_label.csv'
    existpath = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic'
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    study1 = df1['study'].tolist()
    dicom_id = df1['dicom_id'].tolist()

    study2= df2['study'].tolist()
    write_pd = []
    count = 0
    location = []
    for i, s2 in enumerate(study2) :
        try:
            loc = study1.index(int(s2[1:]))
            rname = dicom_id[loc]
            if os.path.exists(os.path.join(existpath, rname + '.jpg')):
                write_pd.append(df2.iloc[i:i+1])
                location.append(rname)
        except:
            count += 1

    # print(len(write_pd))
    # print(len(location))
    newdf2 = pd.concat(write_pd, axis=0).reset_index(drop=True)  ## the index will make wrong
    print(newdf2.shape)
    loca = pd.DataFrame(data={'location': location})
    print(loca.shape)
    df = pd.concat([loca, newdf2], axis=1)
    print(df.shape)
    df.to_csv('/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimi-label_finall.csv', index=False)
    




def write_csv_size():
    spath = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimi-label_finall.csv'
    root = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic'
    df = pd.read_csv(spath)
    location = df['location'].tolist()
    W = []
    H = []
    for n in location:
            img = cv2.imread(os.path.join(root, n + '.jpg'), cv2.IMREAD_GRAYSCALE)
            width, height = img.shape
            W.append(width)
            H.append(height)

    df1 = pd.DataFrame({'width': W, 'height': H})
    pd.concat([df1, df], axis=1).to_csv('/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimi-label_size_finall.csv', index=False)

def test_simple():
    # data_root = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic'
    # print(data_root[:-5])

    # pkl_path = os.path.join(data_root[:-5], 'mimic_pkl',
    #                     '{}.pkl'.format('78afcfdb-9d7b6a67-d1242fcf-8e35083e-b3c0e547'))
    # with open(pkl_path, 'rb') as f:
    #         gt_labels = pkl.load(f) 
    import itertools
    a ="GEeks"
 
    l = list(itertools.combinations_with_replacement(a, 3))
    print("COMBINATIONS WITH REPLACEMENTS OF STRING GEeks OF SIZE 2.")
    print(l)

def partition(lst, n):
    """
    python partition list
    :param lst: list
    :param n: partitionSize
    :return:
    """
    division = len(lst) / float(n)
    return [list(lst)[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]


def chunk(lst, size):
    """
    python chunk list
    :param lst: list
    :param size: listSize
    :return:
    """
    return [list(lst)[int(round(size * i)): int(round(size * (i + 1)))] for i in range(int(len(lst) / float(size)) + 1)]

def obtain_mean_std(count=1000):
    img_prefix = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic'
    ann_file = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/img_id.txt'
    img_ids = mmcv.list_from_file(ann_file)
    img_ids_list = chunk(img_ids, count)
    means = []
    stds = []
    for files in img_ids_list:
        bin = []
        for file in files:
            img = mmcv.imread(os.path.join(img_prefix, file + '.jpg'))
            bin += img[:,:, 0].flatten().tolist()
        means.append(np.mean(bin))
        stds.append(np.std(bin))
    print('The means is {0}, the std is {1}'.format(np.mean(means),  np.mean(stds)))

def stastic_data():
    pkl_path = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_split.pkl'
    txt_path = '/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/img_id.txt'
    gt_path = '/data/haoranlai/Project/DistributionBalancedLoss/data/mimicall/mimic_pkl'
    with open(pkl_path, 'rb') as f:
        split_data = pkl.load(f)
    img_ids = mmcv.list_from_file(txt_path)
    label = []
    for fi in img_ids:
        with open(os.path.join(gt_path, fi + '.pkl'), 'rb') as f:
            gt = pkl.load(f)
            label.append(np.expand_dims(gt, axis=0))
    label = np.concatenate(label, axis=0)
    for i in list(split_data['tail']):
        t = np.sum(label[:, i:i+1], axis=0)
        print('class {} number {}'.format(i, t[0]))

def test_model():
    import torch
    path = '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_multi_net_init_resnet50_DB_0.1_0.1_0.1/latest.pth'
    model = torch.load(path)['state_dict']
    print(model['heads.0.fc_cls.weight'])
    print(model['heads.1.fc_cls.weight'])
    print(model['heads.2.fc_cls.weight'])
    print(model['heads.3.fc_cls.weight'])
    print(model['heads.4.fc_cls.weight'])
    print(model['heads.5.fc_cls.weight'])
    t= 0

def sigmoid(x):
    if isinstance(x,list):
        x = np.asarray(x)
    return 1 / (1 + np.exp(-x))

def plot_confusion(y_pred, y_true):
    y_pred = np.asarray(y_pred) > 0

    # 计算多标签混淆矩阵
    cm = multilabel_confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵图像
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    for i in range(2):
        for j in range(2):
            ax[i, j].matshow(cm[i, j], cmap=plt.cm.Blues, alpha=0.5)
            for x in range(2):
                for y in range(2):
                    ax[i, j].text(x, y, str(cm[i, j][y, x]), va='center', ha='center')
            ax[i, j].set_xlabel('Predicted label')
            ax[i, j].set_ylabel('True label')
            ax[i, j].set_xticks([0, 1])
            ax[i, j].set_yticks([0, 1])
    plt.tight_layout()

    # 保存混淆矩阵图像
    plt.savefig('multilabel_confusion_matrix.png')



def plot_condition_prob():
    # 生成模拟数据
    # labels = ['A', 'B', 'C', 'D']
    # data = np.random.rand(len(labels), len(labels))

    # # 计算条件概率
    # data = data / data.sum(axis=1)[:, np.newaxis]

    with open('/data/haoranlai/Project/DistributionBalancedLoss/appendix_used/mimic/class_freq.pkl', 'rb') as f:
        data = pkl.load(f)
    
    # con_prob = data['condition_prob']

    
    sort_index = np.argsort(data['class_freq'])[::-1]
    sort_class_num = np.sort(data['class_freq'])[::-1]

    gt_labels = np.asarray(data['gt_labels'])[:, sort_index]
    con_prob = condition_prob(gt_labels) 


    labels = np.asarray([i for i in range(sort_class_num.shape[0])])

    # t = 0

    # 绘制热图
    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(con_prob, cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)

    # 设置图形属性
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    # ax.set_xlabel('Class i')
    # ax.set_ylabel('Class j')

    # ax.set_title('Conditional Probability Heat Map')

    # 保存图形
    plt.savefig('conditional_probability_heat_map.png')
    plt.close()


def LTDistribution_map():
    # 生成长尾分布数据
    # data = np.random.lognormal(0, 1, size=10000)
    with open('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_name.pkl', 'rb') as f:
        data1 = pkl.load(f)
    
    with open('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_freq.pkl', 'rb') as f:
        data2 = pkl.load(f)

    sort_index = np.argsort(data2['class_freq'])[::-1]
    sort_class_num = np.sort(data2['class_freq'])[::-1]


    # 类别名称列表
    categories = np.asarray([i for i in range(data2['condition_prob'].shape[0])])

    # 各个类别的样本数目列表
    counts = sort_class_num
    counts = np.log(counts)
    # 绘制直方图
    sns.set_style("white")
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    plt.bar(categories, counts,  color='lightskyblue')

    # 添加x轴和y轴标签
    # plt.xlabel('Categories')
    # plt.ylabel('Log(Counts)')

    # 展示图像
    # plt.show()

    # # 设置图形标题和坐标轴标签
    # plt.title("Long-tailed Distribution")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency (log scale)")
    # labels = np.asarray([i for i in range(data2['condition_prob'].shape[0])])

    # plt.semilogy(labels,  data2['class_freq'])
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Data (Log Scale)')
    # plt.show()

    # 将图像保存为文件
    plt.savefig('dataset_distribution.png')
    plt.close()


def print_performance(per_cls_acc, save_path, metric='bacc', color='lightskyblue'):
    with open('/data/haoranlai/Project/DistributionBalancedLoss/appendix/mimic/class_freq.pkl', 'rb') as f:
        data2 = pkl.load(f)
    categories = np.asarray([i for i in range(data2['condition_prob'].shape[0])])
    sort_index = np.argsort(data2['class_freq'])[::-1]
    per_cls_acc = per_cls_acc[sort_index]
    sns.set_style("white")
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(10, 3), dpi=200)
    ax.bar(categories, per_cls_acc * 100,  label=save_path, color=color)
#  'font.family':'Times New Roman'

    # 添加x轴和y轴标签
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.xlabel('Categories', fontsize=18)
    # plt.ylabel(metric, fontsize=18)
    # plt.set_xticklabels(plt.get_xticklabels(), rotation=0)
    # plt.add_artist(plt.Rectangle((0,0), 10, 10, color='r'))
    plt.legend(loc='upper left', fontsize=18)
    plt.savefig(save_path + '_'  + metric + '.png')
    plt.close()

def read_final_result():
    cfg = mmcv.Config.fromfile('/data/haoranlai/Project/DistributionBalancedLoss/configs/mimic/LT_resnet50_noise_label.py')
    dataset = build_dataset(cfg.data.test)
    path_list = [
        '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_celoss_resnet50_pfc_DB/gt_and_results_eDB/latest.pkl',

        '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_FCloss_resnet50_pfc_DB/gt_and_results_eDB/latest.pkl',

        # '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_resnet50_pfc_DB/gt_and_results_e4.pkl',
        '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_resnet50_pfc_DB_v1/gt_and_results_e5.pkl',

        '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_LamdaReasmpleloss_resnet50_pfc_DB_10/gt_and_results_e3.pkl',

        '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_LamdaNoiseReasmpleloss_resnet50_pfc_DB_10_LL_R/gt_and_results_e3.pkl',

        '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_LamdaNoiseReasmpleloss_resnet50_pfc_DB_10_LL_Ct/gt_and_results_e5.pkl'
    ]

    name_list = ['CE Loss', 'Focal Loss', 'DB Loss', 'ANR', 'ANR-LLA', 'ANR-LLM']
    colors = ['skyblue', 'deepskyblue', 'steelblue']
  
    for i, (outname, path) in enumerate(zip(name_list,path_list)):
        print(outname)
        with open(path, 'rb') as f:
            data = pkl.load(f)[0]
        gt_labels = np.array(data['gt_labels']) 
        outputs = np.array(data['outputs'])
        metrics = []
        for split, selected in dataset.class_split.items():
            # if split == 'head':
            #     selected = selected - set([0])
            selected = list(selected)
            selected_outputs = outputs[:, selected]
            selected_gt_labels = gt_labels[:, selected]
            classes = np.asarray(dataset.CLASSES)[selected]
            mAP, APs = eval_map(selected_outputs, selected_gt_labels, classes, print_summary=False)
            micro_f1, macro_f1, weighted_f1 = eval_F1(selected_outputs, selected_gt_labels)
            acc, per_cls_acc = eval_acc(selected_outputs, selected_gt_labels)
            bacc, per_cls_acc = eval_bacc(selected_outputs, selected_gt_labels)

            mAPA, per_cls_acc = eval_precision(selected_outputs, selected_gt_labels)
            mARA, per_cls_acc = eval_recall(selected_outputs, selected_gt_labels)
            sen, spe = eval_SE(selected_outputs, selected_gt_labels)
            macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(selected_outputs, selected_gt_labels)


            metrics.append([split, mAP, micro_f1, macro_f1, weighted_f1, acc, bacc, mAPA, mARA, sen, spe, macro_auc, micro_auc, weighted_auc])

        # outputs = outputs[:, 1:]
        # gt_labels = gt_labels[:, 1:]
   


        mAP, APs = eval_map(outputs, gt_labels, dataset, print_summary=False)
        micro_f1, macro_f1, weighted_f1 = eval_F1(outputs, gt_labels)
        acc, per_cls_acc = eval_acc(outputs, gt_labels)
        bacc, per_cls_acc = eval_bacc(outputs, gt_labels)  
        if outname == 'DB Loss':
              base_per_bacc = per_cls_acc

        if i > 2:
            print_performance(per_cls_acc - base_per_bacc, save_path=outname, metric='bacc', color=colors[i - 3])
        mAPA, _ = eval_precision(outputs, gt_labels)
        mARA, _ = eval_recall(outputs, gt_labels)
        sen, spe = eval_SE(outputs, gt_labels)
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(outputs, gt_labels)
        if outname == 'DB Loss':
              base_per_auc = per_auc
        if i > 2:
            print_performance(per_auc - base_per_auc, save_path=outname, metric='auc', color=colors[i - 3])

        if outname == 'ANR':
            base_per_bacc = per_cls_acc
            base_per_auc = per_auc

        metrics.append(['Total', mAP, micro_f1, macro_f1, weighted_f1, acc, bacc, mAPA, mARA, sen, spe, macro_auc, micro_auc, weighted_auc])
        for split, mAP, micro_f1, macro_f1, weighted_f1, acc, bacc, mAPA, mARA, sen, spe, macro_auc, micro_auc, weighted_auc in metrics:
            print('Split:{:>6s} mAP:{:.4f}  acc:{:.4f} bacc:{:.4f} micro_f1:{:.4f}  macro_f1:{:.4f} weighted_f1:{:.4f} mAPA:{:.4f} mARA:{:.4f} Sensitivity:{:.4f} Specificity:{:.4f} macro_auc:{:.4f} micro_auc:{:.4f} weighted_auc:{:.4f}'.format(
                split, mAP, acc, bacc, micro_f1, macro_f1, weighted_f1, mAPA, mARA, sen, spe,  macro_auc, micro_auc, weighted_auc))
            

def read_NVUM_result():
    cfg = mmcv.Config.fromfile('/data/haoranlai/Project/DistributionBalancedLoss/configs/mimic/LT_resnet50_noise_label.py')
    dataset = build_dataset(cfg.data.test)
    path = '/data/haoranlai/Project/NVUM/Experiment/NVCM_ClassAware_v1/predict/gt_output_4.pkl'
    with open(path, 'rb') as f:
        data = pkl.load(f)
    gt_labels = np.array(data['gt']) 
    outputs = np.array(data['preds'])
    metrics = []
    for split, selected in dataset.class_split.items():
        # if split == 'head':
        #     selected = selected - set([0])
        selected = list(selected)
        selected_outputs = outputs[:, selected]
        selected_gt_labels = gt_labels[:, selected]
        classes = np.asarray(dataset.CLASSES)[selected]
        mAP, APs = eval_map(selected_outputs, selected_gt_labels, classes, print_summary=False)
        micro_f1, macro_f1, weighted_f1 = eval_F1(selected_outputs, selected_gt_labels)
        acc, per_cls_acc = eval_acc(selected_outputs, selected_gt_labels)
        bacc, per_cls_acc = eval_bacc(selected_outputs, selected_gt_labels)

        mAPA, per_cls_acc = eval_precision(selected_outputs, selected_gt_labels)
        mARA, per_cls_acc = eval_recall(selected_outputs, selected_gt_labels)
        sen, spe = eval_SE(selected_outputs, selected_gt_labels)
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(selected_outputs, selected_gt_labels)


        metrics.append([split, mAP, micro_f1, macro_f1, weighted_f1, acc, bacc, mAPA, mARA, sen, spe, macro_auc, micro_auc, weighted_auc])

    # outputs = outputs[:, 1:]
    # gt_labels = gt_labels[:, 1:]



    mAP, APs = eval_map(outputs, gt_labels, dataset, print_summary=False)
    micro_f1, macro_f1, weighted_f1 = eval_F1(outputs, gt_labels)
    acc, per_cls_acc = eval_acc(outputs, gt_labels)
    bacc, per_cls_acc = eval_bacc(outputs, gt_labels)  
    mAPA, per_cls_acc = eval_precision(outputs, gt_labels)
    mARA, per_cls_acc = eval_recall(outputs, gt_labels)
    sen, spe = eval_SE(outputs, gt_labels)
    macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(outputs, gt_labels)
    metrics.append(['Total', mAP, micro_f1, macro_f1, weighted_f1, acc, bacc, mAPA, mARA, sen, spe, macro_auc, micro_auc, weighted_auc])
    for split, mAP, micro_f1, macro_f1, weighted_f1, acc, bacc, mAPA, mARA, sen, spe, macro_auc, micro_auc, weighted_auc in metrics:
        print('Split:{:>6s} mAP:{:.4f}  acc:{:.4f} bacc:{:.4f} micro_f1:{:.4f}  macro_f1:{:.4f} weighted_f1:{:.4f} mAPA:{:.4f} mARA:{:.4f} Sensitivity:{:.4f} Specificity:{:.4f} macro_auc:{:.4f} micro_auc:{:.4f} weighted_auc:{:.4f}'.format(
            split, mAP, acc, bacc, micro_f1, macro_f1, weighted_f1, mAPA, mARA, sen, spe,  macro_auc, micro_auc, weighted_auc))
        
def test_tsne():
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # 生成一些随机数据
    X = np.random.rand(100, 1)

    # 初始化t-SNE模型
    tsne = TSNE(n_components=1, random_state=0)

    # 计算降维后的数据点
    X_tsne = tsne.fit_transform(X)

    # 绘制t-SNE平面图
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne, 1- X_tsne, s=50)
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('tsne.png')
    plt.close()

# def func():
#     import math
#     h(p) = math.log(1 + math.e **  (p))
#     f(p) = (1 / b) * math.log(1 + math.e **  (b *p))
#     g(p)= (1 / (1 + b * (1 - 1 /  (1 + math.e ** (-p))))) * math.log(1 + math.e ** ((1 + b * (1 - 1 /  (1 + math.e ** (-p)))) * p))


def read_jpg_pkl():
    path = '/data/haoranlai/Project/LTML-MIMIC-CXR/jpg_file_path.pkl'
    csv_path = '/data/haoranlai/Project/LTML-MIMIC-CXR/data/mimicall/mimi-label_finall.csv'
    with open(path, 'rb') as f:
        jpg_path = pkl.load(f)

    all_df = pd.read_csv(csv_path)
    location = all_df['location'].tolist()
    
    jpg_path = [n.replace('/home1/haoranlai/Dataset/MIMIC-CXR/files/', './files/') for n in jpg_path]

    single_name = [n.split('/')[-1].replace('.jpg', '') for n in jpg_path]
    new_location = []
    for n in location:
        index = single_name.index(n)
        new_location.append(jpg_path[index])
    all_df['location'] = new_location
    all_df.to_csv('/data/haoranlai/Project/LTML-MIMIC-CXR/data/mimicall/LTML_MIMIC_CXR_label_.csv', index=False)


def read_split_result():
    cfg = mmcv.Config.fromfile('/data/haoranlai/Project/LTML-MIMIC-CXR/configs/mimic/LTML_resnet50_ANR_LLA.py')
    dataset = build_dataset(cfg.data.test)
    # path_list = [
    #     '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_celoss_resnet50_pfc_DB/gt_and_results_eDB/latest.pkl',

    #     '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_FCloss_resnet50_pfc_DB/gt_and_results_eDB/latest.pkl',

    #     # '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_resnet50_pfc_DB/gt_and_results_e4.pkl',
    #     '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_resnet50_pfc_DB_v1/gt_and_results_e5.pkl',

    #     '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_LamdaReasmpleloss_resnet50_pfc_DB_10/gt_and_results_e3.pkl',

    #     '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_LamdaNoiseReasmpleloss_resnet50_pfc_DB_10_LL_R/gt_and_results_e3.pkl',

    #     '/data/haoranlai/Project/DistributionBalancedLoss/work_dirs/LT_mimic39_LamdaNoiseReasmpleloss_resnet50_pfc_DB_10_LL_Ct/gt_and_results_e5.pkl'
    # ]

    # name_list = ['CE Loss', 'Focal Loss', 'DB Loss', 'ANR', 'ANR-LLA', 'ANR-LLM']
    # colors = ['skyblue', 'deepskyblue', 'steelblue']

    for i in range(1):
        # path = '/data/haoranlai/Project/LTML-MIMIC-CXR/work_dirs/LTML_MIMIC_CXR_resnet50_ANR_LLM/gt_and_results_e{0}.pkl'.format(i + 1)
        path = '/data/haoranlai/Project/LTML-MIMIC-CXR/work_dirs/LTML_MIMIC_CXR_resnet50_ANR_LLM/gt_and_results.pkl'
    # for i, (outname, path) in enumerate(zip(name_list,path_list)):
        # print(outname)
        with open(path, 'rb') as f:
            data = pkl.load(f)[0]
        gt_labels = np.array(data['gt_labels']) 
        outputs = np.array(data['outputs'])
        metrics = []
        for split, selected in dataset.class_split.items():
            # if split == 'head':
            #     selected = selected - set([0])
            selected = list(selected)
            selected_outputs = outputs[:, selected]
            selected_gt_labels = gt_labels[:, selected]
            classes = np.asarray(dataset.CLASSES)[selected]
            mAP, APs = eval_map(selected_outputs, selected_gt_labels, classes, print_summary=False)
            micro_f1, macro_f1, weighted_f1 = eval_F1(selected_outputs, selected_gt_labels)
            acc, per_cls_acc = eval_acc(selected_outputs, selected_gt_labels)
            bacc, per_cls_acc = eval_bacc(selected_outputs, selected_gt_labels)

            mAPA, per_cls_acc = eval_precision(selected_outputs, selected_gt_labels)
            mARA, per_cls_acc = eval_recall(selected_outputs, selected_gt_labels)
            sen, spe = eval_SE(selected_outputs, selected_gt_labels)
            macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(selected_outputs, selected_gt_labels)


            metrics.append([split, mAP, micro_f1, macro_f1, weighted_f1, acc, bacc, mAPA, mARA, sen, spe, macro_auc, micro_auc, weighted_auc])
        mAP, APs = eval_map(outputs, gt_labels, dataset, print_summary=False)
        micro_f1, macro_f1, weighted_f1 = eval_F1(outputs, gt_labels)
        acc, per_cls_acc = eval_acc(outputs, gt_labels)
        bacc, per_cls_acc = eval_bacc(outputs, gt_labels)  
        mAPA, per_cls_acc = eval_precision(outputs, gt_labels)
        mARA, per_cls_acc = eval_recall(outputs, gt_labels)
        sen, spe = eval_SE(outputs, gt_labels)
        macro_auc, micro_auc, weighted_auc, per_auc = eval_auc(outputs, gt_labels)
        metrics.append(['Total', mAP, micro_f1, macro_f1, weighted_f1, acc, bacc, mAPA, mARA, sen, spe, macro_auc, micro_auc, weighted_auc])
        for split, mAP, micro_f1, macro_f1, weighted_f1, acc, bacc, mAPA, mARA, sen, spe, macro_auc, micro_auc, weighted_auc in metrics:
            print('Split:{:>6s} mAP:{:.4f}  acc:{:.4f} bacc:{:.4f} micro_f1:{:.4f}  macro_f1:{:.4f} weighted_f1:{:.4f} mAPA:{:.4f} mARA:{:.4f} Sensitivity:{:.4f} Specificity:{:.4f} macro_auc:{:.4f} micro_auc:{:.4f} weighted_auc:{:.4f}'.format(
                split, mAP, acc, bacc, micro_f1, macro_f1, weighted_f1, mAPA, mARA, sen, spe,  macro_auc, micro_auc, weighted_auc))


if __name__ == '__main__':
    # read_pkl()
    # read_result()
    # read_my_pkl()
    # mimic_image_file_cmp()
    # map_study_name()
    # write_csv_size()
    # write_pkl()
    # reobtain_train()

    # obtain_mean_std()

    # test_simple()

    # stastic_data()

    # test_model()

    # read_final_result()

    # read_NVUM_result()

    # plot_condition_prob()
    # LTDistribution_map()
    # test_tsne()
    # read_jpg_pkl()

    read_split_result()