import tensorflow as tf
import os
import csv
import numpy as np

dataPath = os.path.join('data', 'train')
dicPath = os.path.join('data', 'hf_round1_arrythmia.txt')
labelPath = os.path.join('data', 'hf_round1_label.txt')
tfRecordPath = os.path.join('data.tfrecord')
limit = 24000
blocksize = 1000

def readDic():
    dic = {}
    code = 0
    data = np.genfromtxt(dicPath, encoding = 'UTF-8', dtype = 'str')
    for d in data:
        dic[d] = code
        code = code + 1

    return dic


dic = readDic()

def makeLabelSet(data):
    s = np.zeros(len(dic))
    for d in data:
        if d in dic:
            s[dic[d]] = 1
    return s

def readData(path):
    data = []
    path = os.path.join(dataPath, path)
    lines = np.genfromtxt(path, encoding = 'UTF-8', dtype = int, skip_header = 1)
    for I, II, V1, V2, V3, V4, V5, V6 in lines:
        data.append([I, II, V1, V2, V3, V4, V5, V6])
    return np.asarray(data).astype(np.float32)

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def readLabel():
    with open(labelPath, 'r', encoding = 'UTF-8') as f:
        reader = csv.reader(f, delimiter = '\t')
        count = 0
        writer = tf.io.TFRecordWriter('data%02d.tfrecord' % (count / blocksize + 1))
        for row in reader:
            path = row[0]
            result = row[3:]
            x = readData(path)
            y = makeLabelSet(result)
            features = {
            'x':_floats_feature(x.reshape(-1)),
            'y':_floats_feature(y)
            }
            example = tf.train.Example(features=tf.train.Features(feature = features))
            writer.write(example.SerializeToString())
            count = count + 1
            if count >= limit:
                break
            if (count % blocksize == 0):
                writer.close()
                writer = tf.io.TFRecordWriter('data%02d.tfrecord' % (count / blocksize +1))
        writer.close()

print('start')
readLabel()
print('finish')
