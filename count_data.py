# -*- coding: UTF-8 -*-
import extractFromDB
import json
from preprocessor import preprocessor
import XLSFactory as xf
import re
import numpy as np


def clean_label(file_name):
    labels = xf.read(file_name, 0)
    res = {}
    words = []
    for key, values in labels.items():
        key = clean_words(key)
        label = preprocessor.preprocessLabelToWord(key)
        if len(label) == 1:
            if len(label[0]) > 2 and label[0] != 'type' and label[0] != 'kind'\
                    and label[0] != 'state' and label[0] != 'size' and label[0] != 'team'\
                    and label[0] != 'pro':
                words += label
                res[label[0]] = {}

    for label, values in labels.items():
        has_label = False
        for i in range(len(words)):
            if words[i] in label:
                has_label = True
                res[words[i]][label] = values
                break
        if has_label:
            continue
        else:
            res[label] = {label: values}
    return res


def clean_words(s):
    res = ""
    words = re.split(r'[-_]', s)
    for word in words:
            res += word+" "
    return res


def count_label_repo_num():
    labels = xf.read2("./resource/content related.xls", 0)
    print json.dumps(labels, encoding="utf-8", indent=2)
    print 'read done'
    res = extractFromDB.count(labels)
    xf.write(res, "./output/content related.xls", 'lables', 'occur_num', 'repo_num')


def save_label_map(output_file):
    labels = xf.read2("./data/resource/content related.xls", 0)
    f = open(output_file, "w")
    f.write(json.dumps(labels, encoding="utf-8", indent=2))
    f.close()


def label_map(input_label, label_dic):
    for key, value in label_dic.items():
        if input_label in value:
            return key, value[input_label]


# def label2vec(label):
#     res = np.zeros(len(label_list), dtype=float)
#     i = 0
#     for l in label_list:
#         if label == l:
#             res[i] = 1
#             break
#         i += 1
#     return res


def fetch_range(output_file):
    temp = []
    temp.append(xf.read_range("./data/baseline/httpclient.xls"))
    temp.append(xf.read_range("./data/baseline/jackrabbit.xls"))
    temp.append(xf.read_range("./data/baseline/lucene.xls"))
    print temp
    f = open(output_file, "w")
    f.write(json.dumps(temp, encoding="utf-8", indent=2))
    f.close()


fetch_range("./data/output/cmp_range.json")
# save_label_map("./data/resource/label_map.json")
# f = open("./data/resource/label_map.json")
# label_map = json.loads(f.read())
# f.close()
#
# label_list = []
# for key in label_map:
#     label_list.append(key)
# print label_list
#
# print label2vec("doc")
# print "defect" in map[0]

