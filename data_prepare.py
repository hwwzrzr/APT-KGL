import json

from glob import glob
import dgl
import torch
import numpy as np
import pandas as pd
from meta import set_path ,get_path,get_full_h_graph
from dgl.data.utils import save_graphs,load_graphs
import pickle
import time
def loadjson(path):
    data = []
    with open(path) as f:
        for line in f:
            dic = json.loads(line)
            data.append(dic)
    f.close()
    return data
def labelallgraph():
    path = glob('data/graph_v4/'+'*.json')
    malpath = []
    for file in path:
        data = loadjson(file)
        for dic in data:
            if '/phpstudy/www/DVWA/hackable/uploads/b.php' in  dic['path'] and file not in malpath:
                malpath.append(file)
    normalpath = list(set(path).difference(set(malpath)))
    normalnode =[]
    malnode = []
    for dic in normalpath:
        with open(dic) as f:
            for line in f:
                data = json.loads(line)
                normalnode.append(data['link'][0])
                break
    for dic in malpath:
        with open(dic) as f:
            for line in f:
                data = json.loads(line)
                malnode.append(data['link'][0])
                break
    normaldictnode = {'node':normalnode}
    maldictnode = {'node': malnode}
    with open('data/graph_v4/label/normal.json','w') as f:
        json.dump(normaldictnode,f)
    with open('data/graph_v4/label/mal.json', 'w') as f:
        json.dump(maldictnode, f)
def labelallgraph_v2():
    mal = []
    normal = []
    node = []
    path = 'data/graph_v4/label/all.json'
    data = loadjson(path)
    for dic in data:
        if '/phpstudy/www/DVWA/hackable/uploads/b.php' in  dic['path'] and dic['link'][0] not in mal:
            mal.append(dic['link'][0])
        if dic['link'][0] not in node:
            node.append(dic['link'][0])
    normal = list(set(node).difference(set(mal)))
    normaldictnode = {'node':normal}
    maldictnode = {'node': mal}
    with open('data/graph_v4/label/normal.json','w') as f:
        json.dump(normaldictnode,f)
    with open('data/graph_v4/label/mal.json', 'w') as f:
        json.dump(maldictnode, f)
def load_all_link():
    path = glob('data/graph_v4/'+'*.json')
    data = []
    for file in path:
        with open(file) as f:
            for line in f:
                dic = json.loads(line)
                data.append(dic)
    return data
def get_metapath(data):
    metapath = []
    for dic in data:
        if dic['type'] not in metapath:
            metapath.append(dic['type'])
    return metapath
def load_one_jsondata(path):
    with open(path) as f:
        for line in f:
            return json.loads(line)
def nodelabel_to_csv(data,mal):
    node = []
    for dic in data:
        if dic['link'][0] not in node:
            node.append(dic['link'][0])
    print(len(node))
    label  = []
    for dic in node:
        label.append(0 if dic not in mal['node'] else 1)
    df = pd.DataFrame(columns=('node','label'))
    df['node'] = node
    df['label'] = label
    df.to_csv('data/graph_v4/label/label.csv')
def save_path(list,name):
    path = 'data/graph_v3/label/%s.pickle'%name
    file  = open(path,'wb')
    pickle.dump(list,file)
    file.close()
def set_node_id(data):
    path = 'data/graph_v4/label/all.json'
    node = []
    for dic in data:
        if dic['link'][0] not in node:
            node.append(dic['link'][0])
    with open(path,'w') as f:
        for i,dic in enumerate(data):
            for index,id in enumerate(node):
                if dic['link'][0] == id:
                    dic['link'][0] = index
                    if dic['type'][0] == 'SUBJECT_PROCESS' and dic['type'][2] == 'SUBJECT_PROCESS':
                        for index2,id2 in enumerate(node):
                            if dic['link'][1] == id2:
                                dic['link'][1] == index2
            json.dump(dic,f)
            f.write('\n')
            if i %1000 == 0:
                print((i/len(data))*100)
    return data

def get_all_label():
    nodenum = 13478
    mal = load_one_jsondata('data/graph_v4/label/mal.json')
    mal_li = mal['node']
    label = np.zeros((nodenum))
    label[mal_li] = 1
    return label
def write_feature_list(data):
    feature_list = []
    path  = 'data/graph_v4/label/feature_list.json'
    for dic in data:
        for feat in dic['label'][0]:
            if feat not in feature_list:
                feature_list.append(feat)
        for feat in dic['label'][1]:
            if feat not in feature_list:
                feature_list.append(feat)
    feature_list = {'type':feature_list}
    with open(path,'w') as f:
        json.dump(feature_list,f)
    f.close()
def set_feature(data,feature_list,temp):
    feat = []
    if not data:
        feat = ['PT0']
    else:
        for dic in data:
            for  type in dic['label'][0]:
                if type not in feat:
                    feat.append(type)
            for type in dic['label'][1]:
                if type not in feat:
                    feat.append(type)
        if not feat:
            feat= ['PT0']
    for k in feat:
        temp[feature_list.index(k)] =1
    temp = temp.tolist()
    return temp
def get_feature_matrix(data,feature_list):
    feature_matrix =[]
    li = feature_list['type']

    for i,dic in enumerate(data):
        num = len(li)
        temp = np.zeros((num))
        feature_matrix.append(set_feature(dic,li,temp))
    return feature_matrix
def save_feature_matrix():
    data  = load_all_link()
    feature_list = load_one_jsondata('data/graph_v4/label/feature_list.json')
    new_data = sorted_data(data,13478)
    feature_matrix = get_feature_matrix(new_data,feature_list)
    np.savetxt('data/graph_v4/label/feature_matrix.txt',feature_matrix)
def sorted_data(data,nodenum):
    new_data = []
    for i in range(0,nodenum):
        li = []
        new_data.append(li)
    for dic in data:
        new_data[dic['link'][0]].append(dic)
    return new_data
def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()
def load_all_data():
    num_nodes = 13478
    num_classes =2
    feature_matrix = np.loadtxt('data/graph_v4/label/feature_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v4/label/mal.json')['node']
    shuffle_mal_index = np.random.permutation(mal_index)
    train_mal_idx = shuffle_mal_index[0:6]
    val_mal_idx = shuffle_mal_index[7:8]
    test_mal_idx = shuffle_mal_index[8:]
    features = torch.FloatTensor(feature_matrix)
    label = get_all_label()
    g = get_full_h_graph()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    new_idx = np.setdiff1d(idx,mal_index)
    shuffle_idx = np.random.permutation(new_idx)
    train_idx =np.append(shuffle_idx[0:9434],train_mal_idx)
    val_idx = np.append(shuffle_idx[9435:10782],val_mal_idx)
    test_idx = np.append(shuffle_idx[10783:],test_mal_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask
if __name__ =='__main__':
    # g, label, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask = load_all_data()
    # # print(g)
    # g_meta1 = dgl.metapath_reachable_graph(g,['s_fork_s'])
    process_embedding = torch.load('data/embedding/process_embedding.pkl')
    print(process_embedding)
    # set_feature_list(data)
    # metapath = load_one_jsondata('data/graph_v4/label/meta.json')
    # print(metapath.keys())
    # set_path(data,metapath)
    # hg = get_full_h_graph()






