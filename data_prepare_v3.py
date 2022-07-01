import json

from glob import glob
import dgl
import torch
import numpy as np
import pandas as pd
from meta import set_path ,get_path,get_full_h_graph
from meta_v2 import get_full_h_graph_v2,set_meta_paths,get_full_homogeneous_graph
from dgl.data.utils import save_graphs,load_graphs
import pickle
import time
import random
def set_graph_link():
    data  = load_all_link()
    process_feature = np.zeros((84864, 10))
    file_feature = np.zeros((15433, 10))
    li = load_one_jsondata('data/darpa/label/feature_list_v2.json')['type']
    num = len(li)
    temp = np.zeros((num))
    meta = load_one_jsondata('data/graph_v6/label/meta_v2.json')
    attribute = load_one_jsondata('data/graph_v6/label/attribute_list_v2.json')
    dic_link = load_one_jsondata('data/graph_v6/label/link.json')
    num_of_data = len(data)
    for count,dic in enumerate(data):
        if count %1000 == 0 :
            print(round(count/num_of_data,2)*100,'%')
        for i in meta.items():
            if dic['type'] in i[1]:
                if dic['type'][2] == 'RECORD_MEMORY_OBJECT' or dic['type'][2] == 'RECORD_NET_FLOW_OBJECT' or dic['type'][
                    2] == 'FILE_OBJECT_UNIX_SOCKET' or dic['type'][2] == 'IPC_OBJECT_PIPE_UNNAMED':
                    dic_link[i[0]].append([dic['link'][0],0])
                else:
                    dic_link[i[0]].append(dic['link'])
        for k in dic['label'][0]:
            if k in li:

                if [dic['link'][0], attribute[k]] not in dic_link['s_contain_attribute']:
                    dic_link['s_contain_attribute'].append([dic['link'][0], attribute[k]])
                temp[li.index(k)] = 1
        temp = temp.tolist()
        process_feature[dic['link'][0]] = temp
        temp = np.zeros((num))
        if dic['type'][2] == 'FILE_OBJECT_FILE' or dic['type'][2] == 'FILE_OBJECT_DIR' or dic['type'][
            2] == 'FILE_OBJECT_CHAR':
            for k in dic['label'][1]:
                if k in li:
                    if [dic['link'][1], attribute[k]] not in dic_link['f_contain_attribute']:
                        dic_link['f_contain_attribute'].append([dic['link'][1], attribute[k]])
                    temp[li.index(k)] = 1
            temp = temp.tolist()
            file_feature[dic['link'][1]] = temp
            temp = np.zeros((num))
    with open('data/darpa/label/full_link.json','w') as f:
        json.dump(dic_link,f)
    f.close()
    np.savetxt('data/darpa/label/process_matrix.txt',process_feature)
    np.savetxt('data/darpa/label/file_matrix.txt',file_feature)
def loadjson(path):
    data = []
    with open(path) as f:
        for line in f:
            dic = json.loads(line)
            data.append(dic)
    f.close()
    return data
def loadfile(path):
    data = []
    with open(path) as f:
        dic = f.readlines()
        for i in dic :
            i = i.strip()
            data.append(i)
    return data
def labelallgraph():
    path = glob('data/graph_v5/'+'*.json')
    malpath = []
    filedata = loadfile('data/graph_v5/label/malfile.txt')
    for file in path:
        data = loadjson(file)
        for dic in data:
            for i in filedata:
                if i in  dic['path'][0] and file not in malpath:
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
    with open('data/graph_v5/label/normal.json','w') as f:
        json.dump(normaldictnode,f)
    with open('data/graph_v5/label/mal.json', 'w') as f:
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
    path1 = glob('data/graph_v4/' + '*.json')
    path2 = glob('data/graph_v5/'+'*.json')
    path3 = glob('data/graph_v7/'+'*.json')
    path4 = glob('data/darpa/A/graph_v2/'+'*.json')
    path5 = glob('data/darpa/B/graph/' + '*.json')
    data = []
    for file in path1:
        with open(file) as f:
            for line in f:
                dic = json.loads(line)
                data.append(dic)
    for file in path2:
        with open(file) as f:
            for line in f:
                dic = json.loads(line)
                data.append(dic)
    for file in path3:
        with open(file) as f:
            for line in f:
                dic = json.loads(line)
                data.append(dic)
    for file in path4:
        with open(file) as f:
            for line in f:
                dic = json.loads(line)
                data.append(dic)
    for file in path5:
        with open(file) as f:
            for line in f:
                dic = json.loads(line)
                data.append(dic)
    return data
def load_all_link_v2():
    path1 = glob('data/darpa/A/graph_v2/' + '*.json')
    path2 = glob('data/darpa/B/graph/' + '*.json')
    data = []
    for file in path1:
        with open(file) as f:
            for line in f:
                dic = json.loads(line)
                data.append(dic)
    for file in path2:
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

def get_all_label_v2():
    nodenum = 100307
    mal = load_one_jsondata('data/darpa/label/malid.json')
    mal_li = mal['id']
    label = np.zeros((nodenum))
    label[mal_li] = 1
    return label

def get_all_label():
    nodenum = 34327
    mal = load_one_jsondata('data/graph_v7/label/malid.json')
    mal_li = mal['id']
    label = np.zeros((nodenum))
    label[mal_li] = 1
    return label
def get_all_label_darpa():
    nodenum = 84864
    mal = load_one_jsondata('data/darpa/label/malid.json')
    mal_li = mal['id']
    label = np.zeros((nodenum))
    label[mal_li] = 1
    return label
def get_mal_data():
    data = load_all_link()
    mal = []
    malid = load_one_jsondata('data/graph_v7/label/malid.json')['id']
    for dic in data:
        if dic['link'][0] in malid:
            mal.append(dic)
    path = 'data/graph_v7/label/maldata.json'
    with open(path, 'w') as f:
        for dic in mal:
            json.dump(dic, f)
            f.write('\n')
    f.close()
def write_feature_list(data):
    feature_list = []
    path  = 'data/darpa/label/feature_list.json'
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
        if k not in feature_list:
            continue
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
    feature_list = load_one_jsondata('data/darpa/label/feature_list.json')
    new_data = sorted_data(data,84864)
    feature_matrix = get_feature_matrix(new_data,feature_list)
    np.savetxt('data/darpa/label/feature_matrix.txt',feature_matrix)
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
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    num_nodes = 34327
    num_classes =2
    feature_matrix = np.loadtxt('data/graph_v7/label/feature_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v7/label/malid.json')['id']
    shuffle_mal_index = np.random.permutation(mal_index)
    train_mal_idx = shuffle_mal_index[0:476]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[477:]
    features = torch.FloatTensor(feature_matrix)
    label = get_all_label()
    g = get_full_h_graph()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    new_idx = np.setdiff1d(idx,mal_index)
    shuffle_idx = np.random.permutation(new_idx)
    train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    test_idx = np.append(shuffle_idx[14202:21148], test_mal_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    np.random.shuffle(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,idx
def separate_mal_index(li,mal_ID):
    mal_index=[]
    darpa_mal_index=[]
    for i in li:
        if i >mal_ID:
            darpa_mal_index.append(i)
        else:
            mal_index.append(i)
    return mal_index,darpa_mal_index
def load_all_data_v2():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    num_nodes = 34327
    num_files = 10074
    num_attributes = 6
    num_classes =2
    feature_matrix = np.loadtxt('data/graph_v7/label/process_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v7/label/malid.json')['id']
    shuffle_mal_index = np.random.permutation(mal_index)
    train_mal_idx = shuffle_mal_index[0:476]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[477:]
    features = torch.FloatTensor(feature_matrix)
    label = get_all_label()
    g = get_full_h_graph_v2()
    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    file_idx = np.arange(num_files)
    attr_idx = np.arange(num_attributes)
    process_based_g = get_random_walk_graph(g,idx,meta_paths_process)
    file_based_g = get_random_walk_graph(g,file_idx,meta_paths_file)
    attribute_based_g = get_random_walk_graph(g,attr_idx,meta_paths_attribute)
    IPC_based_g = get_random_walk_graph(g,[0],meta_paths_IPC)
    net_based_g = get_random_walk_graph(g,[0],meta_paths_net)
    mem_based_g = get_random_walk_graph(g,[0],meta_paths_mem)
    sock_based_g = get_random_walk_graph(g,[0],meta_paths_sock)
    new_idx = np.setdiff1d(idx,mal_index)
    shuffle_idx = np.random.permutation(new_idx)
    # train_idx =np.append(shuffle_idx[0:23552],train_mal_idx)
    # val_idx = np.append(shuffle_idx[23553:26916],val_mal_idx)
    # test_idx = np.append(shuffle_idx[26917:],test_mal_idx)
    train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    test_idx = np.append(shuffle_idx[14202:], test_mal_idx)
    train_idx = np.random.permutation(train_idx)
    val_idx = np.random.permutation(val_idx)
    test_idx = np.random.permutation(test_idx)
    idx = np.random.permutation(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,process_based_g,\
           file_based_g,attribute_based_g,IPC_based_g,net_based_g,mem_based_g,sock_based_g
def load_all_data_v3():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    num_nodes = 34327
    num_files = 10074
    num_attributes = 6
    num_classes = 2
    feature_matrix = np.loadtxt('data/graph_v7/label/process_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v7/label/malid.json')['id']
    shuffle_mal_index = np.random.permutation(mal_index)
    # train_mal_idx = shuffle_mal_index[0:24]
    # val_mal_idx = shuffle_mal_index[25:28]
    # test_mal_idx = shuffle_mal_index[29:]
    train_mal_idx = shuffle_mal_index[0:476]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[477:]
    print(len(train_mal_idx), len(test_mal_idx))
    features = torch.FloatTensor(feature_matrix)
    label = get_all_label()
    g = get_full_h_graph_v2()
    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    file_idx = np.arange(num_files)
    attr_idx = np.arange(num_attributes)
    process_based_g = get_random_walk_graph(g,idx,meta_paths_process)
    file_based_g = get_random_walk_graph(g,file_idx,meta_paths_file)
    attribute_based_g = get_random_walk_graph(g,attr_idx,meta_paths_attribute)
    IPC_based_g = get_random_walk_graph(g,[0],meta_paths_IPC)
    net_based_g = get_random_walk_graph(g,[0],meta_paths_net)
    mem_based_g = get_random_walk_graph(g,[0],meta_paths_mem)
    sock_based_g = get_random_walk_graph(g,[0],meta_paths_sock)
    new_idx = np.setdiff1d(idx,mal_index)
    zero_id = torch.load('zero_id_v2.pth')
    zero_id = np.array(zero_id)
    shuffle_idx = np.random.permutation(new_idx)
    shuffle_idx = np.setdiff1d(shuffle_idx, zero_id)
    shuffle_idx =np.random.permutation(shuffle_idx)
    train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    # train_idx=train_idx[0:1]
    val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    test_idx = np.append(shuffle_idx[14202:], test_mal_idx)
    # train_idx = np.append(shuffle_idx[0:30000], train_mal_idx)
    # val_idx = np.append(shuffle_idx[23553:26916], val_mal_idx)
    # test_idx = np.append(shuffle_idx[26917:], test_mal_idx)
    train_idx = np.append(shuffle_idx[0:3000], train_mal_idx)
    val_idx = np.append(shuffle_idx[3001:4001], val_mal_idx)
    test_idx = np.append(shuffle_idx[4002:], test_mal_idx)
    train_idx =np.random.permutation(train_idx)
    val_idx  = np.random.permutation(val_idx)
    test_idx= np.random.permutation(test_idx)
    idx = np.random.shuffle(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,process_based_g,\
           file_based_g,attribute_based_g,IPC_based_g,net_based_g,mem_based_g,sock_based_g
def load_all_data_GPU():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    num_nodes = 34327
    num_files = 10074
    num_attributes = 6
    num_classes = 2
    feature_matrix = np.loadtxt('data/graph_v7/label/process_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v7/label/malid.json')['id']
    shuffle_mal_index = np.random.permutation(mal_index)
    # train_mal_idx = shuffle_mal_index[0:24]
    # val_mal_idx = shuffle_mal_index[25:28]
    # test_mal_idx = shuffle_mal_index[29:]
    train_mal_idx = shuffle_mal_index[0:476]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[477:]
    print(len(train_mal_idx), len(test_mal_idx))
    features = torch.FloatTensor(feature_matrix)
    label = get_all_label()
    g = get_full_h_graph_v2()
    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    file_idx = np.arange(num_files)
    attr_idx = np.arange(num_attributes)
    process_based_g = get_random_walk_graph(g,idx,meta_paths_process)
    file_based_g = get_random_walk_graph(g,file_idx,meta_paths_file)
    attribute_based_g = get_random_walk_graph(g,attr_idx,meta_paths_attribute)
    IPC_based_g = get_random_walk_graph(g,[0],meta_paths_IPC)
    net_based_g = get_random_walk_graph(g,[0],meta_paths_net)
    mem_based_g = get_random_walk_graph(g,[0],meta_paths_mem)
    sock_based_g = get_random_walk_graph(g,[0],meta_paths_sock)
    new_idx = np.setdiff1d(idx,mal_index)
    zero_id = torch.load('zero_id_v2.pth')
    zero_id = np.array(zero_id)
    shuffle_idx = np.random.permutation(new_idx)
    shuffle_idx = np.setdiff1d(shuffle_idx, zero_id)
    shuffle_idx =np.random.permutation(shuffle_idx)
    train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    # train_idx=train_idx[0:1]
    val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    test_idx = np.append(shuffle_idx[14202:], test_mal_idx)
    # train_idx = np.append(shuffle_idx[0:30000], train_mal_idx)
    # val_idx = np.append(shuffle_idx[23553:26916], val_mal_idx)
    # test_idx = np.append(shuffle_idx[26917:], test_mal_idx)
    # train_idx = np.append(shuffle_idx[0:3000], train_mal_idx)
    # val_idx = np.append(shuffle_idx[3001:4001], val_mal_idx)
    # test_idx = np.append(shuffle_idx[4002:6002], test_mal_idx)
    train_idx =np.random.permutation(train_idx)
    val_idx  = np.random.permutation(val_idx)
    test_idx= np.random.permutation(test_idx)
    idx = np.random.shuffle(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,process_based_g,\
           file_based_g,attribute_based_g,IPC_based_g,net_based_g,mem_based_g,sock_based_g
def load_all_data_darpa():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    old_num_nodes = 34327
    old_prcoess_idx = np.arange(old_num_nodes)
    num_nodes = 84864
    num_files = 15433
    num_attributes = 6
    num_classes =2
    feature_matrix = np.loadtxt('data/darpa/label/process_matrix.txt')
    mal_index = load_one_jsondata('data/darpa/label/malid.json')['id']
    darpa_mal_index = load_one_jsondata('data/darpa/label/newmalid.json')['id']
    darpa_index = load_one_jsondata('data/darpa/label/darpaid.json')['id']
    mal_index= np.setdiff1d(mal_index,darpa_mal_index)
    darpa_benign_index = np.setdiff1d(darpa_index,darpa_mal_index)
    # print(darpa_benign_index,darpa_index,darpa_mal_index)
    # mal_index,darpa_mal_index = separate_mal_index(mal_index,34326)

    shuffle_mal_index = np.random.permutation(mal_index)
    shuffle_darpa_mal_index =np.random.permutation(darpa_mal_index)
    train_mal_idx = shuffle_mal_index[0:476]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[477:]
    train_darpa_idx = shuffle_darpa_mal_index[0:4]
    test_darpa_idx = shuffle_darpa_mal_index[4:]

    darpa_mal_index=np.random.permutation(darpa_mal_index)

    features = torch.FloatTensor(feature_matrix)
    label = get_all_label_darpa()
    g = get_full_h_graph_v2()

    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    # old_idx,new_darpa_idx = separate_mal_index(idx,34326)
    file_idx = np.arange(num_files)
    attr_idx = np.arange(num_attributes)
    process_based_g = get_random_walk_graph(g,idx,meta_paths_process)
    file_based_g = get_random_walk_graph(g,file_idx,meta_paths_file)
    attribute_based_g = get_random_walk_graph(g,attr_idx,meta_paths_attribute)
    IPC_based_g = get_random_walk_graph(g,[0],meta_paths_IPC)
    net_based_g = get_random_walk_graph(g,[0],meta_paths_net)
    mem_based_g = get_random_walk_graph(g,[0],meta_paths_mem)
    sock_based_g = get_random_walk_graph(g,[0],meta_paths_sock)
    new_idx = np.setdiff1d(idx,mal_index)
    old_prcoess_idx =np.setdiff1d(old_prcoess_idx,mal_index)
    # new_darpa_idx = np.setdiff1d(new_darpa_idx,darpa_mal_index)
    zero_id = torch.load('zero_id_v2.pth')
    zero_id = np.array(zero_id)
    shuffle_idx = np.random.permutation(old_prcoess_idx)
    shuffle_idx=np.setdiff1d(shuffle_idx, zero_id)
    shuffle_darpa_idx =np.random.permutation(darpa_benign_index)
    # shuffle_darpa_idx=np.random.permutation(new_darpa_idx)
    # train_idx =np.append(shuffle_idx[0:23552],train_mal_idx)
    # val_idx = np.append(shuffle_idx[23553:26916],val_mal_idx)
    # test_idx = np.append(shuffle_idx[26917:],test_mal_idx)
    train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    test_idx = np.append(shuffle_idx[14202:], test_mal_idx)

    darpa_train_idx = np.append(shuffle_darpa_idx[0:260],train_darpa_idx)
    darpa_test_idx = np.append(shuffle_darpa_idx[260:],test_darpa_idx)
    darpa_idx = np.append(shuffle_darpa_idx,darpa_mal_index)
    train_idx = np.append(train_idx,train_darpa_idx)
    # train_idx = np.append(train_idx, darpa_train_idx)
    test_idx = np.append(test_idx,darpa_test_idx)
    train_idx = np.random.permutation(train_idx)
    val_idx = np.random.permutation(val_idx)
    test_idx = np.random.permutation(test_idx)
    darpa_train_idx=np.random.permutation(darpa_train_idx)
    darpa_test_idx=np.random.permutation(darpa_test_idx)

    darpa_idx=np.random.permutation(darpa_idx)
    idx = np.random.permutation(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    darpa_train_mask = get_binary_mask(num_nodes, darpa_train_idx)
    darpa_test_mask = get_binary_mask(num_nodes, darpa_test_idx)
    darpa_mask = get_binary_mask(num_nodes,darpa_idx)

    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,darpa_mal_index,\
           darpa_train_idx,darpa_test_idx,darpa_idx,darpa_train_mask,darpa_test_mask,darpa_mask,process_based_g,\
           file_based_g,attribute_based_g,IPC_based_g,net_based_g,mem_based_g,sock_based_g
def re_sort(list,idx):
    new_list = []
    for id in idx:
        new_list.append(list[id].numpy())
    return torch.from_numpy(np.array(new_list))
def load_all_data_outsample():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    num_nodes = 33646
    num_files = 10074
    num_attributes = 6
    num_classes =2
    feature_matrix = np.loadtxt('data/graph_v6/label/process_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v6/label/mal.json')['node']
    shuffle_mal_index = np.random.permutation(mal_index)

    train_mal_idx = shuffle_mal_index[0:24]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[29:]
    features = torch.FloatTensor(feature_matrix)
    label = get_all_label()
    g = get_full_h_graph_v2()
    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    file_idx = np.arange(num_files)
    attr_idx = np.arange(num_attributes)
    new_idx = np.setdiff1d(idx, mal_index)
    zero_id = torch.load('zero_id.pth')
    zero_id = np.array(zero_id)
    shuffle_idx = np.random.permutation(new_idx)
    shuffle_idx = np.setdiff1d(shuffle_idx, zero_id)
    shuffle_idx = np.random.permutation(shuffle_idx)
    train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    test_idx = np.append(shuffle_idx[14202:], test_mal_idx)
    # train_idx = np.append(shuffle_idx[0:23552], train_mal_idx)
    # val_idx = np.append(shuffle_idx[23553:26916], val_mal_idx)
    # test_idx = np.append(shuffle_idx[26917:], test_mal_idx)
    # train_idx = np.append(shuffle_idx[0:3000], train_mal_idx)
    # val_idx = np.append(shuffle_idx[3001:4001], val_mal_idx)
    # test_idx = np.append(shuffle_idx[4002:6002], test_mal_idx)
    train_idx = np.random.permutation(train_idx)
    val_idx = np.random.permutation(val_idx)
    test_idx = np.random.permutation(test_idx)
    idx = np.random.shuffle(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    # fea = g.ndata['h']['process'][train_idx]
    # fea_mask  = re_sort(g.ndata['h']['process'],train_idx)
    new_g = dgl.node_subgraph(g,{'process':train_idx,'file':file_idx,'IPC':[0],'net':[0],'memory':[0],'socket':[0],'attribute':attr_idx})
    # new_fea = new_g.ndata['h']['process']
    # new_label = sort_label(label,train_idx)

    # print(labels,labels[train_idx])
    # for i,j,k in zip(fea,new_fea,fea_mask):
    #     print(i,j,k)
    process_idx = np.arange(len(train_idx))
    process_based_g = get_random_walk_graph(new_g,process_idx,meta_paths_process)
    file_based_g = get_random_walk_graph(new_g,file_idx,meta_paths_file)
    attribute_based_g = get_random_walk_graph(new_g,attr_idx,meta_paths_attribute)
    IPC_based_g = get_random_walk_graph(new_g,[0],meta_paths_IPC)
    net_based_g = get_random_walk_graph(new_g,[0],meta_paths_net)
    mem_based_g = get_random_walk_graph(new_g,[0],meta_paths_mem)
    sock_based_g = get_random_walk_graph(new_g,[0],meta_paths_sock)

    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,new_g,process_based_g,\
           file_based_g,attribute_based_g,IPC_based_g,net_based_g,mem_based_g,sock_based_g
def load_all_data_outsample_darpa():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    old_num_nodes = 34327
    old_prcoess_idx = np.arange(old_num_nodes)
    num_nodes = 84864
    num_files = 15433
    num_attributes = 6
    num_classes = 2
    feature_matrix = np.loadtxt('data/darpa/label/process_matrix.txt')
    mal_index = load_one_jsondata('data/darpa/label/malid.json')['id']
    darpa_mal_index = load_one_jsondata('data/darpa/label/newmalid.json')['id']
    darpa_index = load_one_jsondata('data/darpa/label/darpaid.json')['id']
    mal_index = np.setdiff1d(mal_index, darpa_mal_index)
    darpa_benign_index = np.setdiff1d(darpa_index, darpa_mal_index)
    # print(darpa_benign_index,darpa_index,darpa_mal_index)
    # mal_index,darpa_mal_index = separate_mal_index(mal_index,34326)

    shuffle_mal_index = np.random.permutation(mal_index)
    shuffle_darpa_mal_index = np.random.permutation(darpa_mal_index)
    train_mal_idx = shuffle_mal_index[0:476]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[477:]
    train_darpa_idx = shuffle_darpa_mal_index[0:4]
    test_darpa_idx = shuffle_darpa_mal_index[4:]

    darpa_mal_index = np.random.permutation(darpa_mal_index)

    features = torch.FloatTensor(feature_matrix)
    label = get_all_label_darpa()
    g = get_full_h_graph_v2()

    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    # old_idx,new_darpa_idx = separate_mal_index(idx,34326)
    file_idx = np.arange(num_files)
    attr_idx = np.arange(num_attributes)

    new_idx = np.setdiff1d(idx, mal_index)
    old_prcoess_idx = np.setdiff1d(old_prcoess_idx, mal_index)
    # new_darpa_idx = np.setdiff1d(new_darpa_idx,darpa_mal_index)
    zero_id = torch.load('zero_id_v2.pth')
    zero_id = np.array(zero_id)
    shuffle_idx = np.random.permutation(old_prcoess_idx)
    shuffle_idx = np.setdiff1d(shuffle_idx, zero_id)
    shuffle_darpa_idx = np.random.permutation(darpa_benign_index)
    # shuffle_darpa_idx=np.random.permutation(new_darpa_idx)
    # train_idx =np.append(shuffle_idx[0:23552],train_mal_idx)
    # val_idx = np.append(shuffle_idx[23553:26916],val_mal_idx)
    # test_idx = np.append(shuffle_idx[26917:],test_mal_idx)
    train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    test_idx = np.append(shuffle_idx[14202:], test_mal_idx)

    darpa_train_idx = np.append(shuffle_darpa_idx[0:260], train_darpa_idx)
    darpa_test_idx = np.append(shuffle_darpa_idx[260:], test_darpa_idx)
    darpa_idx = np.append(shuffle_darpa_idx, darpa_mal_index)
    train_idx = np.append(train_idx, darpa_train_idx)
    test_idx = np.append(test_idx, darpa_test_idx)
    train_idx = np.random.permutation(train_idx)
    val_idx = np.random.permutation(val_idx)
    test_idx = np.random.permutation(test_idx)
    darpa_train_idx = np.random.permutation(darpa_train_idx)
    darpa_test_idx = np.random.permutation(darpa_test_idx)

    darpa_idx = np.random.permutation(darpa_idx)
    idx = np.random.permutation(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    darpa_train_mask = get_binary_mask(num_nodes, darpa_train_idx)
    darpa_test_mask = get_binary_mask(num_nodes, darpa_test_idx)
    darpa_mask = get_binary_mask(num_nodes, darpa_idx)

    new_g = dgl.node_subgraph(g,{'process':train_idx,'file':file_idx,'IPC':[0],'net':[0],'memory':[0],'socket':[0],'attribute':attr_idx})
    # new_fea = new_g.ndata['h']['process']
    # new_label = sort_label(label,train_idx)

    # print(labels,labels[train_idx])
    # for i,j,k in zip(fea,new_fea,fea_mask):
    #     print(i,j,k)
    process_idx = np.arange(len(train_idx))
    process_based_g = get_random_walk_graph(new_g,process_idx,meta_paths_process)
    file_based_g = get_random_walk_graph(new_g,file_idx,meta_paths_file)
    attribute_based_g = get_random_walk_graph(new_g,attr_idx,meta_paths_attribute)
    IPC_based_g = get_random_walk_graph(new_g,[0],meta_paths_IPC)
    net_based_g = get_random_walk_graph(new_g,[0],meta_paths_net)
    mem_based_g = get_random_walk_graph(new_g,[0],meta_paths_mem)
    sock_based_g = get_random_walk_graph(new_g,[0],meta_paths_sock)

    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,darpa_mal_index,\
           darpa_train_idx,darpa_test_idx,darpa_idx,darpa_train_mask,darpa_test_mask,darpa_mask,new_g,process_based_g,\
           file_based_g,attribute_based_g,IPC_based_g,net_based_g,mem_based_g,sock_based_g
def load_all_data_outsample_v2():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    num_nodes = 34327
    num_files = 10074
    num_attributes = 6
    num_classes = 2
    feature_matrix = np.loadtxt('data/darpa/label/process_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v7/label/malid.json')['id']
    shuffle_mal_index = np.random.permutation(mal_index)
    train_mal_idx = shuffle_mal_index[0:476]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[477:]
    features = torch.FloatTensor(feature_matrix)
    label = get_all_label()
    g = get_full_h_graph_v2()
    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    file_idx = np.arange(num_files)
    attr_idx = np.arange(num_attributes)
    new_idx = np.setdiff1d(idx, mal_index)
    zero_id = torch.load('zero_id_v2.pth')
    zero_id = np.array(zero_id)
    shuffle_idx = np.random.permutation(new_idx)
    shuffle_idx = np.setdiff1d(shuffle_idx, zero_id)
    shuffle_idx = np.random.permutation(shuffle_idx)
    train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    test_idx = np.append(shuffle_idx[14202:], test_mal_idx)
    # train_idx = np.append(shuffle_idx[0:3000], train_mal_idx)
    # val_idx = np.append(shuffle_idx[3001:4001], val_mal_idx)
    # test_idx = np.append(shuffle_idx[4002:6002], test_mal_idx)
    train_idx = np.random.permutation(train_idx)
    val_idx = np.random.permutation(val_idx)
    test_idx = np.random.permutation(test_idx)
    idx = np.random.shuffle(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    # fea = g.ndata['h']['process'][train_idx]
    # fea_mask  = re_sort(g.ndata['h']['process'],train_idx)
    new_g = dgl.node_subgraph(g,{'process':train_idx,'file':file_idx,'IPC':[0],'net':[0],'memory':[0],'socket':[0],'attribute':attr_idx})
    # new_fea = new_g.ndata['h']['process']
    # new_label = sort_label(label,train_idx)

    # print(labels,labels[train_idx])
    # for i,j,k in zip(fea,new_fea,fea_mask):
    #     print(i,j,k)
    process_idx = np.arange(len(train_idx))
    process_based_g = get_random_walk_graph(new_g,process_idx,meta_paths_process)
    file_based_g = get_random_walk_graph(new_g,file_idx,meta_paths_file)
    attribute_based_g = get_random_walk_graph(new_g,attr_idx,meta_paths_attribute)
    IPC_based_g = get_random_walk_graph(new_g,[0],meta_paths_IPC)
    net_based_g = get_random_walk_graph(new_g,[0],meta_paths_net)
    mem_based_g = get_random_walk_graph(new_g,[0],meta_paths_mem)
    sock_based_g = get_random_walk_graph(new_g,[0],meta_paths_sock)

    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,new_g,process_based_g,\
           file_based_g,attribute_based_g,IPC_based_g,net_based_g,mem_based_g,sock_based_g
def load_all_data_v4():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    num_nodes = 33646
    num_files = 10074
    num_attributes = 6
    num_classes = 2
    feature_matrix = np.loadtxt('data/graph_v6/label/process_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v6/label/mal.json')['node']
    shuffle_mal_index = np.random.permutation(mal_index)
    train_mal_idx = shuffle_mal_index[0:24]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[29:]

    features = torch.FloatTensor(feature_matrix)
    label = get_all_label()
    g = get_full_h_graph_v2()
    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    file_idx = np.arange(num_files)
    attr_idx = np.arange(num_attributes)
    new_idx = np.setdiff1d(idx, mal_index)
    shuffle_idx = np.random.permutation(new_idx)
    train_idx = np.append(shuffle_idx[0:23552], train_mal_idx)
    val_idx = np.append(shuffle_idx[23553:26916], val_mal_idx)
    test_idx = np.append(shuffle_idx[26917:], test_mal_idx)
    sub_train_g = dgl.node_subgraph(g,{'process':train_idx,'file':file_idx,'IPC':[0],'memory':[0],'socket':[0],'net':[0],'attribute':attr_idx})
    process_based_g = get_random_walk_graph(sub_train_g, train_idx, meta_paths_process)

    file_based_g = get_random_walk_graph(sub_train_g, file_idx, meta_paths_file)
    attribute_based_g = get_random_walk_graph(sub_train_g, attr_idx, meta_paths_attribute)
    IPC_based_g = get_random_walk_graph(sub_train_g, [0], meta_paths_IPC)
    net_based_g = get_random_walk_graph(sub_train_g, [0], meta_paths_net)
    mem_based_g = get_random_walk_graph(sub_train_g, [0], meta_paths_mem)
    sock_based_g = get_random_walk_graph(sub_train_g, [0], meta_paths_sock)
    train_idx = np.random.shuffle(train_idx)
    val_idx = np.random.shuffle(val_idx)
    test_idx = np.random.shuffle(test_idx)
    idx = np.random.shuffle(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return g, label, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, process_based_g, \
           file_based_g, attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g,sub_train_g
def load_all_data_v5():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    num_nodes = 34327
    num_files = 10074
    num_attributes = 6
    num_classes =2
    feature_matrix = np.loadtxt('data/graph_v7/label/process_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v7/label/mal.json')['node']
    shuffle_mal_index = np.random.permutation(mal_index)

    train_mal_idx = shuffle_mal_index[0:476]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[477:]
    features = torch.FloatTensor(feature_matrix)
    label = get_all_label()
    g = get_full_h_graph_v2()
    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    file_idx = np.arange(num_files)
    attr_idx = np.arange(num_attributes)
    new_idx = np.setdiff1d(idx, mal_index)
    zero_id = torch.load('zero_id.pth')
    zero_id = np.array(zero_id)
    shuffle_idx = np.random.permutation(new_idx)
    shuffle_idx = np.setdiff1d(shuffle_idx, zero_id)
    shuffle_idx = np.random.permutation(shuffle_idx)
    train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    test_idx = np.append(shuffle_idx[14202:], test_mal_idx)
    # train_idx = np.append(shuffle_idx[0:3000], train_mal_idx)
    # val_idx = np.append(shuffle_idx[3001:4001], val_mal_idx)
    # test_idx = np.append(shuffle_idx[4002:6002], test_mal_idx)
    train_idx = np.random.permutation(train_idx)
    val_idx = np.random.permutation(val_idx)
    test_idx = np.random.permutation(test_idx)
    idx = np.random.shuffle(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    # fea = g.ndata['h']['process'][train_idx]
    # fea_mask  = re_sort(g.ndata['h']['process'],train_idx)
    new_g = dgl.node_subgraph(g,{'process':train_idx,'file':file_idx,'IPC':[0],'net':[0],'memory':[0],'socket':[0],'attribute':attr_idx})
    # new_fea = new_g.ndata['h']['process']
    # new_label = sort_label(label,train_idx)

    # print(labels,labels[train_idx])
    # for i,j,k in zip(fea,new_fea,fea_mask):
    #     print(i,j,k)
    process_idx = np.arange(len(train_idx))
    process_based_g = get_random_walk_graph(new_g,process_idx,meta_paths_process)
    file_based_g = get_random_walk_graph(new_g,file_idx,meta_paths_file)
    attribute_based_g = get_random_walk_graph(new_g,attr_idx,meta_paths_attribute)
    IPC_based_g = get_random_walk_graph(new_g,[0],meta_paths_IPC)
    net_based_g = get_random_walk_graph(new_g,[0],meta_paths_net)
    mem_based_g = get_random_walk_graph(new_g,[0],meta_paths_mem)
    sock_based_g = get_random_walk_graph(new_g,[0],meta_paths_sock)

    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,new_g,process_based_g,\
           file_based_g,attribute_based_g,IPC_based_g,net_based_g,mem_based_g,sock_based_g
def load_all_data_homogeneous():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    num_all =100297
    old_num_nodes = 34327
    old_prcoess_idx = np.arange(old_num_nodes)
    num_nodes = 100307
    num_files = 15433
    num_attributes = 6
    num_classes = 2
    idx = np.arange(num_nodes)
    feature_matrix = np.loadtxt('data/darpa/label/process_matrix.txt')
    mal_index = load_one_jsondata('data/darpa/label/malid.json')['id']
    darpa_mal_index = load_one_jsondata('data/darpa/label/newmalid.json')['id']
    darpa_index = load_one_jsondata('data/darpa/label/darpaid.json')['id']
    mal_index = np.setdiff1d(mal_index, darpa_mal_index)
    darpa_benign_index = np.setdiff1d(darpa_index, darpa_mal_index)
    # print(darpa_benign_index,darpa_index,darpa_mal_index)
    # mal_index,darpa_mal_index = separate_mal_index(mal_index,34326)

    shuffle_mal_index = np.random.permutation(mal_index)
    shuffle_darpa_mal_index = np.random.permutation(darpa_mal_index)
    train_mal_idx = shuffle_mal_index[0:476]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[477:]
    train_darpa_idx = shuffle_darpa_mal_index[0:4]
    test_darpa_idx = shuffle_darpa_mal_index[4:]

    darpa_mal_index = np.random.permutation(darpa_mal_index)

    features = torch.FloatTensor(feature_matrix)

    label = get_all_label_v2()
    g = get_full_homogeneous_graph()
    label = torch.LongTensor(label)
    new_idx = np.setdiff1d(idx, mal_index)
    old_prcoess_idx = np.setdiff1d(old_prcoess_idx, mal_index)
    # new_darpa_idx = np.setdiff1d(new_darpa_idx,darpa_mal_index)
    # zero_id = torch.load('zero_id_v2.pth')
    # zero_id = np.array(zero_id)
    shuffle_idx = np.random.permutation(old_prcoess_idx)
    # shuffle_idx = np.setdiff1d(shuffle_idx, zero_id)
    shuffle_darpa_idx = np.random.permutation(darpa_benign_index)
    # shuffle_darpa_idx=np.random.permutation(new_darpa_idx)
    # train_idx =np.append(shuffle_idx[0:23552],train_mal_idx)
    # val_idx = np.append(shuffle_idx[23553:26916],val_mal_idx)
    # test_idx = np.append(shuffle_idx[26917:],test_mal_idx)
    train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    test_idx = np.append(shuffle_idx[14202:], test_mal_idx)

    darpa_train_idx = np.append(shuffle_darpa_idx[0:260], train_darpa_idx)
    darpa_test_idx = np.append(shuffle_darpa_idx[260:], test_darpa_idx)
    darpa_idx = np.append(shuffle_darpa_idx, darpa_mal_index)
    train_idx = np.append(train_idx, train_darpa_idx)
    test_idx = np.append(test_idx, darpa_test_idx)
    train_idx = np.random.permutation(train_idx)
    val_idx = np.random.permutation(val_idx)
    test_idx = np.random.permutation(test_idx)
    darpa_train_idx = np.random.permutation(darpa_train_idx)
    darpa_test_idx = np.random.permutation(darpa_test_idx)

    darpa_idx = np.random.permutation(darpa_idx)
    idx = np.random.permutation(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    darpa_train_mask = get_binary_mask(num_nodes, darpa_train_idx)
    darpa_test_mask = get_binary_mask(num_nodes, darpa_test_idx)
    darpa_mask = get_binary_mask(num_nodes, darpa_idx)
    #模拟数据集
    # np.random.seed()
    # # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    # num_all = 44411
    # num_nodes = 34327
    # num_files = 10074
    # num_attributes = 6
    # num_classes = 2
    # feature_matrix = np.loadtxt('data/graph_v7/label/process_matrix.txt')
    # mal_index = load_one_jsondata('data/graph_v7/label/malid.json')['id']
    # shuffle_mal_index = np.random.permutation(mal_index)
    # train_mal_idx = shuffle_mal_index[0:476]
    # val_mal_idx = shuffle_mal_index[25:28]
    # test_mal_idx = shuffle_mal_index[477:]
    # features = torch.FloatTensor(feature_matrix)
    # idx = np.arange(num_nodes)
    # new_idx = np.setdiff1d(idx, mal_index)
    # shuffle_idx = np.random.permutation(new_idx)
    # train_idx = np.append(shuffle_idx[0:12000], train_mal_idx)
    # val_idx = np.append(shuffle_idx[12001:14201], val_mal_idx)
    # test_idx = np.append(shuffle_idx[14202:21148], test_mal_idx)
    # np.random.shuffle(train_idx)
    # np.random.shuffle(val_idx)
    # np.random.shuffle(test_idx)
    # np.random.shuffle(idx)
    # train_mask = get_binary_mask(num_all, train_idx)
    # val_mask = get_binary_mask(num_all, val_idx)
    # test_mask = get_binary_mask(num_all, test_idx)
    return g, label, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask,darpa_train_mask,darpa_test_mask
def get_random_walk_graph(g,idx,meta_paths):
    meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
    _cached_coalesced_graph = {}
    for meta_path in meta_paths:
        new_g = dgl.sampling.RandomWalkNeighborSampler(g, termination_prob=0.5, num_neighbors=10, num_random_walks=2,
                                                       num_traversals=5, metapath=meta_path)
        _cached_coalesced_graph[meta_path] = new_g(idx)
    return _cached_coalesced_graph
def set_new_malid():
    malid = load_one_jsondata('data/graph_v6/label/mal.json')['node']
    for i in range(33647, 34327):
        malid.append(i)
    malid_json = {'id': malid}
    with open('data/graph_v7/label/malid.json', 'w') as f:
        json.dump(malid_json, f)
def create_new_mal_data():
    malid = load_one_jsondata('data/graph_v6/label/mal.json')['node']
    maldata = loadjson('data/graph_v6/label/maldata.json')
    malfile = load_one_jsondata('data/graph_v6/label/malfile.json')['file']
    maldata_path = 'data/graph_v6/label/new_maldata.json'
    cycle_time = 20
    # malid = [13306]
    final_new_mal_data = []
    all_data = []
    for i in malid:
        one_id_data = []
        for data in maldata:
            if data['link'][0] == i:
                one_id_data.append(data)
        all_data.append(one_id_data)
    for i,id in enumerate(malid):
        current_data = all_data[i]
        for j in range(cycle_time):
            new_one_id_data = []
            for data in current_data:
                copy_data =data
                flag = 1
                drop_rate = random.random()
                for filename in malfile:
                    if filename in data['path']:
                        flag = 0
                if copy_data['type'][2] == 'SUBJECT_PROCESS':
                    flag = 0
                if flag == 1 and drop_rate > 0.3:
                    new_one_id_data.append(copy_data)
                if flag == 0:
                        new_one_id_data.append(copy_data)
            final_new_mal_data.append(new_one_id_data)
    new_id = 33647
    for i,dic in enumerate(final_new_mal_data):
        current_id = new_id+i
        path = 'data/graph_v7/%s.json'%current_id
        # print(new_id,i)
        for data in dic:
            data['link'][0]=current_id
            if data['type'][2] == 'SUBJECT_PROCESS':
                data['link'][1] = current_id
                # print(data)
        with open(path,'w') as f:
            for data in dic:
                json.dump(data, f)
                f.write('\n')

        # for data in one_id_data:
        #     print(data)
    # with open(maldata_path, 'w') as f:
    #     for dic in final_new_mal_data:
    #         for one_data in dic:
    #             json.dump(one_data, f)
    #             f.write('\n')

    # f.close()
def get_node_count(data):
    pcnt=0
    fcnt=0
    for i in data:
        if i['link'][0]>pcnt:
            pcnt =i['link'][0]
        if 'FILE' in i['type'][2]:
            if i['link'][1]>fcnt:
                fcnt=i['link'][1]
    return pcnt,fcnt
if __name__ =='__main__':
    # 获取依赖图数据
    # data = load_all_link()

    # data就是第一步的依赖图数据，保存特征矩阵，save_feature_matrix()里面sorted_data那个数字代表进程数量，记得改掉。
    # write_feature_list(data)
    # save_feature_matrix()

    # 获取异构图的所有邻接矩阵，里面process_feature第一个代表进程数量，file_feature第一个代表文件数量，记得改掉。
    # 里面feature_list_v2.json、meta_v2.json、attribute_list_v2.json、link.json都是预先设好的
    # set_graph_link()

    # 构建异构图，里面ipc_matrix.txt、net_matrix.txt、mem_matrix.txt、soc_matrix.txt、attr_matrix.txt都是设好的
    # 函数最后三行为图节点添加数据，如果缺了啥比如上面graph_dict里面没有memory就把什么删掉。
    # hg = get_full_h_graph_v2()
    # print(hg)

    # g, label, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask=load_all_data_homogeneous()

    # data =load_one_jsondata('data/darpa/label/newid.json')
    # data = torch.load('data/embedding_v2/dataset_list_32.pkl')

    # data = load_all_link_v2()

    # darpa_index = load_one_jsondata('data/darpa/label/darpaid.json')['id']
    # print(len(darpa_index))

    g, labels, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, darpa_mal_index, \
    darpa_train_idx, darpa_test_idx, darpa_idx, darpa_train_mask, darpa_test_mask, darpa_mask, process_based_g, \
    file_based_g, attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g = load_all_data_darpa()
    print(g)