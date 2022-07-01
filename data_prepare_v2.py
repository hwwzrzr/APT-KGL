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
    path2 = glob('data/graph_v5/'+'*.json')
    path1 = glob('data/graph_v4/'+'*.json')
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
def load_all_link_v2():

    path = glob('data/graph_v5/' + '*.json')
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
    nodenum = 33646
    mal = load_one_jsondata('data/graph_v5/label/mal.json')
    mal_li = mal['node']
    label = np.zeros((nodenum))
    label[mal_li] = 1
    return label
def get_mal_data():
    data = load_all_link()
    mal = []
    malid = [13486, 20825, 20830, 14679, 14682, 14684, 14715, 25583, 27097, 15380, 15395, 15397, 15458, 15659, 13482, 13483, 18224, 18353, 18453, 18454, 18458, 18510, 14163, 20489]
    for dic in data:
        if dic['link'][0] in malid:
            mal.append(dic)
    path = 'data/graph_v5/label/maldata.json'
    with open(path, 'w') as f:
        for dic in mal:
            json.dump(dic, f)
            f.write('\n')
    f.close()
def write_feature_list(data):
    feature_list = []
    path  = 'data/graph_v5/label/feature_list.json'
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
    feature_list = load_one_jsondata('data/graph_v5/label/feature_list.json')
    new_data = sorted_data(data,33646)
    feature_matrix = get_feature_matrix(new_data,feature_list)
    np.savetxt('data/graph_v5/label/feature_matrix.txt',feature_matrix)
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
    num_nodes = 33646
    num_classes =2
    feature_matrix = np.loadtxt('data/graph_v5/label/feature_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v5/label/mal.json')['node']
    shuffle_mal_index = np.random.permutation(mal_index)
    train_mal_idx = shuffle_mal_index[0:24]
    val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[29:]
    features = torch.FloatTensor(feature_matrix)
    label = get_all_label()
    g = get_full_h_graph()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    new_idx = np.setdiff1d(idx,mal_index)
    shuffle_idx = np.random.permutation(new_idx)
    # test_idx = np.append(shuffle_idx[0:23552], shuffle_mal_index)
    # val_idx = np.append(shuffle_idx[23553:26916], shuffle_mal_index)
    # train_idx = np.append(shuffle_idx[26917:], shuffle_mal_index)
    train_idx =np.append(shuffle_idx[0:23552],train_mal_idx)
    val_idx = np.append(shuffle_idx[23553:26916],val_mal_idx)
    test_idx = np.append(shuffle_idx[26917:],test_mal_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    np.random.shuffle(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return g,label,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,idx

def load_all_data_v1():
    np.random.seed()
    # split = lambda a: map(lambda b: a[b:b + 512], range(0, len(a), 512))
    num_nodes = 33646
    num_classes =2
    feature_matrix = np.loadtxt('data/graph_v5/label/feature_matrix.txt')
    mal_index = load_one_jsondata('data/graph_v5/label/mal.json')['node']
    shuffle_mal_index = np.random.permutation(mal_index)
    train_mal_idx = shuffle_mal_index[0:24]
    # val_mal_idx = shuffle_mal_index[25:28]
    test_mal_idx = shuffle_mal_index[25:]
    features = torch.FloatTensor(feature_matrix)
    print(features[train_mal_idx],features[test_mal_idx])
    label = get_all_label()
    label = torch.LongTensor(label)
    idx = np.arange(num_nodes)
    new_idx = np.setdiff1d(idx,mal_index)
    shuffle_idx = np.random.permutation(new_idx)
    # test_idx = np.append(shuffle_idx[0:23552], shuffle_mal_index)
    # val_idx = np.append(shuffle_idx[23553:26916], shuffle_mal_index)
    # train_idx = np.append(shuffle_idx[26917:], shuffle_mal_index)
    train_idx =np.append(shuffle_idx[0:23552],train_mal_idx)
    test_idx = np.append(shuffle_idx[23553:],test_mal_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    np.random.shuffle(idx)
    train_mask = get_binary_mask(num_nodes, train_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return label,features,num_classes, train_idx, test_idx,train_mask, test_mask,idx

if __name__ =='__main__':
    # data = loadjson('data/graph_v5/label/maldata.json')
    # temp = []
    # temp_data = []
    # for dic in data:
    #     if dic not in temp:
    #         temp.append(dic)
    # for i in data:
    #     print(i)
    g, label, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask,idx = load_all_data()
    print(g)
    meta_paths = [['s_fork_s'], ['s_changeprincipal_s'], ['s_execute_s'], ['s_exit_s'], ['s_clone_s'],
                  ['s_fork_s', 's_write_IPC', 'IPC_write_s'], ['s_close_IPC', 'IPC_close_s'],
                  ['s_read_IPC', 'IPC_read_s'], ['s_mmp_IPC', 'IPC_mmp_s'], ['s_open_f', 'f_close_s'],
                  ['s_read_f', 'f_read_s'], ['s_write_f', 'f_write_s'], ['s_create_f', 'f_create_s'],
                  ['s_unlink_f', 'f_unlink_s'], ['s_loadlibrary_f', 'f_loadlibrary_s'], ['s_update_f', 'f_update_s'],
                  ['s_modify_f', 'f_modify_s'], ['s_rename_f', 'f_rename_s'], ['s_mmp_f', 'f_mmp_s'],
                  ['s_truncate_f', 'f_truncate_s'], ['s_mmp_m', 'm_mmp_s'], ['s_mprotect_m', 'm_mprotect_s'],
                  ['s_connect_n', 'n_connect_s'], ['s_send_n', 'n_send_s'], ['s_recv_n', 'n_recv_s'],
                  ['s_read_n', 'n_read_s'], ['s_close_n', 'n_close_s'], ['s_accept_n', 'n_accept_s'],
                  ['s_write_n', 'n_write_s'], ['s_accept_sock', 'sock_accept_s'], ['s_write_sock', 'sock_write_s'],
                  ['s_read_sock', 'sock_read_s'], ['s_connect_sock', 'sock_connect_s'], ['s_recv_sock', 'sock_recv_s'],
                  ['s_send_sock', 'sock_send_s']]
    meta_paths_v1 = [['s_fork_s','s_read_f','f_write_s']]
    # emb = torch.randn([35,33646,64])
    # full_seq_emb = []
    # Rnn = torch.nn.LSTM(input_size=64,hidden_size=32,num_layers=1,batch_first=True)
    for i,meta_path in enumerate(meta_paths_v1):
        meta_emb = torch.zeros((33646,3,64))
        print(i)
        new_g = dgl.sampling.random_walk(g,idx,metapath=meta_path)
        seq = new_g[0]
        typ = new_g[1]
        print(typ)
        mask =torch.range(1,3).long()
        print(mask)
        # mask_v1 = torch.nonzero(typ)
        # mask = torch.where(typ==4)
        seq_mat = seq[:,mask]
        print(seq_mat)
    #     emb_features = emb[i]
    #     for j,node in enumerate(idx):
    #         node_emb = torch.zeros((4,64))
    #         for k,seq in enumerate(seq_mat[j]):
    #             if seq != -1:
    #                 node_emb[k] = emb_features[seq]
    #
    #         meta_emb[node] = node_emb
    #
    #     out,_ = Rnn(meta_emb,None)
    #     print(out)
        # full_seq_emb.append(seq_emb)












