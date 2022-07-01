import json
import pickle
import dgl
from dgl.data.utils import save_graphs
import torch
import numpy as np
def load_one_jsondata(path):
    with open(path) as f:
        for line in f:
            return json.loads(line)
def get_metapath():
    meta_paths = [['s_fork_s'], ['s_changeprincipal_s'], ['s_execute_s'], ['s_clone_s'],
                  ['s_fork_s', 's_write_IPC', 'IPC_write_s'],
                  ['s_read_IPC', 'IPC_read_s'], ['s_mmp_IPC', 'IPC_mmp_s'],
                  ['s_open_f', 'f_contain_attribute', 'attribute_contain_f', 'f_close_s'],
                  ['s_read_f', 'f_contain_attribute', 'attribute_contain_f', 'f_read_s'],
                  ['s_write_f', 'f_contain_attribute', 'attribute_contain_f', 'f_write_s'],
                  ['s_create_f', 'f_contain_attribute', 'attribute_contain_f', 'f_create_s'],
                  ['s_loadlibrary_f', 'f_contain_attribute', 'attribute_contain_f', 'f_loadlibrary_s'],
                  ['s_update_f', 'f_contain_attribute', 'attribute_contain_f', 'f_update_s'],
                  ['s_modify_f', 'f_contain_attribute', 'attribute_contain_f', 'f_modify_s'],
                  ['s_rename_f', 'f_contain_attribute', 'attribute_contain_f', 'f_rename_s'],
                  ['s_mmp_f', 'f_contain_attribute', 'attribute_contain_f', 'f_mmp_s'],
                  ['s_truncate_f', 'f_contain_attribute', 'attribute_contain_f', 'f_truncate_s'],
                  ['s_mmp_m', 'm_mmp_s'],
                  ['s_connect_n', 'n_connect_s'], ['s_send_n', 'n_recv_s'],
                  ['s_read_n', 'n_read_s'],
                  ['s_write_n', 'n_write_s'], ['s_write_sock', 'sock_write_s'],
                  ['s_read_sock', 'sock_read_s'], ['s_connect_sock', 'sock_connect_s'],
                  ['s_send_sock', 'sock_recv_s']]
    return meta_paths
def write_meta():
    meta = {'s_fork_s': [['SUBJECT_PROCESS', 'EVENT_FORK', 'SUBJECT_PROCESS']],
            's_changeprincipal_s': [['SUBJECT_PROCESS', 'EVENT_CHANGE_PRINCIPAL', 'SUBJECT_PROCESS']],
            's_execute_s': [['SUBJECT_PROCESS', 'EVENT_EXECUTE', 'SUBJECT_PROCESS']],
            's_exit_s': [['SUBJECT_PROCESS', 'EVENT_EXIT', 'SUBJECT_PROCESS']],
            's_clone_s': [['SUBJECT_PROCESS', 'EVENT_CLONE', 'SUBJECT_PROCESS']],
            's_write_IPC': [['SUBJECT_PROCESS', 'EVENT_WRITE', 'IPC_OBJECT_PIPE_UNNAMED']],
            's_close_IPC': [['SUBJECT_PROCESS', 'EVENT_CLOSE', 'IPC_OBJECT_PIPE_UNNAMED']],
            's_read_IPC': [['SUBJECT_PROCESS', 'EVENT_READ', 'IPC_OBJECT_PIPE_UNNAMED']],
            's_mmp_IPC': [['SUBJECT_PROCESS', 'EVENT_MMAP', 'IPC_OBJECT_PIPE_UNNAMED']],
            's_open_f': [['SUBJECT_PROCESS', 'EVENT_OPEN', 'FILE_OBJECT_CHAR'],
                         ['SUBJECT_PROCESS', 'EVENT_OPEN', 'FILE_OBJECT_FILE'],
                         ['SUBJECT_PROCESS', 'EVENT_OPEN', 'FILE_OBJECT_DIR']],
            's_close_f': [['SUBJECT_PROCESS', 'EVENT_CLOSE', 'FILE_OBJECT_CHAR'],
                          ['SUBJECT_PROCESS', 'EVENT_CLOSE', 'FILE_OBJECT_FILE'],
                          ['SUBJECT_PROCESS', 'EVENT_CLOSE', 'FILE_OBJECT_DIR']],
            's_read_f': [['SUBJECT_PROCESS', 'EVENT_CLOSE', 'FILE_OBJECT_FILE'],
                         ['SUBJECT_PROCESS', 'EVENT_READ', 'FILE_OBJECT_DIR'],
                         ['SUBJECT_PROCESS', 'EVENT_READ', 'FILE_OBJECT_CHAR']],
            's_write_f': [['SUBJECT_PROCESS', 'EVENT_WRITE', 'FILE_OBJECT_CHAR'],
                          ['SUBJECT_PROCESS', 'EVENT_WRITE', 'FILE_OBJECT_FILE'],
                          ['SUBJECT_PROCESS', 'EVENT_WRITE', 'FILE_OBJECT_DIR']],
            's_create_f': [['SUBJECT_PROCESS', 'EVENT_CREATE_OBJECT', 'FILE_OBJECT_FILE']],
            's_unlink_f': [['SUBJECT_PROCESS', 'EVENT_UNLINK', 'FILE_OBJECT_FILE'],
                           ['SUBJECT_PROCESS', 'EVENT_UNLINK', 'FILE_OBJECT_DIR']],
            's_loadlibrary_f': [['SUBJECT_PROCESS', 'EVENT_LOADLIBRARY', 'FILE_OBJECT_FILE']],
            's_update_f': [['SUBJECT_PROCESS', 'EVENT_UPDATE', 'FILE_OBJECT_FILE']],
            's_modify_f': [['SUBJECT_PROCESS', 'EVENT_MODIFY_FILE_ATTRIBUTES', 'FILE_OBJECT_FILE'],
                           ['SUBJECT_PROCESS', 'EVENT_MODIFY_FILE_ATTRIBUTES', 'FILE_OBJECT_DIR']],
            's_rename_f': [['SUBJECT_PROCESS', 'EVENT_RENAME', 'FILE_OBJECT_FILE']],
            's_mmap_f': [['SUBJECT_PROCESS', 'EVENT_MMAP', 'FILE_OBJECT_FILE']],
            's_truncate_f': [['SUBJECT_PROCESS', 'EVENT_TRUNCATE', 'FILE_OBJECT_FILE']],
            's_mmap_m': [['SUBJECT_PROCESS', 'EVENT_MMAP', 'RECORD_MEMORY_OBJECT']],
            's_mprotect_m': [['SUBJECT_PROCESS', 'EVENT_MPROTECT', 'RECORD_MEMORY_OBJECT']],
            's_connect_n': [['SUBJECT_PROCESS', 'EVENT_CONNECT', 'RECORD_NET_FLOW_OBJECT']],
            's_send_net': [['SUBJECT_PROCESS', 'EVENT_SENDMSG', 'RECORD_NET_FLOW_OBJECT']],
            's_recv_net': [['SUBJECT_PROCESS', 'EVENT_RECVMSG', 'RECORD_NET_FLOW_OBJECT']],
            's_read_net': [['SUBJECT_PROCESS', 'EVENT_READ', 'RECORD_NET_FLOW_OBJECT']],
            's_close_net': [['SUBJECT_PROCESS', 'EVENT_CLOSE', 'RECORD_NET_FLOW_OBJECT']],
            's_accept_net': [['SUBJECT_PROCESS', 'EVENT_ACCEPT', 'RECORD_NET_FLOW_OBJECT']],
            's_write_net': [['SUBJECT_PROCESS', 'EVENT_WRITE', 'RECORD_NET_FLOW_OBJECT']],
            's_accept_sock': [['SUBJECT_PROCESS', 'EVENT_ACCEPT', 'FILE_OBJECT_UNIX_SOCKET']],
            's_write_sock': [['SUBJECT_PROCESS', 'EVENT_WRITE', 'FILE_OBJECT_UNIX_SOCKET']],
            's_read_sock': [['SUBJECT_PROCESS', 'EVENT_READ', 'FILE_OBJECT_UNIX_SOCKET']],
            's_connect_sock': [['SUBJECT_PROCESS', 'EVENT_CONNECT', 'FILE_OBJECT_UNIX_SOCKET']],
            's_recv_sock': [['SUBJECT_PROCESS', 'EVENT_RECVMSG', 'FILE_OBJECT_UNIX_SOCKET']],
            's_send_sock': [['SUBJECT_PROCESS', 'EVENT_SENDMSG', 'FILE_OBJECT_UNIX_SOCKET']]
            }
    path  = 'data/graph_v3/label/meta.json'
    with open(path,'w') as f:
        json.dump(meta,f)
    f.close()
def save_path(list,name):
    path = 'data/graph_v5/label/%s.pickle' % name
    f = open(path, 'wb')
    pickle.dump(list, f)
    f.close()
def load_pickle(path):
    f = open(path,'rb')
    li = pickle.load(f)
    return li
def get_path():
    s_fork_s = load_pickle('data/graph_v5/label/s_fork_s.pickle')
    s_changeprincipal_s = load_pickle('data/graph_v5/label/s_changeprincipal_s.pickle')
    s_execute_s = load_pickle('data/graph_v5/label/s_execute_s.pickle')
    s_exit_s = load_pickle('data/graph_v5/label/s_exit_s.pickle')
    s_clone_s = load_pickle('data/graph_v5/label/s_clone_s.pickle')
    s_write_IPC = load_pickle('data/graph_v5/label/s_write_IPC.pickle')
    s_close_IPC = load_pickle('data/graph_v5/label/s_close_IPC.pickle')
    s_read_IPC = load_pickle('data/graph_v5/label/s_read_IPC.pickle')
    s_mmp_IPC = load_pickle('data/graph_v5/label/s_mmp_IPC.pickle')
    s_open_f = load_pickle('data/graph_v5/label/s_open_f.pickle')
    s_close_f = load_pickle('data/graph_v5/label/s_close_f.pickle')
    s_read_f = load_pickle('data/graph_v5/label/s_read_f.pickle')
    s_write_f = load_pickle('data/graph_v5/label/s_write_f.pickle')
    s_create_f = load_pickle('data/graph_v5/label/s_create_f.pickle')
    s_unlink_f = load_pickle('data/graph_v5/label/s_unlink_f.pickle')
    s_loadlibrary_f = load_pickle('data/graph_v5/label/s_loadlibrary_f.pickle')
    s_update_f = load_pickle('data/graph_v5/label/s_update_f.pickle')
    s_modify_f = load_pickle('data/graph_v5/label/s_modify_f.pickle')
    s_rename_f = load_pickle('data/graph_v5/label/s_rename_f.pickle')
    s_mmap_f = load_pickle('data/graph_v5/label/s_mmap_f.pickle')
    s_truncate_f = load_pickle('data/graph_v5/label/s_truncate_f.pickle')
    s_mmap_m = load_pickle('data/graph_v5/label/s_mmap_m.pickle')
    s_mprotect_m = load_pickle('data/graph_v5/label/s_mprotect_m.pickle')
    s_connect_n = load_pickle('data/graph_v5/label/s_connect_n.pickle')
    s_send_net = load_pickle('data/graph_v5/label/s_send_net.pickle')
    s_recv_net = load_pickle('data/graph_v5/label/s_recv_net.pickle')
    s_read_net = load_pickle('data/graph_v5/label/s_read_net.pickle')
    s_close_net = load_pickle('data/graph_v5/label/s_close_net.pickle')
    s_accept_net = load_pickle('data/graph_v5/label/s_accept_net.pickle')
    s_write_net = load_pickle('data/graph_v5/label/s_write_net.pickle')
    s_accept_sock = load_pickle('data/graph_v5/label/s_accept_sock.pickle')
    s_write_sock = load_pickle('data/graph_v5/label/s_write_sock.pickle')
    s_read_sock = load_pickle('data/graph_v5/label/s_read_sock.pickle')
    s_connect_sock = load_pickle('data/graph_v5/label/s_connect_sock.pickle')
    s_recv_sock = load_pickle('data/graph_v5/label/s_recv_sock.pickle')
    s_send_sock = load_pickle('data/graph_v5/label/s_send_sock.pickle')
    return s_fork_s, s_changeprincipal_s, s_execute_s, s_exit_s, s_clone_s, s_write_IPC, s_close_IPC, s_read_IPC, s_mmp_IPC, \
           s_open_f, s_close_f, s_read_f, s_write_f, s_create_f, s_unlink_f, s_loadlibrary_f, s_update_f, s_modify_f \
        , s_rename_f, s_mmap_f, s_truncate_f, s_mmap_m, s_mprotect_m, s_connect_n, s_send_net, s_recv_net, s_read_net, s_close_net \
        , s_accept_net, s_write_net, s_accept_sock, s_write_sock, s_read_sock, s_connect_sock, s_recv_sock, s_send_sock
def set_path(data,metapath):
    s_fork_s=[]
    s_changeprincipal_s=[]
    s_execute_s=[]
    s_exit_s=[]
    s_clone_s=[]
    s_write_IPC=[]
    s_close_IPC=[]
    s_read_IPC=[]
    s_mmp_IPC=[]
    s_open_f=[]
    s_close_f = []
    s_read_f=[]
    s_write_f=[]
    s_create_f=[]
    s_unlink_f=[]
    s_loadlibrary_f=[]
    s_update_f=[]
    s_modify_f=[]
    s_rename_f=[]
    s_mmap_f=[]
    s_truncate_f=[]
    s_mmap_m = []
    s_mprotect_m=[]
    s_connect_n=[]
    s_send_net=[]
    s_recv_net=[]
    s_read_net=[]
    s_close_net=[]
    s_accept_net=[]
    s_write_net=[]
    s_accept_sock = []
    s_write_sock=[]
    s_read_sock=[]
    s_connect_sock=[]
    s_recv_sock=[]
    s_send_sock = []
    for dic in data:
        for key,value in metapath.items():
            if dic['type'] in value:
                if key == 's_fork_s':
                    s_fork_s.append(dic)
                if key == 's_changeprincipal_s':
                    s_changeprincipal_s.append(dic)
                if key == 's_execute_s':
                    s_execute_s.append(dic)
                if key == 's_exit_s':
                    s_exit_s.append(dic)
                if key == 's_clone_s':
                    s_clone_s.append(dic)
                if key == 's_write_IPC':
                    s_write_IPC.append(dic)
                if key == 's_close_IPC':
                    s_close_IPC.append(dic)
                if key == 's_read_IPC':
                    s_read_IPC.append(dic)
                if key == 's_mmp_IPC':
                    s_mmp_IPC.append(dic)
                if key == 's_open_f':
                    s_open_f.append(dic)
                if key == 's_close_f':
                    s_close_f.append(dic)
                if key == 's_read_f':
                    s_read_f.append(dic)
                if key == 's_write_f':
                    s_write_f.append(dic)
                if key == 's_create_f':
                    s_create_f.append(dic)
                if key == 's_unlink_f':
                    s_unlink_f.append(dic)
                if key == 's_loadlibrary_f':
                    s_loadlibrary_f.append(dic)
                if key == 's_update_f':
                    s_update_f.append(dic)
                if key == 's_modify_f':
                    s_modify_f.append(dic)
                if key == 's_rename_f':
                    s_rename_f.append(dic)
                if key == 's_mmap_f':
                    s_mmap_f.append(dic)
                if key == 's_truncate_f':
                    s_truncate_f.append(dic)
                if key == 's_mmap_m':
                    s_mmap_m.append(dic)
                if key == 's_mprotect_m':
                    s_mprotect_m.append(dic)
                if key == 's_connect_n':
                    s_connect_n.append(dic)
                if key == 's_send_net':
                    s_send_net.append(dic)
                if key == 's_recv_net':
                    s_recv_net.append(dic)
                if key == 's_read_net':
                    s_read_net.append(dic)
                if key == 's_close_net':
                    s_close_net.append(dic)
                if key == 's_accept_net':
                    s_accept_net.append(dic)
                if key == 's_write_net':
                    s_write_net.append(dic)
                if key == 's_accept_sock':
                    s_accept_sock.append(dic)
                if key == 's_write_sock':
                    s_write_sock.append(dic)
                if key == 's_read_sock':
                    s_read_sock.append(dic)
                if key == 's_connect_sock':
                    s_connect_sock.append(dic)
                if key == 's_recv_sock':
                    s_recv_sock.append(dic)
                if key == 's_send_sock':
                    s_send_sock.append(dic)
    save_path(s_fork_s,'s_fork_s')
    save_path(s_changeprincipal_s, 's_changeprincipal_s')
    save_path(s_execute_s, 's_execute_s')
    save_path(s_exit_s, 's_exit_s')
    save_path(s_clone_s, 's_clone_s')
    save_path(s_write_IPC, 's_write_IPC')
    save_path(s_close_IPC, 's_close_IPC')
    save_path(s_read_IPC, 's_read_IPC')
    save_path(s_mmp_IPC, 's_mmp_IPC')
    save_path(s_open_f, 's_open_f')
    save_path(s_close_f, 's_close_f')
    save_path(s_read_f, 's_read_f')
    save_path(s_write_f, 's_write_f')
    save_path(s_create_f, 's_create_f')
    save_path(s_unlink_f, 's_unlink_f')
    save_path(s_loadlibrary_f, 's_loadlibrary_f')
    save_path(s_update_f, 's_update_f')
    save_path(s_modify_f, 's_modify_f')
    save_path(s_rename_f, 's_rename_f')
    save_path(s_mmap_f, 's_mmap_f')
    save_path(s_truncate_f, 's_truncate_f')
    save_path(s_mmap_m, 's_mmap_m')
    save_path(s_mprotect_m, 's_mprotect_m')
    save_path(s_connect_n, 's_connect_n')
    save_path(s_send_net, 's_send_net')
    save_path(s_recv_net, 's_recv_net')
    save_path(s_read_net, 's_read_net')
    save_path(s_close_net, 's_close_net')
    save_path(s_accept_net, 's_accept_net')
    save_path(s_write_net, 's_write_net')
    save_path(s_accept_sock, 's_accept_sock')
    save_path(s_write_sock, 's_write_sock')
    save_path(s_read_sock, 's_read_sock')
    save_path(s_connect_sock, 's_connect_sock')
    save_path(s_recv_sock, 's_recv_sock')
    save_path(s_send_sock, 's_send_sock')
    return s_fork_s,s_changeprincipal_s,s_execute_s,s_exit_s,s_clone_s,s_write_IPC,s_close_IPC,s_read_IPC,s_mmp_IPC,\
           s_open_f,s_close_f,s_read_f,s_write_f,s_create_f,s_unlink_f,s_loadlibrary_f,s_update_f,s_modify_f\
        ,s_rename_f,s_mmap_f,s_truncate_f,s_mmap_m,s_mprotect_m,s_connect_n,s_send_net,s_recv_net,s_read_net,s_close_net\
        ,s_accept_net,s_write_net,s_accept_sock,s_write_sock,s_read_sock,s_connect_sock,s_recv_sock,s_send_sock
def get_graph(meta,node,edge,num):
    srcnode = []
    dstnode = []
    for dic in meta:
        srcnode.append(dic['link'][0])
        dstnode.append(dic['link'][1])
    return dgl.graph((srcnode,dstnode),node,edge,num)
def get_node_link(meta):
    srcnode = []
    dstnode = []
    for dic in meta:
        srcnode.append(dic['link'][0])
        dstnode.append(dic['link'][1])
    return (srcnode,dstnode)
def get_h_graph(meta,srctype,edge,dsttype,srcnum,dstnum):
    srcnode = []
    dstnode = []
    for dic in meta:
        srcnode.append(dic['link'][0])
        dstnode.append(dic['link'][1])
    src_tu = (srcnode,dstnode)
    dst_tu = (dstnode,srcnode)
    return dgl.bipartite(src_tu,srctype,edge,dsttype,(srcnum,dstnum)),dgl.bipartite(dst_tu,dsttype,edge,srctype,(dstnum,srcnum))
def get_h_graph_v2(meta,srctype,edge,dsttype,srcnum,dstnum):
    srcnode = []
    dstnode = []
    for dic in meta:
        srcnode.append(dic['link'][0])
        dstnode.append(dic['link'][1])
    src_tu = (dstnode,srcnode)
    return dgl.bipartite(src_tu,dsttype,edge,srctype,(dstnum,srcnum))
def get_tuple_link(meta):
    srcnode = []
    dstnode = []
    for dic in meta:
        srcnode.append(dic['link'][0])
        dstnode.append(dic['link'][1])
    src_tu = (srcnode, dstnode)
    # dst_tu = (dstnode, srcnode)
    return src_tu
def get_tuple_link_transpose(meta):
    srcnode = []
    dstnode = []
    for dic in meta:
        srcnode.append(dic['link'][0])
        dstnode.append(dic['link'][1])
    dst_tu = (dstnode, srcnode)
    return dst_tu
def get_link_transpose_v2(data):
    data_trans = []
    for dic in data:
        data_trans.append([dic[1],dic[0]])
    return data_trans
def get_link_v2(idx,link,li):
    return link[li[idx]]
def set_meta_paths():
    meta_paths_process = [['s_fork_s'], ['s_changeprincipal_s'], ['s_execute_s'], ['s_exit_s'], ['s_clone_s'],
                          ['s_fork_s', 's_write_IPC', 'IPC_write_s'], ['s_close_IPC', 'IPC_close_s'],
                          ['s_read_IPC', 'IPC_read_s'], ['s_mmp_IPC', 'IPC_mmp_s'], ['s_open_f', 'f_close_s'],
                          ['s_read_f', 'f_contain_attribute', 'attribute_contain_f', 'f_read_s'],
                          ['s_write_f', 'f_contain_attribute', 'attribute_contain_f', 'f_write_s'],
                          ['s_create_f', 'f_contain_attribute', 'attribute_contain_f', 'f_create_s'],
                          ['s_unlink_f', 'f_contain_attribute', 'attribute_contain_f', 'f_unlink_s'],
                          ['s_loadlibrary_f', 'f_contain_attribute', 'attribute_contain_f', 'f_loadlibrary_s'],
                          ['s_update_f', 'f_contain_attribute', 'attribute_contain_f', 'f_update_s'],
                          ['s_modify_f', 'f_contain_attribute', 'attribute_contain_f', 'f_modify_s'],
                          ['s_rename_f', 'f_contain_attribute', 'attribute_contain_f', 'f_rename_s'],
                          ['s_mmp_f', 'f_contain_attribute', 'attribute_contain_f', 'f_mmp_s'],
                          ['s_truncate_f', 'f_contain_attribute', 'attribute_contain_f', 'f_truncate_s'],
                          ['s_mmp_m', 'm_mmp_s'], ['s_mprotect_m', 'm_mprotect_s'],
                          ['s_connect_n', 'n_connect_s'], ['s_send_n', 'n_send_s'], ['s_recv_n', 'n_recv_s'],
                          ['s_read_n', 'n_read_s'], ['s_close_n', 'n_close_s'], ['s_accept_n', 'n_accept_s'],
                          ['s_write_n', 'n_write_s'], ['s_accept_sock', 'sock_accept_s'],
                          ['s_write_sock', 'sock_write_s'],
                          ['s_read_sock', 'sock_read_s'], ['s_connect_sock', 'sock_connect_s'],
                          ['s_recv_sock', 'sock_recv_s'],
                          ['s_send_sock', 'sock_send_s']]
    meta_paths_IPC = [['IPC_write_s','s_write_IPC'],['IPC_close_s','s_close_IPC'],['IPC_read_s','s_read_IPC'],['IPC_mmp_s','s_mmp_IPC']]
    meta_paths_net = [['n_connect_s','s_connect_n'],['n_send_s','s_send_n'],['n_recv_s','s_recv_n'],['n_read_s','s_read_n'],['n_close_s','s_close_n'],['n_accept_s','s_accept_n']]
    meta_paths_mem = [['m_mmp_s','s_mmp_m'],['m_mprotect_s','s_mprotect_m']]
    meta_paths_sock = [['sock_accept_s','s_accept_sock'],['sock_write_s','s_write_sock'],['sock_read_s','s_read_sock'],['sock_connect_s','s_connect_sock'],['sock_recv_s','s_recv_sock'],['sock_send_s','s_send_sock']]
    meta_paths_file = [['f_contain_attribute', 'attribute_contain_f'],['f_open_s','s_open_f'],['f_read_s','s_read_f', 'f_contain_attribute', 'attribute_contain_f'],
                       ['f_write_s', 's_write_f', 'f_contain_attribute', 'attribute_contain_f'],
                       ['f_create_s','s_create_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_unlink_s','s_unlink_f','f_contain_attribute', 'attribute_contain_f'],
                       [ 'f_loadlibrary_s', 's_loadlibrary_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_update_s','s_update_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_modify_s', 's_modify_f', 'f_contain_attribute', 'attribute_contain_f'],
                       ['f_modify_s','s_modify_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_rename_s','s_rename_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_mmp_s', 's_mmp_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_truncate_s', 's_truncate_f','f_contain_attribute', 'attribute_contain_f']]
    meta_paths_attribute= [['attribute_contain_f','f_contain_attribute'],['attribute_contain_s','s_contain_attribute']]
    meta_paths_process = [['s_fork_s'], ['s_changeprincipal_s'], ['s_execute_s'], ['s_exit_s'], ['s_clone_s'],
                          ['s_open_f', 'f_close_s'],
                          ['s_read_f', 'f_contain_attribute', 'attribute_contain_f', 'f_read_s'],
                          ['s_write_f', 'f_contain_attribute', 'attribute_contain_f', 'f_write_s'],
                          ['s_create_f', 'f_contain_attribute', 'attribute_contain_f', 'f_create_s'],
                          ['s_unlink_f', 'f_contain_attribute', 'attribute_contain_f', 'f_unlink_s'],
                          ['s_loadlibrary_f', 'f_contain_attribute', 'attribute_contain_f', 'f_loadlibrary_s'],
                          ['s_update_f', 'f_contain_attribute', 'attribute_contain_f', 'f_update_s'],
                          ['s_modify_f', 'f_contain_attribute', 'attribute_contain_f', 'f_modify_s'],
                          ['s_rename_f', 'f_contain_attribute', 'attribute_contain_f', 'f_rename_s'],
                          ['s_mmp_f', 'f_contain_attribute', 'attribute_contain_f', 'f_mmp_s'],
                          ['s_truncate_f', 'f_contain_attribute', 'attribute_contain_f', 'f_truncate_s'],
                          ]
    meta_paths_IPC = [['IPC_write_s', 's_write_IPC'], ['IPC_close_s', 's_close_IPC'], ['IPC_read_s', 's_read_IPC'],
                      ['IPC_mmp_s', 's_mmp_IPC']]
    meta_paths_net = [['n_connect_s', 's_connect_n'], ['n_send_s', 's_send_n'], ['n_recv_s', 's_recv_n'],
                      ['n_read_s', 's_read_n'], ['n_close_s', 's_close_n'], ['n_accept_s', 's_accept_n']]
    meta_paths_mem = [['m_mmp_s', 's_mmp_m'], ['m_mprotect_s', 's_mprotect_m']]
    meta_paths_sock = [['sock_accept_s', 's_accept_sock'], ['sock_write_s', 's_write_sock'],
                       ['sock_read_s', 's_read_sock'], ['sock_connect_s', 's_connect_sock'],
                       ['sock_recv_s', 's_recv_sock'], ['sock_send_s', 's_send_sock']]
    meta_paths_file = [['f_contain_attribute', 'attribute_contain_f'],['f_open_s','s_open_f'],['f_read_s','s_read_f', 'f_contain_attribute', 'attribute_contain_f'],
                       ['f_write_s', 's_write_f', 'f_contain_attribute', 'attribute_contain_f'],
                       ['f_create_s','s_create_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_unlink_s','s_unlink_f','f_contain_attribute', 'attribute_contain_f'],
                       [ 'f_loadlibrary_s', 's_loadlibrary_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_update_s','s_update_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_modify_s', 's_modify_f', 'f_contain_attribute', 'attribute_contain_f'],
                       ['f_modify_s','s_modify_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_rename_s','s_rename_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_mmp_s', 's_mmp_f','f_contain_attribute', 'attribute_contain_f'],
                       ['f_truncate_s', 's_truncate_f','f_contain_attribute', 'attribute_contain_f']]
    meta_paths_attribute = [['attribute_contain_f', 'f_contain_attribute'],
                            ['attribute_contain_s', 's_contain_attribute']]
    return meta_paths_process,meta_paths_file,meta_paths_attribute,meta_paths_IPC,meta_paths_net,meta_paths_mem,meta_paths_sock
def set_meta_paths_v2():

    meta_paths_process = [['s_fork_s'],
                          ['s_read_f', 'f_contain_attribute', 'attribute_contain_f', 'f_read_s'],
                          ['s_write_f', 'f_contain_attribute', 'attribute_contain_f', 'f_write_s'],
                          ['s_create_f', 'f_contain_attribute', 'attribute_contain_f', 'f_create_s'],
                          ]
    meta_paths_IPC = [['IPC_write_s', 's_write_IPC'], ['IPC_read_s', 's_read_IPC'],
                      ]
    meta_paths_net = [
                      ['n_read_s', 's_read_n']]
    meta_paths_mem = []
    meta_paths_sock = []
    meta_paths_file = [['f_contain_attribute', 'attribute_contain_f'],['f_read_s','s_read_f', 'f_contain_attribute', 'attribute_contain_f'],
                       ['f_write_s', 's_write_f', 'f_contain_attribute', 'attribute_contain_f'],
                       ['f_create_s','s_create_f','f_contain_attribute', 'attribute_contain_f'],
                       ]
    meta_paths_attribute = [['attribute_contain_f', 'f_contain_attribute'],
                            ['attribute_contain_s', 's_contain_attribute']]
    return meta_paths_process,meta_paths_file,meta_paths_attribute,meta_paths_IPC,meta_paths_net,meta_paths_mem,meta_paths_sock
def get_full_homogeneous_link():
    link = load_one_jsondata('data/darpa/label/full_link.json')
    process_keys = ['s_fork_s', 's_changeprincipal_s', 's_execute_s', 's_exit_s', 's_clone_s']
    file_keys = ['s_open_f', 's_close_f', 's_read_f', 's_write_f', 's_create_f', 's_unlink_f', 's_loadlibrary_f',
                 's_update_f', 's_modify_f', 's_rename_f', 's_mmap_f', 's_truncate_f']
    IPC_keys = ['s_write_IPC', 's_close_IPC', 's_read_IPC', 's_mmp_IPC']
    mem_keys = ['s_mmap_m', 's_mprotect_m']
    net_keys = ['s_connect_net', 's_send_net', 's_recv_net', 's_read_net', 's_close_net', 's_accept_net', 's_write_net']
    soc_keys = ['s_accept_sock', 's_write_sock', 's_read_sock', 's_connect_sock', 's_recv_sock', 's_send_sock']
    s_attr_keys = ['s_contain_attribute']
    f_attr_keys = ['f_contain_attribute']
    new_file_id_start = 84864
    new_project_id_start = 100297
    new_attribute_id_start = 100301
    new_link = []
    for key, value in link.items():
        for i in value:
            if key in file_keys:
                i = [i[0], i[1] + new_file_id_start]
            if key in IPC_keys:
                i = [i[0], i[1] + new_project_id_start]
            if key in mem_keys:
                i = [i[0], i[1] + new_project_id_start + 1]
            if key in net_keys:
                i = [i[0], i[1] + new_project_id_start + 2]
            if key in soc_keys:
                i = [i[0], i[1] + new_project_id_start + 3]
            if key in s_attr_keys:
                i = [i[0], i[1] + new_attribute_id_start]
            if key in f_attr_keys:
                i = [i[0] + new_file_id_start, i[1] + new_attribute_id_start]
            new_link.append(i)
    data = {'link':new_link}
    with open('data/darpa/label/homogeneous_link.json', 'w') as f:
        json.dump(data,f)
    f.close()
def get_full_homogeneous_graph():
    link = load_one_jsondata('data/darpa/label/homogeneous_link.json')
    process_features = torch.from_numpy(np.loadtxt('data/darpa/label/process_matrix.txt'))
    file_features = torch.from_numpy(np.loadtxt('data/darpa/label/file_matrix.txt'))
    ipc_features = torch.from_numpy(np.loadtxt('data/darpa/label/ipc_matrix.txt')).view(-1, 10)
    net_features = torch.from_numpy(np.loadtxt('data/darpa/label/net_matrix.txt')).view(-1, 10)
    mem_features = torch.from_numpy(np.loadtxt('data/darpa/label/mem_matrix.txt')).view(-1, 10)
    soc_features = torch.from_numpy(np.loadtxt('data/darpa/label/soc_matrix.txt')).view(-1, 10)
    attr_features = torch.from_numpy(np.loadtxt('data/darpa/label/attr_matrix.txt'))
    g = dgl.graph(link['link'])
    features = torch.cat((process_features,file_features,ipc_features,net_features,mem_features,soc_features,attr_features),dim=0)
    g.ndata['h']=features
    return g

    return trans_path
def get_full_h_graph_v2():
    link = load_one_jsondata('data/darpa/label/full_link.json')
    process_features = torch.from_numpy(np.loadtxt('data/darpa/label/process_matrix.txt'))
    file_features = torch.from_numpy(np.loadtxt('data/darpa/label/file_matrix.txt'))
    ipc_features = torch.from_numpy(np.loadtxt('data/darpa/label/ipc_matrix.txt')).view(-1,10)
    net_features = torch.from_numpy(np.loadtxt('data/darpa/label/net_matrix.txt')).view(-1,10)
    mem_features = torch.from_numpy(np.loadtxt('data/darpa/label/mem_matrix.txt')).view(-1,10)
    soc_features = torch.from_numpy(np.loadtxt('data/darpa/label/soc_matrix.txt')).view(-1,10)
    attr_features = torch.from_numpy(np.loadtxt('data/darpa/label/attr_matrix.txt'))
    li = ['s_fork_s', 's_changeprincipal_s', 's_execute_s', 's_exit_s', 's_clone_s', 's_write_IPC', 's_close_IPC',
          's_read_IPC', 's_mmp_IPC', 's_open_f', 's_close_f', 's_read_f', 's_write_f', 's_create_f', 's_unlink_f',
          's_loadlibrary_f', 's_update_f', 's_modify_f', 's_rename_f', 's_mmap_f', 's_truncate_f', 's_mmap_m',
          's_mprotect_m', 's_connect_net', 's_send_net', 's_recv_net', 's_read_net', 's_close_net', 's_accept_net',
          's_write_net', 's_accept_sock', 's_write_sock', 's_read_sock', 's_connect_sock', 's_recv_sock', 's_send_sock',
          's_contain_attribute', 'f_contain_attribute']

    graph_dict = {('process', 's_fork_s', 'process'):link[li[0]],
          ('process', 's_changeprincipal_s', 'process'):link[li[1]],
          ('process', 's_execute_s', 'process'):link[li[2]],
          ('process', 's_exit_s', 'process'):link[li[3]],
          ('process', 's_clone_s', 'process'):link[li[4]],
          ('process', 's_write_IPC', 'IPC'):link[li[5]],
          ('IPC', 'IPC_write_s', 'process'):get_link_transpose_v2(link[li[5]]),
          ('process', 's_close_IPC', 'IPC'):link[li[6]],
          ('IPC', 'IPC_close_s', 'process'):get_link_transpose_v2(link[li[6]]),
          ('process', 's_read_IPC', 'IPC'):link[li[7]],
          ('IPC', 'IPC_read_s', 'process'):get_link_transpose_v2(link[li[7]]),
          ('process', 's_mmp_IPC', 'IPC'):link[li[8]],
          ('IPC', 'IPC_mmp_s', 'process'):get_link_transpose_v2(link[li[8]]),
          ('process', 's_open_f', 'file'):link[li[9]],
          ('file', 'f_open_s', 'process'):get_link_transpose_v2(link[li[9]]),
          ('process', 's_close_f', 'file'):link[li[10]],
          ('file', 'f_close_s', 'process'):get_link_transpose_v2(link[li[10]]),
          ('process', 's_read_f', 'file'):link[li[11]],
          ('file', 'f_read_s', 'process'):get_link_transpose_v2(link[li[11]]),
          ('process', 's_write_f', 'file'):link[li[12]],
          ('file', 'f_write_s', 'process'):get_link_transpose_v2(link[li[12]]),
          ('process', 's_create_f', 'file'):link[li[13]],
          ('file', 'f_create_s', 'process'):get_link_transpose_v2(link[li[13]]),
          ('process', 's_unlink_f', 'file'):link[li[14]],
          ('file', 'f_unlink_s', 'process'):get_link_transpose_v2(link[li[14]]),
          ('process', 's_loadlibrary_f', 'file'):link[li[15]],
          ('file', 'f_loadlibrary_s', 'process'):get_link_transpose_v2(link[li[15]]),
          ('process', 's_update_f', 'file'):link[li[16]],
          ('file', 'f_update_s', 'process'):get_link_transpose_v2(link[li[16]]),
          ('process', 's_modify_f', 'file'):link[li[17]],
          ('file', 'f_modify_s', 'process'):get_link_transpose_v2(link[li[17]]),
          ('process', 's_rename_f', 'file'):link[li[18]],
          ('file', 'f_rename_s', 'process'):get_link_transpose_v2(link[li[18]]),
          ('process', 's_mmp_f', 'file'):link[li[19]],
          ('file', 'f_mmp_s', 'process'):get_link_transpose_v2(link[li[19]]),
          ('process', 's_truncate_f', 'file'):link[li[20]],
          ('file', 'f_truncate_s', 'process'):get_link_transpose_v2(link[li[20]]),
          ('process', 's_mmp_m', 'memory'):link[li[21]],
          ('memory', 'm_mmp_s', 'process'):get_link_transpose_v2(link[li[21]]),
          ('process', 's_mprotect_m', 'memory'):link[li[22]],
          ('memory', 'm_mprotect_s', 'process'):get_link_transpose_v2(link[li[22]]),
          ('process', 's_connect_n', 'net'):link[li[23]],
          ('net', 'n_connect_s', 'process'):get_link_transpose_v2(link[li[23]]),
          ('process', 's_send_n', 'net'):link[li[24]],
          ('net', 'n_send_s', 'process'):get_link_transpose_v2(link[li[24]]),
          ('process', 's_recv_n', 'net'):link[li[25]],
          ('net', 'n_recv_s', 'process'):get_link_transpose_v2(link[li[25]]),
          ('process', 's_read_n', 'net'):link[li[26]],
          ('net', 'n_read_s', 'process'):get_link_transpose_v2(link[li[26]]),
          ('process', 's_close_n', 'net'):link[li[27]],
          ('net', 'n_close_s', 'process'):get_link_transpose_v2(link[li[27]]),
          ('process', 's_accept_n', 'net'):link[li[28]],
          ('net', 'n_accept_s', 'process'):get_link_transpose_v2(link[li[28]]),
          ('process', 's_write_n', 'net'):link[li[29]],
          ('net', 'n_write_s', 'process'):get_link_transpose_v2(link[li[29]]),
          ('process', 's_accept_sock', 'socket'):link[li[30]],
          ('socket', 'sock_accept_s', 'process'):get_link_transpose_v2(link[li[30]]),
          ('process', 's_write_sock', 'socket'):link[li[31]],
          ('socket', 'sock_write_s', 'process'):get_link_transpose_v2(link[li[31]]),
          ('process', 's_read_sock', 'socket'):link[li[32]],
          ('socket', 'sock_read_s', 'process'):get_link_transpose_v2(link[li[32]]),
          ('process', 's_connect_sock', 'socket'):link[li[33]],
          ('socket', 'sock_connect_s', 'process'):get_link_transpose_v2(link[li[33]]),
          ('process', 's_recv_sock', 'socket'):link[li[34]],
          ('socket', 'sock_recv_s', 'process'):get_link_transpose_v2(link[li[34]]),
          ('process', 's_send_sock', 'socket'):link[li[35]],
          ('socket', 'sock_send_s', 'process'):get_link_transpose_v2(link[li[35]]),
          ('process','s_contain_attribute','attribute'):link[li[36]],
          ('attribute', 'attribute_contain_s', 'process'):get_link_transpose_v2(link[li[36]]),
          ('file', 'f_contain_attribute', 'attribute'):link[li[37]],
          ('attribute', 'attribute_contain_f', 'file'):get_link_transpose_v2(link[li[37]]),
          }
    hg = dgl.heterograph(graph_dict)

    hg.ndata['h'] = {'process':process_features,'file':file_features,'IPC':ipc_features,'memory':mem_features,
                     'attribute':attr_features,'net':net_features,
                     'socket':soc_features}

    # hg.ndata['h'] = {'process': process_features,'file':file_features,'IPC':ipc_features}
    return hg
if __name__  ==  '__main__':
    # get_full_homogeneous_link()
    print(1)