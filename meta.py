import json
import pickle
import dgl
from dgl.data.utils import save_graphs
import torch
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
            's_connect_net': [['SUBJECT_PROCESS', 'EVENT_CONNECT', 'RECORD_NET_FLOW_OBJECT']],
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
            's_send_sock': [['SUBJECT_PROCESS', 'EVENT_SENDMSG', 'FILE_OBJECT_UNIX_SOCKET']],
            's_contain_attribute':[['SUBJECT_PROCESS', 'CONTAINS', 'ATTRIBUTE']],
            'f_contain_attribute':[['FILE_OBJECT_CHAR','CONTAINS','ATTRIBUTE'],['FILE_OBJECT_DIR','CONTAINS','ATTRIBUTE'],['FILE_OBJECT_FILE','CONTAINS','ATTRIBUTE']]
            }
    path  = 'data/graph_v6/label/meta_v2.json'
    with open(path,'w') as f:
        json.dump(meta,f)
    f.close()
write_meta()
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

def get_full_h_graph():
    nodenum = 33646
    IPCnum = 15629
    filenum = 10074
    memnum = 137400
    netnum = 1030
    socketnum = 7
    s_fork_s, s_changeprincipal_s, s_execute_s, s_exit_s, s_clone_s, s_write_IPC, s_close_IPC, s_read_IPC, s_mmp_IPC, \
    s_open_f, s_close_f, s_read_f, s_write_f, s_create_f, s_unlink_f, s_loadlibrary_f, s_update_f, s_modify_f \
        , s_rename_f, s_mmap_f, s_truncate_f, s_mmap_m, s_mprotect_m, s_connect_n, s_send_net, s_recv_net, s_read_net, s_close_net \
        , s_accept_net, s_write_net, s_accept_sock, s_write_sock, s_read_sock, s_connect_sock, s_recv_sock, s_send_sock \
        = get_path()

    graph_dict = {('process', 's_fork_s', 'process') : get_tuple_link(s_fork_s),
                  ('process', 's_changeprincipal_s', 'process'): get_tuple_link(s_changeprincipal_s),
                  ('process', 's_execute_s', 'process'): get_tuple_link(s_execute_s),
                  ('process', 's_exit_s', 'process'): get_tuple_link(s_exit_s),
                  ('process', 's_clone_s', 'process'): get_tuple_link(s_clone_s),
                  ('process', 's_write_IPC', 'IPC'): get_tuple_link(s_write_IPC),
                  ('IPC', 'IPC_write_s', 'process'): get_tuple_link_transpose(s_write_IPC),
                  ('process', 's_close_IPC', 'IPC'): get_tuple_link(s_close_IPC),
                  ('IPC', 'IPC_close_s', 'process'): get_tuple_link_transpose(s_close_IPC),
                  ('process', 's_read_IPC', 'IPC'): get_tuple_link(s_read_IPC),
                  ('IPC', 'IPC_read_s', 'process'): get_tuple_link_transpose(s_read_IPC),
                  ('process', 's_mmp_IPC', 'IPC'): get_tuple_link(s_mmp_IPC),
                  ('IPC', 'IPC_mmp_s', 'process'): get_tuple_link_transpose(s_mmp_IPC),
                  ('process', 's_open_f', 'file'): get_tuple_link(s_open_f),
                  ('file', 'f_open_s', 'process'): get_tuple_link_transpose(s_open_f),
                  ('process', 's_close_f', 'file'): get_tuple_link(s_close_f),
                  ('file', 'f_close_s', 'process'): get_tuple_link_transpose(s_close_f),
                  ('process', 's_read_f', 'file'): get_tuple_link(s_read_f),
                  ('file', 'f_read_s', 'process'): get_tuple_link_transpose(s_read_f),
                  ('process', 's_write_f', 'file'): get_tuple_link(s_write_f),
                  ('file', 'f_write_s', 'process'): get_tuple_link_transpose(s_write_f),
                  ('process', 's_create_f', 'file'): get_tuple_link(s_create_f),
                  ('file', 'f_create_s', 'process'): get_tuple_link_transpose(s_create_f),
                  ('process', 's_unlink_f', 'file'): get_tuple_link(s_unlink_f),
                  ('file', 'f_unlink_s', 'process'): get_tuple_link_transpose(s_unlink_f),
                  ('process', 's_loadlibrary_f', 'file'): get_tuple_link(s_loadlibrary_f),
                  ('file', 'f_loadlibrary_s', 'process'): get_tuple_link_transpose(s_loadlibrary_f),
                  ('process', 's_update_f', 'file'): get_tuple_link(s_update_f),
                  ('file', 'f_update_s', 'process'): get_tuple_link_transpose(s_update_f),
                  ('process', 's_modify_f', 'file'): get_tuple_link(s_modify_f),
                  ('file', 'f_modify_s', 'process'): get_tuple_link_transpose(s_modify_f),
                  ('process', 's_rename_f', 'file'): get_tuple_link(s_rename_f),
                  ('file', 'f_rename_s', 'process'): get_tuple_link_transpose(s_rename_f),
                  ('process', 's_mmp_f', 'file'): get_tuple_link(s_mmap_f),
                  ('file', 'f_mmp_s', 'process'): get_tuple_link_transpose(s_mmap_f),
                  ('process', 's_truncate_f', 'file'): get_tuple_link(s_truncate_f),
                  ('file', 'f_truncate_s', 'process'): get_tuple_link_transpose(s_truncate_f),
                  ('process', 's_mmp_m', 'memory'): get_tuple_link(s_mmap_m),
                  ('memory', 'm_mmp_s', 'process'): get_tuple_link_transpose(s_mmap_m),
                  ('process', 's_mprotect_m', 'memory'): get_tuple_link(s_mprotect_m),
                  ('memory', 'm_mprotect_s', 'process'): get_tuple_link_transpose(s_mprotect_m),
                  ('process', 's_connect_n', 'net'): get_tuple_link(s_connect_n),
                  ('net', 'n_connect_s', 'process'): get_tuple_link_transpose(s_connect_n),
                  ('process', 's_send_n', 'net'): get_tuple_link(s_send_net),
                  ('net', 'n_send_s', 'process'): get_tuple_link_transpose(s_send_net),
                  ('process', 's_recv_n', 'net'): get_tuple_link(s_recv_net),
                  ('net', 'n_recv_s', 'process'): get_tuple_link_transpose(s_recv_net),
                  ('process', 's_read_n', 'net'): get_tuple_link(s_read_net),
                  ('net', 'n_read_s', 'process'): get_tuple_link_transpose(s_read_net),
                  ('process', 's_close_n', 'net'): get_tuple_link(s_close_net),
                  ('net', 'n_close_s', 'process'): get_tuple_link_transpose(s_close_net),
                  ('process', 's_accept_n', 'net'): get_tuple_link(s_accept_net),
                  ('net', 'n_accept_s', 'process'): get_tuple_link_transpose(s_accept_net),
                  ('process', 's_write_n', 'net'): get_tuple_link(s_write_net),
                  ('net', 'n_write_s', 'process'): get_tuple_link_transpose(s_write_net),
                  ('process', 's_accept_sock', 'socket'): get_tuple_link(s_accept_sock),
                  ('socket', 'sock_accept_s', 'process'): get_tuple_link_transpose(s_accept_sock),
                  ('process', 's_write_sock', 'socket'): get_tuple_link(s_write_sock),
                  ('socket', 'sock_write_s', 'process'): get_tuple_link_transpose(s_write_sock),
                  ('process', 's_read_sock', 'socket'): get_tuple_link(s_read_sock),
                  ('socket', 'sock_read_s', 'process'): get_tuple_link_transpose(s_read_sock),
                  ('process', 's_connect_sock', 'socket'): get_tuple_link(s_connect_sock),
                  ('socket', 'sock_connect_s', 'process'): get_tuple_link_transpose(s_connect_sock),
                  ('process', 's_recv_sock', 'socket'): get_tuple_link(s_recv_sock),
                  ('socket', 'sock_recv_s', 'process'): get_tuple_link_transpose(s_recv_sock),
                  ('process', 's_send_sock', 'socket'): get_tuple_link(s_send_sock),
                  ('socket', 'sock_send_s', 'process'): get_tuple_link_transpose(s_send_sock),
                  }
    hg = dgl.heterograph(graph_dict)
    # hg=dgl.add_self_loop(hg,etype='s_fork_s')
    # hg = dgl.add_self_loop(hg, etype='s_changeprincipal_s')
    # hg = dgl.add_self_loop(hg, etype='s_execute_s')
    # hg = dgl.add_self_loop(hg, etype='s_exit_s')
    # hg = dgl.add_self_loop(hg, etype='s_clone_s')
    return hg
# def save_graph(g):
#     graph_labels = {"glabel": torch.tensor([0, 1])}
#     save_graphs("data/graph_v4/label/graph.bin",g,graph_labels)
# save_graph(get_full_h_graph())
