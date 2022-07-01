import torch
from sklearn.metrics import f1_score
import dgl
from utils import EarlyStopping,get_best_result,collate
from data_prepare_v3 import load_all_data_v2
from torch.utils.data import DataLoader
import numpy as np
import dgl.nn as dglnn
from graph_sampling import set_sample_graph
from meta_v2 import get_metapath
from tqdm import tqdm
from torch.nn import Linear
from random import randint
import torch.nn.functional as F
dgl.batch()
def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model,linear,all_id_matrix,all_typ_matrix,all_logits,
             all_sub_graph,
             labels,idx, mask,h_dict, new_meta_paths,walk_length,
             loss_func,node):

    model.eval()
    total_loss = 0
    # empty_logits = torch.tensor([0, 0]).cuda()
    with torch.no_grad():
        for id in tqdm(idx):
            graph=all_sub_graph[id][0]
            new_h_dict = all_sub_graph[id][1]
            new_label = all_sub_graph[id][2]
        # for id in tqdm(idx):
        #     graph, new_h_dict = set_sample_graph(id, all_id_matrix, all_typ_matrix,
        #                                          h_dict,new_meta_paths,walk_length)
            if graph:
                logits = model(graph, new_h_dict,node)
                all_logits[id] = logits[0]

            else:
                # logits = empty_logits
                # logits = linear(h_dict[node][id].cuda())
                #
                # all_logits[id] = logits
                continue
            total_loss += loss_func(logits, new_label).item()
        # for i,data in enumerate(tqdm(all_sub_graph)):
        #     if data[0]:
        #         logits = model(data[0], data[1],node)
        #         all_logits[i] = logits[0]
        #     else:
        #         all_logits[i]=empty_logits
    # loss = loss_fcn(all_logits[train_mask], labels[train_mask])
    # loss = loss_func(all_logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(all_logits[mask], labels[mask])

    return total_loss, accuracy, micro_f1, macro_f1

def main(args):
    meta_paths = get_metapath()
    g, labels, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, process_based_g, \
    file_based_g, attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g = load_all_data_v2()
    all_id_matrix = []
    all_typ_matrix = []
    idx = np.arange(len(labels))
    # idx =[13306, 1832, 1833, 1831, 13307, 13337, 1854, 1855, 13340, 1856, 13486, 20825, 20830, 14679, 14682, 14684, 14715, 25583, 27097, 15380, 15395, 15397, 15458, 15659, 13482, 13483, 18224, 18353, 18453, 18454, 18458, 18510, 14163, 20489]
    new_meta_paths = []
    for meta_path in meta_paths:
        # walk_times = randint(1, 5)
        # for i in range(walk_times):
            new_g = dgl.sampling.random_walk(g, idx, metapath=meta_path)
            node_id = new_g[0]
            node_typ = new_g[1]
            all_id_matrix.append(node_id)
            all_typ_matrix.append(node_typ)
            new_meta_paths.append(meta_path)
    h_dict = {'process': torch.load('data/embedding/process_embedding_v1.pkl').to('cpu'),
              'file': torch.load('data/embedding/file_embedding_v1.pkl').to('cpu'),
              'IPC': torch.load('data/embedding/IPC_embedding_v1.pkl').to('cpu'),
              'memory': torch.load('data/embedding/mem_embedding_v1.pkl').to('cpu'),
              'net': torch.load('data/embedding/net_embedding_v1.pkl').to('cpu'),
              'attribute': torch.load('data/embedding/attribute_embedding_v1.pkl').to('cpu'),
              'socket': torch.load('data/embedding/sock_embedding_v1.pkl').to('cpu')
              }

    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device']).bool()
    val_mask = val_mask.to(args['device']).bool()
    test_mask = test_mask.to(args['device']).bool()
    from RGCN import RGCN
    model = RGCN(64,128,64,g.etypes).to(args['device'])
    loss_fcn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.001,1.0]).float()).to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    best_train_acc = 0
    best_train_micro_f1 = 0
    best_train_macro_f1 = 0
    best_val_acc = 0
    best_val_micro_f1 = 0
    best_val_macro_f1 = 0
    all_graph = []
    walk_length = 6
    for id in tqdm(idx):
        graph, new_h_dict ,new_label= set_sample_graph(id, all_id_matrix, all_typ_matrix, h_dict,new_meta_paths,labels,walk_length,args['device'])
        all_graph.append((graph,new_h_dict,new_label))
    linear = Linear(64,2).to(args['device'])
    running_loss = 0
    for epoch in range(args['num_epochs']):
        # empty_logits = torch.tensor([0,0]).cuda()
        model.train()
        all_logits = torch.zeros((len(labels),num_classes)).to(args['device'])
        # for id in tqdm(train_idx):
        #     graph, new_h_dict = set_sample_graph(id, all_id_matrix, all_typ_matrix, h_dict,new_meta_paths,walk_length)

        for id in tqdm(train_idx):
            graph=all_graph[id][0]
            new_h_dict = all_graph[id][1]
            new_label = all_graph[id][2]
            if graph:
                logits = model(graph, new_h_dict, 'process')
                # loss = loss_fcn(logits,new_label)
                all_logits[id] = logits[0]
            else:
                # logits = linear(new_h_dict['process'][0])
                # logits =empty_logits
                # logits = linear(h_dict['process'][id].to(args['device']))
                # # loss = loss_fcn(logits,new_label)
                # all_logits[id] = logits
                continue
            # print(id,logits,new_label)
            loss = loss_fcn(logits, new_label)
            # print(id, ':', loss.item())
            running_loss += loss.item()

        # for i,data in enumerate(tqdm(all_graph)) :
        #     if data[0]:
        #         logits = model(data[0],data[1],'process')
        #         all_logits[i] = logits[0]
        #     else:
        #         all_logits[i] =empty_logits
        # all_logits =all_logits.float()
        # loss = loss_fcn(all_logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        train_acc, train_micro_f1, train_macro_f1 = score(all_logits[train_mask], labels[train_mask])
        all_logits_val = torch.zeros((len(labels), num_classes)).to(args['device'])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model,linear,all_id_matrix,all_typ_matrix,
                                                                 all_logits_val,all_graph, labels,val_idx, val_mask,h_dict,
                                                                 new_meta_paths, walk_length,
                                                                 loss_fcn,node='process')
        if train_acc>best_train_acc:
            best_train_acc= train_acc
        if train_micro_f1>best_train_micro_f1:
            best_train_micro_f1 =train_micro_f1
        if train_macro_f1 >best_train_macro_f1:
            best_train_macro_f1 =train_macro_f1
        if val_acc>best_val_acc:
            best_val_acc= val_acc
        if val_micro_f1>best_val_micro_f1:
            best_val_micro_f1 =val_micro_f1
        if val_macro_f1 >best_val_macro_f1:
            best_val_macro_f1 =val_macro_f1
        print('Epoch {:d} | Train Loss {:.4f} |Train acc{:.4f}(best:{:.4f}) |Train Micro f1 {:.4f}(best:{:.4f}) | Train Macro f1 {:.4f}(best:{:.4f}) | '
              'Val Loss {:.4f} |Val acc{:.4f}(best:{:.4f})| Val Micro f1 {:.4f}(best:{:.4f}) | Val Macro f1 {:.4f}(best:{:.4f})'.format(
            epoch + 1,running_loss/len(train_idx),train_acc,best_train_acc, train_micro_f1,best_train_micro_f1, train_macro_f1,best_train_macro_f1, val_loss/len(val_idx),val_acc,best_val_acc, val_micro_f1,best_val_micro_f1 ,val_macro_f1,best_val_macro_f1))
    torch.save(model.state_dict(),'data/embedding/RGCN_sampling.pth')
    all_logits_test = torch.zeros((len(labels), num_classes)).to(args['device'])
    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model,linear,all_id_matrix,all_typ_matrix,
                                                                 all_logits_test,all_graph, labels,test_idx, test_mask,h_dict,
                                                                 new_meta_paths, walk_length,
                                                                 loss_fcn,node='process')
    print('Test loss {:.4f} |Test acc{:.4f}| Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss/len(len(test_idx)), test_acc,test_micro_f1, test_macro_f1))

if __name__ == '__main__':
    import argparse

    from utils_v2 import setup,setup_for_sampling

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
