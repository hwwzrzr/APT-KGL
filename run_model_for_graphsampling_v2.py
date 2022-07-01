import torch
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix
import dgl
from utils import EarlyStopping,get_best_result,collate
from data_prepare_v3 import load_all_data_v3,get_binary_mask,load_all_data_darpa
from torch.utils.data import DataLoader
import numpy as np
import dgl.nn as dglnn
from graph_sampling import set_sample_graph
from meta_v2 import get_metapath
from tqdm import tqdm
from torch.nn import Linear
from random import randint
import time
import torch.nn.functional as F
def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    recall = recall_score(labels,prediction)
    precision = precision_score(labels, prediction)
    tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
    return accuracy, micro_f1, macro_f1,precision,recall,tn, fp, fn, tp

def evaluate(dataset,g,model,linear,all_id_matrix,all_typ_matrix,all_logits,
             all_sub_graph,
             labels,idx, mask,darpa_mask,h_dict, meta_paths,walk_length,
             loss_func,device,node):
    model.eval()
    with torch.no_grad():
        for i,id in enumerate(tqdm(idx)):
            data = dataset[id]
            graph = data[0]
            new_h_dict = data[1]
            new_label = data[2]
            if graph:
                logits = model(graph, new_h_dict,node)
                all_logits[id] = logits[0]
            else:
                continue
    loss = loss_func(all_logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1, precision,recall,tn, fp, fn, tp= score(all_logits[mask], labels[mask])
    darpa_accuracy, darpa_micro_f1, darpa_macro_f1, darpa_precision, darpa_recall, darpa_tn, darpa_fp, darpa_fn, darpa_tp = score(
        all_logits[darpa_mask], labels[darpa_mask])
    return loss, accuracy, micro_f1, macro_f1,precision,recall,tn, fp, fn, tp,darpa_accuracy, darpa_micro_f1, darpa_macro_f1, darpa_precision, darpa_recall, darpa_tn, darpa_fp, darpa_fn, darpa_tp

def main(args):
    meta_paths = get_metapath()
    # g, labels, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, process_based_g, \
    # file_based_g, attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g = load_all_data_v3()
    g, labels, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, darpa_mal_index, \
    darpa_train_idx, darpa_test_idx, darpa_idx, darpa_train_mask, darpa_test_mask, darpa_mask, process_based_g, \
    file_based_g, attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g = load_all_data_darpa()
    all_id_matrix = []
    all_typ_matrix = []
    idx = np.arange(len(labels))
    h_dict = {'process': torch.load('data/embedding_v3/process_embedding.pkl').to('cpu'),
              'file': torch.load('data/embedding_v3/file_embedding.pkl').to('cpu'),
              'IPC': torch.load('data/embedding_v3/IPC_embedding.pkl').to('cpu'),
              'memory': torch.load('data/embedding_v3/mem_embedding.pkl').to('cpu'),
              'net': torch.load('data/embedding_v3/net_embedding.pkl').to('cpu'),
              'attribute': torch.load('data/embedding_v3/attribute_embedding.pkl').to('cpu'),
              'socket': torch.load('data/embedding_v3/sock_embedding.pkl').to('cpu')
              }

    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device']).bool()
    val_mask = val_mask.to(args['device']).bool()
    test_mask = test_mask.to(args['device']).bool()
    darpa_train_mask = darpa_train_mask.to(args['device']).bool()
    darpa_test_mask = darpa_test_mask.to(args['device']).bool()
    from RGCN import RGCN
    model = RGCN(32,128,64,g.etypes).to(args['device'])
    loss_fcn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1,1.0]).float()).to(args['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    best_train_acc = 0
    best_train_micro_f1 = 0
    best_train_macro_f1 = 0
    best_train_recall = 0
    best_val_acc = 0
    best_val_micro_f1 = 0
    best_val_macro_f1 = 0
    best_val_recall = 0
    all_graph = []
    walk_length = 6
    linear = Linear(64,2).to(args['device'])
    dataset = torch.load('data/embedding_v3/all_sub_graph_outsample.pth')

    for epoch in range(args['num_epochs']):
        model.train()
        all_logits = torch.zeros((len(labels),num_classes)).to(args['device'])
        for i,id in enumerate(tqdm(train_idx)):

            data = dataset[id]
            graph = data[0]
            new_h_dict = data[1]
            new_label = data[2]
            if graph:
                logits = model(graph, new_h_dict, 'process')
                # loss = loss_fcn(logits,new_label)
                all_logits[id] = logits[0]
            else:
                continue
        loss = loss_fcn(all_logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, train_micro_f1, train_macro_f1,train_precision, train_recall,train_tn,train_fp,train_fn,train_tp = score(all_logits[train_mask], labels[train_mask])
        darpa_train_acc, darpa_train_micro_f1, darpa_train_macro_f1, darpa_train_precision, darpa_train_recall, darpa_train_tn, darpa_train_fp, darpa_train_fn, darpa_train_tp = score(
            all_logits[darpa_train_mask], labels[darpa_train_mask])
        all_logits_val = torch.zeros((len(labels), num_classes)).to(args['device'])
        # all_logits_val = torch.zeros((len(labels), num_classes)).to(args['device'])
        # val_loss, val_acc, val_micro_f1, val_macro_f1,_,val_recall,_,_,_,_ = evaluate(dataset,g,model,linear,all_id_matrix,all_typ_matrix,
        #                                                          all_logits_val,all_graph, labels,val_idx, val_mask,h_dict,
        #                                                          meta_paths, walk_length,
        #                                                          loss_fcn,args['device'],node='process')
        if train_acc>best_train_acc:
            best_train_acc= train_acc
        if train_micro_f1>best_train_micro_f1:
            best_train_micro_f1 =train_micro_f1
        if train_macro_f1 >best_train_macro_f1:
            best_train_macro_f1 =train_macro_f1
        if train_recall>best_train_recall:
            best_train_recall=train_recall
        # if val_acc>best_val_acc:
        #     best_val_acc= val_acc
        # if val_micro_f1>best_val_micro_f1:
        #     best_val_micro_f1 =val_micro_f1
        # if val_macro_f1 >best_val_macro_f1:
        #     best_val_macro_f1 =val_macro_f1
        print(
            'Epoch {:d} | Train Loss {:.4f} |Train acc{:.4f}(best:{:.4f}) |Train Micro f1 {:.4f}(best:{:.4f}) | Train Macro f1 {:.4f}(best:{:.4f}) | '
            'Train precision {:.4f}|Train recall {:.4f}|Train tn {:.4f}|Train fp {:.4f}|Train fn {:.4f}|Train tp {:.4f}|'
                .format(
                epoch + 1, loss.item(), train_acc, best_train_acc, train_micro_f1, best_train_micro_f1, train_macro_f1,
                best_train_macro_f1, train_precision, train_recall, train_tn, train_fp, train_fn, train_tp, ))
        print('Train acc{:.4f} |Train Micro f1 {:.4f}| Train Macro f1 {:.4f} | '
              'Train precision {:.4f}|Train recall {:.4f}|Train tn {:.4f}|Train fp {:.4f}|Train fn {:.4f}|Train tp {:.4f}|'.format(
            darpa_train_acc, darpa_train_micro_f1, darpa_train_macro_f1, darpa_train_precision,
            darpa_train_recall, darpa_train_tn, darpa_train_fp, darpa_train_fn, darpa_train_tp, ))
    torch.save(model.state_dict(),'data/embedding_v3/RGCN_all_outsampling.pth')
    # model.load_state_dict(torch.load('data/embedding_v3/RGCN_all_sampling_10_nei_len_32_c0.pth'))
    all_logits_test = torch.zeros((len(labels), num_classes)).to(args['device'])
    time_start = time.time()
    test_loss, test_acc, test_micro_f1, test_macro_f1, test_precision, test_recall, test_tn, test_fp, test_fn, test_tp, \
    darpa_accuracy, darpa_micro_f1, darpa_macro_f1, darpa_precision, darpa_recall, darpa_tn, darpa_fp, darpa_fn, darpa_tp= evaluate(dataset,g,model,linear,all_id_matrix,all_typ_matrix,
                                                                 all_logits_test,all_graph, labels,test_idx, test_mask,darpa_test_mask,h_dict,
                                                                 meta_paths, walk_length,
                                                                 loss_fcn,args['device'],node='process')
    print('Test loss {:.4f} |Test acc{:.4f}| Test Micro f1 {:.4f} | Test Macro f1 {:.4f}|'
          'Train precision {:.4f}|Test recall {:.4f}|Test tn {:.4f}|Test fp {:.4f}|Test fn {:.4f}|Test tp {:.4f}|'.format(
        test_loss.item(), test_acc, test_micro_f1, test_macro_f1, test_precision, test_recall, test_tn, test_fp,
        test_fn, test_tp))
    print('Test acc{:.4f} |Test Micro f1 {:.4f}| Test Macro f1 {:.4f} | '
          'Test precision {:.4f}|Test recall {:.4f}|Test tn {:.4f}|Test fp {:.4f}|Test fn {:.4f}|Test tp {:.4f}|'.format(
        darpa_accuracy, darpa_micro_f1, darpa_macro_f1,
        darpa_precision, darpa_recall, darpa_tn, darpa_fp, darpa_fn, darpa_tp, ))
    time_end = time.time()
    print((time_end - time_start), 'second')
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
