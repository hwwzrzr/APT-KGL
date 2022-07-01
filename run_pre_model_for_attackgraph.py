import torch
from sklearn.metrics import f1_score,recall_score,confusion_matrix,precision_score
import dgl
from utils_pre import EarlyStopping,get_best_result,collate
from data_prepare_v3 import load_all_data_v3,load_all_data_darpa
from meta_v2 import set_meta_paths_v2
from torch.utils.data import DataLoader
import numpy as np
import dgl.nn as dglnn
import time
def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    recall = recall_score(labels, prediction)
    tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
    return accuracy, micro_f1, macro_f1,recall,tn, fp, fn, tp
def score_v1(mask,logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    index = []
    mask = mask.tolist()
    for i, (j, k) in enumerate(zip(prediction, labels)):
        if j != k:
            index.append(i)

            # print(j,k,mask[i].index)
    print(index)
    cnt = -1
    for i, ele in enumerate(mask):
        if ele != 0:
            cnt += 1
            if cnt in index:
                print(i)
    accuracy = (prediction == labels).sum() / len(prediction)
    # sample_weight = compute_sample_weight(class_weight='balanced',y=labels)
    micro_f1 = f1_score(labels, prediction, average='micro',sample_weight=None)
    macro_f1 = f1_score(labels, prediction, average='macro',sample_weight=None)
    recall = recall_score(labels,prediction)
    tn,fp,fn,tp =confusion_matrix(labels,prediction).ravel()
    precision = precision_score(labels,prediction)
    # sample_micro_f1 = f1_score(labels, prediction, average='micro', sample_weight=sample_weight)
    # sample_macro_f1 = f1_score(labels, prediction, average='macro', sample_weight=sample_weight)
    # tn, fp, fn, tp
    return accuracy, micro_f1, macro_f1,precision,recall,tn,fp,fn,tp
def evaluate(model, process_based_g,file_based_g,attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g,
              process_features, file_features, attribute_features, IPC_features, net_features, mem_features,
              socket_features,labels,val_idx, mask,darpa_idx,darpa_mask, loss_func):
    model.eval()
    with torch.no_grad():
        # logits = model(process_based_g,file_based_g, features,file_features)
        logits, h,h_file,h_net,h_attribute,h_IPC,h_mem,h_sock=model(process_based_g, file_based_g, attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g,
              process_features, file_features, attribute_features, IPC_features, net_features, mem_features,
              socket_features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1,recall,tn, fp, fn, tp = score(logits[mask], labels[mask])
    darpa_accuracy, darpa_micro_f1,darpa_macro_f1, darpa_recall,darpa_precision, darpa_tn, darpa_fp, darpa_fn, darpa_tp = score_v1(darpa_mask,logits[darpa_mask], labels[darpa_mask])
    return loss, accuracy, micro_f1, macro_f1,recall,tn, fp, fn, tp,darpa_accuracy, darpa_micro_f1,darpa_macro_f1, darpa_recall, darpa_tn, darpa_fp, darpa_fn, darpa_tp
def get_random_walk_graph(g,idx,meta_paths):
    meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
    _cached_coalesced_graph = {}
    for meta_path in meta_paths:
        new_g = dgl.sampling.RandomWalkNeighborSampler(g, termination_prob=0.5, num_neighbors=10, num_random_walks=2,
                                                       num_traversals=5, metapath=meta_path)
        _cached_coalesced_graph[meta_path] = new_g(idx)
    return _cached_coalesced_graph
def main(args):
    start =time.time()
    # g,labels,features,num_classes, train_idx, val_idx, test_idx,train_mask, val_mask, test_mask,process_based_g,\
    #        file_based_g,attribute_based_g,IPC_based_g,net_based_g,mem_based_g,sock_based_g=load_all_data_v3()
    # g, labels, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, darpa_mal_index, \
    # darpa_train_idx, darpa_test_idx, darpa_idx, darpa_train_mask, darpa_test_mask, darpa_mask, process_based_g, \
    # file_based_g, attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g = load_all_data_darpa()
    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths_v2()
    total_graph_list = torch.load('data/create_attack_graph/total_graph_v5.pth')

    g= total_graph_list[0][0]
    print(g)
    process_features= torch.tensor(total_graph_list[0][1]).float().to(args['device'])
    file_features= torch.tensor(total_graph_list[0][2]).float().to(args['device'])
    labels = torch.tensor(total_graph_list[0][3]).to(args['device'])
    # g = g.to(args['device'])
    # labels = labels.to(args['device'])
    # train_mask = train_mask.to(args['device']).bool()
    # val_mask = val_mask.to(args['device']).bool()
    # test_mask = test_mask.to(args['device']).bool()
    # darpa_train_mask=darpa_train_mask.to(args['device']).bool()
    # darpa_test_mask = darpa_test_mask.to(args['device']).bool()
    # process_features = features.to(args['device'])
    # file_features = g.nodes['file'].data['h'].float().to(args['device'])

    attribute_features = torch.from_numpy(np.loadtxt('data/create_attack_graph/label/attr_matrix.txt')).float().to(args['device'])
    IPC_features= torch.from_numpy(np.loadtxt('data/create_attack_graph/label/ipc_matrix.txt')).unsqueeze(0).float().to(args['device'])

    mem_features= torch.from_numpy(np.loadtxt('data/create_attack_graph/label/mem_matrix.txt')).unsqueeze(0).float().to(args['device'])
    net_features = torch.from_numpy(np.loadtxt('data/create_attack_graph/label/net_matrix.txt')).unsqueeze(0).float().to(args['device'])
    socket_features = torch.from_numpy(np.loadtxt('data/create_attack_graph/label/soc_matrix.txt')).unsqueeze(0).float().to(args['device'])
    process_idx = np.arange(len(process_features))
    file_idx = np.arange(len(file_features))
    attr_idx= np.arange(len(attribute_features))
    process_based_g = get_random_walk_graph(g, process_idx, meta_paths_process)
    file_based_g = get_random_walk_graph(g, file_idx, meta_paths_file)

    # attribute_based_g = get_random_walk_graph(g, attr_idx, meta_paths_attribute)

    # IPC_based_g = get_random_walk_graph(g, [0], meta_paths_IPC)
    # net_based_g = get_random_walk_graph(g, [0], meta_paths_net)
    # mem_based_g = get_random_walk_graph(g, [0], meta_paths_mem)
    # sock_based_g = get_random_walk_graph(g, [0], meta_paths_sock)

    from model_pretraining import HAN_v2
    model = HAN_v2(meta_paths_process=meta_paths_process,
                meta_paths_file=meta_paths_file,
                meta_paths_attribute=meta_paths_attribute,

                in_size=process_features.shape[1],
                in_file_size=file_features.shape[1],
                in_attribute_size=attribute_features.shape[1],

                hidden_size=args['hidden_units'],
                out_size=2,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])
    # stopper = EarlyStopping(patience=args['patience'])
    # # loss_fcn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1,1.0]).float()).cuda()
    loss_fcn =torch.nn.CrossEntropyLoss().cuda()
    # # loss_fcn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0]).float()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    best_train_acc = 0
    best_train_micro_f1 = 0
    best_train_macro_f1 = 0
    # best_val_acc = 0
    # best_val_micro_f1 = 0
    # best_val_macro_f1 = 0
    for epoch in range(args['num_epochs']):
        model.train()
        # logits = model(process_based_g,file_based_g,attribute_based_g,net_based_g,features,file_features)
        logits,h,h_file,h_net,h_attribute,h_IPC,h_mem,h_sock= model(process_based_g,file_based_g, process_features,file_features,attribute_features,IPC_features,net_features,mem_features,socket_features)
        torch.save(h, 'data/create_attack_graph/process_embedding_32_v5.pkl')
        torch.save(h_file, 'data/create_attack_graph/file_embedding_32_v5.pkl')
        torch.save(h_net, 'data/create_attack_graph/net_embedding_32_v5.pkl')
        torch.save(h_attribute, 'data/create_attack_graph/attribute_embedding_32_v5.pkl')
        torch.save(h_IPC, 'data/create_attack_graph/IPC_embedding_32_v5.pkl')
        torch.save(h_mem, 'data/create_attack_graph/mem_embedding_32_v5.pkl')
        torch.save(h_sock, 'data/create_attack_graph/sock_embedding_32_v5.pkl')
        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, train_micro_f1, train_macro_f1,train_recall,train_tn,train_fp,train_fn,train_tp = score(logits, labels)
    #     darpa_train_acc, darpa_train_micro_f1, darpa_train_macro_f1, darpa_train_recall,darpa_train_precision, darpa_train_tn, darpa_train_fp, darpa_train_fn, darpa_train_tp= score_v1(darpa_train_mask,logits[darpa_train_mask], labels[darpa_train_mask])
    #     # val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, process_based_g,file_based_g, features,file_features, labels,val_idx, val_mask, loss_fcn)
    #     # val_loss, val_acc, val_micro_f1, val_macro_f1,_,_,_,_,_ =evaluate(model, process_based_g,file_based_g,attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g,
    #     #       process_features, file_features, attribute_features, IPC_features, net_features, mem_features,
    #     #       socket_features,labels,val_idx, val_mask, loss_fcn)
    #     # early_stop = stopper.step(val_loss.data.item(), val_acc, model)
        if train_acc>best_train_acc:
            best_train_acc= train_acc
        if train_micro_f1>best_train_micro_f1:
            best_train_micro_f1 =train_micro_f1
        if train_macro_f1 >best_train_macro_f1:
            best_train_macro_f1 =train_macro_f1
        # if val_acc>best_val_acc:
        #     best_val_acc= val_acc
        # if val_micro_f1>best_val_micro_f1:
        #     best_val_micro_f1 =val_micro_f1
        # if val_macro_f1 >best_val_macro_f1:
        #     best_val_macro_f1 =val_macro_f1
        print(
            'Epoch {:d} | Train Loss {:.4f} |Train acc{:.4f}(best:{:.4f}) |Train Micro f1 {:.4f}(best:{:.4f}) | Train Macro f1 {:.4f}(best:{:.4f}) | '
            'Train recall {:.4f}|Train tn {:.4f}|Train fp {:.4f}|Train fn {:.4f}|Train tp {:.4f}|'
            .format(
                epoch + 1, loss.item(), train_acc, best_train_acc, train_micro_f1, best_train_micro_f1, train_macro_f1,
                best_train_macro_f1, train_recall, train_tn, train_fp, train_fn, train_tp, ))
    #     # print('Train acc{:.4f} |Train Micro f1 {:.4f}| Train Macro f1 {:.4f} | '
    #     #     'Train recall {:.4f}|Train tn {:.4f}|Train fp {:.4f}|Train fn {:.4f}|Train tp {:.4f}|'.format( darpa_train_acc,  darpa_train_micro_f1, darpa_train_macro_f1,
    #     #          darpa_train_recall, darpa_train_tn, darpa_train_fp, darpa_train_fn, darpa_train_tp, ))
    #     # print('Epoch {:d} | Train Loss {:.4f} |Train acc{:.4f}(best:{:.4f}) |Train Micro f1 {:.4f}(best:{:.4f}) | Train Macro f1 {:.4f}(best:{:.4f}) | '
    #     #       'Val Loss {:.4f} |Val acc{:.4f}(best:{:.4f})| Val Micro f1 {:.4f}(best:{:.4f}) | Val Macro f1 {:.4f}(best:{:.4f})'.format(
    #     #     epoch + 1, loss.item(),train_acc,best_train_acc, train_micro_f1,best_train_micro_f1, train_macro_f1,best_train_macro_f1, val_loss.item(),val_acc,best_val_acc, val_micro_f1,best_val_micro_f1 ,val_macro_f1,best_val_macro_f1))
    #     # if early_stop:
    #     #     break
    # # stopper.load_checkpoint(model)
    # test_loss, test_acc, test_micro_f1, test_macro_f1,test_recall,test_tn,test_fp,test_fn,test_tp ,\
    # darpa_accuracy, darpa_micro_f1,darpa_macro_f1, darpa_recall, darpa_tn, darpa_fp, darpa_fn, darpa_tp = evaluate(model, process_based_g,file_based_g,attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g,
    #           process_features, file_features, attribute_features, IPC_features, net_features, mem_features,
    #           socket_features, labels,test_idx, test_mask,darpa_test_idx,darpa_test_mask, loss_fcn)
    # print('Test loss {:.4f} |Test acc{:.4f}| Test Micro f1 {:.4f} | Test Macro f1 {:.4f}|'
    #       'Test recall {:.4f}|Test tn {:.4f}|Test fp {:.4f}|Test fn {:.4f}|Test tp {:.4f}|'.format(
    #     test_loss.item(), test_acc, test_micro_f1, test_macro_f1, test_recall, test_tn, test_fp, test_fn, test_tp))
    # # print('Test acc{:.4f} |Test Micro f1 {:.4f}| Test Macro f1 {:.4f} | '
    # #       'Test recall {:.4f}|Test tn {:.4f}|Test fp {:.4f}|Test fn {:.4f}|Test tp {:.4f}|'.format(
    # #     darpa_accuracy, darpa_micro_f1, darpa_macro_f1,
    # #     darpa_recall, darpa_tn, darpa_fp, darpa_fn, darpa_tp, ))
    end =time.time()
    print(end-start,'s')
if __name__ == '__main__':
    import argparse

    from utils_pre import setup,setup_for_sampling

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