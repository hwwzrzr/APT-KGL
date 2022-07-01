import torch
from sklearn.metrics import f1_score,recall_score,confusion_matrix
import dgl
from utils_pre import EarlyStopping,get_best_result,collate
from data_prepare_v3 import load_all_data_v2,load_all_data_outsample_darpa,re_sort
from meta_v2 import set_meta_paths
from torch.utils.data import DataLoader
import numpy as np
import dgl.nn as dglnn
def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    recall = recall_score(labels, prediction, average='micro')
    tn, fp, fn, tp = confusion_matrix(labels, prediction).ravel()
    return accuracy, micro_f1, macro_f1,recall,tn, fp, fn, tp

def evaluate(model, process_based_g,file_based_g,attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g,
              process_features, file_features, attribute_features, IPC_features, net_features, mem_features,
              socket_features,labels,val_idx, mask, loss_func):
    model.eval()
    with torch.no_grad():
        # logits = model(process_based_g,file_based_g, features,file_features)
        logits, h,h_file,h_net,h_attribute,h_IPC,h_mem,h_sock=model(process_based_g, file_based_g, attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g,
              process_features, file_features, attribute_features, IPC_features, net_features, mem_features,
              socket_features)
    loss = loss_func(logits, labels[mask])
    accuracy, micro_f1, macro_f1,recall,tn, fp, fn, tp = score(logits, labels[mask])

    return loss, accuracy, micro_f1, macro_f1,recall,tn, fp, fn, tp
def main(args):
    g, labels, features, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, darpa_mal_index, \
    darpa_train_idx, darpa_test_idx, darpa_idx, darpa_train_mask, darpa_test_mask, darpa_mask, new_g, process_based_g, \
    file_based_g, attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g=load_all_data_outsample_darpa()
    meta_paths_process, meta_paths_file, meta_paths_attribute, meta_paths_IPC, meta_paths_net, meta_paths_mem, meta_paths_sock = set_meta_paths()
    g = g.to(args['device'])
    labels = re_sort(labels,train_idx)
    labels = labels.to(args['device'])
    # labels =labels[train_mask]
    train_mask = train_mask.to(args['device']).bool()
    val_mask = val_mask.to(args['device']).bool()
    test_mask = test_mask.to(args['device']).bool()
    process_features = new_g.nodes['process'].data['h'].float().to(args['device'])
    file_features = new_g.nodes['file'].data['h'].float().to(args['device'])
    attribute_features = new_g.nodes['attribute'].data['h'].float().to(args['device'])
    IPC_features= new_g.nodes['IPC'].data['h'].float().to(args['device'])
    mem_features= new_g.nodes['memory'].data['h'].float().to(args['device'])
    net_features = new_g.nodes['net'].data['h'].float().to(args['device'])
    socket_features = new_g.nodes['socket'].data['h'].float().to(args['device'])
    from model_pretraining import HAN
    model = HAN(meta_paths_process=meta_paths_process,
                meta_paths_file=meta_paths_file,
                meta_paths_attribute=meta_paths_attribute,
                meta_paths_net=meta_paths_net,
                meta_paths_IPC=meta_paths_IPC,
                meta_paths_mem=meta_paths_mem,
                meta_paths_sock=meta_paths_sock,
                in_size=features.shape[1],
                in_file_size=file_features.shape[1],
                in_attribute_size=attribute_features.shape[1],
                in_IPC_size=IPC_features.shape[1],
                in_net_size=net_features.shape[1],
                in_mem_size=mem_features.shape[1],
                in_sock_size=socket_features.shape[1],
                hidden_size=args['hidden_units'],
                out_size=num_classes,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])
    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1,1.0]).float()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    best_train_acc = 0
    best_train_micro_f1 = 0
    best_train_macro_f1 = 0
    best_val_acc = 0
    best_val_micro_f1 = 0
    best_val_macro_f1 = 0
    for epoch in range(args['num_epochs']):
        model.train()
        # logits = model(process_based_g,file_based_g,attribute_based_g,net_based_g,features,file_features)
        logits,h,h_file,h_net,h_attribute,h_IPC,h_mem,h_sock= model(process_based_g, file_based_g, attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g, process_features,file_features,attribute_features,IPC_features,net_features,mem_features,socket_features)


        loss = loss_fcn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, train_micro_f1, train_macro_f1,train_recall,train_tn,train_fp,train_fn,train_tp = score(logits, labels)
        # val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, process_based_g,file_based_g, features,file_features, labels,val_idx, val_mask, loss_fcn)
        # val_loss, val_acc, val_micro_f1, val_macro_f1,_,_,_,_,_ =evaluate(model, process_based_g,file_based_g,attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g,
        #       process_features, file_features, attribute_features, IPC_features, net_features, mem_features,
        #       socket_features,labels,val_idx, val_mask, loss_fcn)
        # early_stop = stopper.step(val_loss.data.item(), val_acc, model)
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
                best_train_macro_f1, train_recall, train_tn, train_fp, train_fn, train_tp))
        torch.save(h, 'data/embedding_v3/process_embedding_outsample_32_v2.pkl')
        torch.save(h_file, 'data/embedding_v3/file_embedding_outsample_32_v2.pkl')
        torch.save(h_net, 'data/embedding_v3/net_embedding_outsample_32_v2.pkl')
        torch.save(h_attribute, 'data/embedding_v3/attribute_embedding_outsample_32_v2.pkl')
        torch.save(h_IPC, 'data/embedding_v3/IPC_embedding_outsample_32_v2.pkl')
        torch.save(h_mem, 'data/embedding_v3/mem_embedding_outsample_32_v2.pkl')
        torch.save(h_sock, 'data/embedding_v3/sock_embedding_outsample_32_v2.pkl')
        dataset_list = [train_idx,train_mask,test_idx,test_mask]
        darpa_dataset_list = [darpa_train_idx,darpa_train_mask,darpa_test_idx,darpa_test_mask]
        torch.save(dataset_list,'data/embedding_v3/dataset_list_32_v2.pkl')
        torch.save(darpa_dataset_list,'data/embedding_v3/darpa_dataset_list_32_v2.pkl')
        torch.save(new_g,'data/embedding_v3/new_g_32_v2.pkl')
        # print('Epoch {:d} | Train Loss {:.4f} |Train acc{:.4f}(best:{:.4f}) |Train Micro f1 {:.4f}(best:{:.4f}) | Train Macro f1 {:.4f}(best:{:.4f}) | '
        #       'Val Loss {:.4f} |Val acc{:.4f}(best:{:.4f})| Val Micro f1 {:.4f}(best:{:.4f}) | Val Macro f1 {:.4f}(best:{:.4f})'.format(
        #     epoch + 1, loss.item(),train_acc,best_train_acc, train_micro_f1,best_train_micro_f1, train_macro_f1,best_train_macro_f1, val_loss.item(),val_acc,best_val_acc, val_micro_f1,best_val_micro_f1 ,val_macro_f1,best_val_macro_f1))
        # if early_stop:
        #     break
    # stopper.load_checkpoint(model)
    # test_loss, test_acc, test_micro_f1, test_macro_f1,test_recall,test_tn,test_fp,test_fn,test_tp = evaluate(model, process_based_g,file_based_g,attribute_based_g, IPC_based_g, net_based_g, mem_based_g, sock_based_g,
    #           process_features, file_features, attribute_features, IPC_features, net_features, mem_features,
    #           socket_features, labels,test_idx, test_mask, loss_fcn)
    # print('Test loss {:.4f} |Test acc{:.4f}| Test Micro f1 {:.4f} | Test Macro f1 {:.4f}|'
    #       'Test recall {:.4f}|Test tn {:.4f}|Test fp {:.4f}|Test fn {:.4f}|Test tp {:.4f}|'.format(
    #     test_loss.item(), test_acc, test_micro_f1, test_macro_f1, test_recall, test_tn, test_fp, test_fn, test_tp))

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
