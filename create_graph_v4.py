import os
import json
import time
import pandas as pd
def getfilelist(path):
    file = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip(' \n.')
            file.append(line)
    filelist = []
    onefile = []
    for dic in file:
        if 'file' in dic:
            filelist.append(onefile)
            onefile=[]
            onefile.append(dic)
        elif dic  == file[-1]:
            if 'VisitedProcess' in dic:
                onefile.append(dic[15:])
            if 'Label' in dic:
                onefile.append(dic[dic.find(':')+1:])
            filelist.append(onefile)
        else:
            if 'VisitedProcess' in dic:
                onefile.append(dic[15:])
            if 'Label' in dic:
                onefile.append(dic[dic.find(':')+1:])
    filelist.pop(0)
    return filelist
def getprocesslist(path):
    file = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip(' \n.')
            file.append(line)
    processlist = []
    oneprocess = []
    for dic in file:
        if 'pid' in dic:
            processlist.append(oneprocess)
            oneprocess = []
            oneprocess.append(dic)
        elif dic == file[-1]:
            oneprocess.append(dic)
            processlist.append(oneprocess)
        else:
            oneprocess.append(dic)
    processlist.pop(0)
    return processlist
def loadjson(path):
    data = []
    with open(path) as f:
        for line in f:
            dic = json.loads(line)
            data.append(dic)
    f.close()
    return data
def setfileId(filelist):
    filename = []
    for file in filelist:
        if file[0][9:] != '' and file[0][9:] not in filename:
            filename.append(file[0][9:])
    for file in filelist:
        for index, dic in enumerate(filename):
            if file[0][9:] == dic:
                file.insert(1, str(index))
    return filelist
def compressprocess(data):
    file = 'data/darpa/B/process-B_v2.json'
    with open(file, 'w') as f:
        for dic in data:
                compressdata = {"uuid":dic['datum']['uuid'],
                                "basictype":dic['datum']['type'],
                                "cid":dic['datum']['cid'],
                                "localprincipal":dic['datum']['localPrincipal'],
                                "parentsubject":dic['datum']['parentSubject']}
                json.dump(compressdata, f)
                f.write("\n")
def comprcessevent(data):
    file = 'data/darpa/B/event_v2.json'
    with open(file, 'w') as f:
        for dic in data:
                compressdata = {"uuid": dic['datum']['uuid'],
                                "basictype": dic['datum']['type'],
                                "threadid": dic['datum']['threadId'],
                                "subject": dic['datum']['subject'],
                                "parent": dic['datum']['predicateObject']}
                json.dump(compressdata, f)
                f.write("\n")
def countType(data):
    typelist = []
    for dic in data:
        for key in dic['datum'].keys():
            if key not in typelist:
                typelist.append(key)
    print(typelist)
def compressobject(data):
    file = 'data/darpa/B/object_v2.json'
    with open(file, 'w') as f:
        for dic in data:
                    if 'MEMORY' in dic['type'] :
                        compressdata = {
                                        "uuid":dic['datum']['uuid'],
                                        "baseobject":dic['datum']['baseObject'],
                                        "type":dic['type']
                        }
                    if 'FILE' in dic['type'] :
                        compressdata = {
                                        "uuid":dic['datum']['uuid'],
                                        "path":dic['datum']['baseObject']['properties'],
                                        "type":dic['datum']['type']
                        }
                    if 'IPC' in dic['type'] :
                        compressdata = {
                                        "uuid": dic['datum']['uuid'],
                                        "properties": dic['datum']['baseObject']['properties'],
                                        "type": dic['datum']['type']
                        }

                    if 'NET_FLOW' in dic['type'] :
                        compressdata = {
                                        "uuid": dic['datum']['uuid'],
                                        "baseobject":dic['datum']['baseObject'],
                                        "localaddress":dic['datum']['localAddress'],
                                        "localport":dic['datum']['localPort'],
                                        "remoteaddress":dic['datum']['remoteAddress'],
                                        "type":dic['type']
                        }
                    json.dump(compressdata, f)
                    f.write("\n")
def settypejson():
    path = 'data/graph_v5/1.json'
    data = loadjson(path)
    path1= 'data/graph_v5/process.json'
    path2 = 'data/graph_v5/event.json'
    path3 = 'data/graph_v5/object.json'
    with open(path1, 'w') as f1, open(path2, 'w') as f2, open(path3, 'w') as f3:
        for dic in data:
            if dic['type'] == 'RECORD_SUBJECT':
                json.dump(dic,f1)
                f1.write('\n')
            if dic['type'] == 'RECORD_EVENT':
                json.dump(dic, f2)
                f2.write('\n')
            if 'OBJECT' in dic['type'] and 'SRC_SINK_OBJECT' not in dic['type']:
                json.dump(dic, f3)
                f3.write('\n')


def labelprocess(processlist,data):
    file = 'data/darpa/A/process-A_v3.json'
    plist = get_porcess_id(data)
    with open(file, 'w') as f:
        for dic in data:
            label = {"label": ['PT0']}
            processid =getpid(dic,plist)
            for process in processlist:
                if str(dic['cid']) == str(process[0][4:]):
                    label = getplabel(process)
            dic.update(processid)
            dic.update(label)
            json.dump(dic,f)
            f.write("\n")
def labelobject(filelist,data):
    file = 'data/darpa/A/object_v3.json'
    fileli,socketlist = get_file_id(data)
    print(fileli)
    with open(file, 'w') as f:
        # Mid = 14643
        # PRid = 0
        # Iid = 8334
        # Nid = 373
        Mid = 0
        PRid = 0
        Iid = 0
        Nid = 0
        for dic in data:
            if dic['type'] == 'FILE_OBJECT_DIR' or dic['type'] == 'FILE_OBJECT_CHAR' or dic['type'] == 'FILE_OBJECT_FILE':
                label ={"label":['FT0']}
                # ID = {'fid': str(-1)}
                ID = getfileid(dic, fileli)
                for file in filelist:
                    if dic['path']['path'].lower() == file[0][9:].lower():
                        label = getflabel(file)

            if dic['type'] == 'FILE_OBJECT_UNIX_SOCKET':
                label = {"label":['ST0']}
                ID = getsocketid(dic,socketlist)
            if  dic['type'] == 'RECORD_MEMORY_OBJECT':
                label = {"label":['MT0']}
                ID = {'Mid':str(Mid)}
                Mid+=1
            if  dic['type'] == 'RECORD_PRINCIPAL':
                label = {"label":['PR0']}
                ID = {'PRid': str(PRid)}
                PRid += 1
            if dic['type'] == 'IPC_OBJECT_PIPE_UNNAMED':
                label = {"label":['IT0']}
                ID = {'Iid': str(Iid)}
                Iid += 1
            if  dic['type'] == 'RECORD_NET_FLOW_OBJECT':
                label = {"label":['NT0']}
                ID = {'Nid': str(Nid)}
                Nid += 1
            dic.update(ID)
            dic.update(label)
            json.dump(dic, f)
            f.write("\n")
def get_porcess_id(data):
    plist = []
    for dic in data:
        if dic['cid'] not in plist:
            plist.append(dic['cid'])
    return plist
def get_file_id(data):
    filelist = []
    socketlist = []
    for dic in data:
        if dic['type'] == 'FILE_OBJECT_DIR' or dic['type'] == 'FILE_OBJECT_CHAR' or dic['type'] == 'FILE_OBJECT_FILE':
            if dic['path']['path'] not in filelist:
                filelist.append(dic['path']['path'])
        if dic['type'] == 'FILE_OBJECT_UNIX_SOCKET':
            if dic['path']['path'] not in socketlist:
                socketlist.append(dic['path']['path'])
    return filelist,socketlist
def getfileid(dic,filelist):
    currentID = 10074
    fileid = {'fid':-1}
    for index,filename in enumerate(filelist):
        if dic['path']['path'] == filename:
            id = index
            fileid['fid'] = id+currentID
    return fileid
def getsocketid(dic,socketlist):
    currentID = 4
    socketid = {'sid':-1}
    for index,filename in enumerate(socketlist):
        if dic['path']['path'] == filename:
            id = index
            socketid['sid'] = id+currentID
    return socketid
def getpid(dic,plist):
    currentid = 34327
    processid = {'pid':-1}
    for index,pid in enumerate(plist):
        if str(dic['cid']) == str(pid):
            id =index
            processid['pid'] = id+currentid
    return processid
def getplabel(process):
    label  = {"label":['PT0']}
    li = []
    for i in process[1:]:
        if  i :
            li.append(i[6:i.find(';')])
    if li:
        label['label'] = li
    return label

def getflabel(file):
    label = {"label":['FT0']}
    li = []
    for i in file :
        if 'FT' in i :
            li.append(i)
    if li :
        label['label'] = li
    return label
def getlink(process,events,allprocess,allobject,path):
    mal_id = [19289,14536,15913,17609]
    label =0
    if process['cid'] in mal_id:
        label = 1
    with open(path,'w') as f:
        for event in events:
            if event['subject'] == process['uuid']:
                    for oneprocess in allprocess:
                        if event['parent'] == oneprocess['uuid']:
                            onelink = {"link":[process['pid'],oneprocess['pid']],
                                       "path": 'null',
                                       "label":[process['label'],oneprocess['label'],label],
                                       "type":[process['basictype'],event['basictype'],oneprocess['basictype']],
                                       "initlink":[process['cid'],oneprocess['cid']]
                                       }
                            json.dump(onelink, f)
                            f.write("\n")
                            break
                    for oneobject in allobject:
                        if event['parent'] == oneobject['uuid']:
                            if 'FILE_OBJECT' in oneobject['type']:
                                onelink = {"link": [process['pid'], int(list(oneobject.items())[-2][1])],
                                           "path":[oneobject['path']['path']],
                                           "label": [process['label'], oneobject['label'],label],
                                           "type": [process['basictype'], event['basictype'], oneobject['type']],
                                           "initlink":[process['cid'],0]
                                           }
                                json.dump(onelink, f)
                                f.write("\n")
                            else:
                                onelink = {"link": [process['pid'], int(list(oneobject.items())[-2][1])],
                                           "path":'null',
                                           "label": [process['label'], oneobject['label'],label],
                                           "type": [process['basictype'], event['basictype'], oneobject['type']],
                                           "initlink":[process['cid'],0]
                                           }
                                json.dump(onelink, f)
                                f.write("\n")
                                break
    f.close()
                    # if onelink:
                    #     json.dump(onelink, f)
                    #     f.write("\n")
def getlink_v2(process,events,allprocess,allobject,data_list):
    mal_id = [11013,15074,15628]
    label =0

    if process['cid'] in mal_id:
        label = 1

    for index,event in enumerate(events):

            flag = 0
            if event['subject'] == process['uuid']:
                    for oneprocess in allprocess:
                        if event['parent'] == oneprocess['uuid']:

                            onelink = {"link":[process['pid'],oneprocess['pid']],
                                       "path": 'null',
                                       "label":[process['label'],oneprocess['label'],label],
                                       "type":[process['basictype'],event['basictype'],oneprocess['basictype']],
                                       "initlink":[process['cid'],oneprocess['cid']]
                                       }

                            data_list.append(onelink)
                            flag=1
                            break
                    if flag ==0:
                        for oneobject in allobject:
                            if event['parent'] == oneobject['uuid']:
                                if 'FILE_OBJECT' in oneobject['type']:
                                    onelink = {"link": [process['pid'], int(list(oneobject.items())[-2][1])],
                                               "path":[oneobject['path']['path']],
                                               "label": [process['label'], oneobject['label'],label],
                                               "type": [process['basictype'], event['basictype'], oneobject['type']],
                                               "initlink":[process['cid'],0]
                                               }
                                    data_list.append(onelink)

                                    break
                                else:
                                    onelink = {"link": [process['pid'], int(list(oneobject.items())[-2][1])],
                                               "path":'null',
                                               "label": [process['label'], oneobject['label'],label],
                                               "type": [process['basictype'], event['basictype'], oneobject['type']],
                                               "initlink":[process['cid'],0]
                                               }
                                    data_list.append(onelink)

                                    break

def graphtojson(events,allprocess,allobject):
    all_data=[]
    i = 0
    process_list = loadjson('data/darpa/A/anicent-process-A-V1.json')
    uuid = []
    for dic in process_list:
        uuid.append(dic['datum']['uuid'])
    new_process_list = []
    for dic in allprocess:
        if dic['uuid'] in uuid:
            new_process_list.append(dic)
    for process in new_process_list:
        path = 'data/darpa/A/graph_v2/%d_%d.json'%(i,process['cid'])

        # getlink_v2(process,events,allprocess,allobject,all_data)
        getlink(process,events,allprocess,allobject,path)
        if i %100 == 0:
            print("进度为",(i/len(new_process_list)*100),"%")
        i+=1
    # with open('data/darpa/A/all_link.json','w')as f:
    #     for dic in all_data:
    #         json.dump(dic,f)
    #         f.write('\n')
def load_mal_json(path,pid):
    data = []
    with open(path) as f:
        for line in f:
            dic = json.loads(line)
            if 'type' in dic['datum']:
                if dic['datum']['type'] == 'SUBJECT_PROCESS':
                    if dic['datum']['cid'] in pid :
                        data.append(dic)
    f.close()
    return data
def separate_data(path):
    process_path='data/darpa/A/process-A.json'
    event_path= 'data/darpa/A/event-A.json'
    object_path='data/darpa/A/object-A.json'
    with open(path,'r') as f,open(process_path,'w')as f1,open(event_path,'w')as f2,open(object_path,'w')as f3:
        for line in f:
            dic = json.loads(line)
            if 'type' in dic['datum']:
                if 'EVENT' in dic['datum']['type']:
                   json.dump(dic,f2)
                   f2.write('\n')
                elif 'PROCESS' in dic['datum']['type']:
                    json.dump(dic,f1)
                    f1.write('\n')
                elif 'SRCSINK_UNKNOWN' not in dic['datum']['type']:
                    json.dump(dic,f3)
                    f3.write('\n')
if __name__ == '__main__':



    # 获取基本特征
    # filelist = getfilelist('data/darpa/A/filelist-A.txt')
    # processlist = getprocesslist('data/darpa/A/processlist-A.txt')

    # 分离文件
    # separate_data('data/ssh1.json')

    # 压缩数据
    # process = loadjson('data/darpa/B/process-B.json')
    # event = loadjson('data/darpa/B/event_v1.json')
    # object = loadjson('data/darpa/B/object_v1.json')
    # compressobject(object)
    # comprcessevent(event)
    # compressprocess(process)

    # 标记数据
    # process = loadjson('data/darpa/A/process-A_v2.json')
    # object = loadjson('data/darpa/A/object_v2.json')
    # labelobject(filelist, object)
    # labelprocess(processlist,process)

    # 转成依赖图

    # process = loadjson('data/darpa/A/process-A_v3.json')
    # event = loadjson('data/darpa/A/event_v2.json')
    # object = loadjson('data/darpa/A/object_v3.json')
    # graphtojson(event,process,object)
    import  torch
    # a=torch.tensor([[0,1],[1,0]])
    # b=torch.tensor([[0,2],[2,0]])
    #
    # h={'a':a,'b':b}
    # s=h.keys()
    # x=0
    # for u in s:
    #     print(torch.mean(h[u].float(),dim=0))
    #     x+=torch.mean(h[u].float(),dim=0)
    # print(x/len(h.keys()))
    lab = torch.tensor([0,0,0,0])

    print(torch.mean(lab.float()))
    lab[1]= 1 if torch.mean((lab.float()))>0 else 0
    print(lab)



