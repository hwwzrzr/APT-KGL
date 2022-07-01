import os
import json
import time
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
    subjectType= "com.bbn.tc.schema.avro.cdm19.Subject"
    uuidType = "com.bbn.tc.schema.avro.cdm19.UUID"
    file = 'data/process.json'
    with open(file, 'w') as f:
        for dic in data:
            if subjectType in dic['datum']:
                compressdata = {"uuid":dic['datum'][subjectType]['uuid'],
                                "basictype":dic['datum'][subjectType]['type'],
                                "cid":dic['datum'][subjectType]['cid'],
                                "localprincipal":dic['datum'][subjectType]['localPrincipal'],
                                "parentsubject":dic['datum'][subjectType]['parentSubject']}
                json.dump(compressdata, f)
                f.write("\n")
def comprcessevent(data):
    eventType = "com.bbn.tc.schema.avro.cdm19.Event"
    uuidType = "com.bbn.tc.schema.avro.cdm19.UUID"
    file = 'data/event.json'
    with open(file, 'w') as f:
        for dic in data:
            if eventType in dic['datum']:
                compressdata = {"uuid": dic['datum'][eventType]['uuid'],
                                "basictype": dic['datum'][eventType]['type'],
                                "threadid": dic['datum'][eventType]['threadId'],
                                "subject": dic['datum'][eventType]['subject'],
                                "parent": dic['datum'][eventType]['predicateObject']}
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
    objectType = ["com.bbn.tc.schema.avro.cdm19.MemoryObject",
                  "com.bbn.tc.schema.avro.cdm19.Principal",
                  "com.bbn.tc.schema.avro.cdm19.FileObject",
                  "com.bbn.tc.schema.avro.cdm19.IpcObject",
                  "com.bbn.tc.schema.avro.cdm19.NetFlowObject"]
    file = 'data/object.json'
    with open(file, 'w') as f:
        for dic in data:
            for i, type in enumerate(objectType):
                if objectType[i] in dic['datum']:
                    if objectType[i] == "com.bbn.tc.schema.avro.cdm19.MemoryObject":
                        compressdata = {
                                        "uuid":dic['datum'][objectType[i]]['uuid'],
                                        "baseobject":dic['datum'][objectType[i]]['baseObject'],
                                        "type":dic['type']
                        }
                    if objectType[i] == "com.bbn.tc.schema.avro.cdm19.FileObject":
                        compressdata = {
                                        "uuid":dic['datum'][objectType[i]]['uuid'],
                                        "path":dic['datum'][objectType[i]]['baseObject']['properties'],
                                        "type":dic['datum'][objectType[i]]['type']
                        }
                    if objectType[i] == "com.bbn.tc.schema.avro.cdm19.IpcObject":
                        compressdata = {
                                        "uuid": dic['datum'][objectType[i]]['uuid'],
                                        "properties": dic['datum'][objectType[i]]['baseObject']['properties'],
                                        "type": dic['datum'][objectType[i]]['type']
                        }
                    if objectType[i] == "com.bbn.tc.schema.avro.cdm19.Principal":
                        compressdata = {
                                        "uuid": dic['datum'][objectType[i]]['uuid'],
                                        "userid":dic['datum'][objectType[i]]['userId'],
                                        "properties":dic['datum'][objectType[i]]['properties'],
                                        "type":dic['type'],
                        }
                    if objectType[i] == "com.bbn.tc.schema.avro.cdm19.NetFlowObject":
                        compressdata = {
                                        "uuid": dic['datum'][objectType[i]]['uuid'],
                                        "baseobject":dic['datum'][objectType[i]]['baseObject'],
                                        "localaddress":dic['datum'][objectType[i]]['localAddress'],
                                        "localport":dic['datum'][objectType[i]]['localPort'],
                                        "remoteaddress":dic['datum'][objectType[i]]['remoteAddress'],
                                        "type":dic['type']
                        }
                    json.dump(compressdata, f)
                    f.write("\n")
def labelprocess(processlist,data):
    file = 'data/process_v3.json'
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
    file = 'data/object_v3.json'
    fileli,socketlist = get_file_id(data)
    print(fileli)
    with open(file, 'w') as f:
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
                    if dic['path']['map']['path'].lower() == file[0][9:].lower():
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
            if dic['path']['map']['path'] not in filelist:
                filelist.append(dic['path']['map']['path'])
        if dic['type'] == 'FILE_OBJECT_UNIX_SOCKET':
            if dic['path']['map']['path'] not in socketlist:
                socketlist.append(dic['path']['map']['path'])
    return filelist,socketlist
def getfileid(dic,filelist):
    fileid = {'fid':-1}
    for index,filename in enumerate(filelist):
        if dic['path']['map']['path'] == filename:
            id = index
            fileid['fid'] = id
    return fileid
def getsocketid(dic,socketlist):
    socketid = {'sid':-1}
    for index,filename in enumerate(socketlist):
        if dic['path']['map']['path'] == filename:
            id = index
            socketid['sid'] = id
    return socketid
def getpid(dic,plist):
    processid = {'pid':-1}
    for index,pid in enumerate(plist):
        if str(dic['cid']) == str(pid):
            id =index
            processid['pid'] = id
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
    uuidType = "com.bbn.tc.schema.avro.cdm19.UUID"
    events = iter(events)
    with open(path,'w') as f:
        for event in events:
            if event['subject'][uuidType] == process['uuid']:
                    # allprocess = iter(allprocess)
                    # allobject = iter(allobject)
                    for oneprocess in allprocess:
                        if event['parent'][uuidType] == oneprocess['uuid']:
                            onelink = {"link":[process['pid'],oneprocess['pid']],
                                       "path": 'null',
                                       "label":[process['label'],oneprocess['label']],
                                       "type":[process['basictype'],event['basictype'],oneprocess['basictype']]
                                       }
                            json.dump(onelink, f)
                            f.write("\n")
                    for oneobject in allobject:
                        if event['parent'][uuidType] == oneobject['uuid']:
                            if 'FILE_OBJECT' in oneobject['type']:
                                onelink = {"link": [process['pid'], int(list(oneobject.items())[-2][1])],
                                           "path":[oneobject['path']['map']['path']],
                                           "label": [process['label'], oneobject['label']],
                                           "type": [process['basictype'], event['basictype'], oneobject['type']]
                                           }
                                json.dump(onelink, f)
                                f.write("\n")
                            else:
                                onelink = {"link": [process['pid'], int(list(oneobject.items())[-2][1])],
                                           "path":'null',
                                           "label": [process['label'], oneobject['label']],
                                           "type": [process['basictype'], event['basictype'], oneobject['type']]
                                           }
                                json.dump(onelink, f)
                                f.write("\n")
                            break
    f.close()
                    # if onelink:
                    #     json.dump(onelink, f)
                    #     f.write("\n")

def graphtojson(events,allprocess,allobject):
    i = 0
    for process in allprocess:
        path = 'data/graph_v4/%d.json'%i
        getlink(process,events,allprocess,allobject,path)
        if i %100 == 0:
            print("进度为",(i/len(allprocess)*100),"%")
        i+=1
if __name__ == '__main__':
    t0 = time.time()
    # allprocess = loadjson('data/process_v3.json')
    # allobject = loadjson('data/object_v3.json')
    # events = loadjson('data/event.json')
    # graphtojson(events,allprocess,allobject)
    # data = loadjson('data/process.json')
    # filelist = getfilelist('data/filelist.txt')
    # labelobject(filelist,data)
    # filelist = setfileId(filelist)
    # for file in filelist:
    #     print(file)
    # process = loadjson('data/process.json')
    # processlist = getprocesslist('data/processlist.txt')
    # labelprocess(processlist,process)
    # allprocess = loadjson('data/process_v2.json')
    # events = loadjson('data/event.json')
    # allobject = loadjson('data/object_v2.json')
    # graphtojson(events,allprocess,allobject)
    t1 = time.time()
    print(t1-t0)
    # labelobject(filelist,data)


    # data = loadjson('data/process_v2.json')
    # for dic in data:
    #     print(dic['label'])
    # labelobject(filelist,data)
    # data = loadjson('data/process.json')
    # labelprocess(processlist,data)
    # compressprocess(data)
    # comprcessevent(data)
    # compressobject(data)

