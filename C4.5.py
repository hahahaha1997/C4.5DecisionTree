#encoding=utf-8
from math import log
import operator
import pickle

#读取数据集
def createDateset(filename):
    with open(filename, 'r')as csvfile:
        dataset= [line.strip().split(', ') for line in csvfile.readlines()]     #读取文件中的每一行
        dataset=[[int(i) if i.isdigit() else i for i in row] for row in dataset]    #对于每一行中的每一个元素，将行列式数字化并且去除空白保证匹配的正确完成
        cleanoutdata(dataset)   #清洗数据
        del (dataset[-1])       #去除最后一行的空行
        #precondition(dataset)   #预处理数据
        labels=['age','workclass','fnlwgt','education','education-num',
               'marital-status','occupation',
                'relationship','race','sex','capital-gain','capital-loss','hours-per-week',
                'native-country']
        labelType = ['continuous', 'uncontinuous', 'continuous',
                     'uncontinuous',
                     'continuous', 'uncontinuous',
                     'uncontinuous', 'uncontinuous', 'uncontinuous',
                     'uncontinuous', 'continuous', 'continuous',
                     'continuous', 'uncontinuous']

        return dataset,labels,labelType

def cleanoutdata(dataset):#数据清洗
    for row in dataset:
        for column in row:
            if column == '?' or column=='':
                dataset.remove(row)
                break

#计算香农熵/期望信息
def calculateEntropy(dataSet):
    ClassifyCount = {}#分类标签统计字典，用来统计每个分类标签的概率
    for vector in dataSet:
        clasification = vector[-1]  #获取分类
        if not clasification in ClassifyCount.keys():#如果分类暂时不在字典中，在字典中添加对应的值对
            ClassifyCount[clasification] = 0
        ClassifyCount[clasification] += 1         #计算出现次数
    shannonEntropy=0.0
    for key in ClassifyCount:
        probability = float(ClassifyCount[key]) / len(dataSet)      #计算概率
        shannonEntropy -= probability * log(probability,2)   #香农熵的每一个子项都是负的
    return shannonEntropy

# def addFetureValue(feature):

#划分数据集
def splitDataSet(dataSet,featureIndex,value):#根据离散值划分数据集，方法同ID3决策树
    newDataSet=[]
    for vec in dataSet:#将选定的feature的列从数据集中去除
        if vec[featureIndex] == value:
            rest = vec[:featureIndex]
            rest.extend(vec[featureIndex + 1:])
            newDataSet.append(rest)
    return newDataSet


def splitContinuousDataSet(dataSet,feature,continuousValue):#根据连续值来划分数据集
    biggerDataSet = []
    smallerDataSet = []
    for vec in dataSet:
        rest = vec[:feature]
        rest.extend(vec[feature + 1:])#将当前列中的feature列去除，其他的数据保留
        if vec[feature]>continuousValue:#如果feature列的值比最佳分割点的大，则放在biggerDataSet中，否则放在smallerDataSet中
            biggerDataSet.append(rest)
        else:
            smallerDataSet.append(rest)
    return biggerDataSet,smallerDataSet


#不需要将训练集中有，测试集中没有的值补全
# def addFeatureValue(featureListOfValue,feature):
#     feat = [[ 'Private', 'Self-emp-not-inc', 'Self-emp-inc',
#               'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
#             [],[],[],[],[]]
#     for featureValue in feat[feature]: #feat保存的是所有属性特征的所有可能的取值，其结构为feat = [ [val1,val2,val3,…,valn], [], [], [], … ,[] ]
#         featureListOfValue.append(featureValue)

#选择最好的数据集划分方式
def chooseBestSplitWay(dataSet,labelType):
    isContinuous = -1 #判断是否是连续值，是为1，不是为0
    HC = calculateEntropy(dataSet)#计算整个数据集的香农熵(期望信息)，即H(C)，用来和每个feature的香农熵进行比较
    bestfeatureIndex = -1                   #最好的划分方式的索引值，因为0也是索引值，所以应该设置为负数
    gainRatioMax=0.0                        #信息增益率=(期望信息-熵)/分割获得的信息增益，即为GR = IG / split = ( HC - HTC )/ split , gainRatioMax为最好的信息增益率，IG为各种划分方式的信息增益
    IGMAX = 0.0
    continuousValue = -1 #设置如果是连续值的属性返回的最好的划分方式的最好分割点的值
    for feature in range(len(dataSet[0]) -1 ): #计算feature的个数，由于dataset中是包含有类别的，所以要减去类别
        featureListOfValue=[vector[feature] for vector in dataSet] #对于dataset中每一个feature，创建单独的列表list保存其取值，其中是不重复的
        #addFeatureValue(featureListOfValue,feature) #增加在训练集中有，测试集中没有的属性特征的取值
        if labelType[feature] == 'uncontinuous':
            unique=set(featureListOfValue)
            HTC=0.0 #保存HTC，即H（T|C）
            split = 0.0 #保存split(T)
            gainRatio = 0.0 #计算信息增益率
            for value in unique:
                subDataSet = splitDataSet(dataSet,feature,value) #划分数据集
                probability = len(subDataSet) / float(len(dataSet)) #求得当前类别的概率
                split -= probability * log(probability,2) #计算split(T)
                HTC += probability * calculateEntropy(subDataSet) #计算当前类别的香农熵，并和HTC想加，即H(T|C) = H（T1|C）+ H(T2|C) + … + H(TN|C)
            IG=HC-HTC #计算对于该种划分方式的信息增益
            if split == 0:
                split = 1
                gainRatio = IG/split #计算对于该种划分方式的信息增益率
            if gainRatio > gainRatioMax :
                isContinuous = 0
                gainRatioMax = gainRatio
                bestfeatureIndex = feature
        else: #如果feature是连续型的
            featureListOfValue  = set(featureListOfValue)
            sortedValue = sorted(featureListOfValue)
            splitPoint = []
            for i in range(len(sortedValue)-1):#n个value，应该有n-1个分割点splitPoint
                splitPoint.append((float(sortedValue[i])+float(sortedValue[i+1]))/2.0)

            #C4.5修正，不再使用信息增益率来选择最佳分割点
            # for i in range(len(splitPoint)): #对于n-1个分割点，计算每个分割点的信息增益率，来选择最佳分割点
            #     HTC = 0.0
            #     split = 0.0
            #     gainRatio = 0.0
            #     biggerDataSet,smallerDataSet = splitContinuousDataSet(dataSet,feature,splitPoint[i])
            #     print(i)
            #     probabilityBig = len(biggerDataSet)/len(dataSet)
            #     probabilitySmall = len(smallerDataSet)/len(dataSet)
            #     HTC += probabilityBig * calculateEntropy(biggerDataSet)
            #     HTC += probabilityBig * calculateEntropy(smallerDataSet)
            #     if probabilityBig != 0:
            #         split -= probabilityBig * log(probabilityBig,2)
            #     if probabilitySmall != 0:
            #         split -= probabilitySmall *log(probabilitySmall,2)
            #     IG = HC - HTC
            #     if split == 0:
            #         split = 1;
            #     gainRatio = IG/split
            #     if gainRatio>gainRatioMax:
            #         isContinuous = 1
            #         gainRatioMax = gainRatio
            #         bestfeatureIndex = feature
            #         continuousValue = splitPoint[i]
            for i in range(len(splitPoint)):
                HTC = 0.0
                split = 0.0
                gainRatio = 0.0
                biggerDataSet,smallerDataSet = splitContinuousDataSet(dataSet,feature,splitPoint[i])

                probabilityBig = len(biggerDataSet) / len(dataSet)
                probabilitySmall = len(smallerDataSet) / len(dataSet)
                HTC += probabilityBig * calculateEntropy(biggerDataSet)
                HTC += probabilityBig * calculateEntropy(smallerDataSet)
                if probabilityBig != 0:
                    split -= probabilityBig * log(probabilityBig, 2)
                if probabilitySmall != 0:
                    split -= probabilitySmall * log(probabilitySmall, 2)
                IG = HC - HTC
                if IG>IGMAX:
                    IGMAX = IG
                    continuousValue = splitPoint[i]
                N = len(splitPoint)
                D = len(dataSet)
                IG -= log(N,2)/abs(D)
                gainRatio = float(IG)/float(split)
                if gainRatio>gainRatioMax:
                    isContinuous=1
                    gainRatioMax = gainRatio
                    bestfeatureIndex = feature

    return bestfeatureIndex,continuousValue,isContinuous

#返回出现次数最多的类别，避免产生所有特征全部用完无法判断类别的情况
def majority(classList):
    classificationCount = {}
    for i in classList:
        if not i in classificationCount.keys():
            classificationCount[i] = 0
        classificationCount[i] += 1
    sortedClassification = sorted(dict2list(classificationCount),key = operator.itemgetter(1),reverse = True)
    return sortedClassification[0][0]

#dict字典转换为list列表
def dict2list(dic:dict):
    keys=dic.keys()
    values=dic.values()
    lst=[(key,value)for key,value in zip(keys,values)]
    return lst

#创建树
def createTree(dataSet,labels,labelType):
    classificationList = [feature[-1] for feature in dataSet] #产生数据集中的分类列表，保存的是每一行的分类
    if classificationList.count(classificationList[0]) == len(classificationList): #如果分类别表中的所有分类都是一样的，则直接返回当前的分类
        return classificationList[0]
    if len(dataSet[0]) == 1: #如果划分数据集已经到了无法继续划分的程度，即已经使用完了全部的feature，则进行决策
        return majority(classificationList)
    bestFeature,continuousValue,isContinuous = chooseBestSplitWay(dataSet,labelType) #计算香农熵和信息增益来返回最佳的划分方案，bestFeature保存最佳的划分的feature的索引，在C4.5中要判断该feature是连续型还是离散型的,continuousValue是当前的返回feature是continuous的的时候，选择的“最好的”分割点
    bestFeatureLabel = labels[bestFeature] #取出上述的bestfeature的具体值
    Tree = {bestFeatureLabel:{}}
    del(labels[bestFeature]) #删除当前进行划分是使用的feature避免下次继续使用到这个feature来划分
    del(labelType[bestFeature])#删除labelType中的feature类型，保持和labels同步
    if isContinuous == 1 :#如果要划分的feature是连续的
        #构造以当前的feature作为root节点，它的连续序列中的分割点为叶子的子树
        biggerDataSet,smallerDataSet = splitContinuousDataSet(dataSet,bestFeature,continuousValue)#根据最佳分割点将数据集划分为两部分，一个是大于最佳分割值，一个是小于等于最佳分割值
        subLabels = labels[:]
        subLabelType = labelType[:]
        Tree[bestFeatureLabel]['>'+str(continuousValue)] = createTree(biggerDataSet,subLabels,subLabelType)#将大于分割值的数据集加入到树中，并且递归创建这颗子树
        subLabels = labels[:]
        subLabelType = labelType[:]
        Tree[bestFeatureLabel]['<'+str(continuousValue)] = createTree(smallerDataSet,subLabels,subLabelType)#将小于等于分割值的数据集加入到树中，并且递归创建这颗子树
    else:#如果要划分的feature是非连续的，下面的步骤和ID3决策树相同
        #构造以当前的feature作为root节点，它的所有存在的feature取值为叶子的子树
        featureValueList = [feature[bestFeature]for feature in dataSet] #对于上述取出的bestFeature,取出数据集中属于当前feature的列的所有的值
        uniqueValue = set(featureValueList) #去重
        for value in uniqueValue: #对于每一个feature标签的value值，进行递归构造决策树
            subLabels = labels[:]
            subLabelType = labelType[:]
            Tree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLabels,subLabelType)
    return Tree

def storeTree(inputree,filename):
    fw = open(filename, 'wb')
    pickle.dump(inputree, fw)
    fw.close()

def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)

#测试算法
def classify(inputTree,featLabels,testVector,labelType):
    root = list(inputTree.keys())[0] #取出树的第一个标签,即树的根节点
    dictionary = inputTree[root] #取出树的第一个标签下的字典
    featIndex = featLabels.index(root)
    classLabel = '<=50K'
    if labelType[featIndex] == 'uncontinuous':#如果是离散型的标签，则按照ID3决策树的方法来测试算法
        for key in dictionary.keys():#对于这个字典
            if testVector[featIndex] == key:
                if type(dictionary[key]).__name__ == 'dict': #如果还有一个新的字典
                    classLabel = classify(dictionary[key],featLabels,testVector,labelType)#递归向下寻找到非字典的情况，此时是叶子节点，叶子节点保存的肯定是类别
                else:
                    classLabel = dictionary[key]#叶子节点，返回类别
    else:#如果是连续型的标签，则在取出子树的每一个分支的时候，还需要判断是>n还是<=n的情况，只有这两种情况，所以只需要判断是否是其中一种
        firstBranch = list(dictionary.keys())[0] #取出第一个分支，来确定这个double值
        if str(firstBranch).startswith('>'): #如果第一个分支是">n"的情况，则取出n，为1：n
            number = firstBranch[1:]
        else: #如果第一个分支是“<=n”的情况，则取出n，为2：n
            number = firstBranch[2:]
        if float(testVector[featIndex])>float(number):#如果测试向量是>n的情况
            string = '>'+str(number) #设置一个判断string，它是firstBranch的还原，但是为了节省判断branch是哪一种，直接使用字符串连接的方式链接
        else:
            string = "<="+str(number) #设置一个判断string，它是firstBranch的还原，但是为了节省判断branch是哪一种，直接使用字符串连接的方式链接
        for key in dictionary.keys():
            if string == key:
                if type(dictionary[key]).__name__ == 'dict':#如果还有一个新的字典
                    classLabel = classify(dictionary[key],featLabels,testVector,labelType)
                else:
                    classLabel = dictionary[key]
    return classLabel

def test(mytree,labels,filename,labelType,mydate):
    with open(filename, 'r')as csvfile:
        dataset=[line.strip().split(', ') for line in csvfile.readlines()]     #读取文件中的每一行
        dataset=[[int(i) if i.isdigit() else i for i in row] for row in dataset]    #对于每一行中的每一个元素，将行列式数字化并且去除空白保证匹配的正确完成
        cleanoutdata(dataset)   #数据清洗
        del(dataset[0])         #删除第一行和最后一行的空白数据
        del(dataset[-1])
        #precondition(dataset)       #预处理数据集
        clean(dataset,mydate)          #把测试集中的，不存在于训练集中的离散数据清洗掉
        total = len(dataset)
        correct = 0
        error = 0
    for line in dataset:
        result=classify(mytree,labels,line,labelType=labelType)+'.'
        if result==line[14]:     #如果测试结果和类别相同
            correct = correct + 1
        else :
            error = error + 1

    return total,correct,error

#C4.5决策树不需要清洗掉连续性数据
# def precondition(mydate):#清洗连续型数据
#     #continuous:0,2,4,10,11,12
#     for each in mydate:
#         del(each[0])
#         del(each[1])
#         del(each[2])
#         del(each[7])
#         del(each[7])
#         del(each[7])

#C4.5决策树不需要清洗掉测试集中连续值出现了训练集中没有的值的情况，但是离散数据集中还是需要清洗的
def clean(dataset,mydate):#清洗掉测试集中出现了训练集中没有的值的情况
    for i in [1,3,5,6,7,8,9,13]:
        set1=set()
        for row1 in mydate:
            set1.add(row1[i])
        for row2 in dataset:
            if row2[i] not in set1:
               dataset.remove(row2)
        set1.clear()

def main():
    dataSetName = r"C:\Users\yang\Desktop\adult.data"
    mydate, label ,labelType= createDateset(dataSetName)
    labelList = label[:]

    Tree = createTree(mydate, labelList,labelType)

    storeTree(Tree, r'C:\Users\yang\Desktop\tree.txt')  # 保存决策树，避免下次再生成决策树

    #Tree=grabTree(r'C:\Users\yang\Desktop\tree.txt')#读取决策树，如果已经存在tree.txt可以直接使用决策树不需要再次生成决策树
    total, correct, error  = test(Tree, label, r'C:\Users\yang\Desktop\adult.test',labelType,mydate)
    # with open(r'C:\Users\yang\Desktop\trees.txt', 'w')as f:
    #     f.write(str(Tree))
    accuracy = float(correct)/float(total)
    print("准确率：%f" % accuracy)

if __name__ == '__main__':
    main()