
# coding: utf-8

# In[10]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
import random
from matplotlib import pyplot as plt


# In[11]:


maxAcc = 0.0
maxIter = 0
C_Lambda = 1
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
PHI = []
IsSynthetic = False


# In[12]:


def CreateDataMatrixSamePair(filePath):
    target = []
    count= 0
    count2 = 0
    addMatrix =[]
    subMatrix = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if(count<2000):
                target.append(int(row[2]))
                data = []
                data1= []
                with open('GSC_Data.csv', 'rU') as f1:
                    reader1 = csv.reader(f1) 
                    next(reader1)
                    for row1 in reader1:
                        if(row1[0] == row[0]):
                            for column in row1:
                                data.append(column)
                        if(row1[0] == row[1]):
                            for column in row1:
                                data1.append(column)
                            break
                    data=data[1:]
                    data=[int(data) for data in data]
                    data1=data1[1:]
                    data1=[int(data1) for data1 in data1]
                    add = data + data1
                    sub = [abs((data - data1))  for data,data1 in zip(data,data1)]
                    addMatrix.append(add)
                    subMatrix.append(sub)
                count = count +1
    with open("GSC_diffn_pairs.csv", 'rU') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if(count2) <2000:
                target.append(int(row[2]))
                data = []
                data1= []
                with open('GSC_Data.csv', 'rU') as f1:
                    reader1 = csv.reader(f1) 
                    next(reader1)
                    for row1 in reader1:
                        if(row1[0] == row[0]):
                            for column in row1:
                                data.append(column)
                        if(row1[0] == row[1]):
                            for column in row1:
                                data1.append(column)
                            break
                    data=data[1:]
                    data=[int(data) for data in data]
                    data1=data1[1:]
                    data1=[int(data1) for data1 in data1]
                    add = data + data1
                    sub = [abs((data - data1))  for data,data1 in zip(data,data1)]
                    addMatrix.append(add)
                    subMatrix.append(sub)
            count2 = count2 +1
    return addMatrix , subMatrix, target

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma


def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.pinv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


# In[13]:


Add, Subtract, Target =  CreateDataMatrixSamePair('GSC_same_pairs.csv')
shuffledDataMatrix = list(zip(Add, Subtract, Target))
random.shuffle(shuffledDataMatrix)
Add, Subtract, Target = zip(*shuffledDataMatrix)
deladd = (np.where(~np.array(Add).any(axis=0))[0])
delsub = (np.where(~np.array(Subtract).any(axis=0))[0])
Add = np.delete(Add,deladd,axis=1)
Subtract = np.delete(Subtract,delsub,axis=1)
Add = np.transpose(Add)
Subtract = np.transpose(Subtract)
print(np.array(Target).shape)
print(np.array(Add).shape)
print(np.array(Subtract).shape)
#print(Subtract)


# ## Prepare Training Data

# In[14]:


TrainingTarget = np.array(GenerateTrainingTarget(Target,TrainingPercent))
TrainingAddData   = GenerateTrainingDataMatrix(Add,TrainingPercent)
TrainingSubtractData   = GenerateTrainingDataMatrix(Subtract,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingAddData.shape)
print(TrainingSubtractData.shape)


# ## Prepare Validation Data

# In[15]:


ValDataAct = np.array(GenerateValTargetVector(Target,ValidationPercent, (len(TrainingTarget))))
ValDataAdd    = GenerateValData(Add,ValidationPercent, (len(TrainingTarget)))
ValDataSubtract    = GenerateValData(Subtract,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValDataAdd.shape)
print(ValDataSubtract.shape)


# ## Prepare Test Data

# In[16]:


TestDataAct = np.array(GenerateValTargetVector(Target,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestDataAdd = GenerateValData(Add,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
TestDataSubtract= GenerateValData(Subtract,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(TestDataAct.shape)
print(TestDataAdd.shape)
print(TestDataSubtract.shape)


# ## Closed Form Solution For Feature Addition

# In[17]:


ErmsArr = []
AccuracyArr = []
M= 25
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingAddData))
Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(Add, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(Add, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestDataAdd, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValDataAdd, Mu, BigSigma, 100)


# In[18]:


print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set

# In[21]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[22]:


print ("-------Closed Form with Radial Basis Function For Feature Addition-------")
print ('----------------------------------------------------')
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
print ("Accuracy Training   = " + str(float(TrainingAccuracy.split(',')[0])))
print ("Accuracy Validation = " + str(float(ValidationAccuracy.split(',')[0])))
print ("Accuracy Testing    = " + str(float(TestAccuracy.split(',')[0])))


# ## Gradient Descent solution for Linear Regression For Feature Addition

# In[23]:


W_Now        = np.dot(220, W)
La           = 2
learningRate = 0.02
L_Erms_Test  = []
W_Mat        = []

for i in range(0,1601):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    
    #-----------------ValidationData Accuracy---------------------#
VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    
    #-----------------TestingData Accuracy---------------------#
TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
Erms_Test = GetErms(TEST_OUT,TestDataAct)


# In[24]:


print ("-------Gradient Descent For Feature Addition-------")
print ('----------------------------------------------------')
print ("E_rms Training   = " + str(float(Erms_TR.split(',')[1])))
print ("E_rms Validation = " + str(float(Erms_Val.split(',')[1])))
print ("E_rms Testing    = " + str(float(Erms_Test.split(',')[1])))
print ("Accuracy Training   = " + str(float(Erms_TR.split(',')[0])))
print ("Accuracy Validation = " + str(float(Erms_Val.split(',')[0])))
print ("Accuracy Testing    = " + str(float(Erms_Test.split(',')[0])))


# ## Closed Form Solution For Feature Subtraction

# In[25]:


M= 25

kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingSubtractData))
Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(Subtract, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(Subtract, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestDataSubtract, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValDataSubtract, Mu, BigSigma, 100)


# In[26]:


print(Mu.shape)
print(BigSigma.shape)
print(TRAINING_PHI.shape)
print(W.shape)
print(VAL_PHI.shape)
print(TEST_PHI.shape)


# ## Finding Erms on training, validation and test set

# In[28]:


TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))


# In[29]:


print ("-------Closed Form with Radial Basis Function For Feature Subtraction-------")
print ('----------------------------------------------------')
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
print ("Accuracy Training   = " + str(float(TrainingAccuracy.split(',')[0])))
print ("Accuracy Validation = " + str(float(ValidationAccuracy.split(',')[0])))
print ("Accuracy Testing    = " + str(float(TestAccuracy.split(',')[0])))


# ##  Gradient Descent solution for Linear Regression For Feature Subtraction

# In[30]:


W_Now        = np.dot(220, W)
La           = 0.5
learningRate = 0.02
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,1270):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))


# In[31]:


print ("-------Gradient Descent For Feature Subtraction-------")
print ('----------------------------------------------------')
print ("E_rms Training   = " + str(float(Erms_TR.split(',')[1])))
print ("E_rms Validation = " + str(float(Erms_Val.split(',')[1])))
print ("E_rms Testing    = " + str(float(Erms_Test.split(',')[1])))
print ("Accuracy Training   = " + str(float(Erms_TR.split(',')[0])))
print ("Accuracy Validation = " + str(float(Erms_Val.split(',')[0])))
print ("Accuracy Testing    = " + str(float(Erms_Test.split(',')[0])))


# In[32]:


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


# ## Logistic Regression solution for Feature Subtraction

# In[33]:


W_Now        = np.zeros(472)
La           = 2
learningRate = 0.05
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,1600):
    scores = np.dot(W_Now, TrainingSubtractData)
    prediction  = sigmoid(scores)
    #print ('---------Iteration: ' + str(i) + '--------------')
    error =  prediction - TrainingTarget
    Delta_E_D     = np.dot((TrainingSubtractData), error)/TrainingTarget.size
    #La_Delta_E_W  = np.dot(La,W_Now)
    #Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E_D)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next

    #-----------------Training Data Accuracy---------------------#
right=0
wrong=0
TR_TEST=[]
TR_TEST_OUT   =  sigmoid(np.dot(W_Now, TrainingSubtractData)) 
for res in TR_TEST_OUT:
    if (res>0.5):
        TR_TEST.append(1)
    else:
        TR_TEST.append(0)
for i,j in zip(TR_TEST, TrainingTarget):
    if i==j:
        right = right +1
    else:
        wrong = wrong +1
print("Training Accuracy: " + str(right/(right+wrong)*100))
     
    #-----------------Validation Data Accuracy---------------------# 
right=0
wrong=0
VAL_TEST=[] 
VAL_TEST_OUT  = sigmoid(np.dot(W_Now, ValDataSubtract))
for res in VAL_TEST_OUT:
    if (res>0.5):
        VAL_TEST.append(1)
    else:
        VAL_TEST.append(0)
for i,j in zip(VAL_TEST, ValDataAct):
    if i==j:
        right = right +1
    else:
        wrong = wrong +1
print("Validation Accuracy: " + str(right/(right+wrong)*100))
    
    #-----------------Testing Data Accuracy---------------------#
right=0
wrong=0
TEST =[]
TEST_OUT =  sigmoid(np.dot(W_Now, TestDataSubtract))  
for res in TEST_OUT:
    if (res>0.5):
        TEST.append(1)
    else:
        TEST.append(0)
for i,j in zip(TEST, TestDataAct):
    if i==j:
        right = right +1
    else:
        wrong = wrong +1
print("Testing Accuracy: " + str(right/(right+wrong)*100))


# ## Logistic Regression solution for Feature Addition

# In[34]:


W_Now        = np.zeros(935)
La           = 2
learningRate = 0.05
L_Erms_Val   = []
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

for i in range(0,1600):

    scores = np.dot(W_Now, TrainingAddData)
    prediction  = sigmoid(scores)
    #print ('---------Iteration: ' + str(i) + '--------------')
    error =  prediction - TrainingTarget
    Delta_E_D     = np.dot(TrainingAddData, error)
    #La_Delta_E_W  = np.dot(La,W_Now)
    #Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E_D)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next

#-----------------Training Data Accuracy---------------------#
right=0
wrong=0
TR_TEST=[]
TR_TEST_OUT   =  sigmoid(np.dot(W_Now, TrainingAddData)) 
for res in TR_TEST_OUT:
    if (res>0.5):
        TR_TEST.append(1)
    else:
        TR_TEST.append(0)
for i,j in zip(TR_TEST, TrainingTarget):
    if i==j:
        right = right +1
    else:
        wrong = wrong +1
print("Training Accuracy: " + str(right/(right+wrong)*100))
    
    #-----------------ValidationData Accuracy---------------------# 
right=0
wrong=0
VAL_TEST=[] 
VAL_TEST_OUT  = sigmoid(np.dot(W_Now, ValDataAdd))
for res in VAL_TEST_OUT:
    if (res>0.5):
        VAL_TEST.append(1)
    else:
        VAL_TEST.append(0)
for i,j in zip(VAL_TEST, ValDataAct):
    if i==j:
        right = right +1
    else:
        wrong = wrong +1
print("Validation Accuracy: " + str(right/(right+wrong)*100))
    
    #-----------------TestingData Accuracy---------------------#
right=0
wrong=0
TEST =[]
TEST_OUT =  sigmoid(np.dot(W_Now, TestDataAdd))  
for res in TEST_OUT:
    if (res>0.5):
        TEST.append(1)
    else:
        TEST.append(0)
for i,j in zip(TEST, TestDataAct):
    if i==j:
        right = right +1
    else:
        wrong = wrong +1
print("Testing Accuracy: " + str(right/(right+wrong)*100))


# ## Neural Network Solution for Feature Subtraction

# In[35]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
import numpy as np

TrainingTarget1 = np_utils.to_categorical(np.array(TrainingTarget),2)


# In[36]:


input_size = 472
drop_out = 0.2
first_dense_layer_nodes  = 944
second_dense_layer_nodes = 2

def get_model():
    model = Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('sigmoid')) 
    model.add(Dropout(drop_out)) 
    
    model.add(Dense(first_dense_layer_nodes))
    model.add(Activation('sigmoid')) 
    model.add(Dropout(drop_out)) 
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('sigmoid')) 
    model.summary()
    
    model.compile(optimizer='AdaDelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# In[37]:


model = get_model()


# In[38]:


validation_data_split = 0
num_epochs = 1000
model_batch_size = 64
tb_batch_size = 32
early_patience = 100
tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
history = model.fit(np.transpose(TrainingSubtractData)
                    , TrainingTarget1
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb])


# In[39]:


wrong   = 0
right   = 0
predictedTestLabel = []

for i,j in zip(np.transpose(TrainingSubtractData),TrainingTarget):
    y = model.predict(np.array(i).reshape(-1,472))
    predictedTestLabel.append(y.argmax())
    
    if j == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))
print("Training Accuracy: " + str(right/(right+wrong)*100))

wrong   = 0
right   = 0
predictedTestLabel = []
for i,j in zip(np.transpose(ValDataSubtract),ValDataAct):
    y = model.predict(np.array(i).reshape(-1,472))
    predictedTestLabel.append(y.argmax())
    
    if j == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))
print("Validation Accuracy: " + str(right/(right+wrong)*100))


wrong   = 0
right   = 0
predictedTestLabel = []
for i,j in zip(np.transpose(TestDataSubtract),TestDataAct):
    y = model.predict(np.array(i).reshape(-1,472))
    predictedTestLabel.append(y.argmax())
    
    if j == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))
print("Testing Accuracy: " + str(right/(right+wrong)*100))


# ## Neural Network Solution for Feature Addition

# In[40]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
import numpy as np

TrainingTarget1 = np_utils.to_categorical(np.array(TrainingTarget),2)


# In[41]:


input_size = 935
drop_out = 0.2
first_dense_layer_nodes  = 256
second_dense_layer_nodes = 2

def get_model_add():
    model = Sequential()
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('sigmoid')) 
    model.add(Dropout(drop_out)) 
    
    model.add(Dense(first_dense_layer_nodes))
    model.add(Activation('sigmoid')) 
    model.add(Dropout(drop_out)) 
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('sigmoid')) 
    model.summary()
    
    model.compile(optimizer='AdaDelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# In[42]:


model = get_model_add()


# In[43]:


validation_data_split = 0
num_epochs = 1000
model_batch_size = 128
tb_batch_size = 32
early_patience = 10
tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
history = model.fit(np.transpose(TrainingAddData)
                    , TrainingTarget1
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb])


# In[44]:


wrong   = 0
right   = 0
predictedTestLabel = []

for i,j in zip(np.transpose(TrainingAddData),TrainingTarget):
    y = model.predict(np.array(i).reshape(-1,935))
    predictedTestLabel.append(y.argmax())
    
    if j == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))
print("Training Accuracy: " + str(right/(right+wrong)*100))

wrong   = 0
right   = 0
predictedTestLabel = []
for i,j in zip(np.transpose(ValDataAdd),ValDataAct):
    y = model.predict(np.array(i).reshape(-1,935))
    predictedTestLabel.append(y.argmax())
    
    if j == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))
print("Validation Accuracy: " + str(right/(right+wrong)*100))

wrong   = 0
right   = 0
predictedTestLabel = []

for i,j in zip(np.transpose(TestDataAdd),TestDataAct):
    y = model.predict(np.array(i).reshape(-1,935))
    predictedTestLabel.append(y.argmax())
    
    if j == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))
print("Testing Accuracy: " + str(right/(right+wrong)*100))

