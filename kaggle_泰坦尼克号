import pandas as pd
import numpy as np
from pandas import Series,DataFrame

data_train = pd.read_csv('train.csv')
data_train.head()  
data_train.columns  
data_train.info() 
data_train.describe()  

import matplotlib.pyplot as plt
# 解决matplotlib显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

fig = plt.figure(figsize=(20,10))       # 设置图纸，通过figsize设置图纸大小。
fig.set(alpha=0.2)                      # 设置图表颜色alpha参数

# 在一张大图里分列几个小图。分为两行三列，(0,0)代表第一行第一列的位置。
plt.subplot2grid((2,3),(0,0))   
# kind='bar'代表条形图。Survived.value_counts()表示每个类别的数目。横轴表类别，纵轴是数目，适合用条形图表示。
data_train.Survived.value_counts().plot(kind='bar')
plt.title('获救情况(1为获救)')
plt.ylabel('人数')

plt.subplot2grid((2,3),(0,1))
# Pclass也是类别型数据，也用value_counts方法表示出类别和对应数量，用条形图 kind='bar'表示出来。
data_train.Pclass.value_counts().plot(kind='bar')      
plt.title('乘客舱位等级分布')
plt.ylabel('人数')

plt.subplot2grid((2,3),(0,2))
# 因为年龄有很多种，18岁、23岁、35岁等，所以不方便用value_counts()加条形图表示。可以用散点图表示出来。
plt.scatter(data_train.Survived,data_train.Age)
plt.title('按年龄看获救分布(1为获救)')
plt.ylabel('年龄')

plt.subplot2grid((2,3),(1,0),colspan=2)      # colspan=2 代表该图占两张图的宽度。
#查看三个舱位乘客的年龄分布情况,如果用条形图，横轴会出现多个年龄，纵轴代表对应人数，不合适。通过kind='kde'密度图可以直观看出各船舱的年龄特点。
data_train.Age[data_train.Pclass==1].plot(kind='kde')
data_train.Age[data_train.Pclass==2].plot(kind='kde')
data_train.Age[data_train.Pclass==3].plot(kind='kde')
plt.title('各舱位乘客的年龄分布')
plt.xlabel('年龄')
plt.ylabel('密度')
plt.legend(('头等舱','二等舱','三等舱'),loc='best')

plt.subplot2grid((2,3),(1,2))
# 因为Embarked只有三个类别 S、C、Q，适合用value_counts()和条形图表示。分布显示出各个各港口登陆的人数。
data_train.Embarked.value_counts().plot(kind='bar')
plt.title('各登船口岸上船人数')
plt.ylabel('人数')

fig = plt.figure()
fig.set(alpha=0.2)

# 找出未获救乘客的船舱等级，并用value_counts()表示出各等级即对应的人数。
survived_0 = data_train.Pclass[data_train.Survived==0].value_counts()
# 找出获救乘客的船舱等级，并用value_counts()表示出各等级即对应的人数。
survived_1 = data_train.Pclass[data_train.Survived==1].value_counts()
# 构建出一个DataFrame，并做一张图显示出来。
df = pd.DataFrame({'未获救':survived_0,'获救':survived_1})
df.plot(kind='bar',stacked=True)       # stacked=True将数据累积起来，获救与未获救将其分为两类。横轴表类别，纵轴表数量，所以用条形图 bar 。
plt.title('各舱位等级乘客获救情况')
plt.xlabel('乘客舱位等级')
plt.ylabel('人数')

# 与处理乘客舱位等级方式一致。
fig = plt.figure()
fig.set(alpha=0.2)

survived_0 = data_train.Embarked[data_train.Survived==0].value_counts()
survived_1 = data_train.Embarked[data_train.Survived==1].value_counts()
df = pd.DataFrame({'未获救':survived_0,'获救':survived_1})
df.plot(kind='bar',stacked=True)
plt.title('各登陆港口获救情况')
plt.xlabel('登陆港口')
plt.ylabel('人数')

# 看不同性别的获救情况
fig = plt.figure()
fig.set(alpha=0.2)

survived_0 = data_train.Sex[data_train.Survived==0].value_counts()
survived_1 = data_train.Sex[data_train.Survived==1].value_counts()
df = pd.DataFrame({'未获救':survived_0,'获救':survived_1})
df.plot(kind='bar',stacked=True)
plt.title('从性别看获救情况')
plt.xlabel('性别')
plt.ylabel('人数')

fig = plt.figure(figsize=(20,10))
fig.set(alpha=0.65)
plt.title('根据船舱和性别等级看获救情况')

ax1 = fig.add_subplot(141)         # 即创建一张图纸，一行四列，此图是第一列。比较一下与上面创建图的不同之处。
data_train.Survived[data_train.Sex=='female'][data_train.Pclass!=3].value_counts().plot(kind='bar',label='female highclass',color='r')
ax1.set_xticklabels(['获救','未获救'],rotation=0)       # rotation=0 表示 x 轴标签转换为水平方向。
ax1.legend(['女性/高级舱'],loc='best')

ax2 = fig.add_subplot(142,sharey=ax1)    # sharey=ax1 表示和图一共用 y 轴
data_train.Survived[data_train.Sex=='female'][data_train.Pclass==3].value_counts().plot(kind='bar',label='famale low class',color='pink')
ax2.set_xticklabels(['获救','未获救'],rotation=0)
ax2.legend(['女性/低级舱'],loc='best')

ax3 = fig.add_subplot(143,sharey=ax1)
data_train.Survived[data_train.Sex=='male'][data_train.Pclass!=3].value_counts().plot(kind='bar',label='male highclass',color='w')
ax3.set_xticklabels(['未获救','获救'],rotation=0)
ax3.legend(['男性/高级舱'],loc='best')

ax4 = fig.add_subplot(144,sharey=ax1)
data_train.Survived[data_train.Sex=='male'][data_train.Pclass==3].value_counts().plot(kind='bar',label='男性/低级舱',color='steelblue')
ax4.set_xticklabels(['未获救','获救'],rotation=0)
ax4.legend(['男性/低级舱'],loc='best')

plt.show()

# 观察船上兄弟姐妹在一起的个数与获救的情况，看结果发现无明显关系。
g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df

# 观察船上父母小孩在一起的个数与获救的情况，看结果发现无明显关系。
g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df

# ticket是传票编号，是unique的，和最后结果没有什么关系，故不纳入考虑的特征范畴。
# Cabin只有204个乘客有值，查看一下分布。
data_train['Cabin'].value_counts()

fig = plt.figure()
fig.set(alpha=0.2)

survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({'有':survived_cabin,'无':survived_nocabin}).transpose()  # 加transpose()之前 有、无 是标签，0、1是 X 轴，加了后调换一下。
df.plot(kind='bar',stacked=True)
plt.title('有无Cabin值乘客的获救情况')
plt.xlabel('Cabin值有无')
plt.ylabel('人数')

plt.show()

def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df

data_train = set_cabin_type(data_train)
data_train.head()

from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进 RandomForestRegressor 中
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    
    # 将乘客分为已知年龄和未知年龄两部分。已知的特征列和标签列用来训练，未知的特征列用来测试，得出的结果就是预测的年龄。
    known_age = age_df[age_df.Age.notnull()].as_matrix()     # 将其转换为 array 形式，因为该模型接受的 X,y要求该形式。
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    
    # y 即目标年龄
    y = known_age[:,0]
    
    # X 即特征属性值
    X = known_age[:,1:]
    
    # 用 RandomForestRegressor 进行训练
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)
    
    # 训练完毕，对缺失年龄值进行预测
    predictAges = rfr.predict(unknown_age[:,1:])
    
    # 用预测得到的值填补缺失值
    df.loc[(df.Age.isnull()),'Age'] = predictAges
    
    return df,rfr

data_train,rfr = set_missing_ages(data_train)
data_train.head(10)

# 因为逻辑回归建模时，需要输入的特征都是数值型特征
# 我们先对类目型的特征离散/因子化
# 以Cabin为例，原本一个属性维度，因为其取值可以是['yes','no']，而将其平展开为'Cabin_yes','Cabin_no'两个属性
# 原本Cabin取值为yes的，在此处的'Cabin_yes'下取值为1，在'Cabin_no'下取值为0
# 原本Cabin取值为no的，在此处的'Cabin_yes'下取值为0，在'Cabin_no'下取值为1
# 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上，如下所示。

dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix='Pclass')

# 将需要的列连接起来,concat()连接的是DataFrame
df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)

# 删除掉不需要的列
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)
df

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1))
# 转换完毕

# 使用正则选出需要的列，直接挑选出来也行。
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# 此时，已经可以开始训练模型，进行预测了。选择LogisticRegression建模。
from sklearn import linear_model

# y 即 Survived 结果
y = train_np[:,0]

# X 即特征属性值
X = train_np[:,1:]

# 用 LogisticRegression模型开始训练
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)

from sklearn.learning_curve import learning_curve

train_sizes,train_scores,test_scores = learning_curve(clf,X,y,cv=10,train_sizes=np.linspace(0.05,1.0,20),verbose=0)
# 此处 clf 即是训练时所用的模型；
# X,y 即是训练时的 X，y值。
# cv:做cross-validation的时候，数据分成的份数cv，其中一份作为验证集，其余(cv-1)份作为training，这样就可以重复训练 cv 次了。
# train_sizes:以train_sizes=np.linspace(0.05,1.0,20)为例，将训练集均匀切分为20份，分别以 1/20、2/20、3/20...作为(训练集加验证集)交叉验证。

train_scores_mean = np.mean(train_scores,axis=1)
test_scores_mean = np.mean(test_scores,axis=1)

plt.figure()
plt.title('继续努力')
plt.xlabel('训练样本数')
plt.ylabel('得分')

plt.plot(train_sizes,train_scores_mean,'o-',color='b',label='训练集上得分')
plt.plot(train_sizes,test_scores_mean,'o-',color='r',label='交叉验证集上得分')
plt.legend(loc='best')
plt.show()

clf.fit(X,y)
clf

data_test = pd.read_csv('test.csv')
data_test.head()
data_test.info()
data_test.loc[(data_test.Fare.isnull()),'Fare'] = 0

# 按照上面对训练集的数据处理步骤，进行如下数据处理。
data_test = set_cabin_type(data_test)       # 对 Cabin 列处理

tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]   # 选出需要的几列
null_age = tmp_df[tmp_df.Age.isnull()].as_matrix()
X = null_age[:,1:]       # 找出了特征值，直接运用上面对 Age 训练过的模型进行预测。
predictAges = rfr.predict(X)    # 得出了预测值，然后将其填充在空的 Age 值上。
data_test.loc[(data_test.Age.isnull()),'Age'] = predictAges

dummies_Cabin = pd.get_dummies(data_test['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Pclass = pd.get_dummies(data_test['Pclass'],prefix='Pclass')
dummies_Sex = pd.get_dummies(data_test['Sex'],prefix='Sex')

df_test = pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis=1)
df_test.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)

# 对 Age 和 Fare 列进行 scaling 处理
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1))
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1))

# 选择需要的列
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')    # 测试集里是没有 Survivved 列的
test_np = test.as_matrix()

# 对结果进行预测
predictions = clf.predict(test_np)

# 显示出结果
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result

pd.DataFrame({'columns':list(train_df.columns)[1:],'coef':list(clf.coef_.T)})
