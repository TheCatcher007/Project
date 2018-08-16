from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.text import TextCollection
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import os

def proc_text(raw_line):
    """
        处理每行的文本数据
        返回分词结果
    """
    raw_line = str(raw_line)
    # 全部转为小写
    raw_line = raw_line.lower()

    # 去除 b'...' 或 b"..."
    if raw_line[:2] == 'b\'' or raw_line[:2] == 'b"':
        raw_line = raw_line[2:-1]

    # 分词
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(raw_line)

    # 去除停用词
    meaninful_words = [w for w in tokens if w not in stopwords.words('english')]
    return ' '.join(meaninful_words)

def clean_text(raw_text_df):
    """
        清洗原始文本数据
    """
    cln_text_df = pd.DataFrame()
    cln_text_df['date'] = raw_text_df['Date'].values
    cln_text_df['label'] = raw_text_df['Label'].values
    cln_text_df['text'] = ''

    # 处理25列文本数据，['Top1', ..., 'Top25']
    col_list = ['Top' + str(i) for i in range(1, 26)]

    for i, col in enumerate(col_list):
        raw_text_df[col] = raw_text_df[col].apply(proc_text)      # 结果是三列：日期，label（1，0对应涨跌）和col（清洗分词后的一个个单词文本）
        # 合并列
        cln_text_df['text'] = cln_text_df['text'].str.cat(raw_text_df[col], sep=' ')
        print('已处理{}列.'.format(i + 1))

    return cln_text_df

def split_train_test(data_df):
    """
        分割训练集和测试集
    """
    # 训练集时间范围 2008-08-08 ~ 2014-12-31
    train_text_df = data_df.loc['20080808':'20141231', :]
    # 将时间索引替换为整型索引
    train_text_df.reset_index(drop=True, inplace=True)

    # 测试集时间范围 2015-01-02 ~ 2016-07-01
    test_text_df = data_df.loc['20150102':'20160701', :]
    # 将时间索引替换为整型索引
    test_text_df.reset_index(drop=True, inplace=True)

    return train_text_df, test_text_df

def get_word_list_from_data(text_df):
    """
        将数据集中的单词放入到一个列表中
    """
    word_list = []
    for _, r_data in text_df.iterrows():
        word_list += r_data['text'].split(' ')
    return word_list

def extract_feat_from_data(text_df, text_collection, common_words_freqs):
    """
        特征提取
    """
    # 这里只选择TF-IDF特征作为例子
    # 可考虑使用词频或其他文本特征作为额外的特征

    n_sample = text_df.shape[0]
    n_feat = len(common_words_freqs)    # 200维
    common_words = [word for word, _ in common_words_freqs]

    # 初始化
    X = np.zeros([n_sample, n_feat])
    y = np.zeros(n_sample)

    print('提取特征...')
    for i, r_data in text_df.iterrows():
        if (i + 1) % 100 == 0:
            print('已完成{}个样本的特征提取'.format(i + 1))

        text = r_data['text']      # text 列的所有词

        feat_vec = []
        # 计算出TF-IDF值
        for word in common_words:
            if word in text:     # 语料库里最常见的词选取的200个，所以是 200 个word ，所以对应200维。
                # 如果在高频词中，计算TF-IDF值
                tf_idf_val = text_collection.tf_idf(word, text)
            else:
                tf_idf_val = 0    # 不在就是 0

            feat_vec.append(tf_idf_val)

        # 赋值
        X[i, :] = np.array(feat_vec)     # 1 行，200 列。200维特征。
        y[i] = int(r_data['label'])     # 对应的标签
        # 此处是将 x、y都转换成向量和标签，方便训练。

    return X, y

def get_best_model(model, X_train, y_train, params, cv=5):
    """
        交叉验证获取最优模型
        默认5折交叉验证
    """
    clf = GridSearchCV(model, params, cv=cv, verbose=3)
    clf.fit(X_train, y_train)
    return clf.best_estimator_    #得出最佳参数。

raw_text_csv_file = './dataset/Combined_News_DJIA.csv'
cln_text_csv_file = './cln_text.csv'

def main():
    """
        主函数
    """
    # Step 1: 处理数据集
    print('===Step1: 处理数据集===')

    if not os.path.exists(cln_text_csv_file):
        print('清洗数据...')
        # 读取原始csv文件
        raw_text_df = pd.read_csv(raw_text_csv_file)

        # 清洗原始数据
        cln_text_df = clean_text(raw_text_df)
        # 此时数据已清洗完毕。

        # 保存处理好的文本数据
        cln_text_df.to_csv(cln_text_csv_file, index=None)
        print('完成，并保存结果至', cln_text_csv_file)

    print('================\n')

    # Step 2. 查看整理好的数据集，并选取部分数据作为模型的训练
    print('===Step2. 查看数据集===')
    text_data = pd.read_csv(cln_text_csv_file)
    text_data['date'] = pd.to_datetime(text_data['date'])
    text_data.set_index('date', inplace=True)
    print('各类样本数量：')
    print(text_data.groupby('label').size())

    # Step 3. 分割训练集和测试集
    print('===Step3. 分割训练集合测试集===')
    train_text_df, test_text_df = split_train_test(text_data)      # 按照日期先后顺序分割训练集和测试集
    # 查看训练集测试集基本信息
    print('训练集中各类的数据个数：')
    print(train_text_df.groupby('label').size())
    print('测试集中各类的数据个数：')
    print(test_text_df.groupby('label').size())
    print('================\n')

    # Step 4. 特征提取
    print('===Step4. 文本特征提取===')
    # 计算词频  根据训练集的语料库计算词频
    n_common_words = 200

    # 将训练集中的单词拿出来统计词频
    print('统计词频...')
    all_words_in_train = get_word_list_from_data(train_text_df)     # 传入的参数是训练集数据，将DataFrame中text列的所有单词都放在一个列表中。
    fdisk = nltk.FreqDist(all_words_in_train)    # 计算词频
    common_words_freqs = fdisk.most_common(n_common_words)   # 计算出最常见的200个词的频率，即是词和对应的频率。
    print('出现最多的{}个词是：'.format(n_common_words))
    for word, count in common_words_freqs:
        print('{}: {}次'.format(word, count))
    print()

    # 在训练集上提取特征   特征选好就可以训练了
    text_collection = TextCollection(train_text_df['text'].values.tolist())         # 对象构造
    print('训练样本提取特征...')
    #因为计算出最常见的200个词的频率，所以是200维的特征向量
    train_X, train_y = extract_feat_from_data(train_text_df, text_collection, common_words_freqs)
    print('完成')       # 得出的是每个样本计算的 200 维的 tf-idf值和标签。
    print()

    print('测试样本提取特征...')
    test_X, test_y = extract_feat_from_data(test_text_df, text_collection, common_words_freqs)
    print('完成')
    print('================\n')

    # 特征处理
    # 特征范围归一化    将 x 值标准化处理一下
    scaler = StandardScaler()
    tr_feat_scaled = scaler.fit_transform(train_X)
    te_feat_scaled = scaler.transform(test_X)

    # 3.6 特征选择    减少部分特征
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))     # 选取 80% 特征，原理是去掉反差较小的特征。
    tr_feat_scaled_sel = sel.fit_transform(tr_feat_scaled)
    te_feat_scaled_sel = sel.transform(te_feat_scaled)

    # 3.7 PCA降维操作
    pca = PCA(n_components=0.95)  # 保留95%贡献率的特征向量，剩余5%贡献的向量去掉。
    tr_feat_scaled_sel_pca = pca.fit_transform(tr_feat_scaled_sel)
    te_feat_scaled_sel_pca = pca.transform(te_feat_scaled_sel)
    print('特征处理结束')
    print('处理后每个样本特征维度：', tr_feat_scaled_sel_pca.shape[1])
    # 特征自选取后，已全部处理完毕。接下来可以训练了。

    # Step 5. 训练模型
    models = []
    print('===Step5. 训练模型===')
    print('1. 朴素贝叶斯模型：')
    gnb_model = GaussianNB()
    gnb_model.fit(tr_feat_scaled_sel_pca, train_y)
    models.append(['朴素贝叶斯', gnb_model])
    print('完成')
    print()

    print('2. 逻辑回归：')
    lr_param_grid = [
        {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100]}
    ]
    lr_model = LogisticRegression()
    best_lr_model = get_best_model(lr_model,       # get_best_model()用来选取最佳参数。
                                   tr_feat_scaled_sel_pca, train_y,
                                   lr_param_grid, cv=3)
    models.append(['逻辑回归', best_lr_model])
    print('完成')
    print()

    print('3. 支持向量机：')
    svm_param_grid = [
        {'C': [1e-2, 1e-1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]
    svm_model = svm.SVC(probability=True)
    best_svm_model = get_best_model(svm_model,
                                    tr_feat_scaled_sel_pca, train_y,
                                    svm_param_grid, cv=3)
    models.append(['支持向量机', best_svm_model])
    print('完成')
    print()

    print('4. 随机森林：')
    rf_param_grid = [
        {'n_estimators': [10, 50, 100, 150, 200]}
    ]

    rf_model = RandomForestClassifier()
    best_rf_model = get_best_model(rf_model,
                                   tr_feat_scaled_sel_pca, train_y,
                                    rf_param_grid, cv=3)
    rf_model.fit(tr_feat_scaled_sel_pca, train_y)
    models.append(['随机森林', best_rf_model])
    print('完成')
    print()
    # 四个模型全部选好参数并训练完毕。

    # Step 6. 测试模型
    print('===Step6. 测试模型===')
    for i, model in enumerate(models):
        print('{}-{}'.format(i + 1, model[0]))
        # 输出准确率
        print('准确率：', accuracy_score(test_y, model[1].predict(te_feat_scaled_sel_pca)))   # 利用得分判定模型好坏。
        print('AUC：', roc_auc_score(test_y, model[1].predict_proba(te_feat_scaled_sel_pca)[:, 0]))     # 利用AUC 曲线下面积判断模型好坏。
        # 输出混淆矩阵
        print('混淆矩阵')
        print(confusion_matrix(test_y, model[1].predict(te_feat_scaled_sel_pca)))
        print()

if __name__ == '__main__':
    main()
