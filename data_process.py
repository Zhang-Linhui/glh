import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def eigenvalue_processing(file_path):
    # 步骤1: 读取CSV文件
    df = pd.read_csv(file_path)
    #抛弃第一列
    df = df.drop(columns=['customerID'])
    y = df['Churn'].values
    # 抛弃标签列
    df= df.drop(columns=['Churn'])
    # 步骤2: 识别数值特征
    # 对于已经转换为数值类型的列，选择填充NaN值的策略
    # 这里以填充众数为例
    for column in df.columns:  # 跳过第0列
        # 将列转换为数值类型，无法转换的设置为NaN
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column].fillna(df[column].mode()[0], inplace=True)

    numerical_features_to_scale = df.columns 
    # 步骤3: 对特征进行预处理
    # 对最后两列进行归一化，其余数值特征不变
    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), numerical_features_to_scale)
        ])

    X = preprocessor.fit_transform(df)  
    # 步骤4: 保存特征和标签 npy
    np.save('features.npy', X)
    np.save('labels.npy', y)
def adjacency_matrix_processing(file_path):
    # 步骤1: 读取CSV文件
    df = pd.read_csv(file_path)
    #步骤2: 将每一行读取为数字，然后其值减1(csv中编号从1开始，实际从0开始，所以减1)
    df = df.apply(lambda x: x-1)
    #打印前5行
    print(df.head())
    # 步骤3: 保存
    df.to_csv('adjacency_matrix.csv', index=False)

def extract_edge_data(file_path):
    # 步骤1: 读取CSV文件
    df = pd.read_csv(file_path)
    # 步骤2: 提取每行的三个值
    samples = df[['Sample1', 'Sample2']].values
    distances = df['Distance'].values
    # 步骤3: 将距离值转为边的权重
    weights = []
    for distance in distances:
        weights.append(1-distance)
    # 步骤4: 保存为npy
    np.save('samples.npy', samples)
if __name__ == '__main__':
    eigenvalue_processing('raw/特征值2.csv')
    # extract_edge_data('raw/边2.csv')
