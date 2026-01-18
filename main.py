import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os

def load_english_data():
    """
    加载英文数据集（20 Newsgroups）
    """
    print("正在加载20 Newsgroups数据集...")
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, 
                                         remove=('headers', 'footers', 'quotes'), 
                                         shuffle=True, random_state=42)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                        remove=('headers', 'footers', 'quotes'),
                                        shuffle=True, random_state=42)
    
    X_train = newsgroups_train.data
    y_train = newsgroups_train.target
    X_test = newsgroups_test.data
    y_test = newsgroups_test.target
    
    return X_train, X_test, y_train, y_test, newsgroups_train.target_names

def experiment_text_classification():
    """
    实验1：文本分类（传统机器学习）
    比较不同的机器学习算法：随机森林、SVM、逻辑回归、朴素贝叶斯
    """
    print("\n" + "="*80)
    print("实验1：文本分类（传统机器学习）")
    print("="*80)
    
    # 加载数据
    X_train, X_test, y_train, y_test, target_names = load_english_data()
    
    # 使用TF-IDF进行文本表示
    print("\n正在使用TF-IDF向量化文本...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', 
                                 ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"训练集大小: {X_train_tfidf.shape}")
    print(f"测试集大小: {X_test_tfidf.shape}")
    
    # 定义分类器
    classifiers = {
        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='linear', random_state=42, probability=True),
        '逻辑回归': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        '朴素贝叶斯': MultinomialNB(alpha=1.0)
    }
    
    results = {}
    
    print("\n" + "-"*80)
    print("训练和评估分类器...")
    print("-"*80)
    
    for name, clf in classifiers.items():
        print(f"\n正在训练 {name}...")
        clf.fit(X_train_tfidf, y_train)
        
        # 预测
        y_pred = clf.predict(X_test_tfidf)
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classifier': clf
        }
        
        print(f"{name} 准确率: {accuracy:.4f}")
        print(f"\n{name} 详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=target_names))
    
    return results, X_train_tfidf, X_test_tfidf, y_train, y_test, target_names

def experiment_text_clustering():
    """
    实验2：文本聚类
    比较K-means和DBSCAN聚类算法
    """
    print("\n" + "="*80)
    print("实验2：文本聚类")
    print("="*80)
    
    # 加载数据
    X_train, X_test, y_train, y_test, target_names = load_english_data()
    
    # 合并训练集和测试集用于聚类
    X_all = X_train + X_test
    y_all = list(y_train) + list(y_test)
    
    # 使用TF-IDF进行文本表示
    print("\n正在使用TF-IDF向量化文本...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english',
                                 ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_tfidf = vectorizer.fit_transform(X_all)
    
    print(f"文档数量: {X_tfidf.shape[0]}")
    print(f"特征维度: {X_tfidf.shape[1]}")
    
    # 进行降维以便可视化（可选）
    print("\n正在进行PCA降维（用于某些聚类算法）...")
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X_tfidf.toarray())
    
    # K-means聚类
    print("\n" + "-"*80)
    print("K-means 聚类")
    print("-"*80)
    n_clusters = len(target_names)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_tfidf)
    
    # 评估K-means
    ari_kmeans = adjusted_rand_score(y_all, kmeans_labels)
    sil_kmeans = silhouette_score(X_tfidf, kmeans_labels)
    
    print(f"K-means 聚类数量: {n_clusters}")
    print(f"调整兰德指数 (ARI): {ari_kmeans:.4f}")
    print(f"轮廓系数 (Silhouette Score): {sil_kmeans:.4f}")
    
    # DBSCAN聚类
    print("\n" + "-"*80)
    print("DBSCAN 聚类")
    print("-"*80)
    
    # 由于TF-IDF矩阵稀疏，使用PCA降维后的数据进行DBSCAN
    # DBSCAN需要计算距离矩阵，对于大数据集可能很慢
    # 这里使用PCA降维后的数据进行聚类
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    # 转换为密集矩阵用于DBSCAN
    X_dense = X_pca
    dbscan_labels = dbscan.fit_predict(X_dense)
    
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"DBSCAN 发现的聚类数量: {n_clusters_dbscan}")
    print(f"噪声点数量: {n_noise}")
    
    # 评估DBSCAN（排除噪声点）
    if n_clusters_dbscan > 1:
        mask = dbscan_labels != -1
        if mask.sum() > 0:
            ari_dbscan = adjusted_rand_score(np.array(y_all)[mask], dbscan_labels[mask])
            sil_dbscan = silhouette_score(X_dense[mask], dbscan_labels[mask])
            print(f"调整兰德指数 (ARI) [排除噪声]: {ari_dbscan:.4f}")
            print(f"轮廓系数 (Silhouette Score) [排除噪声]: {sil_dbscan:.4f}")
        else:
            ari_dbscan = 0
            sil_dbscan = 0
    else:
        ari_dbscan = 0
        sil_dbscan = 0
    
    results = {
        'K-means': {
            'labels': kmeans_labels,
            'ARI': ari_kmeans,
            'Silhouette': sil_kmeans,
            'n_clusters': n_clusters
        },
        'DBSCAN': {
            'labels': dbscan_labels,
            'ARI': ari_dbscan,
            'Silhouette': sil_dbscan,
            'n_clusters': n_clusters_dbscan,
            'n_noise': n_noise
        }
    }
    
    return results, X_tfidf, y_all, target_names

def main():
    """
    主函数：运行所有实验
    """
    print("="*80)
    print("文本数据管理与分析 - 期末作业")
    print("实验内容：文本分类、文本聚类和深度神经网络")
    print("="*80)
    
    # 实验1：文本分类（传统机器学习）
    classification_results, X_train_tfidf, X_test_tfidf, y_train, y_test, target_names = experiment_text_classification()
    
    # 实验2：文本聚类
    clustering_results, X_tfidf, y_all, target_names = experiment_text_clustering()
    
    # 输出总结
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    print("\n文本分类结果汇总（传统机器学习）:")
    print("-"*80)
    for name, result in classification_results.items():
        print(f"{name}: 准确率 = {result['accuracy']:.4f}")
    
    print("\n文本聚类结果汇总:")
    print("-"*80)
    for name, result in clustering_results.items():
        print(f"{name}:")
        print(f"  聚类数量: {result['n_clusters']}")
        print(f"  调整兰德指数 (ARI): {result['ARI']:.4f}")
        print(f"  轮廓系数: {result['Silhouette']:.4f}")
        if 'n_noise' in result:
            print(f"  噪声点数量: {result['n_noise']}")

if __name__ == "__main__":
    main()
