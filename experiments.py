import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（如果使用中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data():
    """
    加载英文数据集
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

def experiment_different_vectorizations():
    """
    实验：比较不同的文本向量化方法
    """
    print("\n" + "="*80)
    print("实验：不同文本向量化方法比较")
    print("="*80)
    
    X_train, X_test, y_train, y_test, target_names = load_data()
    
    results = {}
    
    # 1. TF-IDF（不同的ngram_range）
    print("\n1. TF-IDF (n-gram范围比较)")
    print("-"*80)
    
    ngram_configs = [
        (1, 1),  # 仅单词
        (1, 2),  # 单词和双词
        (2, 2),  # 仅双词
    ]
    
    for ngram in ngram_configs:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english',
                                     ngram_range=ngram, min_df=2, max_df=0.95)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # 使用逻辑回归作为基准分类器
        clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        key = f"TF-IDF (ngram={ngram})"
        results[key] = accuracy
        print(f"{key}: 准确率 = {accuracy:.4f}, 特征维度 = {X_train_vec.shape[1]}")
    
    # 2. 不同的max_features
    print("\n2. TF-IDF (不同特征数量)")
    print("-"*80)
    
    max_features_list = [1000, 3000, 5000, 10000]
    
    for max_feat in max_features_list:
        vectorizer = TfidfVectorizer(max_features=max_feat, stop_words='english',
                                     ngram_range=(1, 2), min_df=2, max_df=0.95)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        key = f"TF-IDF (max_features={max_feat})"
        results[key] = accuracy
        print(f"{key}: 准确率 = {accuracy:.4f}, 特征维度 = {X_train_vec.shape[1]}")
    
    # 3. 是否使用降维
    print("\n3. TF-IDF + 降维")
    print("-"*80)
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english',
                                 ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 不降维
    clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    accuracy_no_pca = accuracy_score(y_test, y_pred)
    results["TF-IDF (无降维)"] = accuracy_no_pca
    print(f"无降维: 准确率 = {accuracy_no_pca:.4f}, 特征维度 = {X_train_vec.shape[1]}")
    
    # PCA降维
    for n_components in [100, 300, 500]:
        pca = TruncatedSVD(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_vec)
        X_test_pca = pca.transform(X_test_vec)
        
        clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)
        accuracy_pca = accuracy_score(y_test, y_pred)
        
        key = f"TF-IDF + PCA({n_components})"
        results[key] = accuracy_pca
        print(f"PCA降维到{n_components}维: 准确率 = {accuracy_pca:.4f}")
    
    return results

def experiment_parameter_tuning():
    """
    实验：参数调优
    """
    print("\n" + "="*80)
    print("实验：参数调优")
    print("="*80)
    
    X_train, X_test, y_train, y_test, target_names = load_data()
    
    # 使用TF-IDF向量化
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english',
                                 ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    results = {}
    
    # 1. SVM参数调优
    print("\n1. SVM参数调优")
    print("-"*80)
    
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    
    # 使用较小的数据子集进行网格搜索（因为计算量大）
    X_train_small = X_train_vec[:1000]
    y_train_small = y_train[:1000]
    
    svm = SVC(random_state=42, probability=True)
    grid_search = GridSearchCV(svm, param_grid_svm, cv=3, scoring='accuracy', 
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train_small, y_train_small)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
    
    # 在完整训练集上训练最佳模型
    best_svm = grid_search.best_estimator_
    best_svm.fit(X_train_vec, y_train)
    y_pred = best_svm.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    results['SVM (调优后)'] = accuracy
    print(f"测试集准确率: {accuracy:.4f}")
    
    # 2. 随机森林参数调优
    print("\n2. 随机森林参数调优")
    print("-"*80)
    
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='accuracy',
                                  n_jobs=-1, verbose=1)
    grid_search_rf.fit(X_train_small, y_train_small)
    
    print(f"最佳参数: {grid_search_rf.best_params_}")
    print(f"最佳交叉验证得分: {grid_search_rf.best_score_:.4f}")
    
    best_rf = grid_search_rf.best_estimator_
    best_rf.fit(X_train_vec, y_train)
    y_pred = best_rf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    results['随机森林 (调优后)'] = accuracy
    print(f"测试集准确率: {accuracy:.4f}")
    
    return results

def experiment_clustering_parameters():
    """
    实验：聚类算法参数调优
    """
    print("\n" + "="*80)
    print("实验：聚类算法参数调优")
    print("="*80)
    
    X_train, X_test, y_train, y_test, target_names = load_data()
    X_all = X_train + X_test
    y_all = list(y_train) + list(y_test)
    
    # TF-IDF向量化
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english',
                                 ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_tfidf = vectorizer.fit_transform(X_all)
    
    # 降维用于DBSCAN
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X_tfidf.toarray())
    
    results = {}
    
    # 1. K-means不同聚类数
    print("\n1. K-means (不同聚类数)")
    print("-"*80)
    
    n_clusters_list = [2, 3, 4, 5, 6]
    
    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_tfidf)
        
        ari = adjusted_rand_score(y_all, labels)
        sil = silhouette_score(X_tfidf, labels)
        
        key = f"K-means (k={n_clusters})"
        results[key] = {'ARI': ari, 'Silhouette': sil}
        print(f"{key}: ARI = {ari:.4f}, Silhouette = {sil:.4f}")
    
    # 2. DBSCAN不同参数
    print("\n2. DBSCAN (不同参数)")
    print("-"*80)
    
    eps_list = [0.3, 0.5, 0.7, 1.0]
    min_samples_list = [3, 5, 7]
    
    for eps in eps_list:
        for min_samples in min_samples_list:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = dbscan.fit_predict(X_pca)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1:
                mask = labels != -1
                if mask.sum() > 0:
                    ari = adjusted_rand_score(np.array(y_all)[mask], labels[mask])
                    sil = silhouette_score(X_pca[mask], labels[mask])
                    
                    key = f"DBSCAN (eps={eps}, min_samples={min_samples})"
                    results[key] = {
                        'ARI': ari, 
                        'Silhouette': sil,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise
                    }
                    print(f"{key}: ARI = {ari:.4f}, Silhouette = {sil:.4f}, "
                          f"聚类数 = {n_clusters}, 噪声点 = {n_noise}")
    
    return results

def plot_results(classification_results, clustering_results):
    """
    可视化实验结果
    """
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 分类结果可视化
    if classification_results:
        df_class = pd.DataFrame(list(classification_results.items()), 
                               columns=['Method', 'Accuracy'])
        df_class = df_class.sort_values('Accuracy', ascending=False)
        
        axes[0].barh(df_class['Method'], df_class['Accuracy'], color='steelblue')
        axes[0].set_xlabel('准确率 (Accuracy)', fontsize=12)
        axes[0].set_title('文本分类结果比较', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
    
    # 2. 聚类结果可视化
    if clustering_results:
        # 提取ARI分数
        methods = []
        ari_scores = []
        sil_scores = []
        
        for key, value in clustering_results.items():
            if isinstance(value, dict) and 'ARI' in value:
                methods.append(key)
                ari_scores.append(value['ARI'])
                sil_scores.append(value.get('Silhouette', 0))
        
        if methods:
            x = np.arange(len(methods))
            width = 0.35
            
            axes[1].bar(x - width/2, ari_scores, width, label='ARI', color='coral')
            axes[1].bar(x + width/2, sil_scores, width, label='Silhouette', color='lightblue')
            axes[1].set_xlabel('方法', fontsize=12)
            axes[1].set_ylabel('分数', fontsize=12)
            axes[1].set_title('文本聚类结果比较', fontsize=14, fontweight='bold')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(methods, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_results.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存为 experiment_results.png")
    # plt.show()

def main():
    """
    主函数：运行所有扩展实验
    """
    print("="*80)
    print("文本数据管理与分析 - 期末作业")
    print("扩展实验：不同文本表示方法、参数调优")
    print("="*80)
    
    # 实验1：不同向量化方法
    vec_results = experiment_different_vectorizations()
    
    # 实验2：参数调优（耗时较长，可选择运行）
    # param_results = experiment_parameter_tuning()
    
    # 实验3：聚类参数调优
    cluster_results = experiment_clustering_parameters()
    
    # 可视化结果
    plot_results(vec_results, cluster_results)
    
    # 打印总结
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    print("\n不同向量化方法结果:")
    print("-"*80)
    for method, acc in sorted(vec_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method}: {acc:.4f}")

if __name__ == "__main__":
    main()
