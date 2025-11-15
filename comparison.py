import time
import json
import os
import sys

# 嘗試從 traditional_methods.py 和 modern_methods.py 導入函數
try:
    # 導入 traditional_methods.py 中的數據和函數
    from traditional_methods import (
        DOCUMENTS, TEST_TEXTS, ARTICLE_TEXT, sklearn_tfidf_similarity,
        RuleBasedSentimentClassifier, TopicClassifier, StatisticalSummarizer
    )
    # 導入 modern_methods.py 中的 API_KEY 和函數
    from modern_methods import (
        API_KEY, ai_similarity, ai_classify, ai_summarize
    )
except ImportError as e:
    print(f"導入錯誤: 請確保 traditional_methods.py 和 modern_methods.py 在當前目錄中。錯誤: {e}")
    sys.exit(1)

# 確保結果目錄存在
OUTPUT_DIR = 'results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def collect_performance_metrics():
    """執行所有任務並收集處理時間 (C-1)"""
    print("--- 開始收集性能指標 (Part C-1) ---")

    performance_metrics = {}

    # 實例化傳統分類器和摘要器
    rule_classifier = RuleBasedSentimentClassifier()
    topic_classifier = TopicClassifier()
    summarizer = StatisticalSummarizer()

    # --- 相似度計算比較 ---
    print("1. 相似度計算...")
    start_time_tfidf = time.time()
    sklearn_tfidf_similarity(DOCUMENTS)
    time_tfidf = time.time() - start_time_tfidf
    performance_metrics['similarity_tfidf_time'] = f"{time_tfidf:.4f}"

    if API_KEY != "YOUR_API_KEY_HERE":
        start_time_ai_similarity = time.time()
        # 執行一次相似度評估
        ai_similarity(DOCUMENTS[0], DOCUMENTS[1], API_KEY)
        time_ai_similarity = time.time() - start_time_ai_similarity
        performance_metrics['similarity_ai_time'] = f"{time_ai_similarity:.4f}"
    else:
        performance_metrics['similarity_ai_time'] = "N/A (API Key missing)"

    # --- 文本分類比較 ---
    print("2. 文本分類...")
    # 傳統方法（規則分類器）
    start_time_rule_based = time.time()
    for text in TEST_TEXTS:
        rule_classifier.classify(text)
        topic_classifier.classify(text)
    time_rule_based = time.time() - start_time_rule_based
    performance_metrics['classification_rule_time'] = f"{time_rule_based:.4f}"

    # AI 方法
    if API_KEY != "YOUR_API_KEY_HERE":
        start_time_ai_classify = time.time()
        for text in TEST_TEXTS:
            ai_classify(text, API_KEY)
        time_ai_classify = time.time() - start_time_ai_classify
        performance_metrics['classification_ai_time'] = f"{time_ai_classify:.4f}"
    else:
        performance_metrics['classification_ai_time'] = "N/A (API Key missing)"

    # --- 自動摘要比較 ---
    print("3. 自動摘要...")
    # 傳統方法（統計摘要）
    start_time_statistical_summary = time.time()
    summarizer.summarize(ARTICLE_TEXT, ratio=0.3)
    time_statistical_summary = time.time() - start_time_statistical_summary
    performance_metrics['summary_statistical_time'] = f"{time_statistical_summary:.4f}"

    # AI 方法
    if API_KEY != "YOUR_API_KEY_HERE":
        start_time_ai_summary = time.time()
        ai_summarize(ARTICLE_TEXT, 100, API_KEY) # max_length 100
        time_ai_summary = time.time() - start_time_ai_summary
        performance_metrics['summary_ai_time'] = f"{time_ai_summary:.4f}"
    else:
        performance_metrics['summary_ai_time'] = "N/A (API Key missing)"

    print("\n性能指標收集完成。")
    print(json.dumps(performance_metrics, indent=4, ensure_ascii=False))

    # 將結果保存到 performance_metrics.json (作業要求)
    with open(os.path.join(OUTPUT_DIR, "performance_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(performance_metrics, f, indent=4, ensure_ascii=False)

    print(f"結果已保存到 {OUTPUT_DIR}/performance_metrics.json")
    return performance_metrics

def generate_comparison_report(metrics):
    """根據收集的數據生成用於報告的表格內容 (C-1)"""

    report_content = (
        "評估指標 | 傳統方法 (TF-IDF/規則/統計) | 現代方法 (GPT-4o)\n"
        "---|---|---\n"
        "**相似度計算** | | \n"
        f"處理時間 | {metrics.get('similarity_tfidf_time', '?')} 秒 | {metrics.get('similarity_ai_time', '?')} 秒\n"
        f"準確率 | ?% (需手動評估) | ?% (需手動評估)\n"
        f"成本 | $0 (本地運算) | $? (需根據 Token 估算)\n"
        "**文本分類** | | \n"
        f"處理時間 | {metrics.get('classification_rule_time', '?')} 秒 | {metrics.get('classification_ai_time', '?')} 秒\n"
        f"準確率 | ?% (需手動評估) | ?% (需手動評估)\n"
        "支援類別數 | 有限 (需手動定義) | 無限 (由模型決定)\n"
        "**自動摘要** | | \n"
        f"處理時間 | {metrics.get('summary_statistical_time', '?')} 秒 | {metrics.get('summary_ai_time', '?')} 秒\n"
        f"資訊保留度 | ?% (需手動評估) | ?% (需手動評估)\n"
        f"語句通順度 | ?分 (需手動評分) | ?分 (需手動評分)\n"
        "長度控制 | 困難 (基於句子數) | 容易 (通過 Prompt 控制)\n"
    )

    # 將結果保存到 results/summary_comparison.txt (用於輔助報告)
    with open(os.path.join(OUTPUT_DIR, "summary_comparison.txt"), "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"報告草稿已保存到 {OUTPUT_DIR}/summary_comparison.txt，請根據實際運行結果和人工判斷填寫 ? 部分。")


if __name__ == '__main__':
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n*** 警告: modern_methods.py 中的 API_KEY 未設定，AI 相關的性能指標將標記為 N/A。 ***")

    collected_metrics = collect_performance_metrics()
    generate_comparison_report(collected_metrics)