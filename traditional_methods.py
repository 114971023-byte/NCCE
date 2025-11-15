import math
from collections import Counter
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys

# 設置 jieba 輸出安靜模式 - NOTE: The original line was incorrect for this purpose.
# jieba.set_dictionary(jieba.DEFAULT_DICT) # This line caused TypeError, removed.

# ----------------- 測試資料 (與 comparison.py 共享) -----------------
DOCUMENTS = [
    "人工智慧正在改變世界,機器學習是其核心技術",
    "深度學習推動了人工智慧的發展,特別是在圖像識別領域",
    "今天天氣很好,適合出去運動",
    "機器學習和深度學習都是人工智慧的重要分支",
    "運動有益健康,每天都應該保持運動習慣"
]

TEST_TEXTS = [
    "這家餐廳的牛肉麵真的太好吃了,湯頭濃郁,麵條Q彈,下次一定再來!",
    "最新的AI技術突破讓人驚艷,深度學習模型的表現越來越好",
    "這部電影劇情空洞,演技糟糕,完全是浪費時間",
    "每天慢跑5公里,配合適當的重訓,體能進步很多"
]

ARTICLE_TEXT = """人工智慧 (AI) 的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘,
到通勤時的路線規劃,再到工作中的各種輔助工具,AI無處不在。
在醫療領域,AI協助醫生進行疾病診斷,提高了診斷的準確率和效率。
透過分析大量的醫學影像和病歷資料,AI能夠發現肉眼容易忽略的細節,為患者提供更好的治療方案。
教育方面,AI融入化學習系統能夠根據每個學生的學習進度和特點,提供客製化的教學內容。
這種因材施教的方式,讓學習變得更加高效和有趣。
然而,AI的快速發展也帶來了一些挑戰。首先是就業問題,許多傳統工作可能會被AI取代。
其次是隱私和安全問題,AI系統需要大量數據來訓練。如何保護個人隱私成為重要議題。
最後是倫理問題,AI的決策過程往往缺乏透明度,可能會產生偏見或歧視。
面對這些挑戰,我們需要推動AI發展的同時,建立相應的法律法規和倫理準則。
只有這樣,才能確保AI技術真正為人類福祉服務,創造一個更美好的未來。"""

# ----------------- Part A-1: TF-IDF 文本相似度計算 -----------------

def get_tokens_and_word_dicts(documents):
    """將文件列表分詞並計算詞彙計數字典"""
    doc_word_dicts = []
    for doc in documents:
        tokens = list(jieba.cut(doc, cut_all=False))
        doc_word_dicts.append(Counter(tokens))
    return doc_word_dicts

def calculate_tf(word_dict, total_words):
    """計算詞頻 (Term Frequency)"""
    tf_dict = {}
    for word, count in word_dict.items():
        tf_dict[word] = count / total_words if total_words > 0 else 0
    return tf_dict

def calculate_idf(documents, word):
    """計算逆文件頻率 (Inverse Document Frequency)"""
    N = len(documents) # 文件總數
    df = sum(1 for doc in documents if word in doc)

    # 這裡使用常用的 log(N/df) 變形
    return math.log(N / df) if df > 0 else 0

def manual_tfidf_similarity(documents):
    """A-1-1: 手動計算 TF-IDF 向量並計算餘弦相似度"""
    doc_word_dicts = get_tokens_and_word_dicts(documents)
    doc_total_words = [sum(d.values()) for d in doc_word_dicts]

    all_words = set().union(*(d.keys() for d in doc_word_dicts))
    word_to_index = {word: i for i, word in enumerate(all_words)}

    # 1. 計算 TF-IDF 矩陣
    tfidf_matrix = np.zeros((len(documents), len(all_words)))

    for i, (word_dict, total_words) in enumerate(zip(doc_word_dicts, doc_total_words)):
        tf_dict = calculate_tf(word_dict, total_words)
        for word, tf_val in tf_dict.items():
            idf_val = calculate_idf(documents, word)
            tfidf_val = tf_val * idf_val
            j = word_to_index[word]
            tfidf_matrix[i, j] = tfidf_val

    # 2. 計算餘弦相似度
    return cosine_similarity(tfidf_matrix)

def sklearn_tfidf_similarity(documents):
    """A-1-2: 使用 scikit-learn 計算 TF-IDF 並計算相似度矩陣"""

    def jieba_tokenizer(text):
        # 必須搭配分詞器處理中文
        return list(jieba.cut(text, cut_all=False))

    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

# ----------------- Part A-2: 基於規則的文本分類 -----------------

class RuleBasedSentimentClassifier:
    """A-2-1: 情感分類器"""
    def __init__(self):
        self.positive_words = ['好', '棒', '優秀', '喜歡', '推薦', '滿意', '開心', '值得', '精彩', '完美']
        self.negative_words = ['差', '糟', '失望', '討厭', '不推薦', '浪費', '無聊', '爛', '糟糕', '差勁']
        self.negation_words = ['不', '沒', '無', '非', '別']
        self.intensifier_words = {'太': 2.0, '很': 1.5, '非常': 2.0, '超級': 2.5, '一點': 0.5}

    def classify(self, text):
        tokens = list(jieba.cut(text, cut_all=False))
        sentiment_score = 0

        for i, word in enumerate(tokens):
            weight = 1.0

            # 程度副詞加權
            if word in self.intensifier_words:
                weight = self.intensifier_words.get(word, 1.0)

            score = 0
            if word in self.positive_words:
                score = 1 * weight
            elif word in self.negative_words:
                score = -1 * weight

            # 否定詞處理 (檢查前一個詞)
            if score != 0 and i > 0 and tokens[i-1] in self.negation_words:
                score *= -1

            sentiment_score += score

        if sentiment_score > 1.0:
            return "正面"
        elif sentiment_score < -1.0:
            return "負面"
        else:
            return "中性"

class TopicClassifier:
    """A-2-2: 主題分類器"""
    def __init__(self):
        self.topic_keywords = {
            '科技': ['AI', '人工智慧', '電腦', '軟體', '程式', '演算法', '深度學習', '機器學習'],
            '運動': ['運動', '健身', '跑步', '游泳', '球類', '比賽', '體能', '重訓', '慢跑'],
            '美食': ['吃', '食物', '餐廳', '美味', '料理', '烹飪', '牛肉麵', '湯頭', '麵條'],
            '旅遊': ['旅行', '景點', '飯店', '機票', '觀光', '度假']
        }

    def classify(self, text):
        tokens = list(jieba.cut(text, cut_all=False))
        topic_scores = {topic: 0 for topic in self.topic_keywords}

        for token in tokens:
            for topic, keywords in self.topic_keywords.items():
                if token in keywords:
                    topic_scores[topic] += 1

        max_score = 0
        best_topic = "中性/其他"

        for topic, score in topic_scores.items():
            if score > max_score:
                max_score = score
                best_topic = topic

        return best_topic if max_score > 0 else "中性/其他"

# ----------------- Part A-3: 統計式自動摘要 -----------------

class StatisticalSummarizer:
    """A-3: 統計式自動摘要"""
    def __init__(self):
        self.stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '個', '上', '他', '很', '到', '說', '要', '去', '你', '這', '也', '而', '我們', '一個', '可以', '能夠'])
        self.sentence_delimiters = ['。', '！', '？', '\n']

    def _split_sentences(self, text):
        """處理中文標點進行分句"""
        sentences = []
        temp_sentence = ""
        for char in text:
            temp_sentence += char
            if char in self.sentence_delimiters:
                sentences.append(temp_sentence.strip())
                temp_sentence = ""
        if temp_sentence.strip():
            sentences.append(temp_sentence.strip())
        return [s for s in sentences if s]

    def _calculate_word_freq(self, sentences):
        """分詞並計算詞頻 (排除停用詞)"""
        all_tokens = []
        for s in sentences:
            tokens = list(jieba.cut(s, cut_all=False))
            all_tokens.extend([t for t in tokens if t not in self.stop_words])
        return Counter(all_tokens)

    def sentence_score(self, sentence, word_freq, sentence_index, total_sentences):
        """計算句子重要性分數"""
        score = 0
        tokens = list(jieba.cut(sentence, cut_all=False))

        # 1. 包含高頻詞的數量 (加權)
        for token in tokens:
            if token in word_freq:
                score += word_freq[token]

        # 2. 句子位置 (首尾句加權)
        position_weight = 1.0
        if sentence_index == 0 or sentence_index == total_sentences - 1:
            position_weight = 1.5
        score *= position_weight

        # 3. 句子長度
        length = len(tokens)
        if length < 5 or length > 45:
            score *= 0.5

        return score

    def summarize(self, text, ratio=0.3):
        """生成摘要"""
        sentences = self._split_sentences(text)
        if not sentences:
            return ""

        word_freq = self._calculate_word_freq(sentences)

        sentence_scores = {}
        total_sentences = len(sentences)
        for i, sentence in enumerate(sentences):
            sentence_scores[i] = self.sentence_score(sentence, word_freq, i, total_sentences)

        num_sentences = len(sentences)
        num_to_select = max(1, int(num_sentences * ratio))

        sorted_sentences_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
        summary_indices = sorted_sentences_indices[:num_to_select]

        summary_indices.sort()

        summary = "".join([sentences[i] for i in summary_indices])
        return summary

if __name__ == '__main__':
    print("--- 執行 traditional_methods.py 範例 ---")

    # A-1 測試
    print("\n[A-1] TF-IDF 相似度 (Scikit-learn):")
    sim_matrix = sklearn_tfidf_similarity(DOCUMENTS)
    print(f"Doc 1 vs Doc 2 相似度: {sim_matrix[0, 1]:.4f}")

    # A-2 測試
    print("\n[A-2] 規則分類器測試:")
    s_classifier = RuleBasedSentimentClassifier()
    t_classifier = TopicClassifier()
    for text in TEST_TEXTS:
        sentiment = s_classifier.classify(text)
        topic = t_classifier.classify(text)
        print(f"'{text[:10]}...' -> 情感: {sentiment}, 主題: {topic}")

    # A-3 測試
    print("\n[A-3] 統計摘要測試 (Ratio=0.3):")
    summarizer = StatisticalSummarizer()
    summary = summarizer.summarize(ARTICLE_TEXT, ratio=0.3)
    print(summary)