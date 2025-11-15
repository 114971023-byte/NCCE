from openai import OpenAI
import os
import json
import sys

# ******************************************************
# *** 已將用戶提供的 API Key 植入此處 ***
# ******************************************************
# 建議使用環境變數: os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
# 為了方便執行，這裡直接將 Key 設置為變數值
API_KEY = "真實 Key 已移除"
MODEL = "gpt-4o"

def get_openai_client(api_key):
    """初始化 OpenAI 客戶端"""
    if not api_key:
        print("Error: API Key is not set.")
        return None
    try:
        client = OpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=3
        )
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None

# ----------------- Part B-1: 語意相似度計算 -----------------

def ai_similarity(text1, text2, api_key):
    """B-1: 使用 GPT-4o 判斷語意相似度 (返回 0-100 數字)"""
    client = get_openai_client(api_key)
    if not client:
        return -1

    prompt = f"""
請評估以下兩段文字的語意相似度。請只回覆一個0到100的整數。
文字1: {{{text1}}}
文字2: {{{text2}}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是一個專業的文本相似度評估員，只返回一個介於0到100的整數。"},
                {"role": "user", "content": prompt}
            ]
        )
        result_text = response.choices[0].message.content.strip()
        try:
            similarity_score = int(result_text)
            return max(0, min(100, similarity_score))
        except ValueError:
            # print(f"API 返回非數字結果: {result_text}")
            return -1

    except Exception as e:
        # print(f"API 呼叫錯誤: {e}")
        return -1

# ----------------- Part B-2: AI 文本分類 -----------------

def ai_classify(text, api_key):
    """B-2: 使用 GPT-4o 進行多維度分類 (返回 JSON 格式)"""
    client = get_openai_client(api_key)
    if not client:
        return {"sentiment": "N/A", "topic": "N/A", "confidence": 0.0}

    prompt = f"""
請對以下文本進行多維度分類，並以 JSON 格式返回。
1. sentiment (情感傾向): 必須是「正面」、「負面」或「中性」。
2. topic (主要類別): 從「科技」、「運動」、「美食」、「旅遊」中選一個，或「其他」。
3. confidence (信心度): 介於 0.0 到 1.0 之間的小數。
文本: "{text}"
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "你是一個文本分類專家，請嚴格按照用戶的 JSON 格式要求返回結果。"},
                {"role": "user", "content": prompt}
            ]
        )

        json_output = response.choices[0].message.content
        return json.loads(json_output)

    except Exception as e:
        # print(f"API 呼叫錯誤或 JSON 解析失敗: {e}")
        return {"sentiment": "Error", "topic": "Error", "confidence": 0.0}

# ----------------- Part B-3: AI 自動摘要 -----------------

def ai_summarize(text, max_length, api_key):
    """B-3: 使用 GPT-3.5 生成摘要"""
    client = get_openai_client(api_key)
    if not client:
        return "API Client Error"

    prompt = f"""
請為以下文本生成一個精簡、專業的摘要。
摘要長度**必須控制**在大約 {max_length} 個中文字符以內。
請直接輸出摘要內容，不要包含任何額外說明。
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是一個專業的文摘生成器，你的輸出必須嚴格遵守用戶的長度限制。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # print(f"API 呼叫錯誤: {e}")
        return "API Call Failed"

if __name__ == '__main__':
    print("--- 執行 modern_methods.py 範例 ---")

    TEST_DOCUMENTS = ["人工智慧正在改變世界,機器學習是其核心技術",
    "深度學習推動了人工智慧的發展,特別是在圖像識別領域",
    "今天天氣很好,適合出去運動",
    "機器學習和深度學習都是人工智慧的重要分支",
    "運動有益健康,每天都應該保持運動習慣"]

    TEST_TEXT = ["這家餐廳的牛肉麵真的太好吃了,湯頭濃郁,麵條Q彈,下次一定再來!",
    "最新的AI技術突破讓人驚艷,深度學習模型的表現越來越好",
    "這部電影劇情空洞,演技糟糕,完全是浪費時間",
    "每天慢跑5公里,配合適當的重訓,體能進步很多"]

    TEST_ARTICLE = ["人工智慧AI 的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘,",
    "到通勤時的路線規劃,再到工作中的各種輔助工具,AI無處不在。",
    "在醫療領域,AI協助醫生進行疾病診斷,提高了診斷的準確率和效率。",
    "透過分析大量的醫學影像和病歷資料,AI能夠發現肉眼容易忽略的細節,為患者提供更好的治療方案。",
    "教育方面,AI融入化學習系統能夠根據每個學生的學習進度和特點,提供客製化的教學內容。",
    "這種因材施教的方式,讓學習變得更加高效和有趣。",
    "然而,AI的快速發展也帶來了一些挑戰。首先是就業問題,許多傳統工作可能會被AI取代。",
    "其次是隱私和安全問題,AI系統需要大量數據來訓練。如何保護個人隱私成為重要議題。",
    "最後是倫理問題,AI的決策過程往往缺乏透明度,可能會產生偏見或歧視。",
    "面對這些挑戰,我們需要推動AI發展的同時,建立相應的法律法規和倫理準則。",
    "只有這樣,才能確保AI技術真正為人類福祉服務,創造一個更美好的未來。"]

    if API_KEY and API_KEY != "YOUR_API_KEY_HERE":
        print(f"\n[B-1] AI 相似度測試: {ai_similarity(TEST_DOCUMENTS[0], TEST_DOCUMENTS[1], API_KEY)}%")

        print(f"\n[B-2] AI 分類測試:")
        classification_result = ai_classify(TEST_TEXT, API_KEY)
        print(json.dumps(classification_result, indent=4, ensure_ascii=False))

        print(f"\n[B-3] AI 摘要測試 (Max 50字):")
        summary = ai_summarize(TEST_ARTICLE, 50, API_KEY)
        print(summary)
    else:
        print("\n*** 警告: API Key 無效，無法執行 AI 模組測試。***")
