# ===============================================================
# 文件名: app.py
# ===============================================================
import os
import openai
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
# from dotenv import load_dotenv # 不再需要

# --- 初始化 ---

# load_dotenv() # 不再需要

# 创建 Flask 应用实例
app = Flask(__name__)

# 配置 CORS，允许来自任何源的请求
CORS(app) 

# --- 配置 OpenAI API 客户端 ---
# **重要**: 直接在代码中写入API密钥存在安全风险。
# 推荐使用环境变量的方式来管理密钥。
client = openai.OpenAI(
    api_key="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"
)

# --- 算法与辅助函数 ---

def classify_movement(accel_data):
    """
    根据加速度数据判断设备的移动状态。
    这是在前端 classifyMovement 函数的 Python 实现。
    """
    if not accel_data or accel_data.get('x') is None:
        return {"status": "不明"}
    
    try:
        a = {k: float(v) for k, v in accel_data.items()}
        m = (a['x']**2 + a['y']**2 + (a['z'] - 9.8)**2)**0.5
        
        if m < 0.5: return {"status": "静止"}
        if m < 2.0: return {"status": "歩行"}
        if m < 5.0: return {"status": "走行"}
        return {"status": "激しい運動"}
    except (ValueError, TypeError):
        return {"status": "不明"}

# --- API 接口 (Endpoints) ---

@app.route('/api/data', methods=['POST'])
def receive_data():
    """
    接收前端发送的单条实时传感器数据。
    """
    data = request.json
    user_id = data.get('userId', 'unknown_user')
    
    movement_status = classify_movement(data.get('accel'))
    print(f"ユーザーID [{user_id}] からデータ受信。状態: {movement_status['status']}")
    
    return jsonify({"status": "success", "received_movement": movement_status['status']})

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """
    接收前端发送的传感器历史数据，并使用OpenAI进行分析。
    """
    data = request.json
    sensor_history = data.get('history', [])

    if len(sensor_history) < 5:
        return jsonify({"error": "分析するにはデータが不足しています（最低5件必要です）"}), 400

    system_prompt = "あなたは優秀な探偵です。ユーザーから提供されたセンサーデータ履歴を分析し、紛失したデバイスが最も可能性の高い場所を3つ、その理由と共に提案してください。"
    history_str = json.dumps(sensor_history, indent=2, ensure_ascii=False)
    user_prompt = f"""
    以下がセンサーデータの履歴です (直近{len(sensor_history)}件):
    {history_str}
    
    分析のポイント:
    - 位置情報(loc): データがどの場所で途切れたか、または頻繁に記録されているか？
    - 加速度(accel): 最後の移動状態は何か？（静止、歩行、走行など）
    - タイムスタンプ(ts): 最後の記録はいつか？
    
    これらの情報に基づき、最も可能性の高い場所トップ3を、確率と共に報告してください。
    """

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        analysis_text = chat_completion.choices[0].message.content
        return jsonify({"analysis": analysis_text})
        
    except Exception as e:
        print(f"OpenAI API呼び出しエラー: {e}")
        return jsonify({"error": f"OpenAI APIの呼び出し中にエラーが発生しました: {str(e)}"}), 500

@app.route('/api/suggest', methods=['POST'])
def get_suggestion():
    """
    接收AI的初步分析结果，并生成具体的探索建议。
    """
    data = request.json
    analysis_text = data.get('analysis', '')

    if not analysis_text:
        return jsonify({"error": "分析テキストが提供されていません"}), 400

    system_prompt = "あなたは効率的な探索の専門家です。"
    user_prompt = f"""
    以下のAI分析結果に基づいて、紛失したデバイスを見つけるための、具体的で実行可能なステップバイステップの探索プランを提案してください。

    ### AIによる現在の分析結果:
    {analysis_text}

    ### 提案に含めるべき要素:
    1. **最初のステップ:** まず何から始めるべきか。
    2. **探索エリアの優先順位:** どの場所から、どのような順番で探すべきか。
    3. **具体的なアクション:** 「ソファの下を覗く」「カバンの中身をすべて出す」など、具体的な行動を指示してください。

    ユーザーがすぐに行動に移せるような、明確で簡潔な指示リストを作成してください。
    """

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        suggestion_text = chat_completion.choices[0].message.content
        return jsonify({"suggestion": suggestion_text})
    except Exception as e:
        print(f"OpenAI API呼び出しエラー: {e}")
        return jsonify({"error": f"OpenAI APIの呼び出し中にエラーが発生しました: {str(e)}"}), 500

# --- 启动服务器 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

# ===============================================================
# 文件名: requirements.txt
# (python-dotenv 已经移除)
# ===============================================================
# flask
# openai
# flask-cors
