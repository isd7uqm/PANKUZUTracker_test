# ===============================================================
# 文件名: app.py (模型测试版)
# ===============================================================
import os
import openai
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# --- 初始化 ---
app = Flask(__name__, static_folder='static')
CORS(app) 

# --- 配置 OpenAI API 客户端 ---
client = openai.OpenAI(
    api_key="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"
)

# --- 算法与辅助函数 ---
def classify_movement(accel_data):
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

@app.route('/health')
def health_check():
    """一个简单的健康检查接口"""
    print("[DEBUG] Health check endpoint was called.")
    return "OK", 200

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.json
    user_id = data.get('userId', 'unknown_user')
    movement_status = classify_movement(data.get('accel'))
    print(f"[INFO] ユーザーID [{user_id}] からデータ受信。状態: {movement_status['status']}")
    return jsonify({"status": "success", "received_movement": movement_status['status']})

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    print("[DEBUG] /api/analyze endpoint was called.")
    try:
        data = request.json
        sensor_history = data.get('history', [])
        print(f"[DEBUG] Received {len(sensor_history)} records for analysis.")

        if len(sensor_history) < 5:
            return jsonify({"error": "分析するにはデータが不足しています（最低5件必要です）"}), 400

        system_prompt = "あなたは優秀な探偵です。ユーザーから提供されたセンサーデータ履歴を分析し、紛失したデバイスが最も可能性の高い場所を3つ、その理由と共に提案してください。"
        history_str = json.dumps(sensor_history, indent=2, ensure_ascii=False)
        user_prompt = f"""
        以下がセンサーデータの履歴です (直近{len(sensor_history)}件):
        {history_str}
        分析のポイント:
        - 位置情報(loc): データがどの場所で途切れたか、または頻繁に記録されているか？
        - 加速度(accel): 最後の移動状態は何か？
        - タイムスタンプ(ts): 最後の記録はいつか？
        これらの情報に基づき、最も可能性の高い場所トップ3を、確率と共に報告してください。
        """

        print("[DEBUG] Calling OpenAI API for analysis with model gpt-3.5-turbo...")
        chat_completion = client.chat.completions.create(
            # --- 调试修改：使用 gpt-3.5-turbo 进行测试 ---
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        analysis_text = chat_completion.choices[0].message.content
        print("[DEBUG] Successfully received analysis from OpenAI.")
        return jsonify({"analysis": analysis_text})
        
    except Exception as e:
        print(f"[ERROR] An error occurred in /api/analyze: {e}")
        return jsonify({"error": f"サーバー内部でエラーが発生しました: {str(e)}"}), 500

@app.route('/api/suggest', methods=['POST'])
def get_suggestion():
    print("[DEBUG] /api/suggest endpoint was called.")
    try:
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
        3. **具体的なアクション:** 具体的な行動を指示してください。
        ユーザーがすぐに行動に移せるような、明確で簡潔な指示リストを作成してください。
        """
        
        print("[DEBUG] Calling OpenAI API for suggestion with model gpt-3.5-turbo...")
        chat_completion = client.chat.completions.create(
            # --- 调试修改：使用 gpt-3.5-turbo 进行测试 ---
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        suggestion_text = chat_completion.choices[0].message.content
        print("[DEBUG] Successfully received suggestion from OpenAI.")
        return jsonify({"suggestion": suggestion_text})
    except Exception as e:
        print(f"[ERROR] An error occurred in /api/suggest: {e}")
        return jsonify({"error": f"サーバー内部でエラーが発生しました: {str(e)}"}), 500

@app.route('/')
def serve_index():
    """当用户访问根URL时，返回 index.html 文件。"""
    return send_from_directory(app.static_folder, 'index.html')

# --- 启动服务器 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
