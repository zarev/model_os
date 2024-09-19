from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        data = request.json
        # Process the webhook data here
        print(f"Received webhook data: {data}")
        return jsonify({"status": "success", "message": "Webhook received"}), 200
    else:
        return jsonify({"status": "error", "message": "Method not allowed"}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
