from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    # Process the webhook data here
    app.logger.debug(f"Received webhook data: {data}")
    return jsonify({"status": "success", "message": "Webhook received"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)