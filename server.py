from flask import Flask, request, jsonify
import deduplicate
import sys
import traceback
import os

app = Flask(__name__)

@app.route('/deduplicate', methods=['POST'])
def deduplicate_endpoint():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        incoming = data.get("incoming")
        candidates = data.get("candidates", [])
        
        if not incoming:
            return jsonify({"error": "Missing 'incoming' data"}), 400
            
        # Run the pipeline using the imported function
        report = deduplicate.run_pipeline(incoming, candidates)
        
        return jsonify(report)
        
    except Exception as e:
        # Log the full traceback to stderr for debugging
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Run on 0.0.0.0 to be accessible from other machines/containers
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
