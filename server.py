from flask import Flask, request, jsonify,send_from_directory
from flask_cors import CORS

from captioning import load_manifest
from rag import Rag

app = Flask(__name__)
CORS(app)

# Initialize tools
rag = Rag()
rag.load_manifest("data/caption_manifest.json")

@app.route("/")
def index():
    return "Agentic Image Based RAG Server."

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('data/images', filename)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        # Parse JSON request body
        data = request.get_json()
        question = data.get("question")
        k = data.get("k", 3)  # Default to top-3 if not provided

        if not question:
            return jsonify({"error": "Missing 'question' field"}), 400

        # Perform RAG query
        result = rag.ask(question, k)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
