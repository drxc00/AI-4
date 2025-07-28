from flask import Flask

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True)