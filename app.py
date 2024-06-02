from flask import Flask, request, jsonify

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    return "File Uploaded"

if __name__ == '__main__':
    app.run()