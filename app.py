from flask import Flask, render_template, request, send_file
from book_reader import BookReader
import os

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = "./Data/books"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "book" not in request.files:
            return "No file part", 400

        file = request.files["book"]
        if file.filename == "":
            return "No selected file", 400

        # Save the uploaded book
        file_path = os.path.join(UPLOAD_FOLDER, "book.pdf")
        file.save(file_path)

        # Generate audiobook
        reader = BookReader()
        audio_path = reader.read_book()

        return send_file(
            audio_path,
            as_attachment=True,
            mimetype="audio/wav",
            download_name="audiobook.wav",
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
