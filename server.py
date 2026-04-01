from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import os

# Import your OCR pipeline
from main import run_pipeline

app = Flask(__name__, static_folder="frontend", template_folder="frontend")
CORS(app)   # allow frontend connection


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/<path:path>")
def serve_file(path):
    return send_from_directory("frontend", path)


# --------------------------------------------------
# Helper: convert sets → lists (JSON safe)
# --------------------------------------------------
def convert_sets(obj):

    if isinstance(obj, set):
        return list(obj)

    elif isinstance(obj, dict):
        return {k: convert_sets(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [convert_sets(i) for i in obj]

    else:
        return obj


# --------------------------------------------------
# Aadhaar Validation API
# --------------------------------------------------
@app.route("/api/validate", methods=["POST"])
def validate():

    try:

        front = request.files.get("front")
        back = request.files.get("back")
        selfie = request.files.get("selfie")

        if not front:
            return jsonify({"error": "Front image required"}), 400

        # Create temp folder
        temp_dir = tempfile.mkdtemp()

        # Save front
        front_path = os.path.join(temp_dir, "front.jpg")
        front.save(front_path)

        # Save back (optional)
        back_path = None
        if back:
            back_path = os.path.join(temp_dir, "back.jpg")
            back.save(back_path)

        # Save selfie (optional)
        selfie_path = None
        if selfie:
            selfie_path = os.path.join(temp_dir, "selfie.jpg")
            selfie.save(selfie_path)

        # --------------------------------------------------
        # Run OCR + AI pipeline
        # --------------------------------------------------
        result = run_pipeline(front_path, back_path, selfie_path)

        # Convert sets → lists for JSON
        result = convert_sets(result)

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": "Verification pipeline failed",
            "details": str(e)
        }), 500


# --------------------------------------------------
# Start Flask Server
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)