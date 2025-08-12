from flask import Blueprint, request, jsonify

mode_bp = Blueprint("mode", __name__)

# You can store the mode in-memory or in Redis/DB
current_mode = {"mode": "paper"}


@mode_bp.route("/api/set-mode", methods=["POST"])
def set_mode():
    data = request.get_json(silent=True)  # avoid errors if parsing fails
    if not data or "mode" not in data:
        return jsonify({"error": "Missing 'mode' field"}), 400
    mode = data["mode"]
    if mode not in ["live", "paper"]:
        return jsonify({"error": "Invalid mode"}), 400
    current_mode["mode"] = mode
    return jsonify({"status": "ok", "mode": mode}), 200
