import time
import threading
import logging

from flask import Flask
from flask_cors import CORS
from flask import send_from_directory

from automation.run_data_pipeline import run_data_pipeline

app = Flask(__name__)
CORS(app)


class WerkzeugFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not msg.endswith('" 200 -')


# Favicon fun
@app.route("/favicon.svg")
def favicon():
    return send_from_directory("static", "favicon.svg", mimetype="image/svg+xml")


#  Main route
@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml">
        <title>Backend server</title>
      </head>
      <body>
        <h1>Flask is running!</h1>
      </body>
    </html>
    """


if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.addFilter(WerkzeugFilter())

    threading.Thread(target=run_data_pipeline, daemon=True).start()
    app.run(debug=False, port=5000, use_reloader=False)
