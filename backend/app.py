import time
import threading
import logging

from flask import Flask
from flask_cors import CORS
from flask import send_from_directory

from automation.run_data_pipeline import run_data_pipeline

from trend.current_candle import RealTimeCandleBuilder
from services.DWXhandler import start_dwx_client_thread


app = Flask(__name__)
CORS(app)

from route.calendar_route import calendar_bp
from route.indicators import indicators_bp
from route.historic import historic_bp
from route.paper import paper_bp
from route.mode import mode_bp
from route.trend_eval import trend_eval_bp
from route.trend import trend_bp

app.register_blueprint(calendar_bp)
app.register_blueprint(indicators_bp)
app.register_blueprint(historic_bp)
app.register_blueprint(paper_bp)
app.register_blueprint(mode_bp)
app.register_blueprint(trend_eval_bp)
app.register_blueprint(trend_bp)

# -- DWX config
app.config["CANDLE_BUILDER"] = RealTimeCandleBuilder()


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

    dwx_thread = threading.Thread(
        target=start_dwx_client_thread,
        args=(app.config["CANDLE_BUILDER"],),
        daemon=True,
    )
    dwx_thread.start()

    threading.Thread(target=run_data_pipeline, daemon=True).start()
    app.run(debug=False, port=5000, use_reloader=False)
