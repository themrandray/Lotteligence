from flask import Flask
from pathlib import Path
from datetime import datetime

def create_app():
    # Izveido Flask lietotni un konfigurāciju
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "lottelligence-secret-key"

    # Jinja filter: format dates as dd.mm.yyyy
    @app.template_filter("ddmmyyyy")
    def ddmmyyyy(value):
        if value is None:
            return ""
        # datetime/date objects
        if hasattr(value, "strftime"):
            return value.strftime("%d.%m.%Y")
        # strings like "YYYY-MM-DD" or "YYYY-MM-DD ..."
        s = str(value)[:10]
        try:
            return datetime.strptime(s, "%Y-%m-%d").strftime("%d.%m.%Y")
        except ValueError:
            return ""

    # Pamatdirektorijas
    base_dir = Path(__file__).resolve().parent.parent
    outputs_dir = base_dir / "outputs"
    uploads_dir = base_dir / "uploads"

    # Izveido tikai nepieciešamās mapes
    outputs_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Konfigurācija
    app.config["BASE_DIR"] = base_dir
    app.config["OUTPUTS_DIR"] = outputs_dir
    app.config["UPLOAD_DIR"] = uploads_dir

    # Reģistrē maršrutus
    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app