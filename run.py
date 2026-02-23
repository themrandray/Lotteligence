from app import create_app

# Galvenā ieejas vieta Flask lietotnei
app = create_app()

if __name__ == "__main__":
    # Debug režīms lokālai izstrādei
    app.run(debug=True)