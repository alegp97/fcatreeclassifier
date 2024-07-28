from web_app import app
from flask import render_template

@app.route('/user_settings')
def user_settings():
    return render_template('user_settings.html')