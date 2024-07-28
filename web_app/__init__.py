from flask import Flask

app = Flask(__name__)

# Importar rutas
from web_app.routes import main_routes, user_routes, dataset_routes, docker_routes, tree_routes
