from web_app import app
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd

USER_DATASETS_DIRECTORY_PATH = 'fca-decision-tree-classifier\\datasets\\user'

@app.route('/my_datasets')
def my_datasets():
    # Obtener una lista de archivos en el directorio
    files = [f for f in os.listdir(USER_DATASETS_DIRECTORY_PATH) if os.path.isfile(os.path.join(USER_DATASETS_DIRECTORY_PATH, f))]

    # Calcular las estadísticas de filas y columnas para archivos CSV
    file_stats = {}
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(USER_DATASETS_DIRECTORY_PATH, file)
            try:
                df = pd.read_csv(file_path)
                num_rows, num_columns = df.shape
                file_stats[file] = {'num_rows': num_rows, 'num_columns': num_columns}
            except Exception as e:
                file_stats[file] = {'error': str(e)}

    return render_template('my_datasets.html', files=files, file_stats=file_stats)


@app.route('/my_datasets', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    filename = secure_filename(file.filename)
    file.save(os.path.join(USER_DATASETS_DIRECTORY_PATH, filename))
    return redirect(url_for('my_datasets'))

@app.route('/delete-file/<filename>', methods=['POST'])
def delete_file(filename):
    file_path = os.path.join(USER_DATASETS_DIRECTORY_PATH, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('my_datasets'))

@app.route('/preview_file/<filename>', methods=['POST'])
def preview_file(filename):
    try:
        # Construye la ruta completa del archivo CSV
        file_path = os.path.join(USER_DATASETS_DIRECTORY_PATH, filename)

        # Lee el archivo CSV usando pandas
        df = pd.read_csv(file_path)

        # Convierte el DataFrame en una tabla HTML
        html_table = df.to_html(classes='table table-bordered table-striped', index=False)

        return html_table
    except Exception as e:
        return str(e)