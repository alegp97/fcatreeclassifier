import subprocess
from web_app import app
from flask import render_template, request, redirect, url_for
import os

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def docker_call():
    if request.method == 'POST':
        # Guardar el archivo cargado
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Ejecutar In-Close4 en el contenedor Docker (ajustar según las necesidades)
        output_file = filename + "_output.json"
        subprocess.run(["docker", "run", "-v", f"{os.getcwd()}/{UPLOAD_FOLDER}:/data", "inclose4", filename, output_file])

        # Redireccionar a la página de resultados
        return redirect(url_for('results', filename=output_file))

    return render_template('index.html') 





@app.route('/results/<filename>', methods=['GET'])
def results(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return render_template('results.html', data=data)
