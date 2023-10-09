


import subprocess
import socket

def get_local_ip():

    #return "127.0.0.1"
    """Obtener la dirección IP local de la máquina."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # DNS de Google para obtener la dirección IP local
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
    s.close()
    return ip_address

def check_docker_running():
    """Verificar si Docker está en ejecución."""
    try:
        response = subprocess.check_output(["docker", "info"]).decode()
        if "Server Version" in response:
            return True
    except:
        pass
    return False

def run_inclose4(input_file, min_intent, min_extent, output_format, output_sorted_cxt="y"):
    """Ejecutar el programa InClose4 dentro del contenedor Docker con los parámetros dados."""
    
    # Verificar si Docker está corriendo
    if not check_docker_running():
        print("Error: Docker no está corriendo. Por favor, inicia Docker y vuelve a intentarlo.")
        return

    ip_address = get_local_ip()
    
    # Comando base para ejecutar el contenedor Docker
    docker_command = [
        "docker",
        "-H", f"tcp://{ip_address}:8081",
        "run", "-i", "inclose4"
    ]

    # Entradas para el programa InClose4
    inputs = f"{input_file}\n{min_intent}\n{min_extent}\ny\n{output_format}\n{output_sorted_cxt}\n"

    # Ejecutar el contenedor Docker con las entradas
    process = subprocess.Popen(docker_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=inputs.encode())

    # Imprimir la salida
    if stdout:
        print(stdout.decode())
    if stderr:
        print("Errores:", stderr.decode())




run_inclose4("liveinwater.cxt", "100", "100", "6")
