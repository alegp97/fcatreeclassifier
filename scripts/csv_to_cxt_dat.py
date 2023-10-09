import os
import pandas as pd

def convert_csv_to_cxt_and_dat(df, output_dir, base_name):
    # Extract object names and attributes
    object_names = df.iloc[:, 0].tolist()
    attributes = df.columns[1:].tolist()
    
    # Extract binary matrix
    binary_matrix = df.iloc[:, 1:].values
    
    # Create .cxt file
    cxt_filename = os.path.join(output_dir, f"{base_name}.cxt")
    with open(cxt_filename, "w") as cxt_file:
        cxt_file.write("B\n")
        cxt_file.write(f"{len(object_names)}\n")
        cxt_file.write(f"{len(attributes)}\n")
        for obj in object_names:
            cxt_file.write(f"{obj}\n")
        for attr in attributes:
            cxt_file.write(f"{attr}\n")
        for row in binary_matrix:
            cxt_file.write("".join(map(str, row)) + "\n")
    
    # Create .dat file
    dat_filename = os.path.join(output_dir, f"{base_name}.dat")
    with open(dat_filename, "w") as dat_file:
        for row in binary_matrix:
            dat_file.write("".join(map(str, row)) + "\n")
    
    return cxt_filename, dat_filename

def main():
    # Obtener entrada del usuario para la ruta CSV y el directorio de salida
    csv_path = input("Introduce la ruta al archivo CSV: ")
    
    # Comprobar si existe la ruta CSV proporcionada
    if not os.path.exists(csv_path):
        print("El archivo CSV proporcionado no existe.")
        return
    
    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv(csv_path)
    
    # Obtener el directorio de salida (por defecto el directorio actual si no se proporciona)
    output_dir = input("Introduce la ruta de salida (presiona Enter para usar la ruta por defecto): ")

    
    # Comprobar si el directorio de salida existe
    if not os.path.exists(output_dir):
        print("El directorio de salida proporcionado no existe. Creando el directorio...")
        os.makedirs(output_dir)

    # Extract the base name from the input CSV filename
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Convertir el DataFrame a formatos .cxt y .dat
    cxt_filename, dat_filename = convert_csv_to_cxt_and_dat(df, output_dir, base_name)
    
    print(f"Archivo .cxt guardado en: {cxt_filename}")
    print(f"Archivo .dat guardado en: {dat_filename}")

if __name__ == "__main__":
    main()