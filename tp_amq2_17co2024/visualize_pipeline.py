from sklearn.utils import estimator_html_repr
from models.cars_pipeline import CarsPipeline
from sklearn.experimental import enable_iterative_imputer
import webbrowser
import os

# Generar la representación HTML del pipeline
html_content = estimator_html_repr(CarsPipeline())

# Guardar la representación HTML en un archivo, especificando la codificación UTF-8
with open('../../pipeline_visualization.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

# Obtener la ruta absoluta del archivo HTML
file_path = os.path.abspath('../../pipeline_visualization.html')

# Abrir el archivo HTML en el navegador predeterminado
webbrowser.open('file://' + file_path, new=2)

print(f"La visualización del pipeline se ha guardado y abierto en tu navegador como '{file_path}'")
