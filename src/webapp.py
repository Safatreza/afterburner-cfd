from flask import Flask, request, render_template_string, send_file
import tempfile
import os
import yaml
from mesh2d import Mesh2D
from boundary_conditions import BoundaryConditions
from heat_profile_plugin import GaussianHeatProfile
from post_processing import PostProcessor
from compressible_ns_solver import CompressibleNSSolver
import numpy as np

app = Flask(__name__)

HTML = '''
<!doctype html>
<title>Afterburner CFD Web UI</title>
<h2>Upload YAML Config and Run Simulation</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=configfile>
  <input type=submit value=Run>
</form>
{% if resultfile %}
  <h3>Simulation complete!</h3>
  <a href="/download/{{ resultfile }}">Download Results</a>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    resultfile = None
    if request.method == 'POST':
        file = request.files['configfile']
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as tmp:
                file.save(tmp.name)
                with open(tmp.name, 'r') as f:
                    config = yaml.safe_load(f)
                mesh = Mesh2D(config['nx'], config['ny'], config['lx'], config['ly'])
                bc = BoundaryConditions(config, mesh)
                heat_plugin = GaussianHeatProfile(config, mesh)
                solver = CompressibleNSSolver(config, mesh=mesh, bc=bc, heat_plugin=heat_plugin)
                solver.run(n_steps=config.get('n_steps', 100))
                post = PostProcessor(mesh, solver.fields)
                results = post.get_results()
                outname = os.path.join(tempfile.gettempdir(), 'web_results.npz')
                np.savez(outname, **results)
                resultfile = os.path.basename(outname)
    return render_template_string(HTML, resultfile=resultfile)

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(tempfile.gettempdir(), filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 