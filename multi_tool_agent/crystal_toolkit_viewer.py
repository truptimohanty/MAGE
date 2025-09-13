import dash
from dash import dcc, html, Input, Output, callback
import os
import sys

# Try to import crystal toolkit with fallback
try:
    import crystal_toolkit.components as ctc
    from pymatgen.core import Structure

    CRYSTAL_TOOLKIT_AVAILABLE = True
    print("Crystal Toolkit successfully imported!")
    
except ImportError as e:
    print(f"Crystal Toolkit import error: {e}")
    CRYSTAL_TOOLKIT_AVAILABLE = False
    # Create fallback message
    from pymatgen.core import Structure

# Create Dash app
app = dash.Dash(__name__, prevent_initial_callbacks=True)

# Add headers to allow iframe embedding
@app.server.after_request
def after_request(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Global variable to store current structure
current_structure = None
structure_component = None

def create_layout():
    """Create the initial layout"""
    global structure_component
    
    # Create a default structure if none exists
    if current_structure is None:
        # Simple cubic structure as placeholder
        from pymatgen.core.lattice import Lattice
        default_structure = Structure(
            Lattice.cubic(4.0), 
            ["Na", "Cl"], 
            [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        structure_component = ctc.StructureMoleculeComponent(default_structure, id="structure_viewer")
    else:
        structure_component = ctc.StructureMoleculeComponent(current_structure, id="structure_viewer")
    
    layout = html.Div([
        html.Div([
            html.H3("Crystal Structure Viewer", style={'textAlign': 'center', 'margin': '10px'}),
            html.Div(id="file-info", style={'textAlign': 'center', 'margin': '10px'}),
        ]),
        html.Div([
            structure_component.layout()
        ], style={'height': '600px'}),
        dcc.Interval(
            id='interval-component',
            interval=1000,  # Check every second
            n_intervals=0
        ),
        dcc.Store(id='cif-file-store')
    ])
    
    return layout

app.layout = create_layout()

# Register Crystal Toolkit
ctc.register_crystal_toolkit(app=app, layout=app.layout)

@app.callback(
    [Output('file-info', 'children'),
     Output(structure_component.id() if structure_component else 'structure_viewer', 'data')],
    [Input('interval-component', 'n_intervals')]
)
def check_for_cif_file(n_intervals):
    """Check for CIF file to load"""
    cif_file_path = "temp_structure_for_viewer.cif"
    
    if os.path.exists(cif_file_path):
        try:
            # Load structure from CIF file
            structure = Structure.from_file(cif_file_path)
            
            # Create info text
            info_text = f"Loaded: {structure.composition.reduced_formula} | " \
                       f"Space Group: {structure.get_space_group_info()[1]} | " \
                       f"Atoms: {len(structure.sites)}"
            
            return info_text, structure
            
        except Exception as e:
            return f"Error loading CIF: {str(e)}", dash.no_update
    else:
        return "No structure loaded", dash.no_update

def run_crystal_toolkit_server():
    """Run the Crystal Toolkit server"""
    app.run(debug=False, port=8052, host='127.0.0.1')

if __name__ == '__main__':
    run_crystal_toolkit_server()
