import streamlit as st
import asyncio
import os
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import plotly.graph_objects as go
from pymatgen.core import Structure
import numpy as np
import streamlit.components.v1 as components

# Try to import the agent function with error handling
try:
    from agent import call_agent_async
    AGENT_AVAILABLE = True
except ImportError as e:
    st.warning(f"Could not import agent: {e}")
    AGENT_AVAILABLE = False
    
    # Create a mock function for testing
    async def call_agent_async(query):
        return f"Mock response: This is a simulated response for query '{query}'"


def visualize_cif_structure_enhanced(cif_file_path):
    """
    Create enhanced CIF visualization with better atom labeling and bonds
    """
    try:
        # Load structure from CIF file
        structure = Structure.from_file(cif_file_path)
        
        # Get atomic positions and species
        coords = structure.cart_coords
        species = [str(site.specie) for site in structure]
        
        # Color mapping for common elements (CPK colors)
        element_colors = {
            'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00',
            'B': '#FFB5B5', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
            'F': '#90E050', 'Ne': '#B3E3F5', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
            'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30',
            'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
            'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7',
            'Mn': '#9C7AC7', 'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050',
            'Cu': '#C88033', 'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F',
            'As': '#BD80E3', 'Se': '#FFA100', 'Br': '#A62929', 'Kr': '#5CB8D1',
            'Rb': '#702EB0', 'Sr': '#00FF00', 'Y': '#94FFFF', 'Zr': '#94E0E0',
            'Nb': '#73C2C9', 'Mo': '#54B5B5', 'Tc': '#3B9E9E', 'Ru': '#248F8F',
            'Rh': '#0A7D8C', 'Pd': '#006985', 'Ag': '#C0C0C0', 'Cd': '#FFD98F',
            'In': '#A67573', 'Sn': '#668080', 'Sb': '#9E63B5', 'Te': '#D47A00',
            'I': '#940094', 'Xe': '#429EB0', 'Cs': '#57178F', 'Ba': '#00C900',
            'La': '#70D4FF', 'Ce': '#FFFFC7', 'Pr': '#D9FFC7', 'Nd': '#C7FFC7',
            'Pm': '#A3FFC7', 'Sm': '#8FFFC7', 'Eu': '#61FFC7', 'Gd': '#45FFC7',
            'Tb': '#30FFC7', 'Dy': '#1FFFC7', 'Ho': '#00FF9C', 'Er': '#00E675',
            'Tm': '#00D452', 'Yb': '#00BF38', 'Lu': '#00AB24', 'Hf': '#4DC2FF',
            'Ta': '#4DA6FF', 'W': '#2194D6', 'Re': '#267DAB', 'Os': '#266696',
            'Ir': '#175487', 'Pt': '#D0D0E0', 'Au': '#FFD123', 'Hg': '#B8B8D0',
            'Tl': '#A6544D', 'Pb': '#575961', 'Bi': '#9E4FB5', 'Po': '#AB5C00',
            'At': '#754F45', 'Rn': '#428296', 'Fr': '#420066', 'Ra': '#007D00',
            'Ac': '#70ABFA', 'Th': '#00BAFF', 'Pa': '#00A1FF', 'U': '#008FFF',
            'Np': '#0080FF', 'Pu': '#006BFF', 'Am': '#545CF2', 'Cm': '#785CE3',
            'Bk': '#8A4FE3', 'Cf': '#A136D4', 'Es': '#B31FD4', 'Fm': '#B31FBA',
            'Md': '#B30DA6', 'No': '#BD0D87', 'Lr': '#C70066', 'Rf': '#CC0059',
            'Db': '#D1004F', 'Sg': '#D90045', 'Bh': '#E00038', 'Hs': '#E6002E',
            'Mt': '#EB0026'
        }
        
        # Get colors and sizes for each atom
        colors = [element_colors.get(elem, '#CCCCCC') for elem in species]
        
        # Atomic radii for better visualization (in Angstroms)
        atomic_radii = {
            'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76,
            'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41,
            'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
            'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39,
            'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22,
            'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16
        }
        
        # Calculate atom sizes based on atomic radii
        atom_sizes = [max(6, min(20, atomic_radii.get(elem, 1.0) * 8)) for elem in species]
        
        # Create 3D scatter plot with better atom representation
        fig = go.Figure(data=go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=atom_sizes,
                color=colors,
                line=dict(width=1, color='black'),
                opacity=0.9
            ),
            text=species,  # Show element symbols
            textposition="middle center",
            textfont=dict(size=8, color='white'),
            hovertext=[f"<b>{species[i]}</b><br>Position: ({coords[i, 0]:.3f}, {coords[i, 1]:.3f}, {coords[i, 2]:.3f})<br>Atomic radius: {atomic_radii.get(species[i], 'N/A')} Å" 
                      for i in range(len(species))],
            hovertemplate='%{hovertext}<extra></extra>',
            name="Atoms"
        ))
        
        # Add bonds (simple distance-based bonding)
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(coords))
        
        # Add bonds between nearby atoms
        bond_threshold = 3.0  # Angstroms
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                if distances[i, j] < bond_threshold:
                    fig.add_trace(go.Scatter3d(
                        x=[coords[i, 0], coords[j, 0]],
                        y=[coords[i, 1], coords[j, 1]],
                        z=[coords[i, 2], coords[j, 2]],
                        mode='lines',
                        line=dict(color='gray', width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Add unit cell edges
        lattice = structure.lattice
        origin = np.array([0, 0, 0])
        
        # Define unit cell vertices
        vertices = [
            origin,
            lattice.matrix[0],
            lattice.matrix[1],
            lattice.matrix[2],
            lattice.matrix[0] + lattice.matrix[1],
            lattice.matrix[0] + lattice.matrix[2],
            lattice.matrix[1] + lattice.matrix[2],
            lattice.matrix[0] + lattice.matrix[1] + lattice.matrix[2]
        ]
        
        # Define edges of unit cell
        edges = [
            (0, 1), (0, 2), (0, 3),  # from origin
            (1, 4), (1, 5),          # from a
            (2, 4), (2, 6),          # from b
            (3, 5), (3, 6),          # from c
            (4, 7), (5, 7), (6, 7)   # to opposite corner
        ]
        
        # Add unit cell edges to plot
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip',
                name="Unit Cell"
            ))
        
        # Update layout with better styling
        fig.update_layout(
            title=dict(
                text=f"Crystal Structure: {structure.composition.reduced_formula}",
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                aspectmode='cube',
                bgcolor='white',
                xaxis=dict(backgroundcolor='white', gridcolor='lightgray', showbackground=True),
                yaxis=dict(backgroundcolor='white', gridcolor='lightgray', showbackground=True),
                zaxis=dict(backgroundcolor='white', gridcolor='lightgray', showbackground=True),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=False
        )
        
        return fig, structure
        
    except Exception as e:
        st.error(f"Error loading CIF file: {str(e)}")
        return None, None

def visualize_cif_structure_crystal_toolkit(cif_file_path):
    """
    Create Crystal Toolkit visualization by copying CIF file and embedding viewer
    """
    try:
        # Copy the CIF file to a standard location for the Crystal Toolkit viewer
        import shutil
        temp_cif_path = "temp_structure_for_viewer.cif"
        shutil.copy2(cif_file_path, temp_cif_path)
        
        # Load structure for info display
        structure = Structure.from_file(cif_file_path)
        
        # Create iframe to embed Crystal Toolkit viewer
        iframe_html = f"""
        <iframe src="http://localhost:8052" 
                width="600" 
                height="560" 
                frameborder="0"
                style="border: 1px solid #ddd; border-radius: 5px;">
        </iframe>
        """
        
        return iframe_html, structure
        
    except Exception as e:
        st.error(f"Error preparing Crystal Toolkit visualization: {str(e)}")
        return None, None

def visualize_cif_structure(cif_file_path):
    """
    Create a 3D visualization of a CIF structure using pymatgen and Plotly (fallback)
    """
    try:
        # Load structure from CIF file
        structure = Structure.from_file(cif_file_path)
        
        # Get atomic positions and species
        coords = structure.cart_coords
        species = [str(site.specie) for site in structure]
        
        # Color mapping for common elements
        element_colors = {
            'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00',
            'B': '#FFB5B5', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
            'F': '#90E050', 'Ne': '#B3E3F5', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
            'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30',
            'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
            'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7',
            'Mn': '#9C7AC7', 'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050',
            'Cu': '#C88033', 'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F',
            'As': '#BD80E3', 'Se': '#FFA100', 'Br': '#A62929', 'Kr': '#5CB8D1',
            'Rb': '#702EB0', 'Sr': '#00FF00', 'Y': '#94FFFF', 'Zr': '#94E0E0',
            'Nb': '#73C2C9', 'Mo': '#54B5B5', 'Tc': '#3B9E9E', 'Ru': '#248F8F',
            'Rh': '#0A7D8C', 'Pd': '#006985', 'Ag': '#C0C0C0', 'Cd': '#FFD98F',
            'In': '#A67573', 'Sn': '#668080', 'Sb': '#9E63B5', 'Te': '#D47A00',
            'I': '#940094', 'Xe': '#429EB0', 'Cs': '#57178F', 'Ba': '#00C900',
            'La': '#70D4FF', 'Ce': '#FFFFC7', 'Pr': '#D9FFC7', 'Nd': '#C7FFC7',
            'Pm': '#A3FFC7', 'Sm': '#8FFFC7', 'Eu': '#61FFC7', 'Gd': '#45FFC7',
            'Tb': '#30FFC7', 'Dy': '#1FFFC7', 'Ho': '#00FF9C', 'Er': '#00E675',
            'Tm': '#00D452', 'Yb': '#00BF38', 'Lu': '#00AB24', 'Hf': '#4DC2FF',
            'Ta': '#4DA6FF', 'W': '#2194D6', 'Re': '#267DAB', 'Os': '#266696',
            'Ir': '#175487', 'Pt': '#D0D0E0', 'Au': '#FFD123', 'Hg': '#B8B8D0',
            'Tl': '#A6544D', 'Pb': '#575961', 'Bi': '#9E4FB5', 'Po': '#AB5C00',
            'At': '#754F45', 'Rn': '#428296', 'Fr': '#420066', 'Ra': '#007D00',
            'Ac': '#70ABFA', 'Th': '#00BAFF', 'Pa': '#00A1FF', 'U': '#008FFF',
            'Np': '#0080FF', 'Pu': '#006BFF', 'Am': '#545CF2', 'Cm': '#785CE3',
            'Bk': '#8A4FE3', 'Cf': '#A136D4', 'Es': '#B31FD4', 'Fm': '#B31FBA',
            'Md': '#B30DA6', 'No': '#BD0D87', 'Lr': '#C70066', 'Rf': '#CC0059',
            'Db': '#D1004F', 'Sg': '#D90045', 'Bh': '#E00038', 'Hs': '#E6002E',
            'Mt': '#EB0026'
        }
        
        # Get colors for each atom
        colors = [element_colors.get(elem, '#CCCCCC') for elem in species]
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=[f"{species[i]}<br>({coords[i, 0]:.2f}, {coords[i, 1]:.2f}, {coords[i, 2]:.2f})" 
                  for i in range(len(species))],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        # Add unit cell edges
        lattice = structure.lattice
        origin = np.array([0, 0, 0])
        
        # Define unit cell vertices
        vertices = [
            origin,
            lattice.matrix[0],
            lattice.matrix[1],
            lattice.matrix[2],
            lattice.matrix[0] + lattice.matrix[1],
            lattice.matrix[0] + lattice.matrix[2],
            lattice.matrix[1] + lattice.matrix[2],
            lattice.matrix[0] + lattice.matrix[1] + lattice.matrix[2]
        ]
        
        # Define edges of unit cell
        edges = [
            (0, 1), (0, 2), (0, 3),  # from origin
            (1, 4), (1, 5),          # from a
            (2, 4), (2, 6),          # from b
            (3, 5), (3, 6),          # from c
            (4, 7), (5, 7), (6, 7)   # to opposite corner
        ]
        
        # Add unit cell edges to plot
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Update layout with white background
        fig.update_layout(
            title=f"Crystal Structure: {structure.composition.reduced_formula}",
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                aspectmode='cube',
                bgcolor='white',
                xaxis=dict(backgroundcolor='white', gridcolor='lightgray', showbackground=True),
                yaxis=dict(backgroundcolor='white', gridcolor='lightgray', showbackground=True),
                zaxis=dict(backgroundcolor='white', gridcolor='lightgray', showbackground=True)
            ),
            width=700,
            height=500,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig, structure
        
    except Exception as e:
        st.error(f"Error loading CIF file: {str(e)}")
        return None, None

# Configure Streamlit page
st.set_page_config(
    page_title="MatPropAI",
    page_icon="matpropai.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - moved before header
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .status-connected {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-mock {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .example-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    /* Text area styling - dark theme with nice colors */
    textarea {
        color: #e2e8f0 !important;
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        border-radius: 8px !important;
        font-size: 1.4rem !important;
        caret-color: #e2e8f0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    .stTextArea textarea {
        color: #e2e8f0 !important;
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        border-radius: 8px !important;
        font-size: 1.4rem !important;
        caret-color: #e2e8f0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stTextArea"] textarea {
        color: #e2e8f0 !important;
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        border-radius: 8px !important;
        font-size: 1.4rem !important;
        caret-color: #e2e8f0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    /* Text area focus state */
    textarea:focus {
        border-color: #63b3ed !important;
        box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.1) !important;
        outline: none !important;
    }
    .stTextArea textarea:focus {
        border-color: #63b3ed !important;
        box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.1) !important;
        outline: none !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #63b3ed !important;
        box-shadow: 0 0 0 3px rgba(99, 179, 237, 0.1) !important;
        outline: none !important;
    }
    /* Text input styling - reduced font size */
    input {
        color: #000000 !important;
        background-color: #ffffff !important;
        border: 1px solid #ddd !important;
        font-size: 0.85rem !important;
    }
    .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        border: 1px solid #ddd !important;
        font-size: 0.85rem !important;
    }
    /* Example queries text - keep black */
    .example-box {
        color: #000000;
    }
    /* Agent response text styling - larger font with proper wrapping */
    .stExpander .stMarkdown p, .stExpander .stCode {
        font-size: 1.4rem !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    /* Code blocks in responses - larger font with wrapping */
    code {
        font-size: 1.3rem !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    pre code {
        font-size: 1.3rem !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    /* General text in response containers with wrapping */
    .stMarkdown p {
        font-size: 1.4rem !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
    /* Ensure code blocks don't overflow */
    .stCode {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
    }
    /* Status badge in top right corner */
    .status-badge-corner {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 999;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 0.8rem;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Reduce main container top padding */
    .stMainBlockContainer {
        padding-top: 2rem !important;
    }
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Main header with inline logo - positioned right after CSS with reduced spacing
import base64

# Read and encode the logo
with open("matpropai.png", "rb") as f:
    logo_data = base64.b64encode(f.read()).decode()

with open("MAGE.png", "rb") as f:
    logo_name = base64.b64encode(f.read()).decode()

st.markdown(f'''
<div class="main-header" style="margin-top: -1rem; margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: center;">
    <img src="data:image/png;base64,{logo_name}" style="height: 5.5rem; margin-right: 0.1rem; margin-top: 3rem;" />
</div>
''', unsafe_allow_html=True)
st.markdown('<div class="main-header" style="font-size: 1.22rem; color: #666; margin-bottom: 0.5rem; margin-top: -2.0rem; text-align: center;">Materials Agent for Generative and Evaluative design</div>', unsafe_allow_html=True)

st.markdown("""
<script>
// Force text area styling with JavaScript
setTimeout(function() {
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(function(textarea) {
        textarea.style.color = '#000000';
        textarea.style.backgroundColor = '#ffffff';
    });
}, 1000);
</script>
""", unsafe_allow_html=True)

# Agent status - moved to top right corner
if AGENT_AVAILABLE:
    st.markdown('<div class="status-badge-corner">✅ MatProAI Agent Connected</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-badge-corner">⚠️ Mock Mode</div>', unsafe_allow_html=True)

# Sidebar for file upload and settings
with st.sidebar:
    st.header("📁 File Upload")
    uploaded_file = st.file_uploader(
        "Upload CIF File (Optional)",
        type=['cif'],
        help="Upload a CIF structure file for bulk modulus prediction"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Save uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display file info
        st.info(f"📊 File size: {len(uploaded_file.getbuffer())} bytes")
        
        # Add visualization button
        if st.button("🔍 Visualize Structure", use_container_width=True):
            st.session_state.show_visualization = True
            st.session_state.cif_file_path = temp_file_path
    
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        <div style="font-size: 0.9rem;">
        -This application Uses fine tuned LLM to Predict bulk modulus from material composition or structure
        and generates structures (CIF files) from desired bulk modulus values
        </div>
        """, unsafe_allow_html=True)
    
    # Available CIF Files section moved to sidebar
    # st.header("📁 Available CIF Files")
    
    # Get current CIF files (refreshed each time)
    try:
        cif_files = [f for f in os.listdir('.') if f.endswith('.cif')]
    except:
        cif_files = []
    
    # Force refresh of selectbox when refresh button is clicked
    if 'refresh_files' in st.session_state and st.session_state.refresh_files:
        # Clear the selectbox key to force refresh
        if 'cif_selector' in st.session_state:
            del st.session_state.cif_selector
    
    if cif_files:
        st.write("Select any CIF file to visualize:")
        
        # Sort files to show generated ones first, then others
        generated_files = [f for f in cif_files if f.startswith('generated_cif')]
        other_files = [f for f in cif_files if not f.startswith('generated_cif')]
        sorted_files = generated_files + other_files
        
        selected_cif = st.selectbox("Choose CIF file:", sorted_files, key="cif_selector")
        
        
        # Create columns for side-by-side buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Visualize", use_container_width=True):
                st.session_state.show_generated_viz = True
                st.session_state.selected_cif_path = selected_cif
        
        with col2:
            if st.button("📥 Download", use_container_width=True):
                with open(selected_cif, 'rb') as f:
                    st.download_button(
                        label="💾 Download CIF File",
                        data=f.read(),
                        file_name=selected_cif,
                        mime="chemical/x-cif"
                    )
    else:
        st.info("No CIF files found in current directory")

# Agent Response Section with side-by-side layout


# Dynamic messages for when no response is available
import time
import random

dynamic_messages = [
    "🔍 Waiting for your query...",
    #"💡 Ready to analyze materials and predict bulk modulus...",
    # "🧪 Standing by for structure generation requests...",
    "⚡ AI agent ready to process your materials science queries...",
    #"🔬 Ready to help with CIF files and bulk modulus predictions...",
    "📊 Awaiting your next materials property question..."
]

# Dynamic processing messages for when query is being processed
processing_messages = [
    "🧠 Working...",
    "📊 Running...",
    "⚡ Processing ..."
]

# Always show the response expander with processing indicator
processing = st.session_state.get('processing_query', False)
expander_title = "📋 AI Response" if not processing else "Generating Response..."

# Create side-by-side layout for response and visualization
col_response, col_viz = st.columns([4, 3])

with col_response:
    with st.expander(expander_title, expanded=True):
        # Check if we have a stored response to display
        if 'last_agent_response' in st.session_state and st.session_state.last_agent_response:
            st.code(st.session_state.last_agent_response, language="text")
        elif processing:
            # Show random processing message like coin toss
            random_message = random.choice(processing_messages)
            st.markdown(f"**{random_message}**")
            
            # Auto-refresh to create streaming effect during processing
            import threading
            
            def auto_refresh_processing():
                import time
                time.sleep(1.5)  # Slightly longer delay for random messages
                if st.session_state.get('processing_query', False):
                    st.rerun()
            
            # Start auto-refresh in background during processing
            threading.Thread(target=auto_refresh_processing, daemon=True).start()
        else:
            # Show random waiting message like coin toss
            random_message = random.choice(dynamic_messages)
            st.markdown(f"*{random_message}*")
            
            # Auto-refresh to create streaming effect
            import threading
            
            def auto_refresh():
                import time
                time.sleep(2.5)  # Slightly longer delay for random messages
                if not st.session_state.get('processing_query', False) and not st.session_state.get('last_agent_response'):
                    st.rerun()
            
            # Start auto-refresh in background only if not processing
            if not st.session_state.get('processing_query', False):
                threading.Thread(target=auto_refresh, daemon=True).start()

with col_viz:
    # Show Crystal Toolkit visualization if CIF file was generated
    if st.session_state.get('generated_cif_file') and os.path.exists(st.session_state.generated_cif_file):
        st.subheader("🔬 Crystal Structure")
        
        # Crystal Toolkit visualization
        iframe_html, structure = visualize_cif_structure_crystal_toolkit(st.session_state.generated_cif_file)
        if iframe_html and structure:
            st.components.v1.html(iframe_html, height=570, width=650)
            
            # # Show CIF content
            # st.subheader("📄 CIF Content")
            # with open(st.session_state.generated_cif_file, 'r') as f:
            #     cif_content = f.read()
            # st.code(cif_content, language="text")
    else:
        # Only show placeholder when processing a CIF generation query
        if st.session_state.get('processing_query', False) and 'last_agent_response' in st.session_state:
            if 'generate' in st.session_state.get('query_text', '').lower() and 'cif' in st.session_state.get('query_text', '').lower():
                st.markdown("*Generating crystal structure...*")
        # Otherwise show nothing in the visualization column

#showing cif in a show bar
if st.session_state.get('generated_cif_file') and os.path.exists(st.session_state.generated_cif_file):
    # Show CIF content
    st.subheader("📄 CIF Content")
    with open(st.session_state.generated_cif_file, 'r') as f:
        cif_content = f.read()
    st.code(cif_content, language="text")


# Initialize session state for query if not exists
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

# Clear the query text if it's empty to show placeholder
if st.session_state.query_text == "":
    query_value = ""
else:
    query_value = st.session_state.query_text

# Headers on the same line
col_header1, col_header2 = st.columns([2, 1])
with col_header1:
    st.markdown('<p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem; margin-top: 1rem;">Enter your query:</p>', unsafe_allow_html=True)
with col_header2:
    st.markdown('<p style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem; margin-top: 1rem;"><strong>💡 Example Queries</strong></p>', unsafe_allow_html=True)

# Main query input area with example queries on the right
col_query, col_examples = st.columns([2, 1])

with col_query:
    # Main query input (always visible below agent response)
    main_query = st.text_area(
        "Enter your query",
        value=st.session_state.query_text,
        height=160,
        placeholder="Type your question about bulk modulus, CIF generation, or material analysis...",
        key="main_query_input",
        label_visibility="collapsed"
    )
    
    # Update session state when query changes
    st.session_state.query_text = main_query
    
    # Buttons for main query
    col_main_btn1, col_main_btn2, col_main_btn3 = st.columns([1, 1, 2])

with col_examples:
    example_queries = [
        "Predict bulk modulus for Al2O3",
        "What is the bulk modulus of the uploaded structure?",
        "Generate CIF for Bulk Modulus 20 GPa"
    ]
    
    # Add CSS for left-aligned button text
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-child(2) button {
        text-align: left !important;
        justify-content: flex-start !important;
        padding-left: 12px !important;
    }
    div[data-testid="column"]:nth-child(2) button p {
        text-align: left !important;
        margin: 0 !important;
    }
    div[data-testid="column"]:nth-child(2) button div {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    for i, query in enumerate(example_queries):
        if st.button(f"📋 {query}", key=f"example_{i}", use_container_width=True):
            st.session_state.query_text = query
            st.rerun()
    
    # Custom CSS for blue submit button
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-child(1) button[kind="primary"] {
        background-color: #0066CC !important;
        border-color: #0066CC !important;
        color: white !important;
    }
    div[data-testid="column"]:nth-child(1) button[kind="primary"]:hover {
        background-color: #0052A3 !important;
        border-color: #0052A3 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with col_main_btn1:
        main_submit_button = st.button("🚀 Submit Query", type="primary", use_container_width=True, key="main_submit")
    
    with col_main_btn2:
        main_clear_button = st.button("🗑️ Clear", use_container_width=True, key="main_clear")
    
    if main_clear_button:
        st.session_state.query_text = ""
        # Clear agent response as well
        if 'last_agent_response' in st.session_state:
            del st.session_state.last_agent_response
        # Clear processing state
        st.session_state.processing_query = False
        st.session_state.generated_cif_file = None

        st.rerun()

# Handle query template from quick actions
if hasattr(st.session_state, 'query_template'):
    query = st.session_state.query_template
    del st.session_state.query_template
    st.rerun()

# Process query
if main_submit_button and main_query.strip():
    # Set processing state
    st.session_state.processing_query = True
    
    # Add file context only if uploaded and query is file-related
    file_related_keywords = ['cif', 'structure', 'file', 'predict', 'bulk modulus', 'formation energy', 'analyze', 'uploaded']
    query_is_file_related = uploaded_file is not None and any(keyword in main_query.lower() for keyword in file_related_keywords)
    
    if query_is_file_related:
        query_with_context = f"Using uploaded file: {temp_file_path}\n\nQuery: {main_query}"
    else:
        query_with_context = main_query
    
    # Process query with loading indicator
    try:
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        # Run the agent with spinner
        with st.spinner("🔬 Processing your query... This may take a few moments for CIF generation and analysis."):
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    result = loop.run_until_complete(call_agent_async(query_with_context))
                finally:
                    loop.close()
        
        # Display results
        stdout_content = stdout_buffer.getvalue()
        stderr_content = stderr_buffer.getvalue()
        
        if stdout_content:
            # Store the response in session state for persistence
            st.session_state.last_agent_response = stdout_content
            
            # Check if a CIF file was actually generated in this query
            lines = stdout_content.split('\n')
            generated_cif_file = None
            
            # Only look for CIF filename in the current response
            for line in lines:
                if "generated_cif_Bulk_Modulus_" in line and ".cif" in line:
                    import re
                    match = re.search(r'generated_cif_Bulk_Modulus_[\d.]+\.cif', line)
                    if match:
                        generated_cif_file = match.group()
                        break
            
            # Also check if the response contains "CIF Generation Complete" or similar indicators
            if not generated_cif_file:
                for line in lines:
                    if "CIF Generation Complete" in line or "File:" in line and ".cif" in line:
                        import re
                        match = re.search(r'generated_cif_Bulk_Modulus_[\d.]+\.cif', line)
                        if match:
                            generated_cif_file = match.group()
                            break
            
            # Store CIF file info if generated
            if generated_cif_file and os.path.exists(generated_cif_file):
                st.session_state.generated_cif_file = generated_cif_file
                st.success(f"🎉 CIF file generated: {generated_cif_file}")
            
            # Clear processing state and rerun to update the response expander
            st.session_state.processing_query = False
            st.rerun()
        
        # Hide warnings/errors - commented out
        # if stderr_content:
        #     with st.expander("⚠️ Warnings/Errors"):
        #         st.code(stderr_content, language="text")
        
    except Exception as e:
        # Clear processing state on error
        st.session_state.processing_query = False
        st.error(f"❌ Error processing query: {str(e)}")
        st.code(str(e), language="text")
    
    # Clean up temporary file
    if uploaded_file is not None and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
        except:
            pass

elif main_submit_button and not main_query.strip():
    st.warning("⚠️ Please enter a query before submitting!")

# CIF Structure Visualization Section
if hasattr(st.session_state, 'show_visualization') and st.session_state.show_visualization:
    st.header("🔬 Crystal Structure Visualization")
    
    if hasattr(st.session_state, 'cif_file_path'):
        # # Visualization type selector
        # viz_type = st.radio(
        #     "Choose visualization type:",
        #     ["Crystal Toolkit (Professional)"], # "Enhanced Plotly", "Simple Plotly"],
        #     index=0,
        #     help="Crystal Toolkit provides professional crystallographic visualization, Enhanced Plotly provides better atom coloring and bonds, Simple Plotly provides basic 3D visualization."
        # )
        
        if st.button("🔍 Visualize Structure"):
            # Using Crystal Toolkit (Professional) by default
            iframe_html, structure = visualize_cif_structure_crystal_toolkit(st.session_state.cif_file_path)
            if iframe_html and structure:
                st.components.v1.html(iframe_html, height=570)
                
                # Display structure information
                st.subheader("📊 Structure Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Formula:** {structure.composition.reduced_formula}")
                    st.write(f"**Space Group:** {structure.get_space_group_info()[1]}")
                    st.write(f"**Crystal System:** {structure.get_space_group_info()[0]}")
                with col2:
                    st.write(f"**Volume:** {structure.volume:.2f} Ų")
                    st.write(f"**Density:** {structure.density:.2f} g/cm³")
                    st.write(f"**Number of atoms:** {len(structure)}")
            
            st.session_state.show_visualization = False
            st.rerun()

# CIF files section has been moved to sidebar

# Display visualization for generated CIF files
if hasattr(st.session_state, 'show_generated_viz') and st.session_state.show_generated_viz:
    if hasattr(st.session_state, 'selected_cif_path'):
        st.header("🔬 Generated Structure Visualization")
        
        # Add visualization type selector for generated files
        viz_type_gen = st.radio(
            "Choose visualization type:",
            ["Enhanced Plotly", "Simple Plotly", "Crystal Toolkit (Professional)"],
            key="viz_type_generated",
            help="Enhanced Plotly provides better atom coloring and bonds, Simple Plotly provides basic 3D visualization, Crystal Toolkit provides professional crystallographic visualization."
        )
        
        # Buttons to show/hide visualization
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("🔍 Show Visualization", key="show_generated_viz_btn"):
                st.session_state.show_generated_visualization = True
        with col_btn2:
            if st.button("❌ Hide Visualization", key="hide_generated_viz_btn"):
                st.session_state.show_generated_visualization = False
        
        # Show visualization if enabled
        if st.session_state.get('show_generated_visualization', False):
            if viz_type_gen == "Crystal Toolkit (Professional)":
                iframe_html, structure = visualize_cif_structure_crystal_toolkit(st.session_state.selected_cif_path)
                if iframe_html is not None and structure is not None:
                    # Display the Crystal Toolkit viewer
                    st.components.v1.html(iframe_html, height=570)
            elif viz_type_gen == "Enhanced Plotly":
                fig, structure = visualize_cif_structure_enhanced(st.session_state.selected_cif_path)
                if fig is not None and structure is not None:
                    # Display the enhanced Plotly plot
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fig, structure = visualize_cif_structure(st.session_state.selected_cif_path)
                if fig is not None and structure is not None:
                    # Display the simple Plotly plot
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select visualization type and click 'Show Visualization' to display the structure")
            structure = None  # Initialize structure variable
        
        # Display structure information for both visualization types
        if structure is not None:
            
            # Display structure information
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.subheader("📋 Structure Information")
                st.write(f"**Formula:** {structure.composition.reduced_formula}")
                st.write(f"**Space Group:** {structure.get_space_group_info()[1]}")
                st.write(f"**Crystal System:** {structure.get_space_group_info()[0]}")
                st.write(f"**Number of Sites:** {len(structure.sites)}")
                
            with col_info2:
                st.subheader("📐 Lattice Parameters")
                lattice = structure.lattice
                st.write(f"**a:** {lattice.a:.3f} Å")
                st.write(f"**b:** {lattice.b:.3f} Å") 
                st.write(f"**c:** {lattice.c:.3f} Å")
                st.write(f"**α:** {lattice.alpha:.2f}°")
                st.write(f"**β:** {lattice.beta:.2f}°")
                st.write(f"**γ:** {lattice.gamma:.2f}°")
                st.write(f"**Volume:** {lattice.volume:.3f} Ų")
            
            # Add button to hide visualization
            if st.button("❌ Hide Generated Visualization"):
                st.session_state.show_generated_viz = False
                st.rerun()

# Auto-visualization section for newly generated CIF files
if hasattr(st.session_state, 'show_auto_viz') and st.session_state.show_auto_viz:
    if hasattr(st.session_state, 'auto_viz_file'):
        st.header("🔬 Auto-Generated Structure Visualization")
        
        # Add visualization type selector for auto-generated files
        viz_type_auto = st.radio(
            "Choose visualization type:",
            ["Enhanced Plotly", "Simple Plotly", "Crystal Toolkit (Professional)"],
            key="viz_type_auto",
            help="Enhanced Plotly provides better atom coloring and bonds, Simple Plotly provides basic 3D visualization, Crystal Toolkit provides professional crystallographic visualization."
        )
        
        # Create side-by-side layout
        col_viz, col_content = st.columns([2, 1])
        
        with col_viz:
            st.subheader("🔬 3D Visualization")
            # Buttons to show/hide auto-visualization
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("🔍 Show Visualization", key="show_auto_viz_btn"):
                    st.session_state.show_auto_visualization = True
            with col_btn2:
                if st.button("❌ Hide Visualization", key="hide_auto_viz_btn"):
                    st.session_state.show_auto_visualization = False
            
            # Show visualization if enabled
            if st.session_state.get('show_auto_visualization', False):
                if viz_type_auto == "Crystal Toolkit (Professional)":
                    iframe_html, structure = visualize_cif_structure_crystal_toolkit(st.session_state.auto_viz_file)
                    if iframe_html is not None and structure is not None:
                        # Display the Crystal Toolkit viewer
                        st.components.v1.html(iframe_html, height=570)
                elif viz_type_auto == "Enhanced Plotly":
                    fig, structure = visualize_cif_structure_enhanced(st.session_state.auto_viz_file)
                    if fig is not None and structure is not None:
                        # Display the enhanced Plotly plot
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, structure = visualize_cif_structure(st.session_state.auto_viz_file)
                    if fig is not None and structure is not None:
                        # Display the simple Plotly plot
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select visualization type and click 'Show Visualization' to display the structure")
                structure = None  # Initialize structure variable
        
        with col_content:
            # Display CIF content
            st.subheader("📄 CIF Content")
            try:
                with open(st.session_state.auto_viz_file, 'r') as f:
                    cif_content = f.read()
                st.code(cif_content, language="text", height=400)
            except Exception as e:
                st.error(f"Error reading CIF file: {e}")
            
            # Display structure information for auto-generated files
            if 'structure' in locals() and structure is not None:
                st.subheader("📊 Structure Info")
                st.write(f"**Formula:** {structure.composition.reduced_formula}")
                st.write(f"**Space Group:** {structure.get_space_group_info()[1]}")
                st.write(f"**Crystal System:** {structure.get_space_group_info()[0]}")
                st.write(f"**Volume:** {structure.volume:.2f} Ų")
                st.write(f"**Density:** {structure.density:.2f} g/cm³")
                st.write(f"**Atoms:** {len(structure)}")
        
        # Add button to hide auto-visualization
        if st.button("❌ Hide Auto-Visualization"):
            st.session_state.show_auto_viz = False
            st.rerun()


