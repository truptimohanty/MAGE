#!/usr/bin/env python3
"""
Formation Energy API Server
Runs in matgl_env environment to predict formation energies using M3GNet
"""

from flask import Flask, request, jsonify
import matgl
from pymatgen.core import Structure
import os
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variable
model = None

def load_formation_energy_model():
    """Load the M3GNet formation energy model"""
    global model
    try:
        logger.info("Loading M3GNet formation energy model...")
        model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/predict_formation_energy', methods=['POST'])
def predict_formation_energy():
    """
    Predict formation energy for a given CIF file
    Expected JSON payload: {"cif_file_path": "path/to/file.cif"}
    """
    try:
        if model is None:
            return jsonify({
                "status": "error",
                "error": "Model not loaded"
            }), 500
        
        data = request.get_json()
        if not data or 'cif_file_path' not in data:
            return jsonify({
                "status": "error",
                "error": "Missing cif_file_path in request"
            }), 400
        
        cif_file_path = data['cif_file_path']
        
        # Check if file exists
        if not os.path.exists(cif_file_path):
            return jsonify({
                "status": "error",
                "error": f"CIF file not found: {cif_file_path}"
            }), 404
        
        # Load structure from CIF file
        logger.info(f"Loading structure from {cif_file_path}")
        struct = Structure.from_file(cif_file_path)
        
        # Predict formation energy
        logger.info("Predicting formation energy...")
        eform = model.predict_structure(struct)
        formation_energy = float(eform)
        
        logger.info(f"Predicted formation energy: {formation_energy:.3f} eV/atom")
        
        return jsonify({
            "status": "success",
            "formation_energy_eV_per_atom": formation_energy,
            "formula": struct.composition.reduced_formula,
            "num_atoms": len(struct),
            "report": f"Predicted formation energy for {struct.composition.reduced_formula}: {formation_energy:.3f} eV/atom"
        })
        
    except Exception as e:
        logger.error(f"Error predicting formation energy: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/predict_formation_energy_from_cif_content', methods=['POST'])
def predict_formation_energy_from_cif_content():
    """
    Predict formation energy from CIF content directly
    Expected JSON payload: {"cif_content": "CIF file content as string"}
    """
    try:
        if model is None:
            return jsonify({
                "status": "error",
                "error": "Model not loaded"
            }), 500
        
        data = request.get_json()
        if not data or 'cif_content' not in data:
            return jsonify({
                "status": "error",
                "error": "Missing cif_content in request"
            }), 400
        
        cif_content = data['cif_content']
        
        # Save CIF content to temporary file
        temp_cif_path = "temp_formation_energy_prediction.cif"
        with open(temp_cif_path, 'w') as f:
            f.write(cif_content)
        
        try:
            # Load structure from temporary CIF file
            logger.info("Loading structure from CIF content")
            struct = Structure.from_file(temp_cif_path)
            
            # Predict formation energy
            logger.info("Predicting formation energy...")
            eform = model.predict_structure(struct)
            formation_energy = float(eform)
            
            logger.info(f"Predicted formation energy: {formation_energy:.3f} eV/atom")
            
            return jsonify({
                "status": "success",
                "formation_energy_eV_per_atom": formation_energy,
                "formula": struct.composition.reduced_formula,
                "num_atoms": len(struct),
                "report": f"Predicted formation energy for {struct.composition.reduced_formula}: {formation_energy:.3f} eV/atom"
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_cif_path):
                os.remove(temp_cif_path)
        
    except Exception as e:
        logger.error(f"Error predicting formation energy: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Load the model on startup
    if load_formation_energy_model():
        logger.info("Starting Formation Energy API server on http://localhost:5001")
        app.run(host='127.0.0.1', port=5001, debug=False)
    else:
        logger.error("Failed to start server - model could not be loaded")
