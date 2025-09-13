from unsloth import FastLanguageModel
import torch
import argparse
import os
import sys
from dotenv import load_dotenv
from pymatgen.core import Composition, Structure, Lattice
from transformers import TextStreamer
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
import asyncio
import requests
import json

# Load environment variables
load_dotenv()

# Configuration constants
APP_NAME = "Bulk_Modulus_Agent"
USER_ID = "user1234"
SESSION_ID = "session1234"

# Prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}

### Input:
{}

### Response:
{}"""



# Configuration
max_seq_length = 7000
dtype = None  # auto-detect
load_in_4bit = True


# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=r".\BM_lora_model_4bit_mistral7b_unified_epoch10",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)
text_streamer = TextStreamer(tokenizer)


def predict_Bulk_Modulus_from_model_given_structure(cif_file_path: str):
    """
Predict Bulk Modulus using a fine-tuned LLM model.
    Args:
        cif file path (str): Material structure file path, e.g. "Al2O3.cif".
    Returns:
        float: Prdicted Bulk modulus in GPa.
    """

    with open(cif_file_path, "r") as f:
        cif_content = f.read()
    # Prepare input description
    str1 = Structure.from_file(cif_file_path)
    user_input = f"composition:{str1.composition.reduced_formula} cif:{cif_content}"
    #print(f"[Agent] Processing: {str1.composition.reduced_formula}")   

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Predict Bulk modulus (GPa) of the material based on the given description.",
                user_input,
                ""  # leave blank for generation
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    
    # Generate without text streamer to avoid showing full prompt
    outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
    raw_output = tokenizer.batch_decode(outputs)

    # Extract the model response
    response_start = raw_output[0].split("### Response:")[-1]
    bulk_modulus = response_start.strip("</s>").strip()

    #print(f"[Agent Output] Predicted Bulk Modulus for the given {cif_file_path}: {bulk_modulus} GPa")
    
    return {
            "status": "success",
            "report": (
                f"The predicted bulk modulus for the given {cif_file_path}: {bulk_modulus} GPa"
            ),}



def predict_Bulk_Modulus_from_model_given_composition(composition: str):
    """
    Predict Bulk Modulus using a fine-tuned LLM model.
    Args:
        composition (str): Material composition, e.g. "Al2O3".
    Returns:
        float: Prdicted Bulk modulus in GPa.
    """

    # Prepare input description
    comp = Composition(composition)
    user_input = f"composition:{comp.reduced_formula}"
    print(f"[Agent] Processing: {comp.reduced_formula}")

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Predict Bulk modulus (GPa) of the material based on the given description.",
                user_input,
                ""  # leave blank for generation
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    
    # Generate without text streamer to avoid showing full prompt
    outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
    raw_output = tokenizer.batch_decode(outputs)

    # Extract the model response
    response_start = raw_output[0].split("### Response:")[-1]
    bulk_modulus = response_start.strip("</s>").strip()

    #print(f"[Agent Output] Predicted Bulk Modulus for the {composition}: {bulk_modulus} GPa")
    
 
    return {
            "status": "success",
            "report": (
                f"The predicted bulk modulus for {composition}: {bulk_modulus} GPa"
            ),}




def generate_Structure_from_model_given_Bulk_Modulus(bulk_modulus: float):
    """
Generate Structure (CIF) from given Bulk Modulus value in GPa using a fine-tuned LLM model.
    Args:
        bulk_modulus (float): Predicted Bulk Modulus value in GPa.
    Returns:
        str: (cif_text, outfile_path): Generated CIF text and saved file path.
    """

    # Debug: Check if model is loaded (console only)
    sys.stderr.write(f"[DEBUG] Model loaded: {model is not None}\n")
    sys.stderr.write(f"[DEBUG] Tokenizer loaded: {tokenizer is not None}\n")

    # Prepare input description
    user_input = f"Bulk Modulus = {bulk_modulus} GPa"
    sys.stderr.write(f"[Agent Input] {user_input}\n")

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Generate CIF for a material based on the given description.",
                user_input,
                ""  # leave blank for generation
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    
    outputs = model.generate(**inputs, do_sample= True, temperature=1.0, max_new_tokens=3000, use_cache=True)
    raw_output = tokenizer.batch_decode(outputs)

    # Extract the response and clean the CIF content
    response_start = raw_output[0].split("### Response:")[-1]
    #cleaned_cif_content = response_start.strip("<|end_of_text|>").strip()
    cleaned_cif_content = response_start.strip("</s>").strip()

    # Print the CIF content to console only
    sys.stderr.write(f"[DEBUG] Generated CIF Content Length: {len(cleaned_cif_content)} characters\n")
    sys.stderr.write(f"[DEBUG] First 200 chars: {cleaned_cif_content[:200]}...\n")

# Save the cleaned CIF content to the specified path
    outfile = f"generated_cif_Bulk_Modulus_{bulk_modulus}.cif"
    with open(outfile, "w") as f:
        f.write(cleaned_cif_content)
    sys.stderr.write(f"[DEBUG] CIF file saved: {outfile}\n")

    # Predict bulk modulus of the generated CIF
    predicted_result = predict_Bulk_Modulus_from_model_given_structure(outfile)
    predicted_bm = predicted_result['report'] if 'report' in predicted_result else "None"
    
    # Predict formation energy of the generated CIF
    formation_energy_result = predict_formation_energy_from_cif_file(outfile)
    formation_energy = formation_energy_result['report'] if 'report' in formation_energy_result else "None"
    
    # Print file info and predicted bulk modulus
    print(f"✅ CIF Generation Complete!")
    print(f"📁 File: {outfile}")
    print(f"🎯 Predicted BM: {predicted_bm}")
    print(f"⚡ Formation Energy: {formation_energy}")

    return {
            "status": "success",
            "filename": outfile,
            "predicted_bulk_modulus": predicted_bm,
            "formation_energy": formation_energy,
            "cif_content": cleaned_cif_content,
            "report": (
                f" CIF Generation Complete!\n"
                #f"File: {outfile}\n"
                f"predicted bulk modulus: {predicted_bm}\n"
                f"formation energy: {formation_energy}\n\n"
                f"Generated CIF Content:\n{cleaned_cif_content}"
            ),}



def predict_formation_energy_from_cif_file(cif_file_path: str) -> dict:
    """Predict formation energy using M3GNet model via API call.

    Args:
        cif_file_path (str): Path to the CIF file.
        
    Returns:
        dict: status and result or error msg.
    """
    try:
        # Check if formation energy API server is running
        api_url = "http://localhost:5001/predict_formation_energy"
        
        # Prepare the request payload
        payload = {"cif_file_path": cif_file_path}
        
        # Make API call
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if result["status"] == "success":
                return {
                    "status": "success",
                    "formation_energy": result["formation_energy_eV_per_atom"],
                    "formula": result["formula"],
                    "report": result["report"]
                }
            else:
                return {
                    "status": "error",
                    "error_message": result.get("error", "Unknown error from formation energy API")
                }
        else:
            return {
                "status": "error",
                "error_message": f"Formation energy API returned status code {response.status_code}"
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "error_message": "Formation energy API server is not running. Please start it using start_formation_energy_server.bat"
        }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error_message": "Formation energy prediction timed out"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error calling formation energy API: {str(e)}"
        }


root_agent = Agent(

    name="Bulk_Modulus_and_CIF_Generator_Agent",
    model="gemini-2.0-flash",
    # model = llm,
    description=(
        "Agent to predict material properties like bulk modulus, generate CIF structures, and predict formation energies."
    ),
    instruction=(
        "You are a helpful agent who can: \n"
        "1) Predict bulk modulus from composition/structure \n"
        "2) Generate CIF structures from bulk modulus values \n"
        "3) Predict formation energies from CIF files. \n"
        "When calling functions, ALWAYS show the complete function output including file names, predicted values, do not show the contents of the CIF file. Do not modify or summarize the function results."
        
    ),
    tools=[predict_Bulk_Modulus_from_model_given_structure, 
    generate_Structure_from_model_given_Bulk_Modulus,
    predict_Bulk_Modulus_from_model_given_composition,
    predict_formation_energy_from_cif_file],
)


# Session and Runner
async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    return session, runner

# Agent Interaction
async def call_agent_async(query):
    #print(f"[Agent] Received query: {query}")
    content = types.Content(role='user', parts=[types.Part(text=query)])
    session, runner = await setup_session_and_runner()
    events = runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    async for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print(f"[Agent] Response: {final_response}")
            return final_response
    
    return "No response received from agent"


if __name__ == "__main__":
    # Example usage when running directly
    asyncio.run(call_agent_async("Generate CIF for Bulk Modulus 20 GPa"))
