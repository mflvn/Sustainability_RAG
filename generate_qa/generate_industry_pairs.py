import json
import os
import random
from typing import List, Dict

import anthropic
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load industry descriptions
with open('/homes/ml6823/fyp/Thesis/generate_qa/industry_descriptions.json', 'r') as f:
    INDUSTRY_DESCRIPTIONS = json.load(f)

# Define industry groups
INDUSTRY_GROUPS = {
    "Consumer Goods": ["b1-apparel-accessories-and-footwear", "b2-appliance-manufacturing", "b3-building-products-and-furnishings", "b4-e-commerce", "b5-household-and-personal-products", "b6-multiline-and-specialty-retailers-and-distributors"],
    "Extractives and Minerals Processing": ["b7-coal-operations", "b8-construction-materials", "b9-iron-and-steel-producers", "b10-metals-and-mining", "b11-oil-and-gas-exploration-and-production", "b12-oil-and-gas-midstream", "b13-oil-and-gas-refining-and-marketing", "b14-oil-and-gas-services"],
    "Financials": ["b15-asset-management-and-custody-activities", "b16-commercial-banks", "b17-insurance", "b18-investment-banking-and-brokerage", "b19-mortgage-finance"],
    "Food and Beverage": ["b20-agricultural-products", "b21-alcoholic-beverages", "b22-food-retailers-and-distributors", "b23-meat-poultry-and-dairy", "b24-non-alcoholic-beverages", "b25-processed-foods", "b26-restaurants"],
    "Health Care": ["b27-drug-retailers", "b28-health-care-delivery", "b29-health-care-distributors", "b30-managed-care", "b31-medical-equipment-and-supplies"],
    "Infrastructure": ["b32-electric-utilities-and-power-generators", "b33-engineering-and-construction-services", "b34-gas-utilities-and-distributors", "b35-home-builders", "b36-real-estate", "b37-real-estate-services", "b38-waste-management", "b39-water-utilities-and-services"],
    "Renewable Resources and Alternative Energy": ["b40-biofuels", "b41-forestry-management", "b42-fuel-cells-and-industrial-batteries", "b43-pulp-and-paper-products", "b44-solar-technology-and-project-developers", "b45-wind-technology-and-project-developers"],
    "Resource Transformation": ["b46-aerospace-and-defense", "b47-chemicals", "b48-containers-and-packaging", "b49-electrical-and-electronic-equipment", "b50-industrial-machinery-and-goods"],
    "Services": ["b51-casinos-and-gaming", "b52-hotels-and-lodging", "b53-leisure-facilities"],
    "Technology and Communications": ["b54-electronic-manufacturing-services-and-original-design", "b55-hardware", "b56-internet-media-and-services", "b57-semiconductors", "b58-software-and-it-services", "b59-telecommunication-services"],
    "Transportation": ["b60-air-freight-and-logistics", "b61-airlines", "b62-auto-parts", "b63-automobiles", "b64-car-rental-and-leasing", "b65-cruise-lines", "b66-marine-transportation", "b67-rail-transportation", "b68-road-transportation"]
}

# Define schema for LLM response
industry_pairing_schema = {
  "name": "industry_pairing_schema",
  "description": "Suggests industry pairings for IFRS sustainability reporting",
  "type": "object",
  "properties": {
    "pairings": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "industries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Industry codes (first three characters, e.g., 'b1-', 'b2-')"
          },
          "explanation": {
            "type": "string",
            "description": "Reason for comparing these industries in sustainability reporting"
          }
        },
        "required": ["industries", "explanation"]
      }
    }
  },
  "required": ["pairings"]
}

# Use the following industry groups as a reference: 
# {json.dumps(industry_groups, indent=2)}
def consult_llm_for_industry_pairing(client, industry_groups, industry_descriptions):
    prompt = f"""As a specialist consultant on IFRS sustainability reporting standards, please suggest 5 groups of 5 different industries that are most likely to come up when considering reporting standards. These should be industries where comparisons or relationships in sustainability reporting would be particularly relevant or insightful.

    
    Use the following industry groups and descriptions as a reference:

    {json.dumps({group: [f"{code}: {industry_descriptions.get(code, 'No description')}" for code in codes] for group, codes in industry_groups.items()}, indent=2)}

    For each suggestion, provide:
    1. The industries involved (using their codes, e.g., b1-apparel-accessories-and-footwear, b2-appliance-manufacturing, etc.)
    2. A brief explanation of why these industries are relevant to compare in terms of sustainability reporting.

    Use the provided schema to format your response."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "function", "function": {"name": "industry_pairing_schema", "parameters": industry_pairing_schema}}],
        tool_choice={"type": "function", "function": {"name": "industry_pairing_schema"}}
    )

    try:
        tool_call = tool_call = response.choices[0].message.tool_calls[0]
        arguments = tool_call.function.arguments
        return arguments
    except (IndexError, AttributeError) as e:
        print(f"Error parsing LLM response: {e}")
        return None

def generate_industry_pairs():
    llm_suggestions = consult_llm_for_industry_pairing(client, INDUSTRY_GROUPS, INDUSTRY_DESCRIPTIONS)
    
    if not llm_suggestions:
        print("Failed to generate industry pairs using LLM.")
        return []

    pairs = llm_suggestions
    return pairs

def save_industry_pairs(pairs: List[Dict], output_file: str):
    with open(output_file, 'w') as f:
        clean_pairs = json.loads(pairs)
        json.dump(clean_pairs, f, indent=2)

if __name__ == "__main__":
    output_file = "industry_pairs_5group.json"
    pairs = generate_industry_pairs()
    save_industry_pairs(pairs, output_file)
    # print(f"Generated {len(pairs)} industry pairs and saved to {output_file}")