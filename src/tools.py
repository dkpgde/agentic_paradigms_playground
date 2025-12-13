import difflib
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("SCM_Tools")

DB_PARTS = {"Engine": "ID-999", "Tyre": "ID-100", "Windshield": "ID-555", "Brake": "ID-200"}
DB_STOCK = {"ID-999": 4, "ID-100": 200, "ID-555": 15, "ID-200": 50}
DB_SUPPLIERS = {"ID-999": "Stuttgart", "ID-100": "Munich", "ID-555": "Hamburg", "ID-200": "Berlin"}
DB_SHIPPING = {"Stuttgart": 150, "Munich": 50, "Hamburg": 80, "Berlin": 60}

#  INVENTORY TOOLS 

def get_part_id(part_name: str) -> str:
    """
    Retrieves the technical Part ID for a given English part name (e.g., "ID-999" or an error message).
    USE THIS FIRST. You cannot check stock or location without an ID.
    
    Args:
        part_name: The common name of the part (e.g., "Engine", "Tire").
    """
    logger.info(f"get_part_id called with: {part_name}")
    valid_parts = list(DB_PARTS.keys())
    # Exact match
    for k in valid_parts:
        if k.lower() == part_name.lower(): return DB_PARTS[k]
    # Fuzzy match
    matches = difflib.get_close_matches(part_name, valid_parts, n=1, cutoff=0.5)
    return DB_PARTS[matches[0]] if matches else "ERROR: Part not found."

def get_stock_level(part_id: str) -> str:
    """
    Checks the current inventory quantity for a specific Part ID.
    
    Args:
        part_id: The technical ID (must start with "ID-", e.g., "ID-100").
    """
    logger.info(f"get_stock_level called with: {part_id}")
    clean_id = str(part_id).strip()
    if not clean_id.startswith("ID-"): clean_id = f"ID-{clean_id}"
    val = DB_STOCK.get(clean_id)
    return str(val) if val is not None else "ERROR: ID not found in stock DB."

#  LOGISTICS TOOLS 

def get_supplier_location(part_id: str) -> str:
    """
    Finds the city where the supplier for a specific Part ID is located.
    
    Args:
        part_id: The technical ID (must start with "ID-", e.g., "ID-100").
    """
    logger.info(f"get_supplier_location called with: {part_id}")
    clean_id = str(part_id).strip()
    if not clean_id.startswith("ID-"): clean_id = f"ID-{clean_id}"
    return DB_SUPPLIERS.get(clean_id, "ERROR: ID not found in supplier DB.")

def get_shipping_cost(city: str) -> str:
    """
    Calculates the shipping cost to transport items from a specific Supplier City.
    
    Args:
        city: The name of the city (e.g., "Stuttgart", "Berlin").
    """
    logger.info(f"get_shipping_cost called with: {city}")
    city_str = str(city)
    for valid_city in DB_SHIPPING.keys():
        if valid_city.lower() in city_str.lower():
            return f"{DB_SHIPPING[valid_city]} EUR"
    return "ERROR: City not found in logistics DB (Must be Stuttgart, Munich, Hamburg, or Berlin)."