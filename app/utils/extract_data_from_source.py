from app.utils.custom_logging import logger
import json


# ---------------------------------------------------------
# Safe conversion helpers
# ---------------------------------------------------------
def safe_text(value):
    """
    Converts any nested dict/list/other types into a clean string.
    Ensures PostgreSQL always receives SERIALIZABLE strings.
    """
    if value is None:
        return ""

    # If already a clean scalar, return it
    if isinstance(value, (str, int, float, bool)):
        return str(value)

    # If list → comma joined string
    if isinstance(value, list):
        return ", ".join([safe_text(v) for v in value])

    # If dict → JSON string
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    # Fallback
    return str(value)


# ---------------------------------------------------------
# Safe nested extraction
# ---------------------------------------------------------
def get_nested(d, keys, default=None):
    logger.debug(f"[get_nested] Accessing keys={keys} from dict")
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            logger.debug(f"[get_nested] Key {key} not found, returning default")
            return default
    return d


# ---------------------------------------------------------
# Extract SPL info and sanitize all fields
# ---------------------------------------------------------
def extract_spl_info(data_dict):
    logger.info("[extract_spl_info] Extracting SPL info from XML dict")

    doc = data_dict.get("document", {})

    # Basic fields
    setid = safe_text(get_nested(doc, ["setId", "@root"]))
    version = safe_text(get_nested(doc, ["versionNumber", "@value"]))
    effective_date = safe_text(get_nested(doc, ["effectiveTime", "@value"]))

    title = get_nested(doc, ["title"])
    title = safe_text(title if title not in (None, {}, []) else "")

    logger.debug(f"[extract_spl_info] setId={setid}, version={version}, date={effective_date}")

    # Defaults
    ndc = ""
    generic_name = ""
    dosage_form = ""
    label_name = ""
    active_ingredients = []
    inactive_ingredients = []
    description = ""
    indications = ""
    warnings = ""

    # Extract product data elements
    components = get_nested(doc, ["component", "structuredBody", "component"], [])

    if not isinstance(components, list):
        components = [components]

    logger.debug(f"[extract_spl_info] Found {len(components)} SPL sections")

    for comp in components:
        section = comp.get("section", {})
        code_display = safe_text(get_nested(section, ["code", "@displayName"]))

        logger.debug(f"[extract_spl_info] Processing section: {code_display}")

        # ------------------------------------------------------
        # PRODUCT SECTION
        # ------------------------------------------------------
        if code_display == "SPL product data elements section":
            logger.debug("[extract_spl_info] Extracting product info")

            product = get_nested(section, ["subject", "manufacturedProduct", "manufacturedProduct"], {})

            ndc = safe_text(get_nested(product, ["code", "@code"]))
            raw_name = get_nested(product, ["name"])
            label_name = safe_text(raw_name)
            dosage_form = safe_text(get_nested(product, ["formCode", "@displayName"]))
            generic_name = safe_text(get_nested(product, ["asEntityWithGeneric", "genericMedicine", "name"]))

            logger.debug(f"[extract_spl_info] Product: ndc={ndc}, label={label_name}, form={dosage_form}")

            # Ingredients
            ingredients = product.get("ingredient", [])
            if not isinstance(ingredients, list):
                ingredients = [ingredients]

            for ing in ingredients:
                name = safe_text(get_nested(ing, ["ingredientSubstance", "name"]))
                class_code = safe_text(get_nested(ing, ["@classCode"]))

                if class_code == "IACT":
                    inactive_ingredients.append(name)
                else:
                    qty = safe_text(get_nested(ing, ["quantity", "numerator", "@value"]))
                    unit = safe_text(get_nested(ing, ["quantity", "numerator", "@unit"]))

                    if qty and unit:
                        active_ingredients.append(f"{name} {qty}{unit}")
                    else:
                        active_ingredients.append(name)

        # ------------------------------------------------------
        # DESCRIPTION SECTION
        # ------------------------------------------------------
        elif code_display == "DESCRIPTION SECTION":
            logger.debug("[extract_spl_info] Extracting DESCRIPTION")
            text = get_nested(section, ["text", "paragraph"])
            if isinstance(text, list):
                description = safe_text(" ".join([safe_text(p.get("#text")) if isinstance(p, dict) else safe_text(p) for p in text]))
            else:
                description = safe_text(text)

        # ------------------------------------------------------
        # INDICATIONS SECTION
        # ------------------------------------------------------
        elif code_display == "INDICATIONS & USAGE SECTION":
            logger.debug("[extract_spl_info] Extracting INDICATIONS")
            text = get_nested(section, ["text", "paragraph"])
            indications = safe_text(text)

        # ------------------------------------------------------
        # WARNINGS SECTION
        # ------------------------------------------------------
        elif code_display == "WARNINGS SECTION":
            logger.debug("[extract_spl_info] Extracting WARNINGS")
            paragraphs = get_nested(section, ["component"], [])
            if not isinstance(paragraphs, list):
                paragraphs = [paragraphs]

            warning_texts = []
            for p in paragraphs:
                sec = p.get("section", {})
                t = safe_text(get_nested(sec, ["title", "content", "#text"]))
                if t:
                    warning_texts.append(t)
            warnings = safe_text(" ".join(warning_texts))

    logger.info(f"[extract_spl_info] Finished parsing SPL: setId={setid}")

    # ------------------------------------------------------
    # FINAL: return COMPLETELY SANITIZED flat string dict
    # ------------------------------------------------------
    return {
        "setId": safe_text(setid),
        "versionNumber": safe_text(version),
        "effectiveDate": safe_text(effective_date),
        "title": safe_text(title),
        "NDC": safe_text(ndc),
        "genericName": safe_text(generic_name),
        "dosageForm": safe_text(dosage_form),
        "labelName": safe_text(label_name),
        "activeIngredients": safe_text(active_ingredients),
        "inactiveIngredients": safe_text(inactive_ingredients),
        "description": safe_text(description),
        "indications": safe_text(indications),
        "warnings": safe_text(warnings),
    }
