import pandas as pd
from app.utils.custom_logging import logger

def build_text_data(data):
    logger.info("[build_text_data] Starting transformation")

    def safe(v):
        return v if (pd.notna(v) and v != "") else "Not available"

    logger.info(f"[build_text_data] Input rows: {len(data)}")
    logger.info("[build_text_data] Creating text_data column...")

    data = data.assign(
        text_data=lambda df: df.apply(
            lambda row: (
                f"setId: {safe(row['setId'])}, "
                f"genericName: {safe(row['genericName'])}, "
                f"activeIngredients: {safe(row['activeIngredients'])}, "
                f"inactiveIngredients: {safe(row['inactiveIngredients'])}, "
                f"description: {safe(row['description'])}, "
                f"indications: {safe(row['indications'])}, "
                f"warnings: {safe(row['warnings'])}"
            ),
            axis=1
        )
    )

    logger.info("[build_text_data] text_data column generated successfully")
    logger.info("[build_text_data] Preparing outputs: db_data + text_data")

    db_data = data.iloc[:, :8]
    text_data = data.iloc[:, -1]

    logger.info(f"[build_text_data] Returning db_rows={len(db_data)} and text_rows={len(text_data)}")

    return db_data, text_data
