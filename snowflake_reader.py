"""
snowflake_reader.py
Reads PRD_PSAS_ANALYTICS_DB.GOLD_TRANSPORTATION.vw_tsp_lastmile_daily
from Snowflake into a pandas DataFrame.

Authentication: externalbrowser (Okta / Azure AD SSO).
No password or keys required — same login you use in Snowflake's web UI.

Usage:
    from snowflake_reader import load_lastmile_daily
    df = load_lastmile_daily()
"""

from __future__ import annotations

import pandas as pd
import snowflake.connector

_QUERY = """
    SELECT *
    FROM PRD_PSAS_ANALYTICS_DB.GOLD_TRANSPORTATION.vw_tsp_lastmile_daily
"""


def load_lastmile_daily(
    account: str = "MCKESSON-PSAS2",
    user: str = "MASOOD.GHASEMI@MCKESSON.CA",
    warehouse: str = "PRD_PSAS_ANALYTICS_TRANSPORTATION_WH",
    role: str = "PRD_PSAS_ANALYTICS_TRANSPORTATION_FR",
) -> pd.DataFrame:
    """
    Return SELECT * FROM vw_tsp_lastmile_daily as a pandas DataFrame.
    Authenticates via externalbrowser (Okta / Azure AD SSO).
    """
    conn = snowflake.connector.connect(
        account       = account,
        user          = user,
        authenticator = "externalbrowser",
        warehouse     = warehouse,
        role          = role,
        database      = "PRD_PSAS_ANALYTICS_DB",
        schema        = "GOLD_TRANSPORTATION",
    )
    try:
        df = pd.read_sql(_QUERY, conn)
    finally:
        conn.close()

    return df
