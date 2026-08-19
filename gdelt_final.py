import duckdb
# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "gdelt_merged_2018_2025.csv"     # Existing ~1.5GB file
OUTPUT_FILE = "gdelt_monthly_features_fixed.csv"   # Final aggregated output

# =========================================================
# CONNECT TO DUCKDB (DISK-BACKED, RAM SAFE)
# =========================================================

con = duckdb.connect()

# =========================================================
# FULL AGGREGATION QUERY
# =========================================================

query = f"""
COPY (

    -------------------------------------------------------
    -- STEP 0: COUNTRY CODE NORMALIZATION (ISO3 → FIPS)
    -------------------------------------------------------
    WITH country_map AS (
        SELECT * FROM (VALUES
            ('USA','US'), ('IND','IN'), ('RUS','RU'), ('CHN','CN'),
            ('BGD','BD'), ('PAK','PK'), ('JPN','JP'), ('DEU','DE'),
            ('GBR','GB'), ('FRA','FR'), ('NLD','NL'), ('SAU','SA'),
            ('ARE','AE'), ('IDN','ID'), ('VNM','VN'), ('MYS','MY'),
            ('KOR','KR'), ('IRN','IR'), ('IRQ','IQ'), ('TUR','TR'),
            ('UKR','UA'), ('AFG','AF'), ('LKA','LK'), ('NPL','NP'),
            ('EGY','EG')
        ) AS t(iso3, fips)
    ),

    -------------------------------------------------------
    -- STEP 1: NORMALIZE + TYPE SAFETY
    -------------------------------------------------------
    normalized AS (
        SELECT
            e.Event_ID,
            e.Event_Country,
            CAST(e.Year AS INTEGER)  AS Year,
            CAST(e.Month AS INTEGER) AS Month,

            e.EventRootCode,
            e.EventCode,
            e.Goldstein_Score,
            e.NumMentions,
            e.NumSources,
            e.AvgTone,

            cm1.fips AS Actor1_FIPS,
            cm2.fips AS Actor2_FIPS

        FROM read_csv_auto('{INPUT_FILE}') e
        LEFT JOIN country_map cm1
            ON e.Actor1CountryCode = cm1.iso3
        LEFT JOIN country_map cm2
            ON e.Actor2CountryCode = cm2.iso3
    ),

    -------------------------------------------------------
    -- STEP 2: EVENT-LEVEL DEDUPLICATION
    -- Keep the most reported version of each event
    -------------------------------------------------------
    dedup AS (
        SELECT DISTINCT ON (Event_ID)
            *
        FROM normalized
        ORDER BY Event_ID, NumMentions DESC
    ),

    -------------------------------------------------------
    -- STEP 3: CORE MONTHLY AGGREGATION
    -------------------------------------------------------
    base AS (
        SELECT
            Event_Country,
            Year,
            Month,

            COUNT(*) AS Total_Event_Count,
            AVG(Goldstein_Score) AS Avg_Goldstein,
            AVG(AvgTone) AS Avg_Tone,
            SUM(NumMentions) AS Total_Mentions,
            SUM(NumSources) AS Total_Sources,

            -- Severity × Media Attention
            SUM(ABS(Goldstein_Score) * LN(NumMentions + 1)) AS Shock_Intensity

        FROM dedup
        GROUP BY Event_Country, Year, Month
    ),

    -------------------------------------------------------
    -- STEP 4: SHOCK TYPE COUNTS
    -------------------------------------------------------
    shocks AS (
        SELECT
            Event_Country,
            Year,
            Month,

            SUM(EventRootCode IN (18,19,20)) AS Conflict_Event_Count,
            SUM(EventRootCode IN (14,15))    AS Protest_Event_Count,
            SUM(EventCode IN ('163','173')) AS Trade_Shock_Count,
            SUM(EventCode = '172')           AS Sanction_Threat_Count

        FROM dedup
        GROUP BY Event_Country, Year, Month
    ),

    -------------------------------------------------------
    -- STEP 5: DIRECTIONAL SHOCKS (FIXED)
    -------------------------------------------------------
    directional AS (
        SELECT
            Event_Country,
            Year,
            Month,

            -- Incoming: another country acts ON this country
            SUM(
                Actor2_FIPS = Event_Country
                AND Actor1_FIPS IS NOT NULL
                AND Actor1_FIPS != Event_Country
            ) AS Incoming_Shock_Count,

            -- Outgoing: this country acts ON another country
            SUM(
                Actor1_FIPS = Event_Country
                AND Actor2_FIPS IS NOT NULL
                AND Actor2_FIPS != Event_Country
            ) AS Outgoing_Shock_Count

        FROM dedup
        GROUP BY Event_Country, Year, Month
    )

    -------------------------------------------------------
    -- STEP 6: FINAL FEATURE TABLE
    -------------------------------------------------------
    SELECT
        b.Event_Country,
        b.Year,
        b.Month,

        b.Total_Event_Count,
        b.Avg_Goldstein,
        b.Avg_Tone,
        b.Total_Mentions,
        b.Total_Sources,
        b.Shock_Intensity,

        s.Conflict_Event_Count,
        s.Protest_Event_Count,
        s.Trade_Shock_Count,
        s.Sanction_Threat_Count,

        d.Incoming_Shock_Count,
        d.Outgoing_Shock_Count,
        (d.Outgoing_Shock_Count - d.Incoming_Shock_Count) AS Net_Hostility,

        -- Density metrics (safe divide)
        s.Conflict_Event_Count * 1.0 / NULLIF(b.Total_Event_Count, 0) AS Conflict_Density,
        s.Protest_Event_Count * 1.0 / NULLIF(b.Total_Event_Count, 0) AS Protest_Density,
        s.Trade_Shock_Count * 1.0 / NULLIF(b.Total_Event_Count, 0) AS Trade_Shock_Density,
        s.Sanction_Threat_Count * 1.0 / NULLIF(b.Total_Event_Count, 0) AS Sanction_Threat_Density

    FROM base b
    LEFT JOIN shocks s
        USING (Event_Country, Year, Month)
    LEFT JOIN directional d
        USING (Event_Country, Year, Month)

)
TO '{OUTPUT_FILE}'
WITH (HEADER, DELIMITER ',');

"""

# =========================================================
# EXECUTE
# =========================================================

con.execute(query)

print("✅ SUCCESS")
print("• Monthly aggregation complete")
print("• Directional shocks fixed")
print("• Memory-safe (DuckDB)")
print(f"• Output written to: {OUTPUT_FILE}")
