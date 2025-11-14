"""
Available functions for analysis.

All functions defined in `_functions` are created by `client.py`.
"""

from sqlalchemy.sql import func

sql_coalesce = func.coalesce
sql_count = func.count
sql_norm = func.norm
sql_similarity = func.similarity
sql_encode = func.encode
sql_attack_success_any = func.attack_success_any
sql_defense_success_all = func.defense_success_all
sql_first_attack_success = func.first_attack_success
sql_first_defense_success = func.first_defense_success

FUNCTIONS = [
    """
CREATE OR REPLACE FUNCTION norm(arr numeric[])
RETURNS numeric AS $$
DECLARE
    sum_squares numeric := 0;
    i numeric; -- Declaration of the loop variable
BEGIN
    FOREACH i IN ARRAY arr LOOP
        sum_squares := sum_squares + i * i;
    END LOOP;
    RETURN sqrt(sum_squares);
END;
$$ LANGUAGE plpgsql IMMUTABLE;
""",
    """
CREATE OR REPLACE FUNCTION similarity(v1 float[], v2 float[])
RETURNS float AS $$
DECLARE
    dot_product float := 0;
    norm1 float := 0;
    norm2 float := 0;
    i int;
BEGIN
    -- Check if the vectors are not empty and have the same length
    IF v1 IS NULL OR v2 IS NULL OR array_length(v1, 1) != array_length(v2, 1) THEN
        RAISE EXCEPTION 'Vectors must be non-null and of the same length';
    END IF;
    
    -- Calculate dot product and norms of the vectors
    FOR i IN 1..array_length(v1, 1) LOOP
        dot_product := dot_product + v1[i] * v2[i];
        norm1 := norm1 + v1[i] * v1[i];
        norm2 := norm2 + v2[i] * v2[i];
    END LOOP;

    -- Calculate the cosine similarity
    RETURN dot_product / (sqrt(norm1) * sqrt(norm2));
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;
""",
    """
CREATE OR REPLACE FUNCTION sum_input_tokens(jsonb) RETURNS integer
LANGUAGE sql IMMUTABLE AS $$
    SELECT coalesce(
        sum((elem->>'input_tokens')::int),
        0
    )
    FROM jsonb_array_elements($1) AS elem
$$;
""",
    """
CREATE OR REPLACE FUNCTION sum_cached_input_tokens(jsonb) RETURNS integer
LANGUAGE sql IMMUTABLE AS $$
    SELECT coalesce(
        sum((elem->>'cached_input_tokens')::int),
        0
    )
    FROM jsonb_array_elements($1) AS elem
    WHERE (elem->>'cached_input_tokens') IS NOT NULL
$$;
""",
    """
CREATE OR REPLACE FUNCTION sum_output_tokens(jsonb) RETURNS integer
LANGUAGE sql IMMUTABLE AS $$
    SELECT coalesce(
        sum((elem->>'output_tokens')::int),
        0
    )
    FROM jsonb_array_elements($1) AS elem
$$;
""",
    """
CREATE OR REPLACE FUNCTION sum_request_duration(jsonb) RETURNS numeric
LANGUAGE sql IMMUTABLE AS $$
    SELECT round(
        coalesce(
            sum((elem->>'request_duration')::numeric),
            0
        )::numeric,
        2
    )
    FROM jsonb_array_elements($1) AS elem
$$;
""",
    """
CREATE OR REPLACE FUNCTION attack_success_any(traj jsonb)
RETURNS boolean AS $$
DECLARE
    step jsonb;
    val text;
BEGIN
    FOR step IN SELECT value FROM jsonb_array_elements(traj->'steps')
    LOOP
        val := step->'info'->>'attack_success';
        IF val IS NOT NULL AND val::boolean IS TRUE THEN
            RETURN TRUE;
        END IF;
    END LOOP;
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;
""",
    """
CREATE OR REPLACE FUNCTION first_attack_success(traj jsonb)
RETURNS boolean
LANGUAGE sql IMMUTABLE STRICT AS $$
    SELECT attack_success_any($1)
$$;
""",
    """
CREATE OR REPLACE FUNCTION defense_success_all(traj jsonb)
RETURNS boolean AS $$
DECLARE
    step jsonb;
    val text;
BEGIN
    FOR step IN SELECT value FROM jsonb_array_elements(traj->'steps')
    LOOP
        val := step->'info'->>'defense_success';
        IF val IS NOT NULL AND val::boolean IS FALSE THEN
            RETURN FALSE;
        END IF;
    END LOOP;
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;
""",
    """
CREATE OR REPLACE FUNCTION first_defense_success(traj jsonb)
RETURNS boolean
LANGUAGE sql IMMUTABLE STRICT AS $$
    SELECT defense_success_all($1)
$$;
""",
    """
CREATE OR REPLACE FUNCTION async_total_requests(stats jsonb) RETURNS numeric
LANGUAGE sql IMMUTABLE AS $$
    SELECT coalesce((stats->>'total_async_requests')::numeric, 0)
$$;
""",
    """
CREATE OR REPLACE FUNCTION async_cached_requests(stats jsonb) RETURNS numeric
LANGUAGE sql IMMUTABLE AS $$
    SELECT coalesce((stats->>'cached_requests')::numeric, 0)
$$;
""",
]

if __name__ == "__main__":
    for func in FUNCTIONS:
        print(func)
