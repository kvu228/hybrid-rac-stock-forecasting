"""stored procedures for RAC (in-DB computing)

Revision ID: 0007_stored_procedures
Revises: 0006_indexes
Create Date: 2026-04-15
"""

from __future__ import annotations

from alembic import op

revision = "0007_stored_procedures"
down_revision = "0006_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
CREATE OR REPLACE FUNCTION find_similar_patterns(
  query_vec vector(128),
  k_neighbors INTEGER DEFAULT 20,
  similarity_threshold DOUBLE PRECISION DEFAULT 0.7,
  filter_symbol TEXT DEFAULT NULL
)
RETURNS TABLE (
  neighbor_id BIGINT,
  neighbor_symbol TEXT,
  neighbor_label SMALLINT,
  neighbor_return DOUBLE PRECISION,
  cosine_distance DOUBLE PRECISION,
  window_start TIMESTAMPTZ,
  window_end TIMESTAMPTZ
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    pe.id,
    pe.symbol,
    pe.label,
    pe.future_return,
    (pe.embedding <=> query_vec) AS cos_dist,
    pe.window_start,
    pe.window_end
  FROM pattern_embeddings pe
  WHERE (filter_symbol IS NULL OR pe.symbol = filter_symbol)
    AND (pe.embedding <=> query_vec) < (1.0 - similarity_threshold)
  ORDER BY pe.embedding <=> query_vec
  LIMIT k_neighbors;
END;
$$ LANGUAGE plpgsql;
""".strip()
    )

    op.execute(
        """
CREATE OR REPLACE FUNCTION compute_rac_context(
  query_vec vector(128),
  k_neighbors INTEGER DEFAULT 20
)
RETURNS TABLE (
  total_neighbors INTEGER,
  avg_cosine_dist DOUBLE PRECISION,
  label_distribution JSONB,
  avg_future_return DOUBLE PRECISION,
  stddev_future_return DOUBLE PRECISION,
  dominant_label SMALLINT,
  confidence DOUBLE PRECISION
) AS $$
BEGIN
  RETURN QUERY
  WITH neighbors AS (
    SELECT
      pe.label,
      pe.future_return,
      (pe.embedding <=> query_vec) AS cos_dist
    FROM pattern_embeddings pe
    ORDER BY pe.embedding <=> query_vec
    LIMIT k_neighbors
  ),
  label_counts AS (
    SELECT label, COUNT(*)::INTEGER AS cnt
    FROM neighbors
    GROUP BY label
  )
  SELECT
    COUNT(*)::INTEGER,
    AVG(n.cos_dist),
    jsonb_object_agg(lc.label::TEXT, lc.cnt),
    AVG(n.future_return),
    STDDEV(n.future_return),
    (SELECT lc2.label FROM label_counts lc2 ORDER BY lc2.cnt DESC LIMIT 1),
    (SELECT MAX(lc3.cnt)::DOUBLE PRECISION / k_neighbors FROM label_counts lc3)
  FROM neighbors n
  LEFT JOIN label_counts lc ON TRUE;
END;
$$ LANGUAGE plpgsql;
""".strip()
    )

    op.execute(
        """
CREATE OR REPLACE FUNCTION get_distance_to_nearest_sr(
  p_symbol TEXT,
  p_current_price DOUBLE PRECISION
)
RETURNS TABLE (
  dist_to_support DOUBLE PRECISION,
  dist_to_resistance DOUBLE PRECISION
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    (SELECT MIN(ABS(p_current_price - price_level))
     FROM support_resistance_zones
     WHERE symbol = p_symbol AND zone_type = 'SUPPORT' AND is_active = TRUE),
    (SELECT MIN(ABS(price_level - p_current_price))
     FROM support_resistance_zones
     WHERE symbol = p_symbol AND zone_type = 'RESISTANCE' AND is_active = TRUE);
END;
$$ LANGUAGE plpgsql;
""".strip()
    )

    op.execute(
        """
CREATE OR REPLACE FUNCTION compute_full_rac_context(
  query_vec vector(128),
  p_symbol TEXT,
  p_current_price DOUBLE PRECISION,
  k_neighbors INTEGER DEFAULT 20
)
RETURNS TABLE (
  total_neighbors INTEGER,
  avg_cosine_dist DOUBLE PRECISION,
  label_distribution JSONB,
  avg_future_return DOUBLE PRECISION,
  stddev_future_return DOUBLE PRECISION,
  dominant_label SMALLINT,
  knn_confidence DOUBLE PRECISION,
  dist_to_support DOUBLE PRECISION,
  dist_to_resistance DOUBLE PRECISION,
  sr_position_ratio DOUBLE PRECISION,
  neighbor_ids BIGINT[]
) AS $$
BEGIN
  RETURN QUERY
  WITH
  knn AS (
    SELECT
      pe.id,
      pe.label,
      pe.future_return,
      (pe.embedding <=> query_vec) AS cos_dist
    FROM pattern_embeddings pe
    ORDER BY pe.embedding <=> query_vec
    LIMIT k_neighbors
  ),
  knn_stats AS (
    SELECT
      COUNT(*)::INTEGER AS n_total,
      AVG(cos_dist) AS avg_dist,
      AVG(future_return) AS avg_ret,
      STDDEV(future_return) AS std_ret,
      ARRAY_AGG(id ORDER BY cos_dist) AS ids
    FROM knn
  ),
  knn_labels AS (
    SELECT label, COUNT(*)::INTEGER AS cnt
    FROM knn
    GROUP BY label
  ),
  knn_dominant AS (
    SELECT
      label AS dom_label,
      cnt::DOUBLE PRECISION / (SELECT n_total FROM knn_stats) AS conf
    FROM knn_labels
    ORDER BY cnt DESC
    LIMIT 1
  ),
  sr_data AS (
    SELECT
      (SELECT MIN(ABS(p_current_price - price_level))
       FROM support_resistance_zones
       WHERE symbol = p_symbol AND zone_type = 'SUPPORT' AND is_active = TRUE) AS d_support,
      (SELECT MIN(ABS(price_level - p_current_price))
       FROM support_resistance_zones
       WHERE symbol = p_symbol AND zone_type = 'RESISTANCE' AND is_active = TRUE) AS d_resistance
  )
  SELECT
    ks.n_total,
    ks.avg_dist,
    (SELECT jsonb_object_agg(kl.label::TEXT, kl.cnt) FROM knn_labels kl),
    ks.avg_ret,
    ks.std_ret,
    kd.dom_label,
    kd.conf,
    sr.d_support,
    sr.d_resistance,
    CASE
      WHEN sr.d_support IS NOT NULL AND sr.d_resistance IS NOT NULL
        AND (sr.d_support + sr.d_resistance) > 0
      THEN sr.d_support / (sr.d_support + sr.d_resistance)
      ELSE NULL
    END,
    ks.ids
  FROM knn_stats ks
  CROSS JOIN knn_dominant kd
  CROSS JOIN sr_data sr;
END;
$$ LANGUAGE plpgsql;
""".strip()
    )


def downgrade() -> None:
    op.execute(
        """
DROP FUNCTION IF EXISTS compute_full_rac_context(vector(128), TEXT, DOUBLE PRECISION, INTEGER);
DROP FUNCTION IF EXISTS get_distance_to_nearest_sr(TEXT, DOUBLE PRECISION);
DROP FUNCTION IF EXISTS compute_rac_context(vector(128), INTEGER);
DROP FUNCTION IF EXISTS find_similar_patterns(vector(128), INTEGER, DOUBLE PRECISION, TEXT);
""".strip()
    )

