"""RAC procs: ensure enough neighbors via ef_search

Revision ID: 0009_hnsw_ef_search_in_procs
Revises: 0008_stock_ohlcv_unique_key
Create Date: 2026-04-20
"""

from __future__ import annotations

from alembic import op

revision = "0009_hnsw_ef_search_in_procs"
down_revision = "0008_stock_ohlcv_unique_key"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Some pgvector/HNSW configurations may return fewer than k rows when ef_search is too low.
    # We set ef_search locally inside functions to ensure stable Top-K behavior.
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
  PERFORM set_config('hnsw.ef_search', GREATEST(k_neighbors, 100)::text, true);
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
  PERFORM set_config('hnsw.ef_search', GREATEST(k_neighbors, 100)::text, true);
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
    # Revert to previous definitions from revision 0007.
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

