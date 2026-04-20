"""Encoder retrieval diagnostics.

Đo chất lượng không gian embedding của CNN encoder theo góc nhìn KNN:

1) Láng giềng của query theo từng lớp — ma trận ``P@k`` 3x3 so với base rate
   trong ``pattern_embeddings`` để xác định encoder có tách lớp được hay không.
2) PCA 2D của embeddings lấy phân tầng theo lớp — kiểm tra trực quan bằng
   plotly HTML.

Script này KHÔNG sửa mô hình, chỉ đo. Dùng trước khi quyết định retrain.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import torch
from dotenv import load_dotenv

from etl.feature_engineer import WindowRecord, forward_fill_trading_days, generate_windows, train_test_split_by_time
from etl.pipeline import _load_symbols
from ml.cnn_encoder import encode_batch
from ml.embedding_generator import load_encoder

LABEL_NAMES: dict[int, str] = {0: "Down", 1: "Neutral", 2: "Up"}


@dataclass(frozen=True)
class QueryNeighborResult:
    query_label: int
    neighbor_labels: list[int]  # đã loại self, độ dài = k


def _format_vector(vec: np.ndarray) -> str:
    if vec.ndim != 1:
        raise ValueError("Expected 1D vector")
    return "[" + ",".join(f"{float(x):.8f}" for x in vec.tolist()) + "]"


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.removeprefix("postgresql+psycopg://")
    if not url:
        raise SystemExit("DATABASE_URL is required (or pass --database-url).")
    return url


def _fetch_ohlcv_psycopg(
    database_url: str,
    symbols: list[str],
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    """Read OHLCV via raw psycopg — avoids SQLAlchemy+psycopg2 dependency."""
    clauses = ["symbol = ANY(%(symbols)s)"]
    params: dict[str, object] = {"symbols": list(symbols)}
    if start is not None:
        clauses.append("time >= %(start)s")
        params["start"] = start
    if end is not None:
        clauses.append("time <= %(end)s")
        params["end"] = end
    where = " AND ".join(clauses)
    sql = (
        "SELECT time, symbol, open, high, low, close, volume "
        f"FROM stock_ohlcv WHERE {where} ORDER BY symbol, time"
    )
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["time", "symbol", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["time", "symbol", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _fetch_base_rate(conn: psycopg.Connection[tuple[object, ...]]) -> dict[int, int]:
    rows = conn.execute(
        "SELECT label, COUNT(*) FROM pattern_embeddings GROUP BY label ORDER BY label"
    ).fetchall()
    out: dict[int, int] = {}
    for r in rows:
        if r[0] is None:
            continue
        out[int(r[0])] = int(r[1])
    return out


def _to_naive_utc(dt: datetime) -> datetime:
    """Normalize datetime to naive UTC so tz-aware / tz-naive can be compared safely."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _knn_one(
    conn: psycopg.Connection[tuple[object, ...]],
    *,
    query_vec: np.ndarray,
    query_symbol: str,
    query_window_end: datetime,
    k: int,
    overfetch: int,
    leakage_days: int,
) -> list[tuple[int, datetime, int | None]]:
    """Trả về top-k rows (sau khi loại self + rò rỉ gần thời điểm query).

    Rows = (id, window_end, label). Loại row có cùng symbol và window_end nằm
    trong dải ``query_window_end ± leakage_days`` (tránh overlap cửa sổ).
    """
    qv = _format_vector(query_vec)
    q_end = _to_naive_utc(query_window_end)
    low = q_end - timedelta(days=leakage_days)
    high = q_end + timedelta(days=leakage_days)
    rows = conn.execute(
        """
        SELECT pe.id, pe.window_end, pe.label,
               pe.symbol,
               (pe.embedding <=> %(qv)s::vector) AS cos_dist
        FROM pattern_embeddings pe
        ORDER BY pe.embedding <=> %(qv)s::vector
        LIMIT %(limit)s
        """,
        {"qv": qv, "limit": int(k + overfetch)},
    ).fetchall()

    kept: list[tuple[int, datetime, int | None]] = []
    for r in rows:
        nid = int(r[0])
        wend = _to_naive_utc(r[1])
        lab = int(r[2]) if r[2] is not None else None
        sym = str(r[3])
        if sym == query_symbol and low <= wend <= high:
            continue
        kept.append((nid, wend, lab))
        if len(kept) >= k:
            break
    return kept


def _subsample_test(test_recs: list[WindowRecord], n: int, seed: int) -> list[WindowRecord]:
    if n <= 0 or n >= len(test_recs):
        return list(test_recs)
    rng = random.Random(seed)
    return rng.sample(test_recs, n)


def _encode_records(
    encoder: torch.nn.Module,
    recs: list[WindowRecord],
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if not recs:
        return np.zeros((0, 128), dtype=np.float32)
    windows = np.stack([r.data for r in recs], axis=0)
    return encode_batch(encoder, windows, device=device, batch_size=batch_size)


def _compute_retrieval_matrix(
    results: list[QueryNeighborResult],
    *,
    labels: tuple[int, ...] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (count_matrix, fraction_matrix) shape (3, 3).

    count_matrix[i, j] = tổng số neighbor có label=j trong các query có label=i.
    fraction_matrix[i, j] = count_matrix[i, j] / sum_j count_matrix[i, :].
    """
    n = len(labels)
    counts = np.zeros((n, n), dtype=np.int64)
    for r in results:
        i = labels.index(r.query_label) if r.query_label in labels else None
        if i is None:
            continue
        for lab in r.neighbor_labels:
            if lab in labels:
                j = labels.index(lab)
                counts[i, j] += 1
    row_sums = counts.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    fractions = counts / row_sums
    return counts, fractions


def _majority_vote_accuracy(results: list[QueryNeighborResult]) -> tuple[float, dict[int, float]]:
    """Giả lập logic ``dominant_label`` của RAC: mỗi query = argmax tần suất.

    Trả về (accuracy tổng, accuracy theo query label).
    """
    correct = 0
    per_label_correct: Counter[int] = Counter()
    per_label_total: Counter[int] = Counter()
    for r in results:
        if not r.neighbor_labels:
            continue
        c = Counter(r.neighbor_labels)
        pred = c.most_common(1)[0][0]
        per_label_total[r.query_label] += 1
        if pred == r.query_label:
            correct += 1
            per_label_correct[r.query_label] += 1
    total = sum(per_label_total.values())
    overall = (correct / total) if total else float("nan")
    per_label = {
        lab: (per_label_correct[lab] / per_label_total[lab])
        for lab in per_label_total
        if per_label_total[lab] > 0
    }
    return overall, per_label


def _sample_for_pca(
    conn: psycopg.Connection[tuple[object, ...]],
    per_label: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample phân tầng theo lớp: mỗi lớp ``per_label`` embeddings."""
    embs: list[list[float]] = []
    labs: list[int] = []
    conn.execute("SELECT setseed(%s)", (float(seed % 1000) / 1000.0,))
    for lab in (0, 1, 2):
        rows = conn.execute(
            """
            SELECT embedding::text, label
            FROM pattern_embeddings
            WHERE label = %s
            ORDER BY random()
            LIMIT %s
            """,
            (lab, per_label),
        ).fetchall()
        for r in rows:
            vec_text = str(r[0]).strip("[]")
            if not vec_text:
                continue
            vec = np.fromstring(vec_text, sep=",", dtype=np.float32)
            embs.append(vec)
            labs.append(int(r[1]))
    if not embs:
        return np.zeros((0, 128), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(embs, axis=0), np.asarray(labs, dtype=np.int64)


def _save_pca_html(embeddings: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    if embeddings.shape[0] == 0:
        out_path.write_text("<p>No embeddings sampled.</p>", encoding="utf-8")
        return
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:  # pragma: no cover
        raise SystemExit("scikit-learn is required: uv sync") from e
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover
        raise SystemExit("plotly is required: uv sync --dev") from e

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(embeddings)
    fig = go.Figure()
    palette = {0: "#ef5350", 1: "#9e9e9e", 2: "#26a69a"}
    for lab in (0, 1, 2):
        mask = labels == lab
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=xy[mask, 0],
                y=xy[mask, 1],
                mode="markers",
                marker=dict(size=4, opacity=0.55, color=palette[lab]),
                name=f"{LABEL_NAMES[lab]} (n={int(mask.sum())})",
            )
        )
    var = pca.explained_variance_ratio_
    fig.update_layout(
        title=f"PCA 2D of pattern_embeddings (var={var[0]:.2%} + {var[1]:.2%} = {sum(var):.2%})",
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(color="#d1d4dc"),
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend=dict(orientation="h"),
        height=720,
    )
    out_path.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")


def _format_matrix(
    counts: np.ndarray,
    fractions: np.ndarray,
    base_rate: dict[int, float],
    *,
    labels: tuple[int, ...] = (0, 1, 2),
) -> str:
    header = f"{'query \\ neighbor':<16} " + " ".join(f"{LABEL_NAMES[l]:>16}" for l in labels) + "   total"
    lines = [header, "-" * len(header)]
    for i, li in enumerate(labels):
        cells = []
        total_row = int(counts[i].sum())
        for j, lj in enumerate(labels):
            frac = float(fractions[i, j])
            br = base_rate.get(lj, 0.0)
            lift = (frac / br) if br > 0 else float("nan")
            cells.append(f"{frac:>7.2%} (x{lift:>4.2f})")
        lines.append(f"{LABEL_NAMES[li]:<16} " + " ".join(f"{c:>16}" for c in cells) + f"   {total_row:>5}")
    lines.append("")
    lines.append("• Số trong () là lift = P(neighbor=L) / base_rate(L). >1 = encoder kéo đúng hướng.")
    lines.append("• Kỳ vọng encoder tốt: đường chéo (cùng lớp) có lift rõ >1 (thường 1.5-3.0).")
    return "\n".join(lines)


def run_diagnostics(
    *,
    database_url: str,
    symbols: list[str],
    encoder_path: Path,
    start: date | None,
    end: date | None,
    train_ratio: float,
    k: int,
    n_queries: int,
    leakage_days: int,
    device: str,
    batch_size: int,
    pca_per_label: int,
    seed: int,
    out_dir: Path,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading encoder: {encoder_path}")
    encoder = load_encoder(encoder_path)

    print(f"[2/6] Fetching OHLCV from DB ({len(symbols)} symbols)...")
    df = _fetch_ohlcv_psycopg(database_url, symbols, start, end)
    if df.empty:
        raise SystemExit("No OHLCV rows for given symbols/date range.")
    df = forward_fill_trading_days(df)
    print(f"       rows={len(df):,}")

    print("[3/6] Generating windows + chronological split...")
    records = generate_windows(df)
    if not records:
        raise SystemExit("No windows generated.")
    train_recs, test_recs = train_test_split_by_time(records, train_ratio=train_ratio)
    train_hist = Counter(r.label for r in train_recs)
    test_hist = Counter(r.label for r in test_recs)
    print(f"       train={len(train_recs):,} test={len(test_recs):,}")
    print(f"       train label dist = {{Down={train_hist[0]}, Neutral={train_hist[1]}, Up={train_hist[2]}}}")
    print(f"       test  label dist = {{Down={test_hist[0]}, Neutral={test_hist[1]}, Up={test_hist[2]}}}")

    queries = _subsample_test(test_recs, n_queries, seed)
    print(f"       subsampled queries = {len(queries)}")

    print(f"[4/6] Encoding {len(queries)} query windows on {device}...")
    q_embs = _encode_records(encoder, queries, device=device, batch_size=batch_size)

    print(f"[5/6] Running KNN (k={k}, overfetch=20, leakage±{leakage_days}d)...")
    results: list[QueryNeighborResult] = []
    with psycopg.connect(database_url) as conn:
        base_rate_counts = _fetch_base_rate(conn)
        total_base = sum(base_rate_counts.values())
        base_rate = {k_: (v / total_base) for k_, v in base_rate_counts.items()} if total_base else {}
        print(
            "       base rate in pattern_embeddings = "
            + ", ".join(f"{LABEL_NAMES[l]}={base_rate.get(l, 0):.1%}" for l in (0, 1, 2))
        )
        progress_every = max(1, len(queries) // 20)
        for i, (rec, emb) in enumerate(zip(queries, q_embs, strict=True)):
            rows = _knn_one(
                conn,
                query_vec=emb,
                query_symbol=rec.symbol,
                query_window_end=rec.window_end,
                k=k,
                overfetch=20,
                leakage_days=leakage_days,
            )
            labels_only = [lab for (_id, _wend, lab) in rows if lab is not None]
            results.append(QueryNeighborResult(query_label=int(rec.label), neighbor_labels=labels_only))
            if (i + 1) % progress_every == 0:
                print(f"         {i + 1}/{len(queries)}")

    counts, fractions = _compute_retrieval_matrix(results)
    overall_acc, per_label_acc = _majority_vote_accuracy(results)

    print()
    print("=" * 78)
    print("RETRIEVAL CONFUSION (rows = query label, cols = P(neighbor label), x lift)")
    print("=" * 78)
    print(_format_matrix(counts, fractions, base_rate))
    print()
    print(f"Majority-vote accuracy (giống logic dominant_label của RAC):")
    print(f"  overall = {overall_acc:.2%}")
    for lab in (0, 1, 2):
        if lab in per_label_acc:
            print(f"  {LABEL_NAMES[lab]:<7} = {per_label_acc[lab]:.2%}")

    print()
    print(f"[6/6] Sampling embeddings for PCA (per_label={pca_per_label})...")
    with psycopg.connect(database_url) as conn:
        pca_embs, pca_labs = _sample_for_pca(conn, per_label=pca_per_label, seed=seed)
    pca_out = out_dir / "pca_embeddings.html"
    _save_pca_html(pca_embs, pca_labs, pca_out)
    print(f"       saved PCA plot: {pca_out}")

    metrics: dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "encoder_path": str(encoder_path),
        "symbols_count": len(symbols),
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "train_windows": len(train_recs),
        "test_windows": len(test_recs),
        "train_label_dist": {LABEL_NAMES[l]: int(train_hist[l]) for l in (0, 1, 2)},
        "test_label_dist": {LABEL_NAMES[l]: int(test_hist[l]) for l in (0, 1, 2)},
        "base_rate": {LABEL_NAMES[l]: float(base_rate.get(l, 0.0)) for l in (0, 1, 2)},
        "k": k,
        "n_queries": len(queries),
        "leakage_days": leakage_days,
        "retrieval_counts": counts.tolist(),
        "retrieval_fractions": fractions.tolist(),
        "retrieval_lift": [
            [
                (float(fractions[i, j]) / float(base_rate.get(lj, 0.0)))
                if base_rate.get(lj, 0.0) > 0
                else None
                for j, lj in enumerate((0, 1, 2))
            ]
            for i in range(3)
        ],
        "majority_vote_overall_accuracy": overall_acc,
        "majority_vote_per_label_accuracy": {LABEL_NAMES[l]: per_label_acc.get(l) for l in (0, 1, 2)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"       saved metrics: {out_dir / 'metrics.json'}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    def _parse_date(s: str) -> date:
        return date.fromisoformat(s)

    parser = argparse.ArgumentParser(description="Encoder retrieval diagnostics (P@k by class + PCA).")
    parser.add_argument("--database-url", default="", help="Defaults to DATABASE_URL.")
    parser.add_argument("--encoder", type=Path, default=Path("ml/model_store/cnn_encoder.pt"))
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", default="etl/tickers_vn100.txt")
    parser.add_argument("--start", type=_parse_date, default=None)
    parser.add_argument("--end", type=_parse_date, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--n-queries", type=int, default=2000)
    parser.add_argument(
        "--leakage-days",
        type=int,
        default=45,
        help="Exclude neighbors with same symbol within ± this many days of query window_end.",
    )
    parser.add_argument("--device", type=str, default=os.environ.get("TORCH_DEVICE", "cpu"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pca-per-label", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("ml/diagnostics"))
    args = parser.parse_args(argv)

    load_dotenv(override=False)
    database_url = (args.database_url or "").strip() or _database_url()
    if not args.encoder.is_file():
        raise SystemExit(f"Encoder not found: {args.encoder}")

    symbols = _load_symbols(list(args.symbols) if args.symbols else None, args.symbols_file)
    run_diagnostics(
        database_url=database_url,
        symbols=symbols,
        encoder_path=args.encoder,
        start=args.start,
        end=args.end,
        train_ratio=float(args.train_ratio),
        k=int(args.k),
        n_queries=int(args.n_queries),
        leakage_days=int(args.leakage_days),
        device=str(args.device),
        batch_size=int(args.batch_size),
        pca_per_label=int(args.pca_per_label),
        seed=int(args.seed),
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
