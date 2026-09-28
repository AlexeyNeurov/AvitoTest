from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import Stemmer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB


QUERY_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^0-9a-zа-я]+", flags=re.IGNORECASE)
_TOKEN_RE = re.compile(r"[0-9a-zа-я]{2,}", flags=re.IGNORECASE)
_RU_STEMMER = Stemmer.Stemmer("russian")


def normalize_text(value: object) -> str:
    """Normalize Russian free text while retaining digits and Latin brands."""
    text = html.unescape("" if value is None else str(value)).lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def stem_analyzer(text: str) -> list[str]:
    """Tokenize and stem Russian words; Latin brands and digits are retained."""
    tokens = _TOKEN_RE.findall(text.lower().replace("ё", "е"))
    return _RU_STEMMER.stemWords(tokens)


def bm25_weight_matrix(
    counts: sparse.csr_matrix, *, k1: float = 1.5, b: float = 0.75
) -> sparse.csr_matrix:
    """Convert a document-term count matrix to sparse BM25 document weights."""
    matrix = counts.astype(np.float32, copy=True).tocsr()
    document_lengths = np.asarray(matrix.sum(axis=1)).ravel()
    average_length = max(float(document_lengths.mean()), 1.0)
    document_frequency = np.bincount(
        matrix.indices, minlength=matrix.shape[1]
    ).astype(np.float32)
    idf = np.log1p(
        (matrix.shape[0] - document_frequency + 0.5)
        / (document_frequency + 0.5)
    ).astype(np.float32)
    row_lengths = np.repeat(document_lengths, np.diff(matrix.indptr))
    denominator = matrix.data + k1 * (
        1.0 - b + b * row_lengths / average_length
    )
    matrix.data = (
        matrix.data * (k1 + 1.0) / denominator * idf[matrix.indices]
    ).astype(np.float32)
    return matrix


def prepare_item_text(items: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Create field-weighted word and character documents.

    Repeating the concise title makes it more influential than the long and
    often promotional description without needing a dense learning-to-rank
    model.  Description length is capped to keep the index compact.
    """
    titles = items["item_title_raw"].fillna("").map(normalize_text)
    params = (
        items["item_infm_params_text"].fillna("").str.slice(0, 800).map(normalize_text)
    )
    descriptions = (
        items["item_description_raw"].fillna("").str.slice(0, 900).map(normalize_text)
    )
    word_docs = (
        titles + " " + titles + " " + titles + " " + params + " " + descriptions
    ).tolist()
    char_docs = (titles + " " + params.str.slice(0, 250)).tolist()
    return word_docs, char_docs


def build_indexes(items: pd.DataFrame):
    """Fit sparse lexical indexes and return vectorizers plus item matrices."""
    word_docs, char_docs = prepare_item_text(items)
    title_docs = items["item_title_raw"].fillna("").map(normalize_text).tolist()
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        min_df=2,
        max_df=0.997,
        max_features=220_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
        token_pattern=r"(?u)\b[\w-]{2,}\b",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=180_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    title_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.999,
        max_features=280_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
        token_pattern=r"(?u)\b[\w-]{2,}\b",
    )
    bm25_vectorizer = CountVectorizer(
        analyzer=stem_analyzer,
        min_df=2,
        max_features=220_000,
        dtype=np.float32,
    )
    item_word = word_vectorizer.fit_transform(word_docs).tocsr()
    item_char = char_vectorizer.fit_transform(char_docs).tocsr()
    item_title = title_vectorizer.fit_transform(title_docs).tocsr()
    item_bm25_counts = bm25_vectorizer.fit_transform(word_docs).tocsr()
    item_bm25 = bm25_weight_matrix(item_bm25_counts)
    return (
        word_vectorizer,
        item_word,
        char_vectorizer,
        item_char,
        title_vectorizer,
        item_title,
        bm25_vectorizer,
        item_bm25,
    )


def build_train_signals(
    train: pd.DataFrame,
    items: pd.DataFrame,
    char_vectorizer: TfidfVectorizer,
) -> dict:
    """Build leakage-free aggregate signals from historical selected items."""
    item_to_pos = pd.Series(np.arange(len(items), dtype=np.int32), index=items.item_id)
    item_location_ids = set(items.item_location_id.astype(int))
    in_corpus = train[train.item_id.isin(item_to_pos.index)].copy()
    in_corpus["item_pos"] = in_corpus.item_id.map(item_to_pos).astype(np.int32)
    in_corpus["_q"] = in_corpus.search_query.map(normalize_text)

    # Stable popularity prior.  It is deliberately weak and only breaks ties.
    click_counts = in_corpus.groupby("item_pos").size()
    popularity = np.zeros(len(items), dtype=np.float32)
    if len(click_counts):
        popularity[click_counts.index.to_numpy()] = np.log1p(click_counts.to_numpy())
        popularity /= max(float(popularity.max()), 1.0)

    # Query variants are very useful for spelling and phrasing differences.
    exact_text_items: dict[str, list[int]] = {}
    ordered = in_corpus.sort_values(
        ["search_query", "item_rating_reviews_count", "item_rating"],
        ascending=[True, False, False],
        na_position="last",
    )
    for query, group in ordered.groupby("search_query", sort=False):
        exact_text_items[normalize_text(query)] = list(dict.fromkeys(group.item_pos.tolist()))

    exact_location_items: dict[tuple[str, int], list[int]] = {}
    exact_context_items: dict[tuple, list[int]] = {}
    for key, group in in_corpus.groupby(["_q", "search_location_id"], sort=False):
        exact_location_items[(key[0], int(key[1]))] = list(
            dict.fromkeys(group.item_pos.tolist())
        )
    context_keys = [
        "_q",
        "search_location_id",
        "search_is_delivery_search",
        "search_infm_params_text",
        "search_category",
    ]
    for key, group in in_corpus.groupby(context_keys, sort=False, dropna=False):
        normalized_key = (
            key[0],
            int(key[1]),
            int(key[2]),
            normalize_text(key[3]),
            int(key[4]),
        )
        exact_context_items[normalized_key] = list(
            dict.fromkeys(group.item_pos.tolist())
        )

    # A compact query-to-click index transfers evidence from close historical
    # phrasings (for example, singular/plural forms or a short typo).  Only
    # clicked listings still present in the benchmark corpus are retained.
    history_rows = []
    history_item_lists: list[list[int]] = []
    history_grouped = (
        in_corpus.groupby(["_q", "item_pos"]).size().rename("n").reset_index()
    )
    for query, group in history_grouped.groupby("_q", sort=False):
        group = group.sort_values("n", ascending=False)
        history_rows.append(query)
        history_item_lists.append(group.item_pos.astype(int).head(15).tolist())
    history_matrix = char_vectorizer.transform(history_rows).tocsr()

    # Learn which microcategories each normalized query selected.  This gives a
    # high-precision boost, while lexical scores can still retrieve other ones.
    query_microcats: dict[str, list[int]] = {}
    micro_counts = (
        train.assign(_q=train.search_query.map(normalize_text))
        .groupby(["_q", "item_microcat_id"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["_q", "n"], ascending=[True, False])
    )
    for query, group in micro_counts.groupby("_q", sort=False):
        query_microcats[query] = group.item_microcat_id.head(3).astype(int).tolist()

    # Structured search filters often name the service family explicitly.
    # Estimate P(microcategory | filter text); the probability scales the boost
    # so a broad filter such as just "Вид услуги" remains weak.
    filter_frame = train.assign(
        _filter=train.search_infm_params_text.map(normalize_text)
    )
    filter_counts = (
        filter_frame[filter_frame._filter.ne("")]
        .groupby(["_filter", "item_microcat_id"])
        .size()
        .rename("n")
        .reset_index()
    )
    filter_totals = filter_counts.groupby("_filter").n.transform("sum")
    filter_counts["share"] = filter_counts.n / filter_totals
    filter_microcats: dict[str, list[tuple[int, float]]] = {}
    for filter_text, group in filter_counts.groupby("_filter", sort=False):
        best = group.sort_values("n", ascending=False).head(4)
        filter_microcats[filter_text] = list(
            zip(best.item_microcat_id.astype(int), best.share.astype(float), strict=False)
        )

    # Search locations absent from the item corpus are higher-level regions.
    # Learn their city distribution only from historical choices.  Cities that
    # together explain the first 90% of clicks form the region core.
    regional_rows = train[~train.search_location_id.isin(item_location_ids)]
    region_location_weights: dict[int, dict[int, float]] = {}
    region_counts = (
        regional_rows.groupby(["search_location_id", "item_location_id"])
        .size()
        .rename("n")
        .reset_index()
    )
    for region_id, group in region_counts.groupby("search_location_id", sort=False):
        group = group.sort_values("n", ascending=False)
        cumulative_before = group.n.cumsum().shift(fill_value=0) / group.n.sum()
        weights = np.where(cumulative_before < 0.90, 1.0, 0.80)
        region_location_weights[int(region_id)] = {
            int(location_id): float(weight)
            for location_id, weight in zip(
                group.item_location_id, weights, strict=False
            )
        }

    # Generalize the microcategory signal to unseen wording.  ComplementNB is
    # fast on sparse character n-grams and handles Russian inflections and
    # common misspellings well.  Pair counts become sample weights so repeated
    # historical choices retain their evidence without duplicating rows.
    micro_train = micro_counts[micro_counts._q.ne("")]
    micro_features = char_vectorizer.transform(micro_train._q)
    microcat_model = ComplementNB(alpha=0.3)
    microcat_model.fit(
        micro_features,
        micro_train.item_microcat_id.astype(int),
        sample_weight=micro_train.n.to_numpy(dtype=np.float64),
    )

    return {
        "popularity": popularity,
        "exact_text_items": exact_text_items,
        "exact_location_items": exact_location_items,
        "exact_context_items": exact_context_items,
        "query_microcats": query_microcats,
        "filter_microcats": filter_microcats,
        "microcat_model": microcat_model,
        "history_matrix": history_matrix,
        "history_item_lists": history_item_lists,
        "item_location_ids": item_location_ids,
        "region_location_weights": region_location_weights,
    }


def _top_sparse_rows(matrix: sparse.csr_matrix, top_k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract largest values from every CSR row without densifying the corpus."""
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        idx = matrix.indices[start:end]
        val = matrix.data[start:end]
        if len(val) > top_k:
            keep = np.argpartition(val, -top_k)[-top_k:]
            idx, val = idx[keep], val[keep]
        result.append((idx, val))
    return result


def retrieve(
    queries: pd.DataFrame,
    items: pd.DataFrame,
    word_vectorizer: TfidfVectorizer,
    item_word: sparse.csr_matrix,
    char_vectorizer: TfidfVectorizer,
    item_char: sparse.csr_matrix,
    title_vectorizer: TfidfVectorizer,
    item_title: sparse.csr_matrix,
    bm25_vectorizer: CountVectorizer,
    item_bm25: sparse.csr_matrix,
    signals: dict,
    *,
    batch_size: int = 128,
    lexical_pool: int = 700,
    top_k: int = 50,
    location_boost: float = 0.65,
    remote_slots: int = 0,
    title_weight: float = 0.0,
    geo_hierarchy: bool = False,
    bm25_weight: float = 0.20,
    filter_char_weight: float = 0.0,
) -> list[list[str]]:
    """Retrieve and rerank up to ``top_k`` item ids for each query."""
    normalized_queries = queries.search_query.map(normalize_text)
    normalized_filters = queries.search_infm_params_text.map(normalize_text)
    # Search filters are kept separate: many contain generic boilerplate that
    # should not dilute the user's short intent query.
    query_word = word_vectorizer.transform(normalized_queries)
    query_char = char_vectorizer.transform(normalized_queries)
    query_title = title_vectorizer.transform(normalized_queries)
    query_bm25 = bm25_vectorizer.transform(normalized_queries).tocsr()
    query_bm25.data.fill(1.0)
    query_filter = word_vectorizer.transform(normalized_filters)
    query_filter_char = char_vectorizer.transform(normalized_filters)
    microcat_logp = signals["microcat_model"].predict_log_proba(query_char)
    microcat_classes = signals["microcat_model"].classes_.astype(int)

    item_ids = items.item_id.to_numpy()
    item_locations = items.item_location_id.to_numpy()
    item_microcats = items.item_microcat_id.to_numpy()
    item_ratings = items.item_rating.fillna(0).to_numpy(dtype=np.float32)
    item_review_counts = items.item_rating_reviews_count.fillna(0).to_numpy(dtype=np.float32)
    item_latitudes = pd.to_numeric(items.item_latitude, errors="coerce").to_numpy()
    item_longitudes = pd.to_numeric(items.item_longitude, errors="coerce").to_numpy()
    location_coords = (
        pd.DataFrame(
            {
                "location": item_locations,
                "latitude": item_latitudes,
                "longitude": item_longitudes,
            }
        )
        .dropna()
        .groupby("location")[["latitude", "longitude"]]
        .median()
    )
    review_prior = np.log1p(item_review_counts)
    if review_prior.max() > 0:
        review_prior /= review_prior.max()

    predictions: list[list[str]] = []
    for batch_start in range(0, len(queries), batch_size):
        batch_end = min(batch_start + batch_size, len(queries))
        word_scores = (query_word[batch_start:batch_end] @ item_word.T).tocsr()
        char_scores = (query_char[batch_start:batch_end] @ item_char.T).tocsr()
        title_scores = (
            query_title[batch_start:batch_end] @ item_title.T
        ).tocsr()
        bm25_scores = (
            query_bm25[batch_start:batch_end] @ item_bm25.T
        ).tocsr()
        history_scores = (
            query_char[batch_start:batch_end] @ signals["history_matrix"].T
        ).tocsr()
        filter_scores = (query_filter[batch_start:batch_end] @ item_word.T).tocsr()
        filter_char_scores = (
            query_filter_char[batch_start:batch_end] @ item_char.T
        ).tocsr()
        word_top = _top_sparse_rows(word_scores, lexical_pool)
        char_top = _top_sparse_rows(char_scores, lexical_pool)
        title_top = _top_sparse_rows(title_scores, lexical_pool)
        bm25_top = _top_sparse_rows(bm25_scores, lexical_pool)
        history_top = _top_sparse_rows(history_scores, 20)
        filter_top = _top_sparse_rows(filter_scores, 350)
        filter_char_top = _top_sparse_rows(filter_char_scores, 350)

        for local_row, query_pos in enumerate(range(batch_start, batch_end)):
            qrow = queries.iloc[query_pos]
            qtext = normalized_queries.iloc[query_pos]
            candidates: dict[int, float] = defaultdict(float)

            w_idx, w_val = word_top[local_row]
            for pos, score in zip(w_idx, w_val, strict=False):
                candidates[int(pos)] += 0.64 * float(score)
            c_idx, c_val = char_top[local_row]
            for pos, score in zip(c_idx, c_val, strict=False):
                candidates[int(pos)] += 0.36 * float(score)
            if title_weight > 0:
                t_idx, t_val = title_top[local_row]
                for pos, score in zip(t_idx, t_val, strict=False):
                    candidates[int(pos)] += title_weight * float(score)
            if bm25_weight > 0:
                b_idx, b_val = bm25_top[local_row]
                maximum = max(float(b_val.max(initial=0.0)), 1e-6)
                for pos, score in zip(b_idx, b_val, strict=False):
                    candidates[int(pos)] += bm25_weight * float(score) / maximum
            f_idx, f_val = filter_top[local_row]
            for pos, score in zip(f_idx, f_val, strict=False):
                candidates[int(pos)] += 0.085 * float(score)
            if filter_char_weight > 0:
                fc_idx, fc_val = filter_char_top[local_row]
                for pos, score in zip(fc_idx, fc_val, strict=False):
                    candidates[int(pos)] += filter_char_weight * float(score)

            # Historical exact-query selections consume only a few slots and
            # can recover active listings already observed in train.
            for rank, pos in enumerate(signals["exact_text_items"].get(qtext, [])[:20]):
                candidates[int(pos)] += 0.30 / math.sqrt(rank + 1)
            location_key = (qtext, int(qrow.search_location_id))
            for rank, pos in enumerate(
                signals["exact_location_items"].get(location_key, [])[:15]
            ):
                candidates[int(pos)] += 0.48 / math.sqrt(rank + 1)
            context_key = (
                qtext,
                int(qrow.search_location_id),
                int(qrow.search_is_delivery_search),
                normalized_filters.iloc[query_pos],
                int(qrow.search_category),
            )
            for rank, pos in enumerate(
                signals["exact_context_items"].get(context_key, [])[:15]
            ):
                candidates[int(pos)] += 0.75 / math.sqrt(rank + 1)

            h_idx, h_val = history_top[local_row]
            history_order = np.argsort(h_val)[::-1]
            for neighbor_rank, neighbor_offset in enumerate(history_order):
                similarity = float(h_val[neighbor_offset])
                if similarity < 0.34:
                    break
                history_query_pos = int(h_idx[neighbor_offset])
                for click_rank, pos in enumerate(
                    signals["history_item_lists"][history_query_pos][:5]
                ):
                    candidates[int(pos)] += (
                        0.14
                        * similarity
                        / math.sqrt(neighbor_rank + 1)
                        / math.sqrt(click_rank + 1)
                    )

            learned_microcats = signals["query_microcats"].get(qtext)
            if learned_microcats:
                predicted_microcats = learned_microcats
            else:
                best = np.argpartition(microcat_logp[query_pos], -3)[-3:]
                best = best[np.argsort(microcat_logp[query_pos, best])[::-1]]
                predicted_microcats = microcat_classes[best].tolist()
            microcat_bonus = {
                microcat: bonus
                for microcat, bonus in zip(predicted_microcats, (0.105, 0.060, 0.035))
            }
            for microcat, share in signals["filter_microcats"].get(
                normalized_filters.iloc[query_pos], []
            ):
                microcat_bonus[microcat] = microcat_bonus.get(microcat, 0.0) + (
                    0.14 * math.sqrt(share)
                )
            query_location = qrow.search_location_id
            query_coord = None
            if int(query_location) in location_coords.index:
                query_coord = location_coords.loc[int(query_location)].to_numpy()
            region_weights = signals["region_location_weights"].get(
                int(query_location), {}
            )
            for pos in list(candidates):
                score = candidates[pos]
                if item_locations[pos] == query_location:
                    score += location_boost
                elif geo_hierarchy and region_weights:
                    score += location_boost * region_weights.get(
                        int(item_locations[pos]), 0.0
                    )
                elif geo_hierarchy and query_coord is not None:
                    # Equirectangular approximation is accurate enough for the
                    # 100 km neighbor threshold and much cheaper than geocoding.
                    lat1 = math.radians(float(query_coord[0]))
                    lat2 = math.radians(float(item_latitudes[pos]))
                    dlat = lat2 - lat1
                    dlon = math.radians(
                        float(item_longitudes[pos]) - float(query_coord[1])
                    )
                    x = dlon * math.cos((lat1 + lat2) / 2.0)
                    distance_km = 6371.0 * math.sqrt(x * x + dlat * dlat)
                    if distance_km <= 100.0:
                        score += location_boost * 0.60
                score += microcat_bonus.get(int(item_microcats[pos]), 0.0)
                score += 0.018 * float(signals["popularity"][pos])
                score += 0.006 * float(review_prior[pos])
                score += 0.002 * float(item_ratings[pos] / 5.0)
                candidates[pos] = score

            ranked = sorted(candidates, key=candidates.get, reverse=True)
            if remote_slots:
                local_ranked = [
                    pos for pos in ranked if item_locations[pos] == query_location
                ]
                remote_ranked = [
                    pos for pos in ranked if item_locations[pos] != query_location
                ]
                chosen = (
                    local_ranked[: max(top_k - remote_slots, 0)]
                    + remote_ranked[:remote_slots]
                )
                if len(chosen) < top_k:
                    chosen_set = set(chosen)
                    chosen.extend(
                        pos for pos in ranked if pos not in chosen_set
                    )
                    chosen = chosen[:top_k]
            else:
                chosen = ranked[:top_k]
            # Sparse retrieval can return fewer than 50 results for rare input.
            # Fill deterministically with popular local items, then global ones.
            if len(chosen) < top_k:
                chosen_set = set(chosen)
                fallback_score = signals["popularity"] + 0.1 * review_prior
                local = np.flatnonzero(item_locations == query_location)
                for pool in (local[np.argsort(fallback_score[local])[::-1]], np.argsort(fallback_score)[::-1]):
                    for pos in pool:
                        pos = int(pos)
                        if pos not in chosen_set:
                            chosen.append(pos)
                            chosen_set.add(pos)
                            if len(chosen) == top_k:
                                break
                    if len(chosen) == top_k:
                        break
            predictions.append(item_ids[np.asarray(chosen, dtype=np.int32)].tolist())
    return predictions


def validate_predictions(
    validation_queries: pd.DataFrame,
    relevant: list[set[str]],
    predictions: list[list[str]],
) -> dict[str, float]:
    """Calculate the competition's macro Recall@50."""
    recalls = [len(set(pred) & truth) / len(truth) for pred, truth in zip(predictions, relevant)]
    return {
        "queries": len(recalls),
        "recall_at_50": float(np.mean(recalls)),
        "median": float(np.median(recalls)),
        "perfect_share": float(np.mean(np.asarray(recalls) == 1.0)),
    }


def blend_predictions(
    baseline: list[list[str]], alternative: list[list[str]], replacements: int
) -> list[list[str]]:
    """Keep the strongest baseline candidates and add novel alternatives."""
    blended: list[list[str]] = []
    keep = 50 - replacements
    for base_row, alt_row in zip(baseline, alternative, strict=False):
        row = list(base_row[:keep])
        used = set(row)
        for item_id in alt_row:
            if item_id not in used:
                row.append(item_id)
                used.add(item_id)
                if len(row) == 50:
                    break
        if len(row) < 50:
            for item_id in base_row[keep:]:
                if item_id not in used:
                    row.append(item_id)
                    used.add(item_id)
                    if len(row) == 50:
                        break
        blended.append(row)
    return blended


def make_validation(train: pd.DataFrame, item_ids: set[str], max_queries: int, seed: int):
    """Create benchmark-like rows with one context per unique query text."""
    subset = train[train.item_id.isin(item_ids)].copy()
    grouped = subset.groupby(QUERY_COLS, sort=False, dropna=False)
    rows: list[dict] = []
    relevant: list[set[str]] = []
    for key, group in grouped:
        rows.append(dict(zip(QUERY_COLS, key)))
        relevant.append(set(group.item_id))
    validation = pd.DataFrame(rows)
    validation["_relevant"] = relevant
    rng = np.random.default_rng(seed)
    # Benchmark query texts are unique.  Choose a single random context for
    # each text so frequent historical phrases do not dominate validation.
    validation["_random"] = rng.random(len(validation))
    validation = (
        validation.sort_values("_random")
        .drop_duplicates("search_query")
        .head(max_queries)
        .reset_index(drop=True)
    )
    relevant = validation.pop("_relevant").tolist()
    validation = validation.drop(columns="_random")
    return validation, relevant


def assert_answer(answer: pd.DataFrame, queries: pd.DataFrame, items: pd.DataFrame) -> None:
    """Fail fast on every format rule stated in the competition brief."""
    assert list(answer.columns) == ["query_id", "answer"]
    assert len(answer) == len(queries)
    assert answer.query_id.is_unique
    assert set(answer.query_id) == set(queries.query_id)
    assert answer.query_id.map(lambda x: isinstance(x, str) and len(x) == 16).all()
    valid_ids = set(items.item_id)
    for value in answer.answer:
        ids = value.split() if value else []
        assert 0 < len(ids) <= 50
        assert len(ids) == len(set(ids))
        assert all(item_id in valid_ids for item_id in ids)
        assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validation-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune-location", action="store_true")
    parser.add_argument("--tune-remote-slots", action="store_true")
    parser.add_argument("--tune-title-weight", action="store_true")
    parser.add_argument("--geo-hierarchy", action="store_true")
    parser.add_argument("--tune-geo", action="store_true")
    parser.add_argument("--tune-bm25", action="store_true")
    parser.add_argument("--tune-filter-char", action="store_true")
    parser.add_argument(
        "--location-boost",
        type=float,
        default=0.65,
        help="Additive score for items in the query location.",
    )
    parser.add_argument(
        "--remote-slots",
        type=int,
        default=0,
        help="Reserve this many of the 50 candidates for other locations.",
    )
    parser.add_argument(
        "--title-weight",
        type=float,
        default=0.0,
        help="Weight of the title-only unigram/bigram similarity.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.20,
        help="Weight of normalized BM25 similarity over Russian stems.",
    )
    parser.add_argument(
        "--filter-char-weight",
        type=float,
        default=0.0,
        help="Weight of character similarity for structured search filters.",
    )
    args = parser.parse_args()

    train = pd.read_parquet(args.data_dir / "train.parquet")
    queries = pd.read_parquet(args.data_dir / "benchmark_queries.parquet")
    items = pd.read_parquet(args.data_dir / "benchmark_items.parquet")

    # The task concerns service listings.  Train confirms that essentially all
    # selected labels have item_category_id == 114, so unrelated corpus rows are
    # safely removed before indexing.
    service_items = items[items.item_category_id == 114].reset_index(drop=True)
    (
        vectorizer_w,
        item_w,
        vectorizer_c,
        item_c,
        vectorizer_t,
        item_t,
        vectorizer_bm25,
        item_bm25,
    ) = build_indexes(service_items)
    signals = build_train_signals(train, service_items, vectorizer_c)

    if args.validate:
        val_queries, relevant = make_validation(
            train, set(service_items.item_id), args.validation_size, args.seed
        )
        held_out_texts = set(val_queries.search_query.map(normalize_text))
        validation_train = train[
            ~train.search_query.map(normalize_text).isin(held_out_texts)
        ]
        validation_signals = build_train_signals(
            validation_train, service_items, vectorizer_c
        )
        location_values = [0.65]
        if args.tune_location:
            location_values = [0.40, 0.50, 0.65, 0.80, 1.20]
        remote_values = [0]
        if args.tune_remote_slots:
            remote_values = [0, 5, 10, 15, 20]
        title_values = [args.title_weight]
        if args.tune_title_weight:
            title_values = [0.0, 0.20]
        title_predictions: dict[float, list[list[str]]] = {}
        geo_values = [args.geo_hierarchy]
        if args.tune_geo:
            geo_values = [False, True]
        bm25_values = [args.bm25_weight]
        if args.tune_bm25:
            bm25_values = [0.0, 0.20, 0.35, 0.50]
        filter_char_values = [args.filter_char_weight]
        if args.tune_filter_char:
            filter_char_values = [0.0, 0.08, 0.15, 0.25]
        for location_boost in location_values:
            for remote_slots in remote_values:
                for title_weight in title_values:
                    for geo_hierarchy in geo_values:
                        for bm25_weight in bm25_values:
                            for filter_char_weight in filter_char_values:
                                val_predictions = retrieve(
                                    val_queries,
                                    service_items,
                                    vectorizer_w,
                                    item_w,
                                    vectorizer_c,
                                    item_c,
                                    vectorizer_t,
                                    item_t,
                                    vectorizer_bm25,
                                    item_bm25,
                                    validation_signals,
                                    location_boost=location_boost,
                                    remote_slots=remote_slots,
                                    title_weight=title_weight,
                                    geo_hierarchy=geo_hierarchy,
                                    bm25_weight=bm25_weight,
                                    filter_char_weight=filter_char_weight,
                                )
                                metrics = validate_predictions(
                                    val_queries, relevant, val_predictions
                                )
                                metrics["location_boost"] = location_boost
                                metrics["remote_slots"] = remote_slots
                                metrics["title_weight"] = title_weight
                                metrics["geo_hierarchy"] = geo_hierarchy
                                metrics["bm25_weight"] = bm25_weight
                                metrics["filter_char_weight"] = filter_char_weight
                                print(json.dumps(metrics, indent=2))
                                if (
                                    args.tune_title_weight
                                    and bm25_weight == bm25_values[0]
                                    and geo_hierarchy == geo_values[0]
                                    and location_boost == location_values[0]
                                    and remote_slots == remote_values[0]
                                ):
                                    title_predictions[title_weight] = val_predictions
        if args.tune_title_weight and {0.0, 0.20}.issubset(title_predictions):
            for replacements in (3, 5, 10):
                blended = blend_predictions(
                    title_predictions[0.0], title_predictions[0.20], replacements
                )
                metrics = validate_predictions(val_queries, relevant, blended)
                metrics["title_blend_replacements"] = replacements
                print(json.dumps(metrics, indent=2))

    predictions = retrieve(
        queries,
        service_items,
        vectorizer_w,
        item_w,
        vectorizer_c,
        item_c,
        vectorizer_t,
        item_t,
        vectorizer_bm25,
        item_bm25,
        signals,
        location_boost=args.location_boost,
        remote_slots=args.remote_slots,
        title_weight=args.title_weight,
        geo_hierarchy=args.geo_hierarchy,
        bm25_weight=args.bm25_weight,
        filter_char_weight=args.filter_char_weight,
    )
    answer = pd.DataFrame(
        {"query_id": queries.query_id.astype(str), "answer": [" ".join(x) for x in predictions]}
    )
    assert_answer(answer, queries, items)
    answer.to_csv(args.output, index=False, encoding="utf-8", lineterminator="\n")
    print(f"saved {len(answer)} rows to {args.output}")


if __name__ == "__main__":
    main()
