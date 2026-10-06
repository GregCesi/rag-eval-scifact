"""Dashboard d'exploration des résultats d'évaluation RAG SciFact.

Lancer : streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from transformers import AutoTokenizer

from rag_eval_scifact import reading_levels
from rag_eval_scifact.compare import load_run as load_run_file
from rag_eval_scifact.origin_labels import (
    build_label_lookup,
    filter_rows_by_categorie,
    label_in_clear,
)
from rag_eval_scifact.passage_detail import (
    find_query_passage_detail,
)
from rag_eval_scifact.passage_detail import (
    load_passage_detail as load_passage_detail_file,
)
from rag_eval_scifact.report import load_campaign_runs
from rag_eval_scifact.run_diff import (
    FILTER_LABELS,
    NOT_FOUND_RANK,
    changed_claims,
    compare_claims,
    filter_claims,
    find_claim,
    list_campaign_dirs,
    list_campaign_run_files,
    rank_comparison_counts,
    unit_badge,
)

RESULTS_DIR = Path("results")
CORPUS_PATH = Path("data/scifact/corpus.jsonl")


@st.cache_data
def load_run(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_campaign_run(path: Path) -> dict:
    """Charge un run de campagne (`results/<campagne>/<run>.json.gz`)."""
    return load_run_file(path)


@st.cache_data
def load_passage_detail(path: Path) -> dict | None:
    """Charge le fichier dérivé des passages d'un run, ou None s'il est absent (EXE-112)."""
    return load_passage_detail_file(path)


@st.cache_data
def load_corpus() -> dict[str, dict]:
    """Charge le corpus {doc_id: {title, text}}."""
    corpus = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {"title": doc["title"], "text": doc["text"]}
    return corpus


@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data
def load_buckets(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_reading_levels_goldens() -> list[dict] | None:
    """Les 339 goldens niveau-de-lecture (EXE-137). `None` si un fichier
    requis manque — la page le dit en une phrase, sans tableau ni diagramme."""
    return reading_levels.load_goldens()


@st.cache_data
def load_v2_grid_runs() -> list[dict]:
    """Les 34 runs de v2-grid, chargés une fois (H4, EXE-137)."""
    return load_campaign_runs(reading_levels.CAMPAGNE)


ANNOTATIONS_PATH = RESULTS_DIR / "v1-annotations.json"


def load_annotations(path: Path) -> dict[str, dict[str, str]]:
    """Charge les annotations depuis un JSON separe.

    Retourne {query_id: {"categorie": ..., "note": ...}}.
    Tolerant : fichier absent -> dict vide.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_annotation(query_id: str, categorie: str, note: str):
    """Sauvegarde une annotation dans le JSON (merge)."""
    annotations = load_annotations(ANNOTATIONS_PATH)
    if categorie or note:
        annotations[query_id] = {"categorie": categorie, "note": note}
    elif query_id in annotations:
        del annotations[query_id]
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


ORIGIN_LABELS_PATH = RESULTS_DIR / "etiquettes-origine.json"


@st.cache_data
def load_origin_labels(path: Path) -> dict | None:
    """Charge le fichier des étiquettes d'origine (EXE-124). None si absent —
    les pages qui l'utilisent n'affichent alors aucune étiquette d'origine."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_at_truncation(text: str, max_tokens: int = 256) -> tuple[str, str]:
    """Découpe le texte à la frontière de troncature du tokenizer.

    Retourne (partie_vue, partie_ignoree).
    """
    tokenizer = load_tokenizer()
    # On encode sans tokens spéciaux ([CLS]/[SEP]) pour compter le contenu pur
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) <= max_tokens:
        return text, ""
    # Décoder les max_tokens premiers tokens pour retrouver le texte correspondant
    seen_text = tokenizer.decode(encoded[:max_tokens], skip_special_tokens=True)
    # Le reste
    lost_text = tokenizer.decode(encoded[max_tokens:], skip_special_tokens=True)
    return seen_text, lost_text


def highlight_evidence_sentences(text: str, evidence_sentences: list[str]) -> str:
    """Surligne les phrases-preuve retrouvées mot pour mot dans `text` (critère
    11, EXE-124). Une phrase absente du texte (environ 20% des cas, H2) est
    simplement laissée non surlignée, jamais signalée comme une erreur."""
    for sentence in evidence_sentences:
        if sentence and sentence in text:
            text = text.replace(
                sentence,
                f'<mark style="background:#f1c40f; color:#111">{sentence}</mark>',
            )
    return text


def _find_span_ignoring_case_and_spaces(
    text: str, sentence: str
) -> tuple[int, int] | None:
    """Position (début, fin) de `sentence` dans `text`, casse et espaces
    ignorés des deux côtés ; None si absente (EXE-126, H1b)."""
    positions = [i for i, ch in enumerate(text) if not ch.isspace()]
    haystack = "".join(text[i] for i in positions).lower()
    needle = "".join(ch for ch in sentence if not ch.isspace()).lower()
    if not needle:
        return None
    idx = haystack.find(needle)
    if idx == -1:
        return None
    return positions[idx], positions[idx + len(needle) - 1] + 1


def highlight_evidence_sentences_after_decode(
    text: str, evidence_sentences: list[str]
) -> str:
    """Comme `highlight_evidence_sentences`, pour un texte redécodé par le
    tokenizer (critère 3, EXE-126, H1b) : le redécodage change la casse et
    les espaces (ex. "genome-wide" devient "genome - wide"), si bien que la
    phrase-preuve d'origine ne s'y retrouve plus telle quelle. On la
    retrouve en ignorant casse et espaces des deux côtés — toujours mot pour
    mot, jamais par rapprochement approximatif."""
    spans = []
    for sentence in evidence_sentences:
        if not sentence:
            continue
        span = _find_span_ignoring_case_and_spaces(text, sentence)
        if span is not None:
            spans.append(span)
    for start, end in sorted(spans, reverse=True):
        text = (
            text[:start]
            + f'<mark style="background:#f1c40f; color:#111">{text[start:end]}</mark>'
            + text[end:]
        )
    return text


def count_unlocated_evidence_sentences(text: str, evidence_sentences: list[str]) -> int:
    """Nombre de phrases-preuve absentes mot pour mot de `text` (critère 8,
    EXE-126) — toujours compté sur le texte brut du document, jamais sur une
    vue redécodée ou tronquée."""
    return sum(1 for s in evidence_sentences if s and s not in text)


def _announce_unlocated_evidence_sentences(
    text: str, evidence_sentences: list[str] | None
) -> None:
    """Ligne « n phrase(s)-preuve non localisée(s) dans ce texte » quand au
    moins une phrase-preuve ne s'y retrouve pas mot pour mot (critère 8,
    EXE-126)."""
    if not evidence_sentences:
        return
    missing = count_unlocated_evidence_sentences(text, evidence_sentences)
    if missing:
        st.caption(f"{missing} phrase(s)-preuve non localisée(s) dans ce texte")


def _evidence_spans(text: str, evidence_sentences: list[str]) -> list[tuple[int, int]]:
    """Positions (début, fin) des phrases-preuve retrouvées mot pour mot dans
    `text` (critère 2, EXE-126) — texte brut, jamais redécodé : recherche
    exacte, comme `highlight_evidence_sentences`."""
    spans = []
    for sentence in evidence_sentences:
        if not sentence:
            continue
        start = text.find(sentence)
        if start != -1:
            spans.append((start, start + len(sentence)))
    return spans


SOURCE_LABELS = {"v1": "Runs v1 historiques"}


def list_sources(results_dir: Path) -> list[str]:
    """Sources de runs proposées dans la barre latérale (critère 1, EXE-113) :
    les runs v1 historiques, puis chaque campagne (dossier à runs `*.json.gz`)."""
    return ["v1"] + list_campaign_dirs(results_dir)


def runs_for_source(results_dir: Path, source: str) -> list[Path]:
    """Fichiers de run proposés pour une source choisie (critère 1, EXE-113).

    `results_dir` contient aussi des annotations et des buckets en `.json` à
    plat : seul le motif `v1-dense-*.json` les en distingue pour la source v1.
    """
    if source == "v1":
        return sorted(results_dir.glob("v1-dense-*.json"), reverse=True)
    return list_campaign_run_files(results_dir, source)


def effective_retriever_config(run_data: dict) -> dict:
    """Config du retriever sous une forme uniforme, pour `unit_badge` et l'affichage.

    Un run v1 (format historique) n'a pas de bloc `retriever` : il est
    toujours dense, document entier, MiniLM — ce bloc synthétique le
    représente sous la même forme qu'un run de campagne (critères 2 et 7,
    EXE-113).
    """
    cfg = run_data["config"]
    if "retriever" in cfg:
        return cfg["retriever"]
    return {
        "name": "dense",
        "model": cfg["model"],
        "max_seq_length": cfg["max_seq_length"],
        "unit": "document",
    }


def effective_rerank_config(run_data: dict) -> dict | None:
    """Config du reranker, ou None si absent (toujours None pour un run v1)."""
    return run_data["config"].get("rerank")


def is_cosine_scored(retriever_cfg: dict, rerank_cfg: dict | None) -> bool:
    """Les scores de ce run sont-ils des cosinus bornés dans [0, 1] (H2, EXE-113) ?

    Seul le dense pur, sans reclassement, respecte cette échelle : BM25, la
    fusion hybride (union ou RRF) et le reranker cross-encoder ne la
    respectent pas — les graphiques qui la supposent ne s'affichent pas pour
    ces runs (« ce qui ne doit pas arriver »)."""
    if rerank_cfg is not None and rerank_cfg.get("name") not in (None, "none"):
        return False
    return retriever_cfg["name"] == "dense"


def dashboard_max_tokens(retriever_cfg: dict) -> int | None:
    """Fenêtre de coupe affichée pour le texte d'un doc de ce run (critère 7,
    EXE-113). None quand ce run ne tronque pas réellement à 256 tokens
    (fenêtre longue, passages, BM25) — sinon `render_doc_text` inventerait une
    coupe qui n'a jamais eu lieu (même règle que la page de comparaison,
    EXE-111) : `unit_badge` tranche, avec un token_count volontairement grand
    pour n'évaluer que les conditions de fenêtre, jamais une longueur réelle.
    """
    return 256 if unit_badge(retriever_cfg, 999_999) == "tronqué" else None


TRUNCATION_BADGE_MARKDOWN = {
    "tronqué": ":orange[tronqué]",
    "complet": ":green[complet]",
    "lu en entier": ":green[lu en entier]",
    "découpé en passages": ":blue[découpé en passages]",
}


def truncation_badge_markdown(retriever_cfg: dict, token_count: int) -> str:
    """Badge coloré de troncature d'un doc attendu, selon la même règle que la
    page de comparaison (critère 7, EXE-113)."""
    return TRUNCATION_BADGE_MARKDOWN[unit_badge(retriever_cfg, token_count)]


def claim_rank_and_score(q: dict) -> tuple[int | None, float | None]:
    """Rang et score du meilleur document attendu, dans le run choisi (critère
    5, EXE-113). Ne relit jamais `v1-buckets.json` pour ces deux valeurs : ce
    fichier ne fournit plus que le regroupement par bucket, figé sur le run v1.
    """
    rank = q["per_query_metrics"]["best_rank"]
    if rank is None:
        return None, None
    return rank, q["retrieved_top100"][rank - 1]["score"]


def campaign_strategy_lines(run_data: dict) -> list[str]:
    """Lignes markdown de la stratégie déclarée par la config d'un run de
    campagne (critère 2, EXE-113) : retriever, modèle, unité, fenêtre, fusion
    si elle s'applique, reranker si activé, date. Jamais la dimension
    d'embedding — le champ `dim` des configs de v2-grid vaut 384 même pour
    Qwen, il est faux (« ce qui ne doit pas arriver »).
    """
    retriever = run_data["config"]["retriever"]
    rerank = run_data["config"].get("rerank")
    lines = [
        f"**Retriever** : `{retriever['name']}`",
        f"**Modèle** : `{retriever['model']}`",
        f"**Unité** : {retriever['unit']}",
    ]
    if retriever["unit"] == "passages":
        lines.append(
            f"**Fenêtre** : {retriever['chunk_size']} tokens "
            f"(chevauchement {retriever['chunk_overlap']})"
        )
    else:
        lines.append(f"**Fenêtre** : {retriever['max_seq_length']} tokens")
    if retriever["name"] == "hybrid":
        fusion = retriever["fusion_mode"]
        if fusion == "rrf":
            fusion = f"rrf (k={retriever['rrf_k']})"
        lines.append(f"**Fusion** : {fusion}")
    if rerank is not None and rerank.get("name") not in (None, "none"):
        lines.append(f"**Reranker** : `{rerank['model']}` (top {rerank['top_n']})")
    lines.append(f"**Date** : {run_data['date']}")
    return lines


def queries_to_dataframe(queries: list[dict], retriever_cfg: dict) -> pd.DataFrame:
    """Aplatit les requêtes en DataFrame pour filtrage/tri interactif."""
    rows = []
    for q in queries:
        m = q["per_query_metrics"]
        rank = m["best_rank"]
        expected = q["expected_docs"]
        token_count = expected[0]["token_count"]
        top1_score = (
            q["retrieved_top100"][0]["score"] if q["retrieved_top100"] else None
        )
        relevant_score = (
            q["retrieved_top100"][rank - 1]["score"] if rank and rank <= 100 else None
        )
        score_gap = (
            (top1_score - relevant_score)
            if (top1_score and relevant_score and rank != 1)
            else None
        )

        # Catégorisation lisible
        if rank is None:
            category = "Non trouvé"
        elif rank == 1:
            category = "Rang 1"
        elif rank <= 5:
            category = "Rang 2-5"
        elif rank <= 10:
            category = "Rang 6-10"
        elif rank <= 50:
            category = "Rang 11-50"
        else:
            category = "Rang 51-100"

        troncature = unit_badge(retriever_cfg, token_count)

        rows.append(
            {
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "best_rank": rank,
                "category": category,
                "token_count": token_count,
                "troncature": troncature,
                "nb_docs_attendus": len(expected),
                "top1_score": top1_score,
                "relevant_score": relevant_score,
                "score_gap": score_gap,
                "found@1": m["found@1"],
                "found@5": m["found@5"],
                "found@10": m["found@10"],
                "found@100": m["found@100"],
            }
        )
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="RAG Eval SciFact", layout="wide")

    # ===== SIDEBAR =====
    st.sidebar.title("RAG Eval SciFact")

    sources = list_sources(RESULTS_DIR)
    selected_source = st.sidebar.selectbox(
        "Source", sources, format_func=lambda s: SOURCE_LABELS.get(s, s)
    )
    run_files = runs_for_source(RESULTS_DIR, selected_source)

    selected_run = None
    if not run_files:
        st.sidebar.info("Aucun run dans cette source.")
    elif selected_source == "v1":
        run_path = st.sidebar.selectbox("Run", run_files, format_func=lambda p: p.stem)
        data = load_run(run_path)
        cfg = data["config"]
        st.sidebar.markdown(f"**Modèle** : `{cfg['model']}`")
        st.sidebar.markdown(
            f"**Dim** : {cfg['dim']}  |  **Max seq** : {cfg['max_seq_length']}"
        )
        st.sidebar.markdown(f"**Date** : {data['date']}")
        selected_run = {"data": data, "can_annotate": True}
    else:
        runs_loaded = {p: load_campaign_run(p) for p in run_files}
        run_path = st.sidebar.selectbox(
            "Run", run_files, format_func=lambda p: runs_loaded[p]["run_name"]
        )
        data = runs_loaded[run_path]
        for line in campaign_strategy_lines(data):
            st.sidebar.markdown(line)
        selected_run = {"data": data, "can_annotate": False}

    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Vue d'ensemble",
            "Exploration interactive",
            "Analyse d'erreurs",
            "Comparer deux runs",
            "Niveaux de lecture",
        ],
    )

    # Charger les buckets si disponibles
    buckets_path = RESULTS_DIR / "v1-buckets.json"
    buckets_data = load_buckets(buckets_path) if buckets_path.exists() else None
    corpus = load_corpus()

    # Étiquettes d'origine (EXE-124) : absentes -> dicts vides, aucune étiquette
    # ni filtre affiché, jamais d'erreur.
    origin_labels = load_origin_labels(ORIGIN_LABELS_PATH)
    label_by_pair, evidence_by_pair, categorie_by_qid = (
        build_label_lookup(origin_labels) if origin_labels else ({}, {}, {})
    )

    # La page de comparaison ne dépend pas du run choisi ci-dessus (critère 9,
    # EXE-113) : elle s'ouvre même si aucun run n'est sélectionnable.
    if page == "Comparer deux runs":
        page_compare(
            buckets_data, corpus, label_by_pair, evidence_by_pair, categorie_by_qid
        )
        return

    # Comme « Comparer deux runs » (critère 1, EXE-137) : cette page ne
    # dépend pas du run choisi dans la barre de gauche, elle s'ouvre même si
    # aucun run n'est sélectionnable.
    if page == "Niveaux de lecture":
        page_reading_levels(corpus)
        return

    if selected_run is None:
        st.info("Choisis un run dans la barre de gauche pour voir cette page.")
        return

    data = selected_run["data"]
    retriever_cfg = effective_retriever_config(data)
    rerank_cfg = effective_rerank_config(data)
    annotations = load_annotations(ANNOTATIONS_PATH)
    queries = data["queries"]
    df = queries_to_dataframe(queries, retriever_cfg)
    queries_by_id = {q["query_id"]: q for q in queries}

    if page == "Vue d'ensemble":
        page_overview(data, df, retriever_cfg, rerank_cfg)
    elif page == "Exploration interactive":
        page_explore(df, queries_by_id, corpus, retriever_cfg)
    elif page == "Analyse d'erreurs":
        page_errors(
            df,
            queries_by_id,
            corpus,
            buckets_data,
            annotations,
            retriever_cfg,
            selected_run["can_annotate"],
            label_by_pair,
            evidence_by_pair,
        )


# ─────────────────────────────────────────────
# PAGE 1 : VUE D'ENSEMBLE
# ─────────────────────────────────────────────
def page_overview(
    data: dict, df: pd.DataFrame, retriever_cfg: dict, rerank_cfg: dict | None
):
    st.header("Vue d'ensemble")

    # Métriques
    metrics = data["metrics"]
    cols = st.columns(6)
    for col, (label, key) in zip(
        cols,
        [
            ("Recall@1", "recall@1"),
            ("Recall@5", "recall@5"),
            ("Recall@10", "recall@10"),
            ("Recall@100", "recall@100"),
            ("nDCG@10", "ndcg@10"),
            ("MRR", "mrr"),
        ],
    ):
        col.metric(label, f"{metrics[key]:.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    # Distribution des rangs
    with col1:
        cat_order = [
            "Rang 1",
            "Rang 2-5",
            "Rang 6-10",
            "Rang 11-50",
            "Rang 51-100",
            "Non trouvé",
        ]
        cat_colors = {
            "Rang 1": "#2ecc71",
            "Rang 2-5": "#27ae60",
            "Rang 6-10": "#f1c40f",
            "Rang 11-50": "#e67e22",
            "Rang 51-100": "#e74c3c",
            "Non trouvé": "#7f8c8d",
        }
        counts = df["category"].value_counts().reindex(cat_order, fill_value=0)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts.index,
                    y=counts.values,
                    marker_color=[cat_colors[c] for c in counts.index],
                    text=counts.values,
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Distribution des rangs", yaxis_title="Requêtes", height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recall par bucket de longueur
    with col2:
        bins = [0, 128, 256, 512, 9999]
        labels = ["≤128", "129-256", "257-512", ">512"]
        df["len_bucket"] = pd.cut(df["token_count"], bins=bins, labels=labels)
        recall_data = (
            df.groupby("len_bucket", observed=True)
            .agg(
                n=("query_id", "count"),
                R1=("found@1", "mean"),
                R10=("found@10", "mean"),
                R100=("found@100", "mean"),
            )
            .reset_index()
        )
        fig = go.Figure()
        for metric, name in [
            ("R1", "Recall@1"),
            ("R10", "Recall@10"),
            ("R100", "Recall@100"),
        ]:
            fig.add_trace(
                go.Bar(
                    name=name,
                    x=recall_data["len_bucket"],
                    y=recall_data[metric],
                    text=[f"{v:.0%}" for v in recall_data[metric]],
                    textposition="auto",
                )
            )
        fig.update_layout(
            barmode="group",
            title="Recall par longueur de doc",
            yaxis_title="Recall",
            height=350,
        )
        for _, row in recall_data.iterrows():
            fig.add_annotation(
                x=row["len_bucket"],
                y=-0.08,
                text=f"n={row['n']}",
                showarrow=False,
                yref="paper",
                font={"size": 10, "color": "gray"},
            )
        st.plotly_chart(fig, use_container_width=True)

    # Scores — seulement à l'échelle cosinus (H2, EXE-113) : BM25, fusion
    # hybride et reranker n'ont pas cette échelle, et un graphique qui la
    # suppose ne s'affiche pas avec des données inventées pour eux (« ce qui
    # ne doit pas arriver »).
    col1, col2 = st.columns(2)
    if not is_cosine_scored(retriever_cfg, rerank_cfg):
        message = (
            "Scores non cosinus (BM25, fusion ou reranker) : "
            "ce graphique ne s'applique pas à ce run."
        )
        col1.info(message)
        col2.info(message)
        return

    with col1:
        fig = px.histogram(
            df,
            x="top1_score",
            nbins=30,
            title="Score cosinus du rang 1",
            labels={"top1_score": "Score cosinus"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        mask = df["relevant_score"].notna()
        fig = px.scatter(
            df[mask],
            x="top1_score",
            y="relevant_score",
            title="Score rang 1 vs Score doc pertinent",
            labels={
                "top1_score": "Score rang 1",
                "relevant_score": "Score doc pertinent",
            },
            opacity=0.5,
        )
        fig.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "gray"}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 2 : EXPLORATION INTERACTIVE
# ─────────────────────────────────────────────
def page_explore(
    df: pd.DataFrame, queries_by_id: dict, corpus: dict, retriever_cfg: dict
):
    st.header("Exploration interactive")
    st.caption("Filtre, trie, clique — explore les requêtes selon n'importe quel axe.")

    # --- Filtres dans la sidebar ---
    st.sidebar.divider()
    st.sidebar.markdown("### Filtres")

    # Filtre par catégorie de rang
    all_cats = [
        "Rang 1",
        "Rang 2-5",
        "Rang 6-10",
        "Rang 11-50",
        "Rang 51-100",
        "Non trouvé",
    ]
    selected_cats = st.sidebar.multiselect(
        "Catégorie de rang", all_cats, default=all_cats
    )

    # Filtre troncature — options dérivées de ce run (critère 7, EXE-113) :
    # un run en passages ou BM25 n'a pas de "Tronqué"/"Complet" à proposer.
    trunc_options = ["Tous"] + sorted(df["troncature"].unique())
    trunc_filter = st.sidebar.radio("Troncature", trunc_options, index=0)

    # Filtre score gap
    score_gap_filter = st.sidebar.checkbox(
        "Seulement les 'presque' (score gap < 0.02)", value=False
    )

    # Filtre rang numérique
    rank_range = st.sidebar.slider(
        "Rang (None = 101)", min_value=1, max_value=101, value=(1, 101)
    )

    # Filtre longueur doc
    tc_range = st.sidebar.slider(
        "Tokens du doc attendu",
        min_value=int(df["token_count"].min()),
        max_value=int(df["token_count"].max()),
        value=(int(df["token_count"].min()), int(df["token_count"].max())),
    )

    # --- Appliquer les filtres ---
    filtered = df[df["category"].isin(selected_cats)].copy()
    if trunc_filter != "Tous":
        filtered = filtered[filtered["troncature"] == trunc_filter]
    if score_gap_filter:
        filtered = filtered[
            (filtered["score_gap"].notna()) & (filtered["score_gap"] < 0.02)
        ]

    # Rang : on remplace None par 101 pour le slider
    filtered["_rank_sortable"] = filtered["best_rank"].fillna(101).astype(int)
    filtered = filtered[
        (filtered["_rank_sortable"] >= rank_range[0])
        & (filtered["_rank_sortable"] <= rank_range[1])
    ]
    filtered = filtered[
        (filtered["token_count"] >= tc_range[0])
        & (filtered["token_count"] <= tc_range[1])
    ]

    # --- Tri ---
    sort_col = st.selectbox(
        "Trier par",
        ["best_rank", "token_count", "top1_score", "score_gap", "query_id"],
        index=0,
    )
    sort_asc = st.checkbox("Ordre croissant", value=True)
    if sort_col == "best_rank":
        filtered = filtered.sort_values("_rank_sortable", ascending=sort_asc)
    else:
        filtered = filtered.sort_values(
            sort_col, ascending=sort_asc, na_position="last"
        )

    # --- Stats du filtre ---
    st.markdown(f"**{len(filtered)}** requêtes correspondent aux filtres")

    # Métriques rapides sur le subset filtré
    if len(filtered) > 0:
        cols = st.columns(4)
        cols[0].metric("Recall@1", f"{filtered['found@1'].mean():.1%}")
        cols[1].metric("Recall@10", f"{filtered['found@10'].mean():.1%}")
        cols[2].metric("Recall@100", f"{filtered['found@100'].mean():.1%}")
        cols[3].metric("Rang moyen", f"{filtered['_rank_sortable'].mean():.1f}")

    # --- Scatter interactif ---
    if len(filtered) > 0:
        scatter_df = filtered.copy()
        scatter_df["rank_display"] = scatter_df["best_rank"].fillna(101)
        fig = px.scatter(
            scatter_df,
            x="token_count",
            y="rank_display",
            color="category",
            color_discrete_map={
                "Rang 1": "#2ecc71",
                "Rang 2-5": "#27ae60",
                "Rang 6-10": "#f1c40f",
                "Rang 11-50": "#e67e22",
                "Rang 51-100": "#e74c3c",
                "Non trouvé": "#7f8c8d",
            },
            hover_data=["query_id", "query_text", "top1_score"],
            labels={"token_count": "Tokens doc", "rank_display": "Rang"},
            title=f"Vue filtrée — {len(filtered)} requêtes",
            opacity=0.7,
        )
        fig.add_hline(y=10, line_dash="dash", line_color="red", opacity=0.3)
        fig.add_vline(x=256, line_dash="dash", line_color="orange", opacity=0.3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Tableau ---
    display_cols = [
        "query_id",
        "query_text",
        "category",
        "best_rank",
        "token_count",
        "troncature",
        "top1_score",
        "relevant_score",
        "score_gap",
    ]
    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=400,
        column_config={
            "query_text": st.column_config.TextColumn("Query", width="large"),
            "top1_score": st.column_config.NumberColumn("Score rang 1", format="%.4f"),
            "relevant_score": st.column_config.NumberColumn(
                "Score pertinent", format="%.4f"
            ),
            "score_gap": st.column_config.NumberColumn("Ecart score", format="%.4f"),
        },
    )

    # --- Détail d'une requête ---
    st.divider()
    st.subheader("Détail d'une requête")
    query_ids = filtered["query_id"].tolist()
    if query_ids:
        chosen_id = st.selectbox(
            "Sélectionner une requête",
            query_ids,
            format_func=lambda qid: f"[{qid}] {queries_by_id[qid]['query_text'][:80]}",
        )
        render_query_detail(queries_by_id[chosen_id], corpus, retriever_cfg)


def _text_with_cut(text: str, max_tokens: int | None) -> str:
    """Retourne le texte avec marqueur de coupe en texte brut.

    `max_tokens=None` (critère 7, EXE-113) : ce run ne tronque pas réellement
    à 256 tokens — jamais de marqueur de coupe inventé."""
    if max_tokens is None:
        return text
    seen, lost = split_at_truncation(text, max_tokens)
    if not lost:
        return text
    return f"{seen}\n\n--- ✂ coupe {max_tokens} tokens ---\n\n{lost}"


def export_query_markdown(
    q: dict,
    corpus: dict,
    ann: dict,
    retriever_cfg: dict,
    label_by_pair: dict[tuple[str, str], str] | None = None,
    evidence_by_pair: dict[tuple[str, str], list[str]] | None = None,
) -> str:
    """Genere le markdown complet d'une query pour copier-coller.

    `label_by_pair`/`evidence_by_pair` (critère 9, EXE-126) ajoutent, pour
    chaque document attendu, son étiquette d'origine en clair et ses
    phrases-preuve."""
    lines = []
    qid = q["query_id"]
    rank = q["per_query_metrics"]["best_rank"]
    expected_ids = {d["doc_id"] for d in q["expected_docs"]}
    max_tokens = dashboard_max_tokens(retriever_cfg)

    lines.append(f"## Query {qid}")
    lines.append(f"**Question** : {q['query_text']}")
    lines.append(
        f"**Meilleur rang** : {rank if rank else 'NON TROUVE dans le top 100'}"
    )
    lines.append("")

    lines.append("### Document(s) attendu(s)")
    lines.append("")
    for ed in q["expected_docs"]:
        doc_id = ed["doc_id"]
        doc_data = corpus.get(doc_id, {})
        tc = ed["token_count"]
        status = unit_badge(retriever_cfg, tc).upper()
        lines.append(
            f"**{doc_data.get('title', '?')}** (`{doc_id}`, {tc} tokens, {status})"
        )
        label = (label_by_pair or {}).get((qid, doc_id))
        if label:
            lines.append(f"**Étiquette d'origine** : {label_in_clear(label)}")
        evidence_sentences = (
            (evidence_by_pair or {}).get((qid, doc_id))
            if label in ("SUPPORT", "CONTRADICT")
            else None
        )
        if evidence_sentences:
            lines.append("**Phrases-preuve** :")
            for sentence in evidence_sentences:
                lines.append(f"- {sentence}")
        lines.append("")
        lines.append(_text_with_cut(doc_data.get("text", ""), max_tokens))
        lines.append("")

    lines.append("### Top-5 retrouve")
    lines.append("")
    for d in q["retrieved_top100"][:5]:
        doc_data = corpus.get(d["doc_id"], {})
        text = doc_data.get("text", "")
        lost = max_tokens and split_at_truncation(text, max_tokens)[1]
        status = "TRONQUE" if lost else "COMPLET"
        hit = " ✅ PERTINENT" if d["doc_id"] in expected_ids else ""
        lines.append(
            f"**Rang {d['rank']}** (score {d['score']:.4f}, {status}) "
            f"— **{doc_data.get('title', '?')}** (`{d['doc_id']}`){hit}"
        )
        lines.append("")
        lines.append(_text_with_cut(text, max_tokens))
        lines.append("")

    if ann.get("categorie") or ann.get("note"):
        lines.append("### Mon annotation")
        lines.append(f"**Categorie** : {ann.get('categorie', '')}")
        lines.append(f"**Note** : {ann.get('note', '')}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# PAGE 3 : ANALYSE D'ERREURS
# ─────────────────────────────────────────────
def page_errors(
    df: pd.DataFrame,
    queries_by_id: dict,
    corpus: dict,
    buckets_data: dict | None,
    annotations: dict[str, dict[str, str]],
    retriever_cfg: dict,
    can_annotate: bool,
    label_by_pair: dict[tuple[str, str], str] | None = None,
    evidence_by_pair: dict[tuple[str, str], list[str]] | None = None,
):
    st.header("Analyse d'erreurs")

    if buckets_data is None:
        st.warning(
            "Fichier `results/v1-buckets.json` absent. "
            "Lancer `python scripts/extract_error_analysis.py` d'abord."
        )
        return

    # --- Construire les listes de queries par bucket ---
    bucket_names = ["miss_100", "deep_miss", "near_miss", "perfect"]
    bucket_labels = {
        "miss_100": "miss_100 — hors top 100",
        "deep_miss": "deep_miss — top 100, hors top 10",
        "near_miss": "near_miss — top 10, pas rang 1",
        "perfect": "perfect — hit rang 1",
    }
    counts = buckets_data["counts"]
    queries_by_bucket: dict[str, list[str]] = {b: [] for b in bucket_names}
    for qid, info in buckets_data["queries"].items():
        queries_by_bucket[info["bucket"]].append(qid)
    # Trier par query_id numerique si possible
    for b in bucket_names:
        queries_by_bucket[b].sort(key=lambda x: int(x) if x.isdigit() else x)

    # --- Selecteur de bucket ---
    bucket_options = [f"{bucket_labels[b]} ({counts[b]})" for b in bucket_names]

    def _on_bucket_change():
        st.session_state.error_query_idx = 0
        st.session_state._error_query_select = 0
        st.session_state.pop("show_export", None)

    selected_idx = st.selectbox(
        "Bucket",
        range(len(bucket_names)),
        format_func=lambda i: bucket_options[i],
        key="_error_bucket_select",
        on_change=_on_bucket_change,
    )
    selected_bucket = bucket_names[selected_idx]
    query_ids = queries_by_bucket[selected_bucket]

    if not query_ids:
        st.info("Aucune query dans ce bucket.")
        return

    # --- Navigation query par query ---
    col_prev, col_select, col_next = st.columns([1, 6, 1])

    # Initialiser et clamper l'index
    if "error_query_idx" not in st.session_state:
        st.session_state.error_query_idx = 0
    st.session_state.error_query_idx = min(
        st.session_state.error_query_idx, len(query_ids) - 1
    )
    current_idx = st.session_state.error_query_idx

    def _on_query_change():
        st.session_state.error_query_idx = st.session_state._error_query_select
        st.session_state.pop("show_export", None)

    def _go_prev():
        idx = st.session_state.error_query_idx
        if idx > 0:
            st.session_state.error_query_idx = idx - 1
            st.session_state._error_query_select = idx - 1
            st.session_state.pop("show_export", None)

    def _go_next():
        idx = st.session_state.error_query_idx
        if idx < len(query_ids) - 1:
            st.session_state.error_query_idx = idx + 1
            st.session_state._error_query_select = idx + 1
            st.session_state.pop("show_export", None)

    with col_prev:
        st.button("< Prec", use_container_width=True, on_click=_go_prev)
    with col_next:
        st.button("Suiv >", use_container_width=True, on_click=_go_next)
    with col_select:
        st.selectbox(
            "Query",
            range(len(query_ids)),
            index=current_idx,
            format_func=lambda i: (
                f"[{i + 1}/{len(query_ids)}] {query_ids[i]} — "
                f"{queries_by_id[query_ids[i]]['query_text'][:70]}"
            ),
            key="_error_query_select",
            on_change=_on_query_change,
        )

    qid = query_ids[st.session_state.error_query_idx]
    q = queries_by_id[qid]
    expected_ids = {d["doc_id"] for d in q["expected_docs"]}
    max_tokens = dashboard_max_tokens(retriever_cfg)

    # --- Annotation ---
    ann = annotations.get(qid, {})
    if can_annotate:
        new_cat = st.text_input(
            "Categorie", value=ann.get("categorie", ""), key=f"ann_cat_{qid}"
        )
        new_note = st.text_area(
            "Note", value=ann.get("note", ""), height=200, key=f"ann_note_{qid}"
        )
        if st.button("Sauvegarder", key=f"ann_save_{qid}"):
            save_annotation(qid, new_cat.strip(), new_note.strip())
            st.toast(f"Annotation query {qid} sauvegardee")
            st.rerun()
    elif ann.get("categorie") or ann.get("note"):
        # Run de campagne (critère 6, EXE-113) : l'annotation v1 se lit, sans
        # bouton de sauvegarde — elle n'est jamais écrite depuis ce run.
        st.markdown("**Annotation v1** (lecture seule)")
        if ann.get("categorie"):
            st.markdown(f"**Categorie** : {ann['categorie']}")
        if ann.get("note"):
            st.markdown(ann["note"])

    # --- Infos bucket : rang/score du run choisi, jamais de v1-buckets.json
    # (critère 5, EXE-113) — ce fichier ne fournit plus que le regroupement
    # par bucket, figé sur le run v1.
    rank, score = claim_rank_and_score(q)
    if rank is not None:
        st.markdown(f"**Rang** : {rank}  |  **Score** : {score:.4f}")
    else:
        st.markdown("**Rang** : aucun doc pertinent dans le top 100")

    st.divider()

    # --- Query ---
    st.markdown("### Query")
    st.info(q["query_text"])

    # --- Docs attendus ---
    st.markdown("### Docs attendus")
    for d in q["expected_docs"]:
        doc_data = corpus.get(d["doc_id"])
        if not doc_data:
            st.markdown(f"`{d['doc_id']}` — *non trouve dans le corpus*")
            continue
        badge = truncation_badge_markdown(retriever_cfg, d["token_count"])
        label = (label_by_pair or {}).get((qid, d["doc_id"]))
        label_line = (
            f"  \n**Étiquette d'origine** : {label_in_clear(label)}" if label else ""
        )
        st.markdown(
            f"**{doc_data['title']}**  \n`{d['doc_id']}` — {d['token_count']} tokens  {badge}{label_line}"
        )
        evidence_sentences = (
            (evidence_by_pair or {}).get((qid, d["doc_id"]))
            if label in ("SUPPORT", "CONTRADICT")
            else None
        )
        render_doc_text(
            doc_data["text"],
            key=f"err_expected_{d['doc_id']}",
            max_tokens=max_tokens,
            evidence_sentences=evidence_sentences,
        )

    # --- Top-5 retrouve ---
    st.markdown("### Top-5 retrouve")
    for i, doc in enumerate(q["retrieved_top100"][:5]):
        doc_id = doc["doc_id"]
        is_relevant = doc_id in expected_ids
        doc_data = corpus.get(doc_id)
        title = doc_data["title"] if doc_data else "?"
        if is_relevant:
            label = f":green[Rang {doc['rank']}] — **{title}** — score {doc['score']:.4f}  :white_check_mark:"
        else:
            label = f"Rang {doc['rank']} — **{title}** — score {doc['score']:.4f}"
        with st.expander(label, expanded=False, key=f"err_top5_{qid}_{i}"):
            st.markdown(f"`{doc_id}`")
            if doc_data:
                render_doc_text(
                    doc_data["text"],
                    key=f"err_ret_{qid}_{doc_id}",
                    max_tokens=max_tokens,
                )

    # --- Export ---
    st.divider()
    if st.button("Copier pour l'IA"):
        st.session_state.show_export = qid
    if st.session_state.get("show_export") == qid:
        md = export_query_markdown(
            q, corpus, ann, retriever_cfg, label_by_pair, evidence_by_pair
        )
        st.code(md, language="markdown")


# ─────────────────────────────────────────────
# PAGE 4 : COMPARER DEUX RUNS
# ─────────────────────────────────────────────
METRIC_LABELS = [
    ("Recall@1", "recall@1"),
    ("Recall@5", "recall@5"),
    ("Recall@10", "recall@10"),
    ("Recall@100", "recall@100"),
    ("nDCG@10", "ndcg@10"),
    ("MRR", "mrr"),
]


def page_compare(
    buckets_data: dict | None,
    corpus: dict,
    label_by_pair: dict[tuple[str, str], str] | None = None,
    evidence_by_pair: dict[tuple[str, str], list[str]] | None = None,
    categorie_by_qid: dict[str, str] | None = None,
):
    st.header("Comparer deux runs")

    campaigns = list_campaign_dirs(RESULTS_DIR)
    if not campaigns:
        st.info(
            "Aucune campagne (dossier avec des runs `*.json.gz`) trouvée sous results/."
        )
        return

    campaign = st.selectbox("Campagne", campaigns, key="compare_campaign")
    run_files = list_campaign_run_files(RESULTS_DIR, campaign)

    if len(run_files) < 2:
        st.info(f"La campagne « {campaign} » a moins de deux runs ({len(run_files)}).")
        return

    loaded_runs = [load_campaign_run(p) for p in run_files]
    runs_by_name = {run["run_name"]: run for run in loaded_runs}
    paths_by_name = {run["run_name"]: p for run, p in zip(loaded_runs, run_files)}
    run_names = sorted(runs_by_name)

    col_a, col_b = st.columns(2)
    name_a = col_a.selectbox("Run A", run_names, index=0, key="compare_run_a")
    name_b = col_b.selectbox(
        "Run B", run_names, index=min(1, len(run_names) - 1), key="compare_run_b"
    )
    run_a, run_b = runs_by_name[name_a], runs_by_name[name_b]
    passage_detail_a = load_passage_detail(paths_by_name[name_a])
    passage_detail_b = load_passage_detail(paths_by_name[name_b])

    st.subheader("Métriques")
    metrics_df = pd.DataFrame(
        [
            {label: run_a["metrics"][key] for label, key in METRIC_LABELS},
            {label: run_b["metrics"][key] for label, key in METRIC_LABELS},
        ],
        index=[f"A — {name_a}", f"B — {name_b}"],
    )
    st.dataframe(
        metrics_df,
        column_config={
            label: st.column_config.NumberColumn(label, format="%.4f")
            for label, _ in METRIC_LABELS
        },
    )

    rows = compare_claims(run_a, run_b)
    counts = rank_comparison_counts(rows)

    st.subheader("Rang du meilleur document attendu, de A vers B")
    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleur dans B", counts["better"])
    c2.metric("Moins bon dans B", counts["worse"])
    c3.metric("Identique", counts["same"])

    st.subheader("Claims dont le rang change")
    filter_choice = st.radio(
        "Filtre",
        ["Tous"] + list(FILTER_LABELS.values()),
        horizontal=True,
        key="compare_filter",
    )
    label_to_key = {v: k for k, v in FILTER_LABELS.items()}
    display_rows = (
        changed_claims(rows)
        if filter_choice == "Tous"
        else filter_claims(rows, label_to_key[filter_choice])
    )

    # Filtre par catégorie d'étiquette d'origine (critère 12, EXE-124) :
    # n'apparaît que si le fichier des étiquettes d'origine est présent.
    if categorie_by_qid:
        categorie_titles = {
            "confirme": "confirme",
            "contredit": "contredit",
            "sans_preuve": "sans preuve",
        }
        selected_categories = st.multiselect(
            "Catégorie (étiquette d'origine)",
            list(categorie_titles),
            default=list(categorie_titles),
            format_func=lambda c: categorie_titles[c],
            key="compare_categorie_filter",
        )
        display_rows = filter_rows_by_categorie(
            display_rows, set(selected_categories), categorie_by_qid
        )

    bucket_by_qid = (
        {qid: info["bucket"] for qid, info in buckets_data["queries"].items()}
        if buckets_data
        else {}
    )

    if not display_rows:
        st.info("Aucun claim ne correspond à ce filtre.")
    else:
        table_df = pd.DataFrame(display_rows)
        table_df["bucket_v1"] = table_df["query_id"].map(bucket_by_qid).fillna("—")
        table_df["rank_diff"] = table_df["rank_b"].fillna(NOT_FOUND_RANK).astype(
            int
        ) - table_df["rank_a"].fillna(NOT_FOUND_RANK).astype(int)
        sort_desc = st.checkbox(
            "Trier par écart de rang absolu décroissant",
            value=True,
            key="compare_sort_desc",
        )
        table_df = table_df.reindex(
            table_df["rank_diff"].abs().sort_values(ascending=not sort_desc).index
        )

        st.dataframe(
            table_df[
                ["query_id", "query_text", "rank_a", "rank_b", "bucket_v1", "rank_diff"]
            ].reset_index(drop=True),
            use_container_width=True,
            height=350,
            column_config={
                "query_text": st.column_config.TextColumn("Claim", width="large"),
                "rank_a": st.column_config.NumberColumn("Rang A"),
                "rank_b": st.column_config.NumberColumn("Rang B"),
                "bucket_v1": st.column_config.TextColumn("Bucket v1"),
                "rank_diff": st.column_config.NumberColumn("Écart (B - A)"),
            },
        )

    st.divider()
    st.subheader("Détail d'un claim")
    text_by_qid = {r["query_id"]: r["query_text"] for r in display_rows}
    qids_available = [r["query_id"] for r in display_rows]

    col_number, col_select = st.columns([1, 3])
    with col_number:
        claim_number = st.text_input(
            "Ouvrir un claim par son numéro",
            key="compare_claim_number",
        ).strip()
    with col_select:
        chosen_from_table = (
            st.selectbox(
                "Ou choisir dans le tableau filtré",
                qids_available,
                format_func=lambda qid: f"[{qid}] {text_by_qid[qid][:80]}",
                key="compare_claim_select",
            )
            if qids_available
            else None
        )

    if claim_number:
        if find_claim(run_a, claim_number) is None:
            st.error(f"Claim {claim_number} introuvable dans le split test.")
        else:
            render_claim_comparison(
                claim_number,
                run_a,
                run_b,
                name_a,
                name_b,
                corpus,
                bucket_by_qid,
                passage_detail_a,
                passage_detail_b,
                label_by_pair,
                evidence_by_pair,
            )
    elif chosen_from_table is not None:
        render_claim_comparison(
            chosen_from_table,
            run_a,
            run_b,
            name_a,
            name_b,
            corpus,
            bucket_by_qid,
            passage_detail_a,
            passage_detail_b,
            label_by_pair,
            evidence_by_pair,
        )


def render_claim_comparison(
    qid: str,
    run_a: dict,
    run_b: dict,
    name_a: str,
    name_b: str,
    corpus: dict,
    bucket_by_qid: dict[str, str],
    passage_detail_a: dict | None = None,
    passage_detail_b: dict | None = None,
    label_by_pair: dict[tuple[str, str], str] | None = None,
    evidence_by_pair: dict[tuple[str, str], list[str]] | None = None,
):
    query_a = find_claim(run_a, qid)
    query_b = find_claim(run_b, qid)

    rank_a = query_a["per_query_metrics"]["best_rank"]
    rank_b = query_b["per_query_metrics"]["best_rank"]

    def _fmt_rank(rank: int | None) -> str:
        return str(rank) if rank is not None else "hors top 100"

    st.markdown(
        f"**Rang A** : {_fmt_rank(rank_a)}  |  **Rang B** : {_fmt_rank(rank_b)}  |  "
        f"**Bucket v1** : {bucket_by_qid.get(qid, '—')}"
    )

    st.markdown("### Query")
    st.info(query_a["query_text"])

    st.markdown("### Docs attendus")
    retriever_a = run_a["config"]["retriever"]
    retriever_b = run_b["config"]["retriever"]
    expected_b_by_id = {d["doc_id"]: d for d in query_b["expected_docs"]}
    for d_a in query_a["expected_docs"]:
        doc_id = d_a["doc_id"]
        d_b = expected_b_by_id.get(doc_id, d_a)
        doc_data = corpus.get(doc_id, {})
        token_count_a, token_count_b = d_a["token_count"], d_b["token_count"]
        badge_a = unit_badge(retriever_a, token_count_a)
        badge_b = unit_badge(retriever_b, token_count_b)
        token_label = (
            f"{token_count_a} tokens"
            if token_count_a == token_count_b
            else f"{token_count_a} tokens (A) / {token_count_b} tokens (B)"
        )
        label = (label_by_pair or {}).get((qid, doc_id))
        label_line = (
            f"  \n**Étiquette d'origine** : {label_in_clear(label)}" if label else ""
        )
        st.markdown(
            f"**{doc_data.get('title', '?')}**  \n`{doc_id}` — {token_label}  \n"
            f"A : {badge_a}  |  B : {badge_b}{label_line}"
        )
        evidence_sentences = (
            (evidence_by_pair or {}).get((qid, doc_id))
            if label in ("SUPPORT", "CONTRADICT")
            else None
        )
        if retriever_a["unit"] == "passages" or retriever_b["unit"] == "passages":
            col_pa, col_pb = st.columns(2)
            with col_pa:
                st.caption(f"Passages — A ({name_a})")
                render_doc_passages_or_text(
                    doc_id,
                    doc_data,
                    retriever_a,
                    passage_detail_a,
                    qid,
                    key=f"compare_expected_passages_a_{qid}_{doc_id}",
                    evidence_sentences=evidence_sentences,
                )
            with col_pb:
                st.caption(f"Passages — B ({name_b})")
                render_doc_passages_or_text(
                    doc_id,
                    doc_data,
                    retriever_b,
                    passage_detail_b,
                    qid,
                    key=f"compare_expected_passages_b_{qid}_{doc_id}",
                    evidence_sentences=evidence_sentences,
                )
        else:
            show_cut = badge_a == "tronqué" or badge_b == "tronqué"
            render_doc_text(
                doc_data.get("text", ""),
                key=f"compare_expected_{qid}_{doc_id}",
                max_tokens=256 if show_cut else None,
                evidence_sentences=evidence_sentences,
            )

    st.divider()
    expected_ids = {d["doc_id"] for d in query_a["expected_docs"]}
    top10_a = query_a["retrieved_top100"][:10]
    top10_b = query_b["retrieved_top100"][:10]
    ids_a = {d["doc_id"] for d in top10_a}
    ids_b = {d["doc_id"] for d in top10_b}

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"#### A — {name_a}")
        render_top10_side(
            top10_a,
            expected_ids,
            ids_b,
            corpus,
            retriever_a,
            passage_detail_a,
            qid,
            key_prefix=f"cmp_a_{qid}",
        )
    with col_b:
        st.markdown(f"#### B — {name_b}")
        render_top10_side(
            top10_b,
            expected_ids,
            ids_a,
            corpus,
            retriever_b,
            passage_detail_b,
            qid,
            key_prefix=f"cmp_b_{qid}",
        )

    st.divider()
    if st.button("Copier pour l'IA", key=f"compare_export_btn_{qid}"):
        st.session_state.compare_show_export = qid
    if st.session_state.get("compare_show_export") == qid:
        md = export_compare_markdown(
            qid, run_a, run_b, name_a, name_b, corpus, label_by_pair, evidence_by_pair
        )
        st.code(md, language="markdown")


def export_compare_markdown(
    qid: str,
    run_a: dict,
    run_b: dict,
    name_a: str,
    name_b: str,
    corpus: dict,
    label_by_pair: dict[tuple[str, str], str] | None = None,
    evidence_by_pair: dict[tuple[str, str], list[str]] | None = None,
) -> str:
    """Genere le markdown d'un claim compare entre deux runs, pour copier-coller.

    `label_by_pair`/`evidence_by_pair` (critère 9, EXE-126) ajoutent, pour
    chaque document attendu, son étiquette d'origine en clair et ses
    phrases-preuve — sans eux, l'export n'en disait rien."""
    query_a = find_claim(run_a, qid)
    query_b = find_claim(run_b, qid)

    lines = [
        f"## Claim {qid} — {name_a} vs {name_b}",
        f"**Query** : {query_a['query_text']}",
        "",
    ]

    lines.append("### Documents attendus")
    lines.append("")
    for d in query_a["expected_docs"]:
        doc_id = d["doc_id"]
        doc_data = corpus.get(doc_id, {})
        lines.append(
            f"**{doc_data.get('title', '?')}** (`{doc_id}`, {d['token_count']} tokens)"
        )
        label = (label_by_pair or {}).get((qid, doc_id))
        if label:
            lines.append(f"**Étiquette d'origine** : {label_in_clear(label)}")
        evidence_sentences = (
            (evidence_by_pair or {}).get((qid, doc_id))
            if label in ("SUPPORT", "CONTRADICT")
            else None
        )
        if evidence_sentences:
            lines.append("**Phrases-preuve** :")
            for sentence in evidence_sentences:
                lines.append(f"- {sentence}")
        lines.append("")
        lines.append(doc_data.get("text", ""))
        lines.append("")

    for label, query in ((name_a, query_a), (name_b, query_b)):
        lines.append(f"### Top 10 — {label}")
        lines.append("")
        for d in query["retrieved_top100"][:10]:
            doc_data = corpus.get(d["doc_id"], {})
            lines.append(
                f"Rang {d['rank']} (score {d['score']:.4f}) — "
                f"{doc_data.get('title', '?')} (`{d['doc_id']}`)"
            )
        lines.append("")

    return "\n".join(lines)


def render_top10_side(
    docs: list[dict],
    expected_ids: set[str],
    other_side_ids: set[str],
    corpus: dict,
    retriever_cfg: dict,
    passage_detail: dict | None,
    qid: str,
    key_prefix: str,
):
    for doc in docs:
        doc_id = doc["doc_id"]
        doc_data = corpus.get(doc_id, {})
        title = doc_data.get("title", "?")
        badges = ""
        if doc_id in expected_ids:
            badges += "  :white_check_mark: **attendu**"
        if doc_id not in other_side_ids:
            badges += "  :large_blue_circle: *absent de l'autre run*"
        label = f"Rang {doc['rank']} — **{title}** — score {doc['score']:.4f}{badges}"
        with st.expander(label, key=f"{key_prefix}_{doc_id}"):
            st.markdown(f"`{doc_id}`")
            render_doc_passages_or_text(
                doc_id,
                doc_data,
                retriever_cfg,
                passage_detail,
                qid,
                key=f"{key_prefix}_{doc_id}_text",
            )


def render_doc_passages_or_text(
    doc_id: str,
    doc_data: dict,
    retriever_cfg: dict,
    passage_detail: dict | None,
    qid: str,
    key: str,
    evidence_sentences: list[str] | None = None,
):
    """Texte d'un doc, annoté de ses passages si un fichier dérivé existe (EXE-112).

    `evidence_sentences` (critère 2, EXE-126) surligne les phrases-preuve,
    y compris dans l'unité passages — avant EXE-126, ce chemin ne recevait
    jamais ce paramètre.
    """
    _announce_unlocated_evidence_sentences(doc_data.get("text", ""), evidence_sentences)

    if retriever_cfg["unit"] != "passages":
        text = doc_data.get("text", "")
        if evidence_sentences:
            st.markdown(
                highlight_evidence_sentences(text, evidence_sentences),
                unsafe_allow_html=True,
            )
        else:
            st.write(text)
        return

    query_detail = find_query_passage_detail(passage_detail, qid)
    doc_passages = (query_detail or {}).get("docs", {}).get(doc_id, [])
    if not doc_passages:
        st.caption("passages non disponibles pour ce run")
        text = doc_data.get("text", "")
        if evidence_sentences:
            st.markdown(
                highlight_evidence_sentences(text, evidence_sentences),
                unsafe_allow_html=True,
            )
        else:
            st.write(text)
        return

    full_text = f"{doc_data.get('title', '')} {doc_data.get('text', '')}"
    render_doc_text_with_passages(
        full_text, doc_passages, key=key, evidence_sentences=evidence_sentences
    )


def render_doc_text_with_passages(
    text: str,
    doc_passages: list[dict],
    key: str,
    evidence_sentences: list[str] | None = None,
):
    """Texte complet d'un doc, avec un badge au début de chaque passage et le
    passage au meilleur score surligné sur toute sa portée (EXE-112, critères
    4, 5, 6). `char_start`/`char_end` sont relatifs à `text` (titre + texte,
    exactement ce que le run a découpé et indexé).

    `evidence_sentences` (critère 2, EXE-126) surligne en plus les
    phrases-preuve retrouvées mot pour mot dans `text` ; leur marqueur se
    distingue de celui du meilleur passage (critère 5)."""
    best = max(doc_passages, key=lambda p: p["score"])
    evidence_spans = _evidence_spans(text, evidence_sentences or [])
    cuts = sorted(
        {0, len(text), best["char_start"], best["char_end"]}
        | {p["char_start"] for p in doc_passages}
        | {pos for span in evidence_spans for pos in span}
    )
    badges_at_start: dict[int, list[dict]] = {}
    for p in doc_passages:
        badges_at_start.setdefault(p["char_start"], []).append(p)

    html_parts = []
    for i in range(len(cuts) - 1):
        start, end = cuts[i], cuts[i + 1]
        segment = text[start:end]
        prefix = ""
        for p in badges_at_start.get(start, []):
            is_best = p is best
            color = "#2ecc71" if is_best else "#7f8c8d"
            prefix += (
                f'<span style="background:{color}; color:#111; font-size:0.7em; '
                f"padding:1px 5px; border-radius:3px; margin-right:4px; "
                f'font-weight:bold">score {p["score"]:.4f}</span>'
            )
        if any(s <= start and end <= e for s, e in evidence_spans):
            segment = f'<mark style="background:#f1c40f; color:#111">{segment}</mark>'
        if start >= best["char_start"] and end <= best["char_end"]:
            segment_html = (
                '<span style="background:#1a2e1a; border-bottom:2px solid #2ecc71">'
                f"{segment}</span>"
            )
        else:
            segment_html = segment
        html_parts.append(prefix + segment_html)

    st.markdown(
        f'<div style="font-size:0.85em; line-height:1.7">{"".join(html_parts)}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# COMPOSANT : DÉTAIL D'UNE REQUÊTE
# ─────────────────────────────────────────────
def render_query_detail(q: dict, corpus: dict, retriever_cfg: dict):
    m = q["per_query_metrics"]
    rank = m["best_rank"]
    expected_ids = {d["doc_id"] for d in q["expected_docs"]}
    max_tokens = dashboard_max_tokens(retriever_cfg)

    # ── Query ──
    st.markdown("### Query")
    st.info(q["query_text"])

    # ── Doc(s) attendu(s) ──
    st.markdown(
        f"### Doc attendu  —  {'Rang ' + str(rank) if rank else 'Non trouvé dans le top 100'}"
    )
    for d in q["expected_docs"]:
        doc_data = corpus.get(d["doc_id"])
        if doc_data:
            badge = truncation_badge_markdown(retriever_cfg, d["token_count"])
            st.markdown(
                f"**{doc_data['title']}**  \n`{d['doc_id']}` — {d['token_count']} tokens  {badge}"
            )
            render_doc_text(
                doc_data["text"], key=f"expected_{d['doc_id']}", max_tokens=max_tokens
            )
        else:
            st.markdown(
                f"`{d['doc_id']}` ({d['token_count']} tok) — *texte non trouvé dans le corpus*"
            )

    # ── Top retrieved ──
    st.markdown("### Top 10 retrieved")
    for i, doc in enumerate(q["retrieved_top100"][:10]):
        doc_id = doc["doc_id"]
        is_relevant = doc_id in expected_ids
        doc_data = corpus.get(doc_id)
        title = doc_data["title"] if doc_data else "?"

        # Header de chaque doc
        if is_relevant:
            label = f":green[Rang {doc['rank']}] — **{title}** — score {doc['score']:.4f}  :white_check_mark:"
        else:
            label = f"Rang {doc['rank']} — **{title}** — score {doc['score']:.4f}"

        with st.expander(label, key=f"explore_ret_{q['query_id']}_{i}"):
            st.markdown(f"`{doc_id}`")
            if doc_data:
                render_doc_text(
                    doc_data["text"],
                    key=f"ret_{q['query_id']}_{doc_id}",
                    max_tokens=max_tokens,
                )


def render_doc_text(
    text: str,
    key: str,
    max_tokens: int | None = 256,
    evidence_sentences: list[str] | None = None,
):
    """Affiche le texte d'un doc avec marqueur visuel de troncature.

    `max_tokens=None` affiche le texte entier, sans coupure (EXE-111 : un run
    qui ne tronque pas à 256 tokens ne doit montrer aucune coupure).

    `evidence_sentences` (EXE-124, critère 11) surligne les phrases-preuve
    d'un document SUPPORT ou CONTRADICT ; appliqué après la coupure de
    troncature, pour qu'une phrase-preuve ne change jamais où cette coupure
    tombe. Le texte redécodé par le tokenizer (partie vue et partie perdue)
    diffère de l'original en casse et en espaces : la recherche mot pour mot
    les ignore des deux côtés dans ce cas précis (critère 3, EXE-126, H1b).
    """
    _announce_unlocated_evidence_sentences(text, evidence_sentences)
    if max_tokens is None:
        seen = text
        if evidence_sentences:
            seen = highlight_evidence_sentences(seen, evidence_sentences)
        st.markdown(
            f'<div style="background:#1a2e1a; border-left:4px solid #2ecc71; '
            f'padding:12px; border-radius:4px; font-size:0.85em; line-height:1.5">'
            f"{seen}</div>",
            unsafe_allow_html=True,
        )
        return
    seen, lost = split_at_truncation(text, max_tokens)
    if evidence_sentences:
        seen = highlight_evidence_sentences_after_decode(seen, evidence_sentences)
        if lost:
            lost = highlight_evidence_sentences_after_decode(lost, evidence_sentences)
    if not lost:
        # Doc complet — tout est vert
        st.markdown(
            f'<div style="background:#1a2e1a; border-left:4px solid #2ecc71; '
            f'padding:12px; border-radius:4px; font-size:0.85em; line-height:1.5">'
            f"{seen}</div>",
            unsafe_allow_html=True,
        )
    else:
        # Doc tronqué — partie vue en vert, partie ignorée en rouge
        st.markdown(
            f'<div style="font-size:0.85em; line-height:1.5; border-radius:4px; overflow:hidden">'
            # Partie vue
            f'<div style="background:#1a2e1a; border-left:4px solid #2ecc71; padding:12px">'
            f"{seen}"
            f"</div>"
            # Séparateur
            f'<div style="background:#4a3000; padding:4px 12px; font-size:0.8em; font-weight:bold; '
            f'border-left:4px solid #e67e22">'
            f"--- TRONCATURE (256 tokens) — ce qui suit n'a PAS ete lu par le modele ---"
            f"</div>"
            # Partie ignorée
            f'<div style="background:#2e1a1a; border-left:4px solid #e74c3c; padding:12px; opacity:0.7">'
            f"{lost}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# PAGE 5 : NIVEAUX DE LECTURE
# ─────────────────────────────────────────────
def reading_levels_table_frames(
    rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deux DataFrame alignées (mêmes lignes et colonnes, EXE-137, critère 3) :
    le texte affiché (`trouvé/effectif`) et la part retrouvée, qui pilote la
    couleur de fond sans être elle-même affichée."""
    columns = [reading_levels.GROUP_LABELS[name] for name in reading_levels.GROUP_NAMES]
    index = [row["run_name"] for row in rows]
    display_df = pd.DataFrame(
        {
            reading_levels.GROUP_LABELS[name]: [
                f"{row['counts'][name]['found']}/{row['counts'][name]['total']}"
                for row in rows
            ]
            for name in reading_levels.GROUP_NAMES
        },
        index=index,
        columns=columns,
    )
    rates_df = pd.DataFrame(
        {
            reading_levels.GROUP_LABELS[name]: [
                reading_levels.group_rate(row, name) for row in rows
            ]
            for name in reading_levels.GROUP_NAMES
        },
        index=index,
        columns=columns,
    )
    return display_df, rates_df


def render_reading_levels_table(rows: list[dict]) -> None:
    """Critère 3 : un tableau coloré, une ligne par run, une colonne par
    groupe — la couleur encode la part retrouvée, le texte le compte brut."""
    display_df, rates_df = reading_levels_table_frames(rows)
    styled = display_df.style.background_gradient(
        cmap="RdYlGn", gmap=rates_df.to_numpy(), vmin=0.0, vmax=1.0, axis=None
    )
    st.dataframe(styled, use_container_width=True, height=min(400, 40 + 35 * len(rows)))


def render_reading_levels_comparison_chart(
    rows: list[dict], run_names: list[str]
) -> None:
    """Critère 9 : diagramme en barres comparant les runs choisis, groupe par
    groupe, au seuil courant."""
    by_name = {row["run_name"]: row for row in rows}
    fig = go.Figure()
    for run_name in run_names:
        row = by_name.get(run_name)
        if row is None:
            continue
        fig.add_trace(
            go.Bar(
                name=run_name,
                x=[reading_levels.GROUP_LABELS[g] for g in reading_levels.GROUP_NAMES],
                y=[
                    reading_levels.group_rate(row, g)
                    for g in reading_levels.GROUP_NAMES
                ],
            )
        )
    fig.update_layout(
        barmode="group",
        title="Part retrouvée par groupe",
        yaxis_title="Part retrouvée",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_reranker_effect_chart(diff_rows: list[dict]) -> None:
    """Critère 10 : écart avec − sans reranker, en nombre de goldens, par
    groupe, pour chaque stratégie de base."""
    fig = go.Figure()
    for name in reading_levels.GROUP_NAMES:
        fig.add_trace(
            go.Bar(
                name=reading_levels.GROUP_LABELS[name],
                x=[row["base"] for row in diff_rows],
                y=[row["diffs"][name] for row in diff_rows],
            )
        )
    fig.update_layout(
        barmode="group",
        title="Effet du reranker (avec − sans), en nombre de goldens",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_reading_levels_group_detail(rows: list[dict]) -> None:
    """Critère 11 : une ligne par golden du groupe choisi — affirmation,
    document, rang dans le run, verdict du juge, phrase citée et raison."""
    for row in rows:
        rank_text = (
            f"rang {row['rank']}"
            if row["rank"] is not None
            else "hors des 100 premiers"
        )
        st.markdown(
            f"**[{row['query_id']}] {row['claim_text']}**  \n"
            f"`{row['doc_id']}` — {row['doc_title']}  \n"
            f"{rank_text} — verdict du juge : {row['verdict'] or '—'}"
        )
        if row["evidence"]:
            st.caption(f"« {row['evidence']} »")
        if row["reason"]:
            st.caption(row["reason"])
        st.divider()


def page_reading_levels(corpus: dict):
    st.header("Niveaux de lecture")
    st.caption(
        "Pour chaque stratégie de v2-grid, quels goldens elle retrouve selon "
        "le niveau de lecture donné par le juge Claude (campagne v3-juge)."
    )

    goldens = load_reading_levels_goldens()
    if goldens is None:
        st.info(
            "`results/etiquettes-origine.json` ou "
            "`results/v3-juge/jugements-claude.json` est absent : "
            "cette page n'affiche ni tableau ni diagramme."
        )
        return

    agreement = reading_levels.judge_agreement_with_annotators()
    agreement_text = f"{agreement:.2%}" if agreement is not None else "indisponible"
    st.caption(
        "Niveaux issus d'un seul juge (Claude) ; accord de ce juge avec les "
        f"étiquettes d'origine, sur les documents attendus : {agreement_text}. "
        "Groupes direct, vocabulaire et raisonnement : 55 à 71 goldens."
    )

    categorie_choice = st.radio(
        "Restreindre aux goldens",
        ["Tous", "confirme", "contredit"],
        horizontal=True,
        key="reading_levels_categorie",
    )
    filtered_goldens = (
        goldens
        if categorie_choice == "Tous"
        else reading_levels.filter_goldens_by_categorie(goldens, categorie_choice)
    )

    threshold_label = st.radio(
        "Seuil",
        reading_levels.THRESHOLD_LABELS,
        index=reading_levels.THRESHOLD_LABELS.index(
            reading_levels.DEFAULT_THRESHOLD_LABEL
        ),
        horizontal=True,
        key="reading_levels_threshold",
    )
    threshold = reading_levels.THRESHOLD_RANKS[threshold_label]

    runs = load_v2_grid_runs()
    rows = reading_levels.build_table_rows(filtered_goldens, runs, threshold)

    group_labels = [reading_levels.GROUP_LABELS[n] for n in reading_levels.GROUP_NAMES]
    sort_label = st.selectbox(
        "Trier par", group_labels, key="reading_levels_sort_column"
    )
    sort_group = next(
        n
        for n in reading_levels.GROUP_NAMES
        if reading_levels.GROUP_LABELS[n] == sort_label
    )
    rows = reading_levels.sort_rows_by_group(rows, sort_group)

    render_reading_levels_table(rows)

    st.divider()
    st.subheader("Comparer des runs, groupe par groupe")
    all_run_names = [row["run_name"] for row in rows]
    default_comparison = [
        name
        for name in reading_levels.DEFAULT_COMPARISON_RUN_NAMES
        if name in all_run_names
    ]
    compared_runs = st.multiselect(
        "Runs comparés (1 à 6)",
        all_run_names,
        default=default_comparison,
        max_selections=6,
        key="reading_levels_compared_runs",
    )
    if compared_runs:
        render_reading_levels_comparison_chart(rows, compared_runs)

    st.divider()
    st.subheader("Effet du reranker")
    diff_rows = reading_levels.reranker_diff_rows(filtered_goldens, runs, threshold)
    render_reranker_effect_chart(diff_rows)

    st.divider()
    st.subheader("Détail d'un groupe")
    col_run, col_group = st.columns(2)
    with col_run:
        detail_run_name = st.selectbox(
            "Run", all_run_names, key="reading_levels_detail_run"
        )
    with col_group:
        detail_group_label = st.selectbox(
            "Groupe", group_labels, key="reading_levels_detail_group"
        )
    detail_group = next(
        n
        for n in reading_levels.GROUP_NAMES
        if reading_levels.GROUP_LABELS[n] == detail_group_label
    )

    runs_by_name = {run["run_name"]: run for run in runs}
    detail_run = runs_by_name.get(detail_run_name)
    if detail_run is None:
        return
    detail_rows = reading_levels.build_group_detail_rows(
        filtered_goldens, detail_run, detail_group, corpus
    )
    detail_rows = reading_levels.sort_rows_beyond_threshold_first(
        detail_rows, threshold
    )
    render_reading_levels_group_detail(detail_rows)


if __name__ == "__main__":
    main()
