# src/ohnomitools/evolution.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
import polars as pl
from scipy.stats import fisher_exact, mannwhitneyu, norm

from .config import PATHS, ProjectPaths

logger = logging.getLogger("OhnomiTools.evolution")

# Cladi Ensembl più giovani di (o uguali a) Sarcopterygii:
# le SSD con LCA in questo insieme sono troppo recenti rispetto alle 1R/2R-WGD (~500 Mya)
YOUNG_SSD_CLADES: set[str] = {
    "Sarcopterygii",
    "Tetrapoda",
    "Amniota",
    "Mammalia",
    "Theria",
    "Eutheria",
    "Boreoeutheria",
    "Euarchontoglires",
    "Primates",
    "Haplorrhini",
    "Simiiformes",
    "Catarrhini",
    "Hominoidea",
    "Hominidae",
    "Homininae",
    "Homo sapiens",
}

# Cladi antichi coerenti con epoca >= WGD (più antichi di Sarcopterygii)
ANCIENT_SSD_CLADES: set[str] = {
    "Euteleostomi",
    "Gnathostomata",
    "Vertebrata",
    "Chordata",
    "Deuterostomia",
    "Bilateria",
    "Eumetazoa",
    "Metazoa",
    "Opisthokonta",
    "Eukaryota",
    "Cellular organisms",
}


@dataclass(frozen=True)
class Phase2Outputs:
    host_lncrna_map: Path
    wgd_intragenic_pairs: Path
    ssd_intragenic_pairs: Path
    lncrna_phylostratigraphy: Path
    enrichment_summary_csv: Path


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Intervallo di confidenza di Wilson al 95% per una proporzione binomiale k/n."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _fit_logistic_irls(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int = 35,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stima esattamente i coefficienti beta e gli errori standard (matrice di Fisher)
    di una regressione logistica binaria tramite IRLS (Iteratively Reweighted Least Squares).
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
        H = X.T @ (w[:, None] * X)
        grad = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(H + 1e-6 * np.eye(p), grad)
        beta += step
        if float(np.max(np.abs(step))) < tol:
            break

    eta = np.clip(X @ beta, -30.0, 30.0)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1.0 - mu), 1e-12, None)
    H = X.T @ (w[:, None] * X)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    return beta, se


class EvolutionAnalyzer:
    """
    Fase 2: Identificazione evolutiva degli Ohno-lncRNA tramite doppio approccio:
    - Ramo A (Intragenico / Modello del Passeggero): incrocio genomico su coppie host
      WGD vs SSD con controllo di conservazione dello strand relativo.
    - Ramo B (Filostratigrafia LncBook v2.0): profilazione dell'età evolutiva (`Age`)
      e isolamento della coorte al nodo `17. Euteleostomi` (inclusi lincRNA intergenici).
    - Intersezione A ∩ B: filtro filostratigrafico progressivo applicato al Ramo A
      con controllo di normalizzazione sulla lunghezza dei geni host.
    """

    def __init__(self, paths: ProjectPaths = PATHS) -> None:
        self.paths = paths

    def build_lncrna_phylostratigraphy(self, force_rebuild: bool = False) -> pl.DataFrame:
        """
        Estrae una riga univoca per gene lncRNA dal Parquet di conservazione LncBook v2.0
        (che contiene una riga per specie confrontata), salvando `Age`, `Symbol` e
        numero di specie in cui il lncRNA è conservato.
        """
        out_path = self.paths.silver / "lncbook_gene_phylostratigraphy.parquet"
        if out_path.exists() and not force_rebuild:
            return pl.read_parquet(out_path)

        lf_cons = pl.scan_parquet(self.paths.silver_lncbook_cons)
        cols = lf_cons.collect_schema().names()

        gene_col = "Gene ID" if "Gene ID" in cols else "gene_id"
        age_col = "Age" if "Age" in cols else "age"
        sym_col = "Symbol" if "Symbol" in cols else None
        pres_col = "#Present in Species" if "#Present in Species" in cols else None

        aggs: list[pl.Expr] = [
            pl.col(age_col).drop_nulls().first().alias("lncrna_age"),
        ]
        if sym_col:
            aggs.append(pl.col(sym_col).drop_nulls().first().alias("lncrna_symbol"))
        if pres_col:
            aggs.append(pl.col(pres_col).max().alias("n_species_present"))

        df_phylo = (
            lf_cons.group_by(gene_col)
            .agg(aggs)
            .rename({gene_col: "lncrna_gene_id"})
            .with_columns([
                pl.col("lncrna_age")
                .str.extract(r"^(\d+)\.", 1)
                .cast(pl.Int32)
                .alias("age_node_num"),
                pl.col("lncrna_age")
                .str.contains("Euteleostomi", literal=True)
                .fill_null(False)
                .alias("is_euteleostomi"),
            ])
            .sort("lncrna_gene_id")
            .collect()
        )

        self.paths.save_parquet_safe(df_phylo, out_path)
        return df_phylo

    def map_intragenic_lncrnas_to_hosts(
        self,
        strict_containment: bool = True,
        force_rebuild: bool = False,
    ) -> pl.DataFrame:
        """
        Esegue un'unica intersezione spaziale vettorizzata tra tutti i geni `protein_coding`
        di Ensembl (~19k) e tutti i geni lncRNA di LncBook v2.0 (~95k) per cromosoma.
        """
        out_path = self.paths.silver / "intragenic_host_lncrna_map.parquet"
        if out_path.exists() and not force_rebuild:
            return pl.read_parquet(out_path)

        df_pc = pl.read_parquet(self.paths.silver_ensembl_pc_genes).select([
            pl.col("gene_id").alias("host_gene_id"),
            pl.col("gene_name").alias("host_gene_name"),
            pl.col("chr"),
            pl.col("start").alias("host_start"),
            pl.col("end").alias("host_end"),
            pl.col("strand").alias("host_strand"),
        ])

        df_lnc = pl.read_parquet(self.paths.silver_lncrna_genes).select([
            pl.col("gene_id").alias("lncrna_gene_id"),
            pl.col("gene_alias").alias("lncrna_alias"),
            pl.col("chr"),
            pl.col("start").alias("lnc_start"),
            pl.col("end").alias("lnc_end"),
            pl.col("strand").alias("lnc_strand"),
            pl.col("n_isoforms"),
        ])

        df_phylo = self.build_lncrna_phylostratigraphy(force_rebuild=False)
        df_lnc = df_lnc.join(df_phylo, on="lncrna_gene_id", how="left")

        chunks: list[pl.DataFrame] = []
        for chrom in sorted(df_pc["chr"].unique().to_list()):
            pc_chr = df_pc.filter(pl.col("chr") == chrom)
            lnc_chr = df_lnc.filter(pl.col("chr") == chrom)
            if pc_chr.is_empty() or lnc_chr.is_empty():
                continue

            joined = pc_chr.join(lnc_chr, on="chr", how="inner")
            if strict_containment:
                matched = joined.filter(
                    (pl.col("lnc_start") >= pl.col("host_start"))
                    & (pl.col("lnc_end") <= pl.col("host_end"))
                )
            else:
                matched = joined.filter(
                    (pl.col("lnc_start") <= pl.col("host_end"))
                    & (pl.col("lnc_end") >= pl.col("host_start"))
                )

            if not matched.is_empty():
                chunks.append(matched)

        df_map = (
            pl.concat(chunks, how="vertical")
            .with_columns([
                pl.when(pl.col("host_strand") == pl.col("lnc_strand"))
                .then(pl.lit("sense"))
                .when(
                    pl.col("host_strand").is_not_null()
                    & pl.col("lnc_strand").is_not_null()
                )
                .then(pl.lit("antisense"))
                .otherwise(pl.lit("unknown"))
                .alias("rel_strand"),
                (pl.col("host_end") - pl.col("host_start")).alias("host_length_bp"),
                (pl.col("lnc_end") - pl.col("lnc_start")).alias("lnc_length_bp"),
            ])
            .sort(["chr", "host_start", "lnc_start"])
        )

        host_counts = df_map.group_by("host_gene_id").agg(
            pl.col("lncrna_gene_id").n_unique().alias("n_lncrnas_in_host")
        )
        df_map = df_map.join(host_counts, on="host_gene_id", how="left")

        self.paths.save_parquet_safe(df_map, out_path)
        logger.info(
            f"Mappatura intragenica completata: {len(df_map):,} relazioni host-lncRNA "
            f"({df_map['host_gene_id'].n_unique():,} geni host, "
            f"{df_map['lncrna_gene_id'].n_unique():,} geni lncRNA unici)."
        )
        return df_map

    def _pair_intragenic_lncrnas(
        self,
        df_host_pairs: pl.DataFrame,
        df_host_map: pl.DataFrame,
        duplication_type: str,
        require_conserved_rel_strand: bool = True,
        sense_only: bool = False,
    ) -> pl.DataFrame:
        """
        Proietta le coppie di geni codificanti duplicati (WGD o SSD) sulle coppie di
        lncRNA ospitati su entrambi i lati (`gene_id_1` e `gene_id_2`).
        """
        hm = df_host_map
        if sense_only:
            hm = hm.filter(pl.col("rel_strand") == "sense")

        map_1 = hm.select([
            pl.col("host_gene_id").alias("gene_id_1"),
            pl.col("lncrna_gene_id").alias("lncrna_1"),
            pl.col("lncrna_alias").alias("lncrna_alias_1"),
            pl.col("lnc_strand").alias("lnc_strand_1"),
            pl.col("rel_strand").alias("rel_strand_1"),
            pl.col("lncrna_age").alias("lncrna_age_1"),
            pl.col("age_node_num").alias("age_node_1"),
            pl.col("n_lncrnas_in_host").alias("n_lncrnas_host_1"),
        ])

        map_2 = hm.select([
            pl.col("host_gene_id").alias("gene_id_2"),
            pl.col("lncrna_gene_id").alias("lncrna_2"),
            pl.col("lncrna_alias").alias("lncrna_alias_2"),
            pl.col("lnc_strand").alias("lnc_strand_2"),
            pl.col("rel_strand").alias("rel_strand_2"),
            pl.col("lncrna_age").alias("lncrna_age_2"),
            pl.col("age_node_num").alias("age_node_2"),
            pl.col("n_lncrnas_in_host").alias("n_lncrnas_host_2"),
        ])

        df_pairs = (
            df_host_pairs.lazy()
            .join(map_1.lazy(), on="gene_id_1", how="inner")
            .join(map_2.lazy(), on="gene_id_2", how="inner")
            .filter(pl.col("lncrna_1") != pl.col("lncrna_2"))
            .with_columns([
                pl.lit(duplication_type).alias("duplication_type"),
                (pl.col("rel_strand_1") == pl.col("rel_strand_2")).alias("conserved_rel_strand"),
                (
                    (pl.col("n_lncrnas_host_1") == 1) & (pl.col("n_lncrnas_host_2") == 1)
                ).alias("is_strict_1to1_host"),
                pl.when(pl.col("lncrna_1") < pl.col("lncrna_2"))
                .then(pl.concat_str([pl.col("lncrna_1"), pl.lit("_"), pl.col("lncrna_2")]))
                .otherwise(pl.concat_str([pl.col("lncrna_2"), pl.lit("_"), pl.col("lncrna_1")]))
                .alias("lncrna_pair_key"),
            ])
        )

        if require_conserved_rel_strand:
            df_pairs = df_pairs.filter(pl.col("conserved_rel_strand"))

        return df_pairs.collect()

    def _standardize_ssd_clade_column(self, df_ssd: pl.DataFrame) -> pl.DataFrame:
        """
        Normalizza la colonna del Last Common Ancestor (LCA) negli SSD e aggiunge
        `is_ancient_ssd` (più antico di Sarcopterygii).
        """
        lca_candidates = [
            "ancestor_clade",
            "Paralogue last common ancestor with Human",
            "Human paralogue last common ancestor with Human",
        ]
        found_col = next((c for c in lca_candidates if c in df_ssd.columns), None)

        if found_col is None:
            return df_ssd.with_columns([
                pl.lit(None, dtype=pl.Utf8).alias("ancestor_clade"),
                pl.lit(True).alias("is_ancient_ssd"),
            ])

        if found_col != "ancestor_clade":
            df_ssd = df_ssd.rename({found_col: "ancestor_clade"})

        return df_ssd.with_columns(
            pl.when(pl.col("ancestor_clade").is_in(list(ANCIENT_SSD_CLADES)))
            .then(pl.lit(True))
            .when(pl.col("ancestor_clade").is_in(list(YOUNG_SSD_CLADES)))
            .then(pl.lit(False))
            .otherwise(pl.lit(None))
            .alias("is_ancient_ssd")
        )

    def compute_host_pair_enrichment(
        self,
        df_wgd_hosts: pl.DataFrame,
        df_ssd_hosts: pl.DataFrame,
        df_wgd_lnc_pairs: pl.DataFrame,
        df_ssd_lnc_pairs: pl.DataFrame,
        label_ssd: str = "SSD_all",
    ) -> dict[str, float | int | str]:
        """
        Test esatto di Fisher a livello di COPPIE DI HOST (ogni coppia di geni codificanti
        conta 1) per valutare se le coppie WGD ospitano coppie di lncRNA con orientamento
        coerente più spesso delle coppie SSD.
        """
        n_wgd_total = df_wgd_hosts["pair_key"].n_unique()
        n_ssd_total = df_ssd_hosts["pair_key"].n_unique()

        wgd_pos_keys = set(df_wgd_lnc_pairs["pair_key"].unique().to_list())
        ssd_host_keys = set(df_ssd_hosts["pair_key"].unique().to_list())
        ssd_pos_keys = set(df_ssd_lnc_pairs["pair_key"].unique().to_list()) & ssd_host_keys

        a = len(wgd_pos_keys)
        b = n_wgd_total - a
        c = len(ssd_pos_keys)
        d = n_ssd_total - c

        odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")

        return {
            "Confronto": f"WGD vs {label_ssd}",
            "WGD_host_pairs_totali": n_wgd_total,
            "WGD_con_coppia_lncRNA": a,
            "WGD_percentuale": round(100.0 * a / n_wgd_total, 3) if n_wgd_total else 0.0,
            "SSD_host_pairs_totali": n_ssd_total,
            "SSD_con_coppia_lncRNA": c,
            "SSD_percentuale": round(100.0 * c / n_ssd_total, 3) if n_ssd_total else 0.0,
            "Odds_Ratio": round(float(odds_ratio), 4),
            "Fisher_p_value": float(p_value),
        }

    def run_phase2(
        self,
        strict_containment: bool = True,
        require_conserved_rel_strand: bool = True,
        sense_only: bool = False,
        force_rebuild: bool = False,
    ) -> tuple[Phase2Outputs, pl.DataFrame]:
        """
        Esegue la Fase 2 base (Celle 2–4 del Notebook 02) e salva i Parquet in Silver
        e la tabella riassuntiva in Gold.
        """
        _ = self.build_lncrna_phylostratigraphy(force_rebuild=force_rebuild)
        df_host_map = self.map_intragenic_lncrnas_to_hosts(
            strict_containment=strict_containment,
            force_rebuild=force_rebuild,
        )

        df_wgd_hosts = pl.read_parquet(self.paths.silver_wgd_pairs)
        df_ssd_hosts = self._standardize_ssd_clade_column(
            pl.read_parquet(self.paths.silver_ssd_pairs)
        )

        df_wgd_lnc = self._pair_intragenic_lncrnas(
            df_wgd_hosts,
            df_host_map,
            duplication_type="WGD",
            require_conserved_rel_strand=require_conserved_rel_strand,
            sense_only=sense_only,
        )
        df_ssd_lnc = self._pair_intragenic_lncrnas(
            df_ssd_hosts,
            df_host_map,
            duplication_type="SSD",
            require_conserved_rel_strand=require_conserved_rel_strand,
            sense_only=sense_only,
        )

        out_wgd_lnc = self.paths.silver / "wgd_intragenic_lncrna_pairs.parquet"
        out_ssd_lnc = self.paths.silver / "ssd_intragenic_lncrna_pairs.parquet"
        self.paths.save_parquet_safe(df_wgd_lnc, out_wgd_lnc)
        self.paths.save_parquet_safe(df_ssd_lnc, out_ssd_lnc)

        df_ssd_ancient_hosts = df_ssd_hosts.filter(pl.col("is_ancient_ssd") == True)
        df_wgd_sense = df_wgd_lnc.filter(pl.col("rel_strand_1") == "sense")
        df_ssd_sense = df_ssd_lnc.filter(pl.col("rel_strand_1") == "sense")
        df_wgd_antisense = df_wgd_lnc.filter(pl.col("rel_strand_1") == "antisense")
        df_ssd_antisense = df_ssd_lnc.filter(pl.col("rel_strand_1") == "antisense")

        summary_rows = [
            self.compute_host_pair_enrichment(
                df_wgd_hosts, df_ssd_hosts, df_wgd_lnc, df_ssd_lnc, label_ssd="SSD_tutte (sense+antisense)"
            ),
            self.compute_host_pair_enrichment(
                df_wgd_hosts, df_ssd_ancient_hosts, df_wgd_lnc, df_ssd_lnc, label_ssd="SSD_antiche_preSarcopterygii (sense+antisense)"
            ),
            self.compute_host_pair_enrichment(
                df_wgd_hosts, df_ssd_ancient_hosts, df_wgd_sense, df_ssd_sense, label_ssd="SSD_antiche (solo SENSE)"
            ),
            self.compute_host_pair_enrichment(
                df_wgd_hosts, df_ssd_ancient_hosts, df_wgd_antisense, df_ssd_antisense, label_ssd="SSD_antiche (solo ANTISENSE)"
            ),
        ]
        df_summary = pl.DataFrame(summary_rows)

        out_summary = self.paths.gold / "phase2_wgd_vs_ssd_enrichment.csv"
        df_summary.write_csv(out_summary)
        logger.info(f"Tabella statistica Fase 2 salvata in Gold: {out_summary}")

        outputs = Phase2Outputs(
            host_lncrna_map=self.paths.silver / "intragenic_host_lncrna_map.parquet",
            wgd_intragenic_pairs=out_wgd_lnc,
            ssd_intragenic_pairs=out_ssd_lnc,
            lncrna_phylostratigraphy=self.paths.silver / "lncbook_gene_phylostratigraphy.parquet",
            enrichment_summary_csv=out_summary,
        )
        return outputs, df_summary

    # =========================================================================
    # NUOVI METODI: CONTROLLO LUNGHEZZA HOST & CURVA DI SENSIBILITÀ A ∩ B
    # =========================================================================

    def _prepare_host_pair_feature_table(
        self,
        use_ancient_ssd: bool = True,
    ) -> pl.DataFrame:
        """
        Costruisce una tabella con una riga per ciascuna coppia di geni host (WGD e SSD)
        completa di `len_1`, `len_2`, `min_len`, `max_len`, `geom_mean_len` e relativi
        log10 per la normalizzazione statistica.
        """
        df_wgd = pl.read_parquet(self.paths.silver_wgd_pairs).select([
            "pair_key", "start_1", "end_1", "start_2", "end_2"
        ]).unique(subset=["pair_key"]).with_columns(pl.lit(1, dtype=pl.Int8).alias("is_wgd"))

        df_ssd_raw = self._standardize_ssd_clade_column(
            pl.read_parquet(self.paths.silver_ssd_pairs)
        )
        if use_ancient_ssd:
            df_ssd_raw = df_ssd_raw.filter(pl.col("is_ancient_ssd") == True)

        df_ssd = df_ssd_raw.select([
            "pair_key", "start_1", "end_1", "start_2", "end_2"
        ]).unique(subset=["pair_key"]).with_columns(pl.lit(0, dtype=pl.Int8).alias("is_wgd"))

        return (
            pl.concat([df_wgd, df_ssd], how="vertical")
            .with_columns([
                (pl.col("end_1") - pl.col("start_1")).clip(lower_bound=1).cast(pl.Float64).alias("len_1"),
                (pl.col("end_2") - pl.col("start_2")).clip(lower_bound=1).cast(pl.Float64).alias("len_2"),
            ])
            .with_columns([
                pl.min_horizontal("len_1", "len_2").alias("min_len"),
                pl.max_horizontal("len_1", "len_2").alias("max_len"),
                (pl.col("len_1") * pl.col("len_2")).sqrt().alias("geom_mean_len"),
            ])
            .with_columns([
                pl.col("min_len").log10().alias("log10_min_len"),
                pl.col("max_len").log10().alias("log10_max_len"),
                pl.col("geom_mean_len").log10().alias("log10_geom_len"),
            ])
        )

    def _fit_length_adjusted_or(
        self,
        df_features: pl.DataFrame,
        positive_pair_keys: set[str],
    ) -> dict[str, float]:
        """
        Stima l'Odds Ratio aggiustato per lunghezza (OR_adj) tramite regressione logistica:
            logit(P(has_lnc_pair = 1)) = b0 + b_wgd * I_WGD + b1 * log10(min_len) + b2 * log10(max_len)
        """
        df_model = df_features.with_columns(
            pl.col("pair_key").is_in(list(positive_pair_keys)).cast(pl.Float64).alias("y")
        )
        y = df_model["y"].to_numpy()
        is_wgd = df_model["is_wgd"].to_numpy().astype(np.float64)

        wgd_pos = int(y[is_wgd == 1].sum())
        ssd_pos = int(y[is_wgd == 0].sum())
        if wgd_pos < 3 or ssd_pos < 3:
            return {
                "OR_adj_length": float("nan"),
                "OR_adj_CI95_low": float("nan"),
                "OR_adj_CI95_high": float("nan"),
                "Logistic_z": float("nan"),
                "Logistic_p_value": float("nan"),
            }

        lmin = df_model["log10_min_len"].to_numpy()
        lmax = df_model["log10_max_len"].to_numpy()
        # Standardizzazione dei predittori di lunghezza per stabilità numerica
        lmin_z = (lmin - lmin.mean()) / (lmin.std() + 1e-12)
        lmax_z = (lmax - lmax.mean()) / (lmax.std() + 1e-12)
        X = np.column_stack([np.ones_like(y), is_wgd, lmin_z, lmax_z])

        beta, se = _fit_logistic_irls(y, X)
        b_wgd = float(beta[1])
        se_wgd = float(se[1])
        z_stat = b_wgd / se_wgd
        p_val = float(2.0 * norm.sf(abs(z_stat)))

        return {
            "OR_adj_length": round(float(np.exp(b_wgd)), 4),
            "OR_adj_CI95_low": round(float(np.exp(b_wgd - 1.959964 * se_wgd)), 4),
            "OR_adj_CI95_high": round(float(np.exp(b_wgd + 1.959964 * se_wgd)), 4),
            "Logistic_z": round(z_stat, 3),
            "Logistic_p_value": p_val,
        }

    def analyze_host_length_confounding(
        self,
        use_ancient_ssd: bool = True,
        n_bins: int = 8,
    ) -> dict[str, pl.DataFrame | dict]:
        """
        Verifica in modo rigoroso se l'arricchimento intragenico WGD vs SSD è un artefatto
        della maggiore lunghezza dei geni ohnologhi oppure sopravvive a parità di lunghezza:
        1. Confronto distribuzione lunghezze WGD vs SSD (Mann-Whitney U).
        2. Analisi stratificata su `n_bins` quantili di lunghezza media geometrica (Mantel-Haenszel OR).
        3. Regressione logistica multivariata controllando `log10(min_len)` e `log10(max_len)`.
        """
        wgd_lnc_path = self.paths.silver / "wgd_intragenic_lncrna_pairs.parquet"
        ssd_lnc_path = self.paths.silver / "ssd_intragenic_lncrna_pairs.parquet"
        if not wgd_lnc_path.exists() or not ssd_lnc_path.exists():
            raise FileNotFoundError(
                "Esegui prima `analyzer.run_phase2()` per generare i Parquet intragenici in Silver."
            )

        df_feat = self._prepare_host_pair_feature_table(use_ancient_ssd=use_ancient_ssd)
        wgd_pos_keys = set(pl.read_parquet(wgd_lnc_path)["pair_key"].unique().to_list())
        ssd_pos_keys = set(pl.read_parquet(ssd_lnc_path)["pair_key"].unique().to_list())
        all_pos_keys = wgd_pos_keys | ssd_pos_keys

        df_eval = df_feat.with_columns(
            pl.col("pair_key").is_in(list(all_pos_keys)).cast(pl.Int8).alias("has_lnc_pair")
        )

        # 1. Confronto diretto delle lunghezze host (Mann-Whitney U)
        wgd_geom = df_eval.filter(pl.col("is_wgd") == 1)["geom_mean_len"].to_numpy()
        ssd_geom = df_eval.filter(pl.col("is_wgd") == 0)["geom_mean_len"].to_numpy()
        _, mw_p = mannwhitneyu(wgd_geom, ssd_geom, alternative="two-sided")

        # 2. Fisher grezzo
        a_tot = int(df_eval.filter((pl.col("is_wgd") == 1) & (pl.col("has_lnc_pair") == 1)).height)
        b_tot = int(df_eval.filter((pl.col("is_wgd") == 1) & (pl.col("has_lnc_pair") == 0)).height)
        c_tot = int(df_eval.filter((pl.col("is_wgd") == 0) & (pl.col("has_lnc_pair") == 1)).height)
        d_tot = int(df_eval.filter((pl.col("is_wgd") == 0) & (pl.col("has_lnc_pair") == 0)).height)
        or_crude, p_crude = fisher_exact([[a_tot, b_tot], [c_tot, d_tot]], alternative="two-sided")

        # 3. Stratificazione per quantili di lunghezza media geometrica + Mantel-Haenszel OR
        geom_all = df_eval["geom_mean_len"].to_numpy()
        edges = np.quantile(geom_all, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] -= 1.0
        edges[-1] += 1.0
        bin_idx = np.digitize(geom_all, edges[1:-1], right=True)

        df_binned = df_eval.with_columns(pl.Series("length_bin", bin_idx))
        bin_rows: list[dict] = []
        mh_num = 0.0
        mh_den = 0.0

        for b_id in range(n_bins):
            sub = df_binned.filter(pl.col("length_bin") == b_id)
            lo_kb = round(float(edges[b_id]) / 1000.0, 1)
            hi_kb = round(float(edges[b_id + 1]) / 1000.0, 1)
            med_kb = round(float(sub["geom_mean_len"].median()) / 1000.0, 1)

            a = int(sub.filter((pl.col("is_wgd") == 1) & (pl.col("has_lnc_pair") == 1)).height)
            b = int(sub.filter((pl.col("is_wgd") == 1) & (pl.col("has_lnc_pair") == 0)).height)
            c = int(sub.filter((pl.col("is_wgd") == 0) & (pl.col("has_lnc_pair") == 1)).height)
            d = int(sub.filter((pl.col("is_wgd") == 0) & (pl.col("has_lnc_pair") == 0)).height)
            n_w = a + b
            n_s = c + d
            n_tot = n_w + n_s

            if n_tot > 0:
                mh_num += (a * d) / n_tot
                mh_den += (b * c) / n_tot

            or_bin, p_bin = (
                fisher_exact([[a, b], [c, d]], alternative="two-sided")
                if (n_w > 0 and n_s > 0)
                else (float("nan"), float("nan"))
            )
            w_lo, w_hi = _wilson_ci(a, n_w)
            s_lo, s_hi = _wilson_ci(c, n_s)

            bin_rows.append({
                "Bin": b_id + 1,
                "Range_kb": f"{max(0.0, lo_kb):.1f}–{hi_kb:.1f} kb",
                "Median_kb": med_kb,
                "WGD_n": n_w,
                "WGD_pos": a,
                "WGD_pct": round(100.0 * a / n_w, 2) if n_w else 0.0,
                "WGD_ci_low": round(100.0 * w_lo, 2),
                "WGD_ci_high": round(100.0 * w_hi, 2),
                "SSD_n": n_s,
                "SSD_pos": c,
                "SSD_pct": round(100.0 * c / n_s, 2) if n_s else 0.0,
                "SSD_ci_low": round(100.0 * s_lo, 2),
                "SSD_ci_high": round(100.0 * s_hi, 2),
                "Bin_OR": round(float(or_bin), 3),
                "Bin_p_value": float(p_bin),
            })

        or_mh = round(mh_num / mh_den, 4) if mh_den > 0 else float("nan")

        # 4. Regressione logistica multivariata continua
        log_res = self._fit_length_adjusted_or(df_feat, all_pos_keys)

        df_bins = pl.DataFrame(bin_rows)
        summary_dict = {
            "SSD_control_set": "SSD_antiche_preSarcopterygii" if use_ancient_ssd else "SSD_tutte",
            "WGD_median_geom_len_kb": round(float(np.median(wgd_geom)) / 1000.0, 2),
            "SSD_median_geom_len_kb": round(float(np.median(ssd_geom)) / 1000.0, 2),
            "Length_ratio_WGD_over_SSD": round(float(np.median(wgd_geom) / np.median(ssd_geom)), 3),
            "MannWhitney_length_p": float(mw_p),
            "OR_crude_Fisher": round(float(or_crude), 4),
            "Fisher_crude_p": float(p_crude),
            "OR_MantelHaenszel_stratified": or_mh,
            **log_res,
        }
        df_summary = pl.DataFrame([summary_dict])

        df_bins.write_csv(self.paths.gold / "phase2_length_control_bins.csv")
        df_summary.write_csv(self.paths.gold / "phase2_length_control_summary.csv")
        logger.info(
            f"Controllo lunghezza completato: OR grezzo={summary_dict['OR_crude_Fisher']} | "
            f"OR Mantel-Haenszel={or_mh} | OR logistico aggiustato={log_res['OR_adj_length']} "
            f"(p={log_res['Logistic_p_value']:.2e})"
        )

        return {"summary": df_summary, "bins": df_bins, "metrics": summary_dict}

    def run_ab_sensitivity_curve(
        self,
        use_ancient_ssd: bool = True,
        export_three_lanes: bool = True,
    ) -> pl.DataFrame:
        """
        Calcola la curva di sensibilità dell'Intersezione A ∩ B lungo soglie
        filostratigrafiche crescenti (da Tutte le età fino a 17. Euteleostomi),
        affiancando Odds Ratio grezzo (Fisher) e Odds Ratio aggiustato per lunghezza.
        Se `export_three_lanes=True`, salva in Silver le 3 coorti ufficiali per la Fase 3.
        """
        wgd_lnc_path = self.paths.silver / "wgd_intragenic_lncrna_pairs.parquet"
        ssd_lnc_path = self.paths.silver / "ssd_intragenic_lncrna_pairs.parquet"
        df_wgd_lnc = pl.read_parquet(wgd_lnc_path)
        df_ssd_lnc = pl.read_parquet(ssd_lnc_path)
        df_feat = self._prepare_host_pair_feature_table(use_ancient_ssd=use_ancient_ssd)

        ssd_allowed_keys = set(df_feat.filter(pl.col("is_wgd") == 0)["pair_key"].to_list())
        df_ssd_lnc = df_ssd_lnc.filter(pl.col("pair_key").is_in(list(ssd_allowed_keys)))

        n_wgd_tot = int(df_feat.filter(pl.col("is_wgd") == 1).height)
        n_ssd_tot = int(df_feat.filter(pl.col("is_wgd") == 0).height)

        thresholds: list[tuple[str, pl.Expr]] = [
            ("1. Ramo A puro (Tutte le età)", pl.lit(True)),
            ("2. A ∩ B (Entrambi >= 14. Mammalia)", (pl.col("age_node_1") >= 14) & (pl.col("age_node_2") >= 14)),
            ("3. A ∩ B (Entrambi >= 15. Amniota)", (pl.col("age_node_1") >= 15) & (pl.col("age_node_2") >= 15)),
            ("4. A ∩ B (Entrambi >= 16. Tetrapoda)", (pl.col("age_node_1") >= 16) & (pl.col("age_node_2") >= 16)),
            (
                "5. A ∩ B (Asimm.: 17. Eut + >=15. Amn)",
                ((pl.col("age_node_1") == 17) & (pl.col("age_node_2") >= 15))
                | ((pl.col("age_node_2") == 17) & (pl.col("age_node_1") >= 15)),
            ),
            ("6. A ∩ B Strict (Entrambi = 17. Euteleostomi)", (pl.col("age_node_1") == 17) & (pl.col("age_node_2") == 17)),
        ]

        rows: list[dict] = []
        for label, expr in thresholds:
            w_sub = df_wgd_lnc.filter(expr)
            s_sub = df_ssd_lnc.filter(expr)

            w_keys = set(w_sub["pair_key"].unique().to_list())
            s_keys = set(s_sub["pair_key"].unique().to_list())

            a = len(w_keys)
            b = n_wgd_tot - a
            c = len(s_keys)
            d = n_ssd_tot - c

            or_crude, p_crude = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            adj = self._fit_length_adjusted_or(df_feat, w_keys | s_keys)

            rows.append({
                "Soglia_A_cap_B": label,
                "WGD_host_pos": a,
                "WGD_pct": round(100.0 * a / n_wgd_tot, 3),
                "WGD_lnc_pairs": len(w_sub),
                "SSD_host_pos": c,
                "SSD_pct": round(100.0 * c / n_ssd_tot, 3),
                "SSD_lnc_pairs": len(s_sub),
                "OR_crude": round(float(or_crude), 4),
                "Fisher_p": float(p_crude),
                "OR_adj_length": adj["OR_adj_length"],
                "OR_adj_CI95_low": adj["OR_adj_CI95_low"],
                "OR_adj_CI95_high": adj["OR_adj_CI95_high"],
                "Logistic_p": adj["Logistic_p_value"],
            })

        df_curve = pl.DataFrame(rows)
        df_curve.write_csv(self.paths.gold / "phase2_ab_sensitivity_curve.csv")

        if export_three_lanes:
            # Via 1 (Ramo A puro): già in silver/wgd_intragenic_lncrna_pairs.parquet
            # Via 2 (Intersezione A ∩ B Gold Standard): entrambi >= 15. Amniota con flag per 16/17
            df_via2_wgd = df_wgd_lnc.filter(
                (pl.col("age_node_1") >= 15) & (pl.col("age_node_2") >= 15)
            ).with_columns([
                ((pl.col("age_node_1") >= 16) & (pl.col("age_node_2") >= 16)).alias("is_tier_tetrapoda"),
                ((pl.col("age_node_1") == 17) & (pl.col("age_node_2") == 17)).alias("is_tier_euteleostomi_strict"),
            ])
            df_via2_ssd = df_ssd_lnc.filter(
                (pl.col("age_node_1") >= 15) & (pl.col("age_node_2") >= 15)
            ).with_columns([
                ((pl.col("age_node_1") >= 16) & (pl.col("age_node_2") >= 16)).alias("is_tier_tetrapoda"),
                ((pl.col("age_node_1") == 17) & (pl.col("age_node_2") == 17)).alias("is_tier_euteleostomi_strict"),
            ])
            self.paths.save_parquet_safe(df_via2_wgd, self.paths.silver / "lane2_ab_wgd_pairs.parquet")
            self.paths.save_parquet_safe(df_via2_ssd, self.paths.silver / "lane2_ab_ssd_pairs.parquet")

            # Via 3 (Ramo B puro): coorte 17. Euteleostomi con etichetta intragenico vs intergenico
            df_phylo = self.build_lncrna_phylostratigraphy(force_rebuild=False)
            df_hmap = pl.read_parquet(self.paths.silver / "intragenic_host_lncrna_map.parquet")
            intra_ids = set(df_hmap["lncrna_gene_id"].unique().to_list())
            wgd_host_ids = set(pl.read_parquet(self.paths.silver_wgd_genes)["gene_id"].unique().to_list())
            wgd_intra_ids = set(
                df_hmap.filter(pl.col("host_gene_id").is_in(list(wgd_host_ids)))["lncrna_gene_id"]
                .unique()
                .to_list()
            )

            df_via3 = (
                df_phylo.filter(pl.col("is_euteleostomi"))
                .with_columns([
                    pl.col("lncrna_gene_id").is_in(list(intra_ids)).alias("is_intragenic"),
                    pl.col("lncrna_gene_id").is_in(list(wgd_intra_ids)).alias("is_in_wgd_host"),
                ])
                .with_columns(
                    pl.when(pl.col("is_in_wgd_host"))
                    .then(pl.lit("intragenic_WGD_host"))
                    .when(pl.col("is_intragenic"))
                    .then(pl.lit("intragenic_other_host"))
                    .otherwise(pl.lit("intergenic_lincRNA"))
                    .alias("genomic_context")
                )
            )
            self.paths.save_parquet_safe(df_via3, self.paths.silver / "lane3_ramoB_euteleostomi_cohort.parquet")
            logger.info("Esportate in Silver le coorti ufficiali delle 3 vie per la Fase 3.")

        return df_curve

    def plot_phase2_diagnostics(
        self,
        length_res: dict,
        df_ab_curve: pl.DataFrame,
        save_fig: bool = True,
    ) -> tuple[Path | None, Path | None]:
        """
        Genera la figura diagnostica a 2 pannelli per la Fase 2:
        - Pannello A: Probabilità di ospitare coppie lncRNA (WGD vs SSD antiche)
          stratificata per bin di lunghezza media geometrica dei geni host (con CI 95%).
        - Pannello B: Confronto Odds Ratio grezzo vs aggiustato per lunghezza lungo
          la curva filostratigrafica dell'Intersezione A ∩ B.
        """
        import matplotlib.pyplot as plt

        df_bins: pl.DataFrame = length_res["bins"]
        metrics: dict = length_res["metrics"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), dpi=140)
        fig.patch.set_facecolor("white")

        # --- Pannello A: Stratificazione per Lunghezza Host ---
        x_bins = np.arange(len(df_bins))
        w_pct = df_bins["WGD_pct"].to_numpy()
        w_err = np.vstack([
            np.clip(w_pct - df_bins["WGD_ci_low"].to_numpy(), 0, None),
            np.clip(df_bins["WGD_ci_high"].to_numpy() - w_pct, 0, None),
        ])
        s_pct = df_bins["SSD_pct"].to_numpy()
        s_err = np.vstack([
            np.clip(s_pct - df_bins["SSD_ci_low"].to_numpy(), 0, None),
            np.clip(df_bins["SSD_ci_high"].to_numpy() - s_pct, 0, None),
        ])

        ax1.errorbar(
            x_bins - 0.08, w_pct, yerr=w_err, fmt="o-", color="#D62728",
            linewidth=2.2, markersize=6, capsize=4, label="Coppie Host WGD (Ohnologhi)"
        )
        ax1.errorbar(
            x_bins + 0.08, s_pct, yerr=s_err, fmt="s--", color="#1F77B4",
            linewidth=2.0, markersize=6, capsize=4, label="Coppie Host SSD Antiche (>Sarcopterygii)"
        )
        ax1.set_xticks(x_bins)
        ax1.set_xticklabels(df_bins["Range_kb"].to_list(), rotation=35, ha="right", fontsize=8.5)
        ax1.set_xlabel("Lunghezza media geometrica della coppia host $\\sqrt{L_1 L_2}$ (quantili)", fontsize=10)
        ax1.set_ylabel("Coppie host con coppia lncRNA coerente (%)", fontsize=10)
        ax1.set_title(
            f"A) Controllo Bias di Lunghezza Host\n"
            f"OR grezzo = {metrics['OR_crude_Fisher']:.2f}  |  "
            f"OR Mantel-Haenszel = {metrics['OR_MantelHaenszel_stratified']:.2f}  |  "
            f"OR adj (Logit) = {metrics['OR_adj_length']:.2f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(frameon=True, fontsize=9, loc="upper left")

        # --- Pannello B: Curva di Sensibilità A ∩ B ---
        y_pos = np.arange(len(df_ab_curve))
        short_labels = [
            "Ramo A (Tutte)",
            "A ∩ B (≥14 Mammalia)",
            "A ∩ B (≥15 Amniota)",
            "A ∩ B (≥16 Tetrapoda)",
            "A ∩ B (17 + ≥15)",
            "A ∩ B (=17 Eut. strict)",
        ]
        labels = short_labels[:len(df_ab_curve)]

        or_crude = df_ab_curve["OR_crude"].to_numpy()
        or_adj = df_ab_curve["OR_adj_length"].to_numpy()
        ci_lo = df_ab_curve["OR_adj_CI95_low"].to_numpy()
        ci_hi = df_ab_curve["OR_adj_CI95_high"].to_numpy()

        ax2.axvline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Nessun arricchimento (OR = 1)")
        ax2.scatter(
            or_crude, y_pos - 0.14, color="#FF7F0E", marker="D", s=55,
            zorder=4, label="OR grezzo (Fisher)"
        )

        valid = np.isfinite(or_adj)
        if np.any(valid):
            adj_err = np.vstack([
                np.clip(or_adj[valid] - ci_lo[valid], 0, None),
                np.clip(ci_hi[valid] - or_adj[valid], 0, None),
            ])
            ax2.errorbar(
                or_adj[valid], y_pos[valid] + 0.14, xerr=adj_err,
                fmt="o", color="#2CA02C", markersize=7, capsize=4, linewidth=2.0,
                zorder=5, label="OR aggiustato per lunghezza (CI 95%)"
            )

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Odds Ratio (WGD vs SSD Antiche)", fontsize=10)
        ax2.set_title(
            "B) Curva di Sensibilità Intersezione A ∩ B\n"
            "(Effetto del filtro filostratigrafico LncBook v2.0)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax2.grid(True, axis="x", linestyle=":", alpha=0.5)
        ax2.legend(frameon=True, fontsize=8.5, loc="lower right")

        plt.tight_layout()

        png_path = pdf_path = None
        if save_fig:
            png_path = self.paths.gold / "phase2_length_control_and_ab_curve.png"
            pdf_path = self.paths.gold / "phase2_length_control_and_ab_curve.pdf"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            logger.info(f"Grafici diagnostici Fase 2 salvati in Gold: {png_path.name}, {pdf_path.name}")

        plt.show()
        return png_path, pdf_path
