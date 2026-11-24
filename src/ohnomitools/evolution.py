# src/ohnomitools/evolution.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
import polars as pl
import requests
from scipy.stats import chi2, fisher_exact, mannwhitneyu, norm

from .config import PATHS, ProjectPaths

logger = logging.getLogger("OhnomiTools.evolution")

Z_95 = 1.959963984540054

MAKINO_SINGH_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/fmottes/wgd-network-motifs/main/Paralogues/MakinoSinghStrict/HUMAN.Pairs.WGD.Protcod.Strict.tsv",
    "https://raw.githubusercontent.com/fmottes/wgd-network-motifs/master/Paralogues/MakinoSinghStrict/HUMAN.Pairs.WGD.Protcod.Strict.tsv",
)

# Cladi Ensembl <= Sarcopterygii (esclusi dalle SSD antiche)
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

# Cladi strettamente più antichi di Sarcopterygii
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


def _wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """CI di Wilson al 95% per una proporzione k/n."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _compute_mantel_haenszel(
    strata_tables: list[tuple[int, int, int, int]],
    z: float = Z_95,
) -> dict[str, float]:
    """
    Calcola OR_MH, CI 95% di Robins-Breslow-Greenland e chi2_MH con correzione di continuità
    su una lista di tabelle 2x2 per strato: (a, b, c, d).
    """
    R = S = 0.0
    sum_pr = sum_ps_qr = sum_qs = 0.0
    sum_diff = sum_var = 0.0

    for a, b, c, d in strata_tables:
        n1 = a + b
        n0 = c + d
        m1 = a + c
        m0 = b + d
        N = n1 + n0
        if N <= 0:
            continue

        r_s = (a * d) / N
        s_s = (b * c) / N
        p_s = (a + d) / N
        q_s = (b + c) / N

        R += r_s
        S += s_s
        sum_pr += p_s * r_s
        sum_ps_qr += p_s * s_s + q_s * r_s
        sum_qs += q_s * s_s

        if N > 1:
            sum_diff += a - (n1 * m1) / N
            sum_var += (n1 * n0 * m1 * m0) / (N * N * (N - 1))

    if R > 0.0 and S > 0.0:
        or_mh = R / S
        var_log_or = (
            sum_pr / (2.0 * R * R)
            + sum_ps_qr / (2.0 * R * S)
            + sum_qs / (2.0 * S * S)
        )
        se_log_or = float(np.sqrt(max(0.0, var_log_or)))
        log_or = float(np.log(or_mh))
        ci_lo = float(np.exp(log_or - z * se_log_or))
        ci_hi = float(np.exp(log_or + z * se_log_or))
    else:
        or_mh = ci_lo = ci_hi = float("nan")

    if sum_var > 0.0:
        num = max(0.0, abs(sum_diff) - 0.5)
        chi2_mh = (num * num) / sum_var
        p_mh = float(chi2.sf(chi2_mh, df=1))
    else:
        chi2_mh = p_mh = float("nan")

    return {
        "OR_MantelHaenszel_stratified": round(float(or_mh), 4) if np.isfinite(or_mh) else float("nan"),
        "OR_MH_CI95_low": round(ci_lo, 4) if np.isfinite(ci_lo) else float("nan"),
        "OR_MH_CI95_high": round(ci_hi, 4) if np.isfinite(ci_hi) else float("nan"),
        "MH_chi2": round(float(chi2_mh), 4) if np.isfinite(chi2_mh) else float("nan"),
        "MH_p_value": p_mh,
    }


def _fit_logistic_irls(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Regressione logistica binaria tramite IRLS; restituisce (beta, se)."""
    _, p = X.shape
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
    """Fase 2: Ramo A intragenico, Ramo B filostratigrafico, controllo lunghezza e curva A ∩ B."""

    def __init__(self, paths: ProjectPaths = PATHS) -> None:
        self.paths = paths

    def _get_sprotein_lncrna_ids(self, exclude_sprotein: str | None) -> set[str]:
        """Restituisce gli HSALNG da escludere per purezza non codificante ('all' | 'ms_only')."""
        if exclude_sprotein is None:
            return set()
        if exclude_sprotein not in {"all", "ms_only"}:
            raise ValueError("`exclude_sprotein` deve essere None, 'all' oppure 'ms_only'.")
        if not self.paths.silver_lncbook_sprotein.exists():
            raise FileNotFoundError(f"File SmProt assente: {self.paths.silver_lncbook_sprotein}")

        df_sp = pl.read_parquet(self.paths.silver_lncbook_sprotein).with_columns(
            pl.col("Gene ID").cast(pl.Utf8).str.strip_chars().alias("lncrna_gene_id")
        )
        if exclude_sprotein == "ms_only":
            df_sp = df_sp.filter(
                pl.col("Experimental Evidence")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .str.contains("mass")
                .fill_null(False)
            )
        return set(df_sp["lncrna_gene_id"].unique().to_list())

    def load_makino_singh_strict_hosts(
        self,
        raw_path: Path | None = None,
        download_if_missing: bool = True,
    ) -> dict[str, pl.DataFrame | dict]:
        """
        Carica (o scarica in bronze/) `MakinoSinghStrict` da fmottes/wgd-network-motifs,
        mappa i simboli HGNC sui geni Ensembl protein-coding canonici e costruisce:
        - `wgd_makino`: coppie host WGD MakinoSinghStrict in formato canonico ENSG
        - `ssd_decontaminated`: `silver_ssd_pairs` privato dell'overlap con MakinoSinghStrict
        - `stats`: conteggi di overlap e decontaminazione
        """
        tsv_path = raw_path or (self.paths.bronze / "makino_singh_strict_wgd_pairs.tsv")
        if not tsv_path.exists():
            if not download_if_missing:
                raise FileNotFoundError(f"File MakinoSinghStrict non trovato: {tsv_path}")
            downloaded = False
            last_err = None
            for url in MAKINO_SINGH_URLS:
                try:
                    resp = requests.get(url, timeout=30)
                    if resp.status_code == 200 and len(resp.text) > 1000:
                        tsv_path.parent.mkdir(parents=True, exist_ok=True)
                        tsv_path.write_text(resp.text, encoding="utf-8")
                        downloaded = True
                        logger.info(f"Scaricato MakinoSinghStrict in {tsv_path.name}")
                        break
                except Exception as exc:
                    last_err = exc
            if not downloaded:
                raise RuntimeError(f"Download MakinoSinghStrict fallito: {last_err}")

        df_ms_raw = (
            pl.read_csv(tsv_path, separator="\t", has_header=False, new_columns=["sym_1", "sym_2"])
            .with_columns([
                pl.col("sym_1").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
                pl.col("sym_2").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
            ])
            .filter(
                (pl.col("sym_1") != "OHNOLOG1")
                & (pl.col("sym_2") != "OHNOLOG2")
                & pl.col("sym_1").is_not_null()
                & pl.col("sym_2").is_not_null()
                & (pl.col("sym_1") != pl.col("sym_2"))
            )
            .with_columns(
                pl.when(pl.col("sym_1") < pl.col("sym_2"))
                .then(pl.concat_str([pl.col("sym_1"), pl.lit("_"), pl.col("sym_2")]))
                .otherwise(pl.concat_str([pl.col("sym_2"), pl.lit("_"), pl.col("sym_1")]))
                .alias("sym_pair_key")
            )
            .unique(subset=["sym_pair_key"])
        )

        # Mappa deterministica HGNC -> ENSG protein_coding canonico
        df_pc = (
            pl.read_parquet(self.paths.silver_ensembl_pc_genes)
            .with_columns([
                pl.col("gene_name").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("_sym"),
                (pl.col("end") - pl.col("start") + 1).alias("_len"),
            ])
            .filter(pl.col("_sym").is_not_null() & (pl.col("_sym") != ""))
            .sort(["_len", "gene_id"], descending=[True, False])
            .unique(subset=["_sym"], keep="first")
        )

        pc1 = df_pc.select([
            pl.col("_sym").alias("sym_1"),
            pl.col("gene_id").alias("_g1"),
            pl.col("gene_name").alias("_n1"),
            pl.col("chr").alias("_c1"),
            pl.col("start").alias("_s1"),
            pl.col("end").alias("_e1"),
            pl.col("strand").alias("_st1"),
        ])
        pc2 = df_pc.select([
            pl.col("_sym").alias("sym_2"),
            pl.col("gene_id").alias("_g2"),
            pl.col("gene_name").alias("_n2"),
            pl.col("chr").alias("_c2"),
            pl.col("start").alias("_s2"),
            pl.col("end").alias("_e2"),
            pl.col("strand").alias("_st2"),
        ])

        swap = pl.col("_g1") > pl.col("_g2")
        df_wgd_ms = (
            df_ms_raw.join(pc1, on="sym_1", how="inner")
            .join(pc2, on="sym_2", how="inner")
            .filter(pl.col("_g1") != pl.col("_g2"))
            .select([
                pl.when(swap).then(pl.col("_g2")).otherwise(pl.col("_g1")).alias("gene_id_1"),
                pl.when(swap).then(pl.col("_g1")).otherwise(pl.col("_g2")).alias("gene_id_2"),
                pl.when(swap).then(pl.col("_n2")).otherwise(pl.col("_n1")).alias("gene_name_1"),
                pl.when(swap).then(pl.col("_n1")).otherwise(pl.col("_n2")).alias("gene_name_2"),
                pl.when(swap).then(pl.col("_c2")).otherwise(pl.col("_c1")).alias("chr_1"),
                pl.when(swap).then(pl.col("_s2")).otherwise(pl.col("_s1")).alias("start_1"),
                pl.when(swap).then(pl.col("_e2")).otherwise(pl.col("_e1")).alias("end_1"),
                pl.when(swap).then(pl.col("_st2")).otherwise(pl.col("_st1")).alias("strand_1"),
                pl.when(swap).then(pl.col("_c1")).otherwise(pl.col("_c2")).alias("chr_2"),
                pl.when(swap).then(pl.col("_s1")).otherwise(pl.col("_s2")).alias("start_2"),
                pl.when(swap).then(pl.col("_e1")).otherwise(pl.col("_e2")).alias("end_2"),
                pl.when(swap).then(pl.col("_st1")).otherwise(pl.col("_st2")).alias("strand_2"),
            ])
            .with_columns(
                pl.concat_str([pl.col("gene_id_1"), pl.lit("_"), pl.col("gene_id_2")]).alias("pair_key")
            )
            .unique(subset=["pair_key"])
        )

        # Decontaminazione del controllo SSD (match sia su pair_key ENSG sia su simboli HGNC)
        df_ssd_raw = self._standardize_ssd_clade_column(pl.read_parquet(self.paths.silver_ssd_pairs))
        ms_ensg_keys = set(df_wgd_ms["pair_key"].to_list())
        ms_sym_keys = set(df_ms_raw["sym_pair_key"].to_list())

        if "gene_name_1" in df_ssd_raw.columns and "gene_name_2" in df_ssd_raw.columns:
            s1 = pl.col("gene_name_1").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
            s2 = pl.col("gene_name_2").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
            df_ssd_tagged = df_ssd_raw.with_columns(
                pl.when(s1 < s2)
                .then(pl.concat_str([s1, pl.lit("_"), s2]))
                .otherwise(pl.concat_str([s2, pl.lit("_"), s1]))
                .alias("_ssd_sym_key")
            )
            df_ssd_clean = df_ssd_tagged.filter(
                ~pl.col("pair_key").is_in(list(ms_ensg_keys))
                & ~pl.col("_ssd_sym_key").fill_null("").is_in(list(ms_sym_keys))
            ).drop("_ssd_sym_key")
        else:
            df_ssd_clean = df_ssd_raw.filter(~pl.col("pair_key").is_in(list(ms_ensg_keys)))

        ohno_keys = set(pl.read_parquet(self.paths.silver_wgd_pairs)["pair_key"].to_list())
        stats = {
            "makino_raw_pairs": len(df_ms_raw),
            "makino_mapped_ensg_pairs": len(df_wgd_ms),
            "overlap_with_ohnologs_com_ensg": len(ms_ensg_keys & ohno_keys),
            "ssd_raw_pairs": len(df_ssd_raw),
            "ssd_removed_makino_overlap": len(df_ssd_raw) - len(df_ssd_clean),
            "ssd_decontaminated_pairs": len(df_ssd_clean),
            "ssd_decontaminated_ancient_pairs": int(df_ssd_clean.filter(pl.col("is_ancient_ssd") == True).height),
        }
        logger.info(
            f"MakinoSinghStrict mappato: {stats['makino_mapped_ensg_pairs']:,}/{stats['makino_raw_pairs']:,} coppie ENSG | "
            f"SSD decontaminato: rimosse {stats['ssd_removed_makino_overlap']:,} coppie."
        )
        return {
            "wgd_makino": df_wgd_ms,
            "ssd_decontaminated": df_ssd_clean,
            "stats": stats,
        }

    def build_lncrna_phylostratigraphy(self, force_rebuild: bool = False) -> pl.DataFrame:
        """Estrae Age, Symbol e #Present in Species a livello di gene lncRNA univoco."""
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
        """Intersezione spaziale vettorizzata per cromosoma tra geni protein_coding e lncRNA."""
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
                (pl.col("host_end") - pl.col("host_start") + 1).alias("host_length_bp"),
                (pl.col("lnc_end") - pl.col("lnc_start") + 1).alias("lnc_length_bp"),
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
        exclude_lncrna_ids: set[str] | None = None,
    ) -> pl.DataFrame:
        """Proietta le coppie di geni host duplicati sulle coppie di lncRNA ospitati."""
        hm = df_host_map
        if sense_only:
            hm = hm.filter(pl.col("rel_strand") == "sense")
        if exclude_lncrna_ids:
            hm = hm.filter(~pl.col("lncrna_gene_id").is_in(list(exclude_lncrna_ids)))

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
        """Normalizza la colonna LCA e aggiunge `is_ancient_ssd` (> Sarcopterygii)."""
        if "is_ancient_ssd" in df_ssd.columns:
            return df_ssd

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

    def _ensure_host_coords(self, df_hosts: pl.DataFrame) -> pl.DataFrame:
        """Garantisce la presenza di start_1/end_1/start_2/end_2 su un DataFrame host."""
        needed = {"start_1", "end_1", "start_2", "end_2"}
        if needed.issubset(set(df_hosts.columns)):
            return df_hosts

        df_pc = pl.read_parquet(self.paths.silver_ensembl_pc_genes).select(["gene_id", "start", "end"])
        c1 = df_pc.rename({"gene_id": "gene_id_1", "start": "start_1", "end": "end_1"})
        c2 = df_pc.rename({"gene_id": "gene_id_2", "start": "start_2", "end": "end_2"})
        drop_cols = [c for c in needed if c in df_hosts.columns]
        base = df_hosts.drop(drop_cols) if drop_cols else df_hosts
        return base.join(c1, on="gene_id_1", how="inner").join(c2, on="gene_id_2", how="inner")

    def compute_host_pair_enrichment(
        self,
        df_wgd_hosts: pl.DataFrame,
        df_ssd_hosts: pl.DataFrame,
        df_wgd_lnc_pairs: pl.DataFrame,
        df_ssd_lnc_pairs: pl.DataFrame,
        label_ssd: str = "SSD_all",
    ) -> dict[str, float | int | str]:
        """Test esatto di Fisher a livello di coppie di geni host."""
        n_wgd_total = df_wgd_hosts["pair_key"].n_unique()
        n_ssd_total = df_ssd_hosts["pair_key"].n_unique()

        wgd_host_keys = set(df_wgd_hosts["pair_key"].unique().to_list())
        ssd_host_keys = set(df_ssd_hosts["pair_key"].unique().to_list())
        wgd_pos_keys = set(df_wgd_lnc_pairs["pair_key"].unique().to_list()) & wgd_host_keys
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
        df_wgd_hosts: pl.DataFrame | None = None,
        df_ssd_hosts: pl.DataFrame | None = None,
        exclude_sprotein: str | None = None,
        save_outputs: bool = True,
        output_tag: str | None = None,
    ) -> tuple[Phase2Outputs, pl.DataFrame]:
        """Esegue la Fase 2 base (supportando dataset host alternativi e filtro SmProt)."""
        is_canonical = (
            df_wgd_hosts is None
            and df_ssd_hosts is None
            and exclude_sprotein is None
            and output_tag is None
        )

        _ = self.build_lncrna_phylostratigraphy(force_rebuild=force_rebuild)
        df_host_map = self.map_intragenic_lncrnas_to_hosts(
            strict_containment=strict_containment,
            force_rebuild=force_rebuild,
        )
        sp_exclude = self._get_sprotein_lncrna_ids(exclude_sprotein)

        wgd_hosts = (
            df_wgd_hosts if df_wgd_hosts is not None
            else pl.read_parquet(self.paths.silver_wgd_pairs)
        )
        ssd_hosts = self._standardize_ssd_clade_column(
            df_ssd_hosts if df_ssd_hosts is not None
            else pl.read_parquet(self.paths.silver_ssd_pairs)
        )

        df_wgd_lnc = self._pair_intragenic_lncrnas(
            wgd_hosts,
            df_host_map,
            duplication_type="WGD",
            require_conserved_rel_strand=require_conserved_rel_strand,
            sense_only=sense_only,
            exclude_lncrna_ids=sp_exclude,
        )
        df_ssd_lnc = self._pair_intragenic_lncrnas(
            ssd_hosts,
            df_host_map,
            duplication_type="SSD",
            require_conserved_rel_strand=require_conserved_rel_strand,
            sense_only=sense_only,
            exclude_lncrna_ids=sp_exclude,
        )

        out_wgd_lnc = self.paths.silver / "wgd_intragenic_lncrna_pairs.parquet"
        out_ssd_lnc = self.paths.silver / "ssd_intragenic_lncrna_pairs.parquet"
        if save_outputs and is_canonical:
            self.paths.save_parquet_safe(df_wgd_lnc, out_wgd_lnc)
            self.paths.save_parquet_safe(df_ssd_lnc, out_ssd_lnc)

        df_ssd_ancient_hosts = ssd_hosts.filter(pl.col("is_ancient_ssd") == True)
        df_wgd_sense = df_wgd_lnc.filter(pl.col("rel_strand_1") == "sense")
        df_ssd_sense = df_ssd_lnc.filter(pl.col("rel_strand_1") == "sense")
        df_wgd_antisense = df_wgd_lnc.filter(pl.col("rel_strand_1") == "antisense")
        df_ssd_antisense = df_ssd_lnc.filter(pl.col("rel_strand_1") == "antisense")

        summary_rows = [
            self.compute_host_pair_enrichment(
                wgd_hosts, ssd_hosts, df_wgd_lnc, df_ssd_lnc, label_ssd="SSD_tutte (sense+antisense)"
            ),
            self.compute_host_pair_enrichment(
                wgd_hosts, df_ssd_ancient_hosts, df_wgd_lnc, df_ssd_lnc, label_ssd="SSD_antiche_preSarcopterygii (sense+antisense)"
            ),
            self.compute_host_pair_enrichment(
                wgd_hosts, df_ssd_ancient_hosts, df_wgd_sense, df_ssd_sense, label_ssd="SSD_antiche (solo SENSE)"
            ),
            self.compute_host_pair_enrichment(
                wgd_hosts, df_ssd_ancient_hosts, df_wgd_antisense, df_ssd_antisense, label_ssd="SSD_antiche (solo ANTISENSE)"
            ),
        ]
        df_summary = pl.DataFrame(summary_rows)

        suffix = f"_{output_tag}" if output_tag else ""
        out_summary = self.paths.gold / f"phase2_wgd_vs_ssd_enrichment{suffix}.csv"
        if save_outputs and (is_canonical or output_tag is not None):
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
    # CONTROLLO LUNGHEZZA HOST & CURVA DI SENSIBILITÀ A ∩ B
    # =========================================================================

    def _prepare_host_pair_feature_table(
        self,
        use_ancient_ssd: bool = True,
        df_wgd_hosts: pl.DataFrame | None = None,
        df_ssd_hosts: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Tabella di lunghezza per coppia host (L = end - start + 1)."""
        wgd_src = (
            df_wgd_hosts if df_wgd_hosts is not None
            else pl.read_parquet(self.paths.silver_wgd_pairs)
        )
        wgd_src = self._ensure_host_coords(wgd_src)
        df_wgd = (
            wgd_src.select(["pair_key", "start_1", "end_1", "start_2", "end_2"])
            .unique(subset=["pair_key"])
            .with_columns(pl.lit(1, dtype=pl.Int8).alias("is_wgd"))
        )

        ssd_src = self._standardize_ssd_clade_column(
            df_ssd_hosts if df_ssd_hosts is not None
            else pl.read_parquet(self.paths.silver_ssd_pairs)
        )
        if use_ancient_ssd:
            ssd_src = ssd_src.filter(pl.col("is_ancient_ssd") == True)
        ssd_src = self._ensure_host_coords(ssd_src)
        df_ssd = (
            ssd_src.select(["pair_key", "start_1", "end_1", "start_2", "end_2"])
            .unique(subset=["pair_key"])
            .with_columns(pl.lit(0, dtype=pl.Int8).alias("is_wgd"))
        )

        return (
            pl.concat([df_wgd, df_ssd], how="vertical")
            .with_columns([
                (pl.col("end_1") - pl.col("start_1") + 1).clip(lower_bound=1).cast(pl.Float64).alias("len_1"),
                (pl.col("end_2") - pl.col("start_2") + 1).clip(lower_bound=1).cast(pl.Float64).alias("len_2"),
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
        wgd_positive_keys: set[str],
        ssd_positive_keys: set[str] | None = None,
    ) -> dict[str, float]:
        """Stima OR_adj tramite regressione logistica su I_WGD, log10(min_len), log10(max_len)."""
        if ssd_positive_keys is None:
            y_expr = pl.col("pair_key").is_in(list(wgd_positive_keys))
        else:
            y_expr = (
                ((pl.col("is_wgd") == 1) & pl.col("pair_key").is_in(list(wgd_positive_keys)))
                | ((pl.col("is_wgd") == 0) & pl.col("pair_key").is_in(list(ssd_positive_keys)))
            )

        df_model = df_features.with_columns(y_expr.cast(pl.Float64).alias("y"))
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
            "OR_adj_CI95_low": round(float(np.exp(b_wgd - Z_95 * se_wgd)), 4),
            "OR_adj_CI95_high": round(float(np.exp(b_wgd + Z_95 * se_wgd)), 4),
            "Logistic_z": round(z_stat, 3),
            "Logistic_p_value": p_val,
        }

    def _resolve_lncrna_pairs_for_analysis(
        self,
        df_wgd_hosts: pl.DataFrame | None,
        df_ssd_hosts: pl.DataFrame | None,
        df_wgd_lnc: pl.DataFrame | None,
        df_ssd_lnc: pl.DataFrame | None,
        exclude_sprotein: str | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Carica o ricostruisce le coppie intrageniche applicando l'eventuale filtro SmProt."""
        sp_exclude = self._get_sprotein_lncrna_ids(exclude_sprotein)

        if df_wgd_lnc is None:
            if df_wgd_hosts is not None:
                df_hmap = self.map_intragenic_lncrnas_to_hosts(force_rebuild=False)
                df_wgd_lnc = self._pair_intragenic_lncrnas(
                    df_wgd_hosts, df_hmap, duplication_type="WGD", exclude_lncrna_ids=sp_exclude
                )
            else:
                wgd_path = self.paths.silver / "wgd_intragenic_lncrna_pairs.parquet"
                if not wgd_path.exists():
                    raise FileNotFoundError("Esegui prima `analyzer.run_phase2()`.")
                df_wgd_lnc = pl.read_parquet(wgd_path)

        if df_ssd_lnc is None:
            if df_ssd_hosts is not None:
                df_hmap = self.map_intragenic_lncrnas_to_hosts(force_rebuild=False)
                df_ssd_lnc = self._pair_intragenic_lncrnas(
                    df_ssd_hosts, df_hmap, duplication_type="SSD", exclude_lncrna_ids=sp_exclude
                )
            else:
                ssd_path = self.paths.silver / "ssd_intragenic_lncrna_pairs.parquet"
                if not ssd_path.exists():
                    raise FileNotFoundError("Esegui prima `analyzer.run_phase2()`.")
                df_ssd_lnc = pl.read_parquet(ssd_path)

        if sp_exclude:
            sp_list = list(sp_exclude)
            df_wgd_lnc = df_wgd_lnc.filter(
                ~pl.col("lncrna_1").is_in(sp_list) & ~pl.col("lncrna_2").is_in(sp_list)
            )
            df_ssd_lnc = df_ssd_lnc.filter(
                ~pl.col("lncrna_1").is_in(sp_list) & ~pl.col("lncrna_2").is_in(sp_list)
            )

        return df_wgd_lnc, df_ssd_lnc

    def analyze_host_length_confounding(
        self,
        use_ancient_ssd: bool = True,
        n_bins: int = 8,
        df_wgd_hosts: pl.DataFrame | None = None,
        df_ssd_hosts: pl.DataFrame | None = None,
        df_wgd_lnc: pl.DataFrame | None = None,
        df_ssd_lnc: pl.DataFrame | None = None,
        exclude_sprotein: str | None = None,
        save_outputs: bool = True,
        output_tag: str | None = None,
    ) -> dict[str, pl.DataFrame | dict]:
        """Controllo confondimento lunghezza: Mann-Whitney, CMH (OR, CI 95% RBG, chi2), IRLS."""
        is_canonical = (
            df_wgd_hosts is None
            and df_ssd_hosts is None
            and df_wgd_lnc is None
            and df_ssd_lnc is None
            and exclude_sprotein is None
            and output_tag is None
        )

        df_feat = self._prepare_host_pair_feature_table(
            use_ancient_ssd=use_ancient_ssd,
            df_wgd_hosts=df_wgd_hosts,
            df_ssd_hosts=df_ssd_hosts,
        )
        wgd_lnc, ssd_lnc = self._resolve_lncrna_pairs_for_analysis(
            df_wgd_hosts, df_ssd_hosts, df_wgd_lnc, df_ssd_lnc, exclude_sprotein
        )
        wgd_pos_keys = set(wgd_lnc["pair_key"].unique().to_list())
        ssd_pos_keys = set(ssd_lnc["pair_key"].unique().to_list())

        df_eval = df_feat.with_columns(
            (
                ((pl.col("is_wgd") == 1) & pl.col("pair_key").is_in(list(wgd_pos_keys)))
                | ((pl.col("is_wgd") == 0) & pl.col("pair_key").is_in(list(ssd_pos_keys)))
            )
            .cast(pl.Int8)
            .alias("has_lnc_pair")
        )

        # 1. Confronto lunghezze host (Mann-Whitney U)
        wgd_geom = df_eval.filter(pl.col("is_wgd") == 1)["geom_mean_len"].to_numpy()
        ssd_geom = df_eval.filter(pl.col("is_wgd") == 0)["geom_mean_len"].to_numpy()
        _, mw_p = mannwhitneyu(wgd_geom, ssd_geom, alternative="two-sided")

        # 2. Fisher grezzo
        a_tot = int(df_eval.filter((pl.col("is_wgd") == 1) & (pl.col("has_lnc_pair") == 1)).height)
        b_tot = int(df_eval.filter((pl.col("is_wgd") == 1) & (pl.col("has_lnc_pair") == 0)).height)
        c_tot = int(df_eval.filter((pl.col("is_wgd") == 0) & (pl.col("has_lnc_pair") == 1)).height)
        d_tot = int(df_eval.filter((pl.col("is_wgd") == 0) & (pl.col("has_lnc_pair") == 0)).height)
        or_crude, p_crude = fisher_exact([[a_tot, b_tot], [c_tot, d_tot]], alternative="two-sided")

        # 3. Stratificazione su n_bins quantili + CMH (Robins-Breslow-Greenland CI + chi2)
        geom_all = df_eval["geom_mean_len"].to_numpy()
        edges = np.quantile(geom_all, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] -= 1.0
        edges[-1] += 1.0
        bin_idx = np.digitize(geom_all, edges[1:-1], right=True)

        df_binned = df_eval.with_columns(pl.Series("length_bin", bin_idx))
        bin_rows: list[dict] = []
        strata_tables: list[tuple[int, int, int, int]] = []

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
            strata_tables.append((a, b, c, d))

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

        mh_res = _compute_mantel_haenszel(strata_tables)

        # 4. Regressione logistica multivariata continua
        log_res = self._fit_length_adjusted_or(df_feat, wgd_pos_keys, ssd_pos_keys)

        df_bins = pl.DataFrame(bin_rows)
        summary_dict = {
            "SSD_control_set": "SSD_antiche_preSarcopterygii" if use_ancient_ssd else "SSD_tutte",
            "WGD_median_geom_len_kb": round(float(np.median(wgd_geom)) / 1000.0, 2),
            "SSD_median_geom_len_kb": round(float(np.median(ssd_geom)) / 1000.0, 2),
            "Length_ratio_WGD_over_SSD": round(float(np.median(wgd_geom) / np.median(ssd_geom)), 3),
            "MannWhitney_length_p": float(mw_p),
            "OR_crude_Fisher": round(float(or_crude), 4),
            "Fisher_crude_p": float(p_crude),
            **mh_res,
            **log_res,
        }
        df_summary = pl.DataFrame([summary_dict])

        suffix = f"_{output_tag}" if output_tag else ""
        if save_outputs and (is_canonical or output_tag is not None):
            df_bins.write_csv(self.paths.gold / f"phase2_length_control_bins{suffix}.csv")
            df_summary.write_csv(self.paths.gold / f"phase2_length_control_summary{suffix}.csv")

        logger.info(
            f"Controllo lunghezza: OR grezzo={summary_dict['OR_crude_Fisher']} | "
            f"OR_MH={mh_res['OR_MantelHaenszel_stratified']} "
            f"[{mh_res['OR_MH_CI95_low']}, {mh_res['OR_MH_CI95_high']}] "
            f"(chi2={mh_res['MH_chi2']}, p={mh_res['MH_p_value']:.2e}) | "
            f"OR_adj={log_res['OR_adj_length']} (p={log_res['Logistic_p_value']:.2e})"
        )

        return {"summary": df_summary, "bins": df_bins, "metrics": summary_dict}

    def run_ab_sensitivity_curve(
        self,
        use_ancient_ssd: bool = True,
        export_three_lanes: bool = True,
        df_wgd_hosts: pl.DataFrame | None = None,
        df_ssd_hosts: pl.DataFrame | None = None,
        df_wgd_lnc: pl.DataFrame | None = None,
        df_ssd_lnc: pl.DataFrame | None = None,
        exclude_sprotein: str | None = None,
        save_outputs: bool = True,
        output_tag: str | None = None,
    ) -> pl.DataFrame:
        """Curva di sensibilità A ∩ B su 6 soglie filostratigrafiche con OR grezzo e aggiustato."""
        is_canonical = (
            df_wgd_hosts is None
            and df_ssd_hosts is None
            and df_wgd_lnc is None
            and df_ssd_lnc is None
            and exclude_sprotein is None
            and output_tag is None
        )

        df_feat = self._prepare_host_pair_feature_table(
            use_ancient_ssd=use_ancient_ssd,
            df_wgd_hosts=df_wgd_hosts,
            df_ssd_hosts=df_ssd_hosts,
        )
        wgd_lnc, ssd_lnc = self._resolve_lncrna_pairs_for_analysis(
            df_wgd_hosts, df_ssd_hosts, df_wgd_lnc, df_ssd_lnc, exclude_sprotein
        )

        wgd_allowed_keys = set(df_feat.filter(pl.col("is_wgd") == 1)["pair_key"].to_list())
        ssd_allowed_keys = set(df_feat.filter(pl.col("is_wgd") == 0)["pair_key"].to_list())
        wgd_lnc = wgd_lnc.filter(pl.col("pair_key").is_in(list(wgd_allowed_keys)))
        ssd_lnc = ssd_lnc.filter(pl.col("pair_key").is_in(list(ssd_allowed_keys)))

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
            w_sub = wgd_lnc.filter(expr)
            s_sub = ssd_lnc.filter(expr)

            w_keys = set(w_sub["pair_key"].unique().to_list())
            s_keys = set(s_sub["pair_key"].unique().to_list())

            a = len(w_keys)
            b = n_wgd_tot - a
            c = len(s_keys)
            d = n_ssd_tot - c

            or_crude, p_crude = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            adj = self._fit_length_adjusted_or(df_feat, w_keys, s_keys)

            rows.append({
                "Soglia_A_cap_B": label,
                "WGD_host_pos": a,
                "WGD_pct": round(100.0 * a / n_wgd_tot, 3) if n_wgd_tot else 0.0,
                "WGD_lnc_pairs": len(w_sub),
                "SSD_host_pos": c,
                "SSD_pct": round(100.0 * c / n_ssd_tot, 3) if n_ssd_tot else 0.0,
                "SSD_lnc_pairs": len(s_sub),
                "OR_crude": round(float(or_crude), 4),
                "Fisher_p": float(p_crude),
                "OR_adj_length": adj["OR_adj_length"],
                "OR_adj_CI95_low": adj["OR_adj_CI95_low"],
                "OR_adj_CI95_high": adj["OR_adj_CI95_high"],
                "Logistic_p": adj["Logistic_p_value"],
            })

        df_curve = pl.DataFrame(rows)
        suffix = f"_{output_tag}" if output_tag else ""
        if save_outputs and (is_canonical or output_tag is not None):
            df_curve.write_csv(self.paths.gold / f"phase2_ab_sensitivity_curve{suffix}.csv")

        if export_three_lanes and is_canonical:
            df_via2_wgd = wgd_lnc.filter(
                (pl.col("age_node_1") >= 15) & (pl.col("age_node_2") >= 15)
            ).with_columns([
                ((pl.col("age_node_1") >= 16) & (pl.col("age_node_2") >= 16)).alias("is_tier_tetrapoda"),
                ((pl.col("age_node_1") == 17) & (pl.col("age_node_2") == 17)).alias("is_tier_euteleostomi_strict"),
            ])
            df_via2_ssd = ssd_lnc.filter(
                (pl.col("age_node_1") >= 15) & (pl.col("age_node_2") >= 15)
            ).with_columns([
                ((pl.col("age_node_1") >= 16) & (pl.col("age_node_2") >= 16)).alias("is_tier_tetrapoda"),
                ((pl.col("age_node_1") == 17) & (pl.col("age_node_2") == 17)).alias("is_tier_euteleostomi_strict"),
            ])
            self.paths.save_parquet_safe(df_via2_wgd, self.paths.silver / "lane2_ab_wgd_pairs.parquet")
            self.paths.save_parquet_safe(df_via2_ssd, self.paths.silver / "lane2_ab_ssd_pairs.parquet")

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

    def run_sensitivity_phase2(
        self,
        label: str,
        df_wgd_hosts: pl.DataFrame | None = None,
        df_ssd_hosts: pl.DataFrame | None = None,
        exclude_sprotein: str | None = None,
        use_ancient_ssd: bool = True,
        n_bins: int = 8,
        output_tag: str | None = None,
    ) -> dict[str, pl.DataFrame | dict | str]:
        """
        Esegue in memoria (senza toccare i Parquet canonici in Silver) l'intera Fase 2
        su dataset WGD/SSD alternativi e/o con filtro purezza SmProt.
        """
        df_hmap = self.map_intragenic_lncrnas_to_hosts(force_rebuild=False)
        sp_exclude = self._get_sprotein_lncrna_ids(exclude_sprotein)

        wgd_h = (
            df_wgd_hosts if df_wgd_hosts is not None
            else pl.read_parquet(self.paths.silver_wgd_pairs)
        )
        ssd_h = self._standardize_ssd_clade_column(
            df_ssd_hosts if df_ssd_hosts is not None
            else pl.read_parquet(self.paths.silver_ssd_pairs)
        )

        df_wgd_lnc = self._pair_intragenic_lncrnas(
            wgd_h, df_hmap, duplication_type="WGD", exclude_lncrna_ids=sp_exclude
        )
        df_ssd_lnc = self._pair_intragenic_lncrnas(
            ssd_h, df_hmap, duplication_type="SSD", exclude_lncrna_ids=sp_exclude
        )

        length_res = self.analyze_host_length_confounding(
            use_ancient_ssd=use_ancient_ssd,
            n_bins=n_bins,
            df_wgd_hosts=wgd_h,
            df_ssd_hosts=ssd_h,
            df_wgd_lnc=df_wgd_lnc,
            df_ssd_lnc=df_ssd_lnc,
            exclude_sprotein=None,  # già applicato
            save_outputs=(output_tag is not None),
            output_tag=output_tag,
        )
        ab_curve = self.run_ab_sensitivity_curve(
            use_ancient_ssd=use_ancient_ssd,
            export_three_lanes=False,
            df_wgd_hosts=wgd_h,
            df_ssd_hosts=ssd_h,
            df_wgd_lnc=df_wgd_lnc,
            df_ssd_lnc=df_ssd_lnc,
            exclude_sprotein=None,  # già applicato
            save_outputs=(output_tag is not None),
            output_tag=output_tag,
        )

        return {
            "label": label,
            "length_control": length_res,
            "ab_curve": ab_curve,
            "n_wgd_hosts": wgd_h["pair_key"].n_unique(),
            "n_ssd_hosts": (
                ssd_h.filter(pl.col("is_ancient_ssd") == True)["pair_key"].n_unique()
                if use_ancient_ssd else ssd_h["pair_key"].n_unique()
            ),
        }

    def plot_phase2_diagnostics(
        self,
        length_res: dict,
        df_ab_curve: pl.DataFrame,
        save_fig: bool = True,
    ) -> tuple[Path | None, Path | None]:
        """Figura diagnostica Fase 2 a 2 pannelli (lunghezza host + curva A ∩ B)."""
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
