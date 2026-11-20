# src/ohnomitools/evolution.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import polars as pl
from scipy.stats import fisher_exact

from .config import PATHS, ProjectPaths

logger = logging.getLogger("OhnomiTools.evolution")

# Cladi Ensembl più giovani di (o uguali a) Sarcopterygii:
# secondo Mottes et al. (2021), le SSD con LCA in questo insieme sono
# troppo recenti rispetto alle 1R/2R-WGD (~500 Mya) e vanno stratificate.
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

# Cladi antichi coerenti con epoca >= WGD (older than Sarcopterygii)
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


class EvolutionAnalyzer:
    """
    Fase 2: Identificazione evolutiva degli Ohno-lncRNA tramite doppio approccio:
    - Ramo A (Intragenico / Modello del Passeggero): incrocio genomico su coppie host
      WGD vs SSD con controllo di conservazione dello strand relativo (Agasso et al., 2025).
    - Ramo B (Filostratigrafia LncBook v2.0): profilazione dell'età evolutiva (`Age`)
      e validazione dell'arricchimento su `17. Euteleostomi` / nodi vertebrati antichi.
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
                # Estrae il numero del nodo filostratigrafico (es. "17. Euteleostomi" -> 17)
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

        Parametri:
        - `strict_containment=True`: il lncRNA deve essere interamente contenuto nei confini
          del gene host (`lnc_start >= host_start` e `lnc_end <= host_end`).
        - Calcola la relazione di strand (`same_strand` = sense vs `opposite_strand` = antisense).
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

        # Intersezione per singolo cromosoma per mantenere la RAM bassissima
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

        # Aggiunge il numero di lncRNA ospitati in ciascun gene host (per filtrare 1:1)
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
        Normalizza il nome della colonna del Last Common Ancestor (LCA) negli SSD
        e aggiunge il flag booleano `is_ancient_ssd` (older than Sarcopterygii, Mottes et al. 2021).
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
        conta 1, eliminando bias da geni lunghi con molti lncRNA) per misurare se le coppie
        WGD ospitano coppie di lncRNA con strand coerente più spesso delle coppie SSD.
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
        Esegue l'intera Fase 2 e salva su Google Drive:
        - `silver/lncbook_gene_phylostratigraphy.parquet`
        - `silver/intragenic_host_lncrna_map.parquet`
        - `silver/wgd_intragenic_lncrna_pairs.parquet`
        - `silver/ssd_intragenic_lncrna_pairs.parquet`
        - `gold/phase2_wgd_vs_ssd_enrichment.csv`
        """
        df_phylo = self.build_lncrna_phylostratigraphy(force_rebuild=force_rebuild)
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

        # Confronti statistici: WGD vs SSD (tutte), WGD vs SSD antiche (>Sarcopterygii), WGD vs SSD solo sense
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
