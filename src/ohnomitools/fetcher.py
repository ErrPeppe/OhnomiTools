# src/ohnomitools/fetcher.py
from __future__ import annotations
from pathlib import Path
import gzip
import logging
import shutil
import requests
import polars as pl

from .config import CANONICAL_CHROMOSOMES, PATHS, ProjectPaths

logger = logging.getLogger("OhnomiTools.fetcher")

# Archivi HTTPS ufficiali Ensembl con BioMart attivo
BIOMART_ARCHIVE_URLS: tuple[str, ...] = (
    "https://e115.ensembl.org/biomart/martservice",
    "https://jun2026.archive.ensembl.org/biomart/martservice",
    "https://sep2025.archive.ensembl.org/biomart/martservice",
    "https://may2025.archive.ensembl.org/biomart/martservice",
)

# Query 1 (solo pagina `feature_page`): catalogo geni umani con coordinate, strand e biotype
BIOMART_GENES_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Attribute name="ensembl_gene_id" />'
    '<Attribute name="external_gene_name" />'
    '<Attribute name="chromosome_name" />'
    '<Attribute name="start_position" />'
    '<Attribute name="end_position" />'
    '<Attribute name="strand" />'
    '<Attribute name="gene_biotype" />'
    '</Dataset>'
    '</Query>'
)

# Query 2 (solo pagina `homologs`): coppie di paraloghi umani con clade ancestrale e % identità
BIOMART_PARALOGUES_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Attribute name="ensembl_gene_id" />'
    '<Attribute name="hsapiens_paralog_ensembl_gene" />'
    '<Attribute name="hsapiens_paralog_subtype" />'
    '<Attribute name="hsapiens_paralog_orthology_type" />'
    '<Attribute name="hsapiens_paralog_perc_id" />'
    '<Attribute name="hsapiens_paralog_perc_id_r1" />'
    '</Dataset>'
    '</Query>'
)


class DataIngestionPipeline:
    """
    Pipeline ETL (Bronze -> Silver Parquet) per LncBook v2.0, OHNOLOGS ed Ensembl BioMart.
    Ottimizzata per Google Colab con storage persistente su Google Drive.
    """

    def __init__(self, paths: ProjectPaths = PATHS) -> None:
        self.paths = paths

    def _decompress_to_scratch(self, gz_path: Path) -> Path:
        """
        Decomprime un file .gz da Google Drive sull'SSD locale (/tmp) di Colab
        per evitare picchi di RAM durante il parsing CSV/GTF in Polars.
        """
        if not gz_path.exists():
            raise FileNotFoundError(f"File Bronze mancante su Google Drive: {gz_path}")

        if gz_path.suffix != ".gz":
            return gz_path

        scratch_target = self.paths.scratch / gz_path.stem
        logger.info(f"Decompressione su /tmp: {gz_path.name} -> {scratch_target.name}")
        with gzip.open(gz_path, "rb") as f_in, open(scratch_target, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
        return scratch_target

    def _download_biomart_tsv(self, xml_query: str, dest_path: Path) -> Path:
        """
        Scarica un TSV da Ensembl BioMart provando in sequenza gli archivi HTTPS
        ufficiali (usando GET) e verificando che la risposta sia un TSV valido.
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_download = self.paths.scratch / dest_path.name
        headers = {"User-Agent": "OhnomiTools/0.1.0 (https://github.com/ErrPeppe/OhnomiTools)"}
        errors: list[str] = []

        for url in BIOMART_ARCHIVE_URLS:
            logger.info(f"Download Ensembl BioMart ({dest_path.name}) da: {url}")
            try:
                with requests.get(
                    url,
                    params={"query": xml_query},
                    headers=headers,
                    stream=True,
                    timeout=300,
                ) as r:
                    r.raise_for_status()
                    chunk_iter = r.iter_content(chunk_size=1024 * 1024)
                    first_chunk = next(chunk_iter, b"")

                    if not first_chunk:
                        raise ValueError("Risposta vuota dal server BioMart.")

                    head_text = first_chunk[:512].decode("utf-8", errors="ignore").lstrip()
                    if head_text.startswith("<") or "Query ERROR" in head_text:
                        preview = head_text.splitlines()[0][:200]
                        raise ValueError(f"Risposta non-TSV dal server BioMart: {preview}")

                    with open(tmp_download, "wb") as f:
                        f.write(first_chunk)
                        for chunk in chunk_iter:
                            if chunk:
                                f.write(chunk)

                shutil.move(str(tmp_download), str(dest_path))
                logger.info(f"Download completato: {dest_path.name}")
                return dest_path

            except Exception as exc:
                if tmp_download.exists():
                    tmp_download.unlink()
                logger.warning(f"Archivio BioMart fallito ({url}): {exc}")
                errors.append(f"{url} -> {exc}")

        raise RuntimeError(
            f"Impossibile scaricare {dest_path.name} dagli archivi Ensembl BioMart:\n"
            + "\n".join(errors)
        )

    def build_lncbook_master_tables(self, force_rebuild: bool = False) -> tuple[Path, Path]:
        """
        Estrae dal GTF puro di LncBook v2.0 il catalogo dei trascritti e quello
        aggregato a livello di gene su cromosomi canonici (chr1..22, chrX, chrY).
        """
        out_tx = self.paths.silver_lncrna_transcripts
        out_genes = self.paths.silver_lncrna_genes

        if out_tx.exists() and out_genes.exists() and not force_rebuild:
            logger.info("Master tables LncBook già presenti in Silver. Skip.")
            return out_tx, out_genes

        tmp_gtf = self._decompress_to_scratch(self.paths.raw_lncbook_gtf)

        try:
            gtf_cols = [
                "chr", "source", "feature", "start", "end",
                "score", "strand", "frame", "attributes"
            ]
            lf = pl.scan_csv(
                tmp_gtf,
                separator="\t",
                has_header=False,
                comment_prefix="#",
                new_columns=gtf_cols,
                schema_overrides={"chr": pl.Utf8, "start": pl.Int64, "end": pl.Int64},
            )

            df_tx = (
                lf.filter(
                    (pl.col("feature") == "transcript")
                    & (pl.col("chr").is_in(CANONICAL_CHROMOSOMES))
                )
                .with_columns([
                    pl.col("attributes").str.extract(r'gene_id "([^"]+)"', 1).alias("gene_id"),
                    pl.col("attributes").str.extract(r'transcript_id "([^"]+)"', 1).alias("transcript_id"),
                    pl.col("attributes").str.extract(r'gene_alias "([^"]+)"', 1).alias("gene_alias"),
                    pl.col("attributes").str.extract(r'transcript_alias "([^"]+)"', 1).alias("transcript_alias"),
                ])
                .select([
                    "gene_id", "transcript_id", "chr", "start", "end",
                    "strand", "gene_alias", "transcript_alias"
                ])
                .collect()
            )

            df_genes = (
                df_tx.group_by(["gene_id", "chr", "strand"])
                .agg([
                    pl.col("start").min().alias("start"),
                    pl.col("end").max().alias("end"),
                    pl.col("transcript_id").n_unique().alias("n_isoforms"),
                    pl.col("gene_alias").drop_nulls().first().alias("gene_alias"),
                ])
                .sort(["chr", "start"])
            )

            self.paths.save_parquet_safe(df_tx, out_tx)
            self.paths.save_parquet_safe(df_genes, out_genes)
            return out_tx, out_genes
        finally:
            if tmp_gtf != self.paths.raw_lncbook_gtf and tmp_gtf.exists():
                tmp_gtf.unlink()

    def convert_csv_dataset_to_silver(
        self,
        raw_gz_path: Path,
        silver_parquet_path: Path,
        separator: str = ",",
        force_rebuild: bool = False,
    ) -> Path:
        """
        Converte in streaming un CSV/TSV compresso di LncBook (Conservazione, miRNA,
        RBP, Espressione) in formato Apache Parquet compresso (zstd) su Google Drive.
        """
        if silver_parquet_path.exists() and not force_rebuild:
            logger.info(f"Parquet già presente in Silver: {silver_parquet_path.name}. Skip.")
            return silver_parquet_path

        tmp_csv = self._decompress_to_scratch(raw_gz_path)
        tmp_parquet = self.paths.scratch / silver_parquet_path.name

        try:
            lf = pl.scan_csv(
                tmp_csv,
                separator=separator,
                infer_schema_length=10000,
                ignore_errors=True,
            )
            lf.sink_parquet(tmp_parquet, compression="zstd")
            silver_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_parquet), str(silver_parquet_path))
            logger.info(f"Convertito in Silver Parquet: {silver_parquet_path.name}")
            return silver_parquet_path
        finally:
            if tmp_csv != raw_gz_path and tmp_csv.exists():
                tmp_csv.unlink()

    def fetch_and_build_ensembl_pc_genes(self, force_rebuild: bool = False) -> Path:
        """
        Scarica da Ensembl BioMart il catalogo dei geni umani e filtra i geni
        `protein_coding` sui cromosomi canonici con coordinate e strand (+/-).
        """
        raw_tsv = self.paths.raw_ensembl_genes
        out_pc = self.paths.silver_ensembl_pc_genes

        if out_pc.exists() and not force_rebuild:
            logger.info("Catalogo Ensembl protein-coding già presente in Silver. Skip.")
            return out_pc

        if not raw_tsv.exists() or force_rebuild:
            self._download_biomart_tsv(BIOMART_GENES_XML, raw_tsv)

        df_raw = pl.read_csv(
            raw_tsv,
            separator="\t",
            schema_overrides={
                "Chromosome/scaffold name": pl.Utf8,
                "Gene start (bp)": pl.Int64,
                "Gene end (bp)": pl.Int64,
                "Strand": pl.Int8,
            },
            ignore_errors=True,
        )

        col_map = {
            "Gene stable ID": "gene_id",
            "Gene name": "gene_name",
            "Chromosome/scaffold name": "scaffold_id",
            "Gene start (bp)": "start",
            "Gene end (bp)": "end",
            "Strand": "strand_num",
            "Gene type": "gene_biotype",
        }
        df_pc = (
            df_raw.rename({k: v for k, v in col_map.items() if k in df_raw.columns})
            .filter(pl.col("gene_biotype") == "protein_coding")
            .with_columns([
                pl.concat_str([pl.lit("chr"), pl.col("scaffold_id")]).alias("chr"),
                pl.when(pl.col("strand_num") == 1)
                .then(pl.lit("+"))
                .when(pl.col("strand_num") == -1)
                .then(pl.lit("-"))
                .otherwise(pl.lit(None))
                .alias("strand"),
            ])
            .filter(pl.col("chr").is_in(CANONICAL_CHROMOSOMES))
            .select(["gene_id", "gene_name", "chr", "start", "end", "strand", "gene_biotype"])
            .unique(subset=["gene_id"])
            .sort(["chr", "start"])
        )

        self.paths.save_parquet_safe(df_pc, out_pc)
        return out_pc

    def build_ohnologs_wgd_tables(self, force_rebuild: bool = False) -> tuple[Path, Path]:
        """
        Normalizza i file TSV senza intestazione del dump OHNOLOGS:
        - `database/genes.tsv` (10 colonne, filtrato su `homo_sapiens`)
        - `ohnologs/homo_sapiens.tsv` (4.650 ohnologhi umani + categoria 1r/2r/both/unk/syn/htf)
        - `database/gene_ohnology.tsv` (filtrato su coppie intra-umane ENSP-ENSP -> ENSG-ENSG)
        """
        out_genes = self.paths.silver_wgd_genes
        out_pairs = self.paths.silver_wgd_pairs

        if out_genes.exists() and out_pairs.exists() and not force_rebuild:
            logger.info("Tabelle OHNOLOGS (WGD) già presenti in Silver. Skip.")
            return out_genes, out_pairs

        # 1. Leggi database/genes.tsv filtrando solo homo_sapiens su cromosomi canonici
        genes_cols = [
            "species_id", "scaffold_id", "segment_id", "family_id",
            "gene_id", "protein_id", "start", "end", "pvc", "pgc"
        ]
        df_human_all = (
            pl.scan_csv(
                self.paths.raw_ohnologs_genes,
                separator="\t",
                has_header=False,
                new_columns=genes_cols,
                schema_overrides={
                    "species_id": pl.Utf8,
                    "scaffold_id": pl.Utf8,
                    "segment_id": pl.Utf8,
                    "family_id": pl.Utf8,
                    "gene_id": pl.Utf8,
                    "protein_id": pl.Utf8,
                    "start": pl.Int64,
                    "end": pl.Int64,
                    "pvc": pl.Utf8,
                    "pgc": pl.Utf8,
                },
            )
            .filter(pl.col("species_id") == "homo_sapiens")
            .with_columns(
                pl.concat_str([pl.lit("chr"), pl.col("scaffold_id")]).alias("chr")
            )
            .filter(pl.col("chr").is_in(CANONICAL_CHROMOSOMES))
            .select([
                "gene_id", "protein_id", "chr", "start", "end",
                "family_id", "segment_id", "pvc", "pgc"
            ])
            .collect()
        )

        # Arricchisci con gene_name e strand (+/-) da Ensembl protein-coding se disponibile
        if self.paths.silver_ensembl_pc_genes.exists():
            df_pc = pl.read_parquet(self.paths.silver_ensembl_pc_genes).select(
                ["gene_id", "gene_name", "strand"]
            )
            df_human_all = df_human_all.join(df_pc, on="gene_id", how="left")
        else:
            df_human_all = df_human_all.with_columns([
                pl.lit(None, dtype=pl.Utf8).alias("gene_name"),
                pl.lit(None, dtype=pl.Utf8).alias("strand"),
            ])

        # 2. Leggi ohnologs/homo_sapiens.tsv (lista ohnologhi umani + categoria)
        df_hs_ohno = pl.read_csv(
            self.paths.raw_ohnologs_human_list,
            separator="\t",
            has_header=False,
            new_columns=["protein_id", "retention_category"],
            schema_overrides={"protein_id": pl.Utf8, "retention_category": pl.Utf8},
        )

        df_wgd_genes = (
            df_human_all.join(df_hs_ohno, on="protein_id", how="inner")
            .unique(subset=["gene_id"])
            .sort(["chr", "start"])
        )
        self.paths.save_parquet_safe(df_wgd_genes, out_genes)

        # 3. Costruisci le coppie WGD intra-umane da database/gene_ohnology.tsv
        if self.paths.raw_ohnologs_pairs.exists():
            prot_map = df_human_all.join(df_hs_ohno, on="protein_id", how="left").select([
                "protein_id", "gene_id", "gene_name", "chr", "start", "end",
                "strand", "family_id", "retention_category"
            ]).unique(subset=["protein_id"])

            lf_raw_pairs = pl.scan_csv(
                self.paths.raw_ohnologs_pairs,
                separator="\t",
                has_header=False,
                new_columns=["protein_id_1", "protein_id_2", "wgd_round"],
                schema_overrides={
                    "protein_id_1": pl.Utf8,
                    "protein_id_2": pl.Utf8,
                    "wgd_round": pl.Utf8,
                },
            )

            lf_p1 = prot_map.lazy().rename({
                "protein_id": "protein_id_1",
                "gene_id": "gene_id_1",
                "gene_name": "gene_name_1",
                "chr": "chr_1",
                "start": "start_1",
                "end": "end_1",
                "strand": "strand_1",
                "family_id": "family_id_1",
                "retention_category": "retention_category_1",
            })
            lf_p2 = prot_map.lazy().rename({
                "protein_id": "protein_id_2",
                "gene_id": "gene_id_2",
                "gene_name": "gene_name_2",
                "chr": "chr_2",
                "start": "start_2",
                "end": "end_2",
                "strand": "strand_2",
                "family_id": "family_id_2",
                "retention_category": "retention_category_2",
            })

            df_wgd_pairs = (
                lf_raw_pairs
                .join(lf_p1, on="protein_id_1", how="inner")
                .join(lf_p2, on="protein_id_2", how="inner")
                .filter(pl.col("gene_id_1") != pl.col("gene_id_2"))
                .with_columns(
                    pl.when(pl.col("gene_id_1") < pl.col("gene_id_2"))
                    .then(pl.concat_str([pl.col("gene_id_1"), pl.lit("_"), pl.col("gene_id_2")]))
                    .otherwise(pl.concat_str([pl.col("gene_id_2"), pl.lit("_"), pl.col("gene_id_1")]))
                    .alias("pair_key")
                )
                .unique(subset=["pair_key"])
                .sort(["chr_1", "start_1"])
                .collect()
            )
            self.paths.save_parquet_safe(df_wgd_pairs, out_pairs)

        return out_genes, out_pairs

    def fetch_and_build_ensembl_ssd_pairs(self, force_rebuild: bool = False) -> Path:
        """
        Scarica i paraloghi umani da Ensembl BioMart (solo pagina `homologs`),
        li arricchisce con coordinate e strand dei geni `protein_coding` e sottrae
        tutte le coppie WGD di OHNOLOGS per ottenere il controllo SSD puro.
        """
        raw_tsv = self.paths.raw_ensembl_paralogues
        out_ssd = self.paths.silver_ssd_pairs

        if out_ssd.exists() and not force_rebuild:
            logger.info("Tabella Ensembl SSD già presente in Silver. Skip.")
            return out_ssd

        if not self.paths.silver_ensembl_pc_genes.exists():
            self.fetch_and_build_ensembl_pc_genes(force_rebuild=False)

        if not raw_tsv.exists() or force_rebuild:
            self._download_biomart_tsv(BIOMART_PARALOGUES_XML, raw_tsv)

        df_raw = pl.read_csv(raw_tsv, separator="\t", ignore_errors=True)
        col_map = {
            "Gene stable ID": "gene_id_1",
            "Human paralogue gene stable ID": "gene_id_2",
            "Human paralogue last common ancestor with Human": "ancestor_clade",
            "Human paralogue homology type": "homology_type",
            "Paralogue %id. target Human gene identical to query gene": "perc_id_1",
            "Paralogue %id. query gene identical to target Human gene": "perc_id_2",
        }
        df_par = df_raw.rename({k: v for k, v in col_map.items() if k in df_raw.columns})

        df_pc = pl.read_parquet(self.paths.silver_ensembl_pc_genes).select([
            "gene_id", "gene_name", "chr", "start", "end", "strand"
        ])
        pc1 = df_pc.rename({c: f"{c}_1" for c in df_pc.columns})
        pc2 = df_pc.rename({c: f"{c}_2" for c in df_pc.columns})

        df_par = (
            df_par.filter(
                pl.col("gene_id_1").is_not_null()
                & pl.col("gene_id_2").is_not_null()
                & (pl.col("gene_id_1") != pl.col("gene_id_2"))
            )
            .join(pc1, on="gene_id_1", how="inner")
            .join(pc2, on="gene_id_2", how="inner")
            .with_columns(
                pl.when(pl.col("gene_id_1") < pl.col("gene_id_2"))
                .then(pl.concat_str([pl.col("gene_id_1"), pl.lit("_"), pl.col("gene_id_2")]))
                .otherwise(pl.concat_str([pl.col("gene_id_2"), pl.lit("_"), pl.col("gene_id_1")]))
                .alias("pair_key")
            )
            .unique(subset=["pair_key"])
        )

        # Sottrazione esatta delle coppie WGD (anti-join su pair_key canonica)
        if self.paths.silver_wgd_pairs.exists():
            wgd_keys = pl.read_parquet(self.paths.silver_wgd_pairs).select(["pair_key"]).unique()
            n_before = len(df_par)
            df_par = df_par.join(wgd_keys, on="pair_key", how="anti")
            logger.info(
                f"Anti-join WGD -> SSD: {n_before:,} paraloghi Ensembl -> "
                f"{len(df_par):,} coppie SSD pure (rimosse {n_before - len(df_par):,} coppie WGD)."
            )

        self.paths.save_parquet_safe(df_par, out_ssd)
        return out_ssd

    def run_all(self, force_rebuild: bool = False) -> dict[str, Path]:
        """
        Esegue l'intera pipeline ETL sui file Bronze disponibili in Google Drive.
        """
        outputs: dict[str, Path] = {}

        if self.paths.raw_lncbook_gtf.exists():
            tx, gn = self.build_lncbook_master_tables(force_rebuild=force_rebuild)
            outputs["transcripts"] = tx
            outputs["genes"] = gn

        if self.paths.raw_lncbook_cons.exists():
            outputs["conservation"] = self.convert_csv_dataset_to_silver(
                self.paths.raw_lncbook_cons,
                self.paths.silver_lncbook_cons,
                separator=",",
                force_rebuild=force_rebuild,
            )

        if self.paths.raw_lncbook_mirna_int.exists():
            outputs["mirna_interactions"] = self.convert_csv_dataset_to_silver(
                self.paths.raw_lncbook_mirna_int,
                self.paths.silver_lncbook_mirna_int,
                separator=",",
                force_rebuild=force_rebuild,
            )

        if self.paths.raw_lncbook_rbp_int.exists():
            outputs["rbp_interactions"] = self.convert_csv_dataset_to_silver(
                self.paths.raw_lncbook_rbp_int,
                self.paths.silver_lncbook_rbp_int,
                separator=",",
                force_rebuild=force_rebuild,
            )

        if self.paths.raw_lncbook_expr.exists():
            outputs["expression"] = self.convert_csv_dataset_to_silver(
                self.paths.raw_lncbook_expr,
                self.paths.silver_lncbook_expr,
                separator=",",
                force_rebuild=force_rebuild,
            )

        # Prima i geni protein-coding Ensembl (forniscono gene_name e strand a WGD e SSD)
        outputs["ensembl_pc_genes"] = self.fetch_and_build_ensembl_pc_genes(
            force_rebuild=force_rebuild
        )

        # Poi le tabelle WGD (forniscono pair_key per l'anti-join sugli SSD)
        if self.paths.raw_ohnologs_human_list.exists() and self.paths.raw_ohnologs_genes.exists():
            wgd_g, wgd_p = self.build_ohnologs_wgd_tables(force_rebuild=force_rebuild)
            outputs["wgd_genes"] = wgd_g
            outputs["wgd_pairs"] = wgd_p

        # Infine le coppie SSD pure da Ensembl BioMart
        outputs["ssd_pairs"] = self.fetch_and_build_ensembl_ssd_pairs(
            force_rebuild=force_rebuild
        )
        return outputs
