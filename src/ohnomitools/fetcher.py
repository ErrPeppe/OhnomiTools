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

# Archivi ufficiali Ensembl con BioMart attivo (in ordine di affidabilità/carico)
BIOMART_ARCHIVE_URLS: tuple[str, ...] = (
    "https://e115.ensembl.org/biomart/martservice",
    "https://sep2025.archive.ensembl.org/biomart/martservice",
    "https://jun2026.archive.ensembl.org/biomart/martservice",
    "https://may2025.archive.ensembl.org/biomart/martservice",
)

BIOMART_PARALOGUES_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Attribute name="ensembl_gene_id" />'
    '<Attribute name="external_gene_name" />'
    '<Attribute name="chromosome_name" />'
    '<Attribute name="start_position" />'
    '<Attribute name="end_position" />'
    '<Attribute name="hsapiens_paralog_ensembl_gene" />'
    '<Attribute name="hsapiens_paralog_associated_gene_name" />'
    '<Attribute name="hsapiens_paralog_chromosome" />'
    '<Attribute name="hsapiens_paralog_chrom_start" />'
    '<Attribute name="hsapiens_paralog_chrom_end" />'
    '<Attribute name="hsapiens_paralog_subtype" />'
    '<Attribute name="hsapiens_paralog_orthology_type" />'
    '<Attribute name="hsapiens_paralog_perc_id" />'
    '<Attribute name="hsapiens_paralog_perc_id_r1" />'
    '</Dataset>'
    '</Query>'
)


class DataIngestionPipeline:
    """
    Pipeline ETL (Bronze -> Silver Parquet) per LncBook v2.0, OHNOLOGS v2 ed Ensembl BioMart.
    Ottimizzata per l'esecuzione su Google Colab con storage su Google Drive.
    """

    def __init__(self, paths: ProjectPaths = PATHS) -> None:
        self.paths = paths

    def _decompress_to_scratch(self, gz_path: Path) -> Path:
        """
        Decomprime un file .gz da Google Drive direttamente sull'SSD locale (/tmp)
        di Colab per evitare picchi di RAM durante il parsing CSV/GTF in Polars.
        """
        if not gz_path.exists():
            raise FileNotFoundError(f"File Bronze mancante su Google Drive: {gz_path}")

        if gz_path.suffix != ".gz":
            return gz_path

        scratch_target = self.paths.scratch / gz_path.stem
        logger.info(f"Decompressione su SSD locale (/tmp): {gz_path.name} -> {scratch_target.name}")
        with gzip.open(gz_path, "rb") as f_in, open(scratch_target, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
        return scratch_target

    def _download_biomart_tsv(self, xml_query: str, dest_path: Path) -> Path:
        """
        Scarica una tabella TSV da Ensembl BioMart provando in sequenza gli archivi
        HTTPS ufficiali (usando GET per evitare errori 405 sui redirect) e verifica
        che il server abbia restituito dati TSV validi invece di una pagina HTML.
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_download = self.paths.scratch / dest_path.name
        headers = {"User-Agent": "OhnomiTools/0.1.0 (https://github.com/ErrPeppe/OhnomiTools)"}
        errors: list[str] = []

        for url in BIOMART_ARCHIVE_URLS:
            logger.info(f"Tentativo download Ensembl BioMart da: {url}")
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
                        preview = head_text.splitlines()[0][:160]
                        raise ValueError(f"Risposta non-TSV dal server BioMart: {preview}")

                    with open(tmp_download, "wb") as f:
                        f.write(first_chunk)
                        for chunk in chunk_iter:
                            if chunk:
                                f.write(chunk)

                shutil.move(str(tmp_download), str(dest_path))
                logger.info(f"Download BioMart completato con successo da {url}")
                return dest_path

            except Exception as exc:
                if tmp_download.exists():
                    tmp_download.unlink()
                logger.warning(f"Archivio BioMart fallito ({url}): {exc}")
                errors.append(f"{url} -> {exc}")

        raise RuntimeError(
            "Impossibile scaricare i paraloghi da tutti gli archivi Ensembl BioMart:\n"
            + "\n".join(errors)
        )

    def build_lncbook_master_tables(self, force_rebuild: bool = False) -> tuple[Path, Path]:
        """
        Estrae dal GTF puro di LncBook v2.0 sia il catalogo dei trascritti sia
        quello aggregato a livello di gene (coordinate canoniche chr1..22, X, Y).
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
        Converte in streaming un CSV/TSV compresso di LncBook (Conservazione, miRNA, RBP)
        in formato Apache Parquet compresso (zstd) su Google Drive.
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

    def build_ohnologs_wgd_tables(self, force_rebuild: bool = False) -> tuple[Path, Path]:
        """
        Normalizza i file TSV di OHNOLOGS v2 (geni e coppie WGD) in formato Parquet.
        """
        out_genes = self.paths.silver_wgd_genes
        out_pairs = self.paths.silver_wgd_pairs

        if out_genes.exists() and out_pairs.exists() and not force_rebuild:
            logger.info("Tabelle OHNOLOGS v2 (WGD) già presenti in Silver. Skip.")
            return out_genes, out_pairs

        if self.paths.raw_ohnologs_genes.exists():
            df_genes = pl.read_csv(
                self.paths.raw_ohnologs_genes,
                separator="\t",
                ignore_errors=True,
            )
            self.paths.save_parquet_safe(df_genes, out_genes)

        if self.paths.raw_ohnologs_pairs.exists():
            df_pairs = pl.read_csv(
                self.paths.raw_ohnologs_pairs,
                separator="\t",
                ignore_errors=True,
            )
            self.paths.save_parquet_safe(df_pairs, out_pairs)

        return out_genes, out_pairs

    def fetch_and_build_ensembl_ssd_pairs(self, force_rebuild: bool = False) -> Path:
        """
        Scarica i paraloghi umani da Ensembl BioMart (se non già presenti in Bronze)
        e costruisce in Silver il dataset di controllo SSD (Small Scale Duplications),
        escludendo le coppie classificate come WGD in OHNOLOGS v2.
        """
        raw_tsv = self.paths.raw_ensembl_paralogues
        out_ssd = self.paths.silver_ssd_pairs

        if out_ssd.exists() and not force_rebuild:
            logger.info("Tabella Ensembl SSD già presente in Silver. Skip.")
            return out_ssd

        if not raw_tsv.exists() or force_rebuild:
            self._download_biomart_tsv(BIOMART_PARALOGUES_XML, raw_tsv)

        df_raw = pl.read_csv(raw_tsv, separator="\t", ignore_errors=True)
        col_map = {
            "Gene stable ID": "gene_id_1",
            "Gene name": "gene_name_1",
            "Chromosome/scaffold name": "chr_1",
            "Gene start (bp)": "start_1",
            "Gene end (bp)": "end_1",
            "Human paralogue gene stable ID": "gene_id_2",
            "Human paralogue associated gene name": "gene_name_2",
            "Human paralogue chromosome/scaffold name": "chr_2",
            "Human paralogue chromosome/scaffold start (bp)": "start_2",
            "Human paralogue chromosome/scaffold end (bp)": "end_2",
            "Human paralogue last common ancestor with Human": "ancestor_clade",
            "Human paralogue homology type": "homology_type",
            "Paralogue %id. target Human gene identical to query gene": "perc_id_1",
            "Paralogue %id. query gene identical to target Human gene": "perc_id_2",
        }
        existing_map = {k: v for k, v in col_map.items() if k in df_raw.columns}
        df_par = df_raw.rename(existing_map)

        # Filtra righe valide e ordina la coppia (A, B) in modo canonico (pair_key)
        df_par = (
            df_par.filter(
                pl.col("gene_id_1").is_not_null()
                & pl.col("gene_id_2").is_not_null()
                & (pl.col("gene_id_1") != pl.col("gene_id_2"))
            )
            .with_columns([
                pl.when(pl.col("gene_id_1") < pl.col("gene_id_2"))
                .then(pl.concat_str([pl.col("gene_id_1"), pl.lit("_"), pl.col("gene_id_2")]))
                .otherwise(pl.concat_str([pl.col("gene_id_2"), pl.lit("_"), pl.col("gene_id_1")]))
                .alias("pair_key")
            ])
            .unique(subset=["pair_key"])
        )

        # Sottrae le coppie WGD di OHNOLOGS v2 se già normalizzate con ID ENSG
        if self.paths.silver_wgd_pairs.exists():
            df_wgd = pl.read_parquet(self.paths.silver_wgd_pairs)
            id_cols = [c for c in df_wgd.columns if "ensg" in c.lower() or "gene_id" in c.lower()]
            if len(id_cols) >= 2:
                c1, c2 = id_cols[0], id_cols[1]
                wgd_keys = (
                    df_wgd.select([
                        pl.when(pl.col(c1) < pl.col(c2))
                        .then(pl.concat_str([pl.col(c1), pl.lit("_"), pl.col(c2)]))
                        .otherwise(pl.concat_str([pl.col(c2), pl.lit("_"), pl.col(c1)]))
                        .alias("pair_key")
                    ])
                    .unique()
                )
                df_par = df_par.join(wgd_keys, on="pair_key", how="anti")

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

        if self.paths.raw_ohnologs_pairs.exists() or self.paths.raw_ohnologs_genes.exists():
            wgd_g, wgd_p = self.build_ohnologs_wgd_tables(force_rebuild=force_rebuild)
            outputs["wgd_genes"] = wgd_g
            outputs["wgd_pairs"] = wgd_p

        outputs["ssd_pairs"] = self.fetch_and_build_ensembl_ssd_pairs(force_rebuild=force_rebuild)
        return outputs
