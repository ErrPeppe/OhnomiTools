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

BIOMART_ARCHIVE_URLS: tuple[str, ...] = (
    "https://e115.ensembl.org/biomart/martservice",
    "https://sep2025.archive.ensembl.org/biomart/martservice",
    "https://jun2026.archive.ensembl.org/biomart/martservice",
    "https://may2025.archive.ensembl.org/biomart/martservice",
)

STRING_INFO_URL = (
    "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
)
STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
)

BIOMART_CANONICAL_CHRS = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y"

BIOMART_GENES_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Filter name="biotype" value="protein_coding" />'
    f'<Filter name="chromosome_name" value="{BIOMART_CANONICAL_CHRS}" />'
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

BIOMART_PARALOGUES_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">'
    '<Dataset name="hsapiens_gene_ensembl" interface="default">'
    '<Filter name="biotype" value="protein_coding" />'
    f'<Filter name="chromosome_name" value="{BIOMART_CANONICAL_CHRS}" />'
    '<Attribute name="ensembl_gene_id" />'
    '<Attribute name="hsapiens_paralog_ensembl_gene" />'
    '<Attribute name="hsapiens_paralog_subtype" />'
    '<Attribute name="hsapiens_paralog_orthology_type" />'
    '<Attribute name="hsapiens_paralog_perc_id" />'
    '<Attribute name="hsapiens_paralog_perc_id_r1" />'
    '</Dataset>'
    '</Query>'
)

LNCBOOK_MIRNA_COLUMNS: list[str] = [
    "gene_id",
    "symbol",
    "unused_col3",
    "transcript_id",
    "start",
    "end",
    "mfe",
    "tools",
    "unused_col9",
    "interaction_id",
    "mirna_id",
]


class DataIngestionPipeline:
    """
    Pipeline ETL (Bronze -> Silver Parquet) per LncBook v2.0, OHNOLOGS, Ensembl BioMart e STRING v12.0.
    """

    def __init__(self, paths: ProjectPaths = PATHS) -> None:
        self.paths = paths

    def _decompress_to_scratch(self, gz_path: Path) -> Path:
        if not gz_path.exists():
            raise FileNotFoundError(f"File Bronze mancante su Google Drive: {gz_path}")
        if gz_path.suffix != ".gz":
            return gz_path
        scratch_target = self.paths.scratch / gz_path.stem
        logger.info(f"Decompressione su /tmp: {gz_path.name} -> {scratch_target.name}")
        with gzip.open(gz_path, "rb") as f_in, open(scratch_target, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
        return scratch_target

    def _download_file(self, url: str, dest_path: Path) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.paths.scratch / dest_path.name
        headers = {"User-Agent": "OhnomiTools/0.1.0 (https://github.com/ErrPeppe/OhnomiTools)"}
        logger.info(f"Download {dest_path.name} da {url}...")
        with requests.get(url, headers=headers, stream=True, timeout=(20, 300)) as r:
            r.raise_for_status()
            written = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        written += len(chunk)
        shutil.move(str(tmp_path), str(dest_path))
        logger.info(f"Download completato: {dest_path.name} ({written / (1024 * 1024):.2f} MB)")
        return dest_path

    def _download_biomart_tsv(self, xml_query: str, dest_path: Path) -> Path:
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
                    timeout=(15, 120),
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

                    bytes_written = 0
                    with open(tmp_download, "wb") as f:
                        f.write(first_chunk)
                        bytes_written += len(first_chunk)
                        for chunk in chunk_iter:
                            if chunk:
                                f.write(chunk)
                                bytes_written += len(chunk)

                shutil.move(str(tmp_download), str(dest_path))
                logger.info(
                    f"Download completato: {dest_path.name} "
                    f"({bytes_written / (1024 * 1024):.2f} MB)"
                )
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
        has_header: bool = True,
        new_columns: list[str] | None = None,
        force_rebuild: bool = False,
    ) -> Path:
        if silver_parquet_path.exists() and not force_rebuild:
            logger.info(f"Parquet già presente in Silver: {silver_parquet_path.name}. Skip.")
            return silver_parquet_path

        tmp_csv = self._decompress_to_scratch(raw_gz_path)
        tmp_parquet = self.paths.scratch / silver_parquet_path.name
        try:
            scan_kwargs: dict = {
                "separator": separator,
                "has_header": has_header,
                "infer_schema_length": 10000,
                "ignore_errors": True,
            }
            if new_columns is not None:
                scan_kwargs["new_columns"] = new_columns

            lf = pl.scan_csv(tmp_csv, **scan_kwargs)
            lf.sink_parquet(tmp_parquet, compression="zstd")
            silver_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_parquet), str(silver_parquet_path))
            logger.info(f"Convertito in Silver Parquet: {silver_parquet_path.name}")
            return silver_parquet_path
        finally:
            if tmp_csv != raw_gz_path and tmp_csv.exists():
                tmp_csv.unlink()

    def build_lncbook_mirna_interactions(self, force_rebuild: bool = False) -> Path:
        """
        Converte `lncrna_mirna_miRandaAndTargetScanAndRNAhybrid_LncBook2.0.csv.gz`
        (che è privo di intestazione) assegnando esplicitamente i nomi delle 11 colonne.
        """
        return self.convert_csv_dataset_to_silver(
            self.paths.raw_lncbook_mirna_int,
            self.paths.silver_lncbook_mirna_int,
            separator=",",
            has_header=False,
            new_columns=LNCBOOK_MIRNA_COLUMNS,
            force_rebuild=force_rebuild,
        )

    def fetch_and_build_ensembl_pc_genes(self, force_rebuild: bool = False) -> Path:
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
        out_genes = self.paths.silver_wgd_genes
        out_pairs = self.paths.silver_wgd_pairs

        if out_genes.exists() and out_pairs.exists() and not force_rebuild:
            logger.info("Tabelle OHNOLOGS (WGD) già presenti in Silver. Skip.")
            return out_genes, out_pairs

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
                low_memory=True,
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
            .with_columns(pl.concat_str([pl.lit("chr"), pl.col("scaffold_id")]).alias("chr"))
            .filter(pl.col("chr").is_in(CANONICAL_CHROMOSOMES))
            .select([
                "gene_id", "protein_id", "chr", "start", "end",
                "family_id", "segment_id", "pvc", "pgc"
            ])
            .collect()
        )

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

        if self.paths.raw_ohnologs_pairs.exists():
            prot_map = df_human_all.join(df_hs_ohno, on="protein_id", how="left").select([
                "protein_id", "gene_id", "gene_name", "chr", "start", "end",
                "strand", "family_id", "retention_category"
            ]).unique(subset=["protein_id"])

            lf_raw_pairs = pl.scan_csv(
                self.paths.raw_ohnologs_pairs,
                separator="\t",
                has_header=False,
                low_memory=True,
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
        raw_tsv = self.paths.raw_ensembl_paralogues
        out_ssd = self.paths.silver_ssd_pairs

        if out_ssd.exists() and not force_rebuild:
            logger.info("Tabella Ensembl SSD già presente in Silver. Skip.")
            return out_ssd

        if not self.paths.silver_ensembl_pc_genes.exists():
            self.fetch_and_build_ensembl_pc_genes(force_rebuild=False)

        if not raw_tsv.exists() or force_rebuild:
            self._download_biomart_tsv(BIOMART_PARALOGUES_XML, raw_tsv)

        col_map = {
            "Gene stable ID": "gene_id_1",
            "Human paralogue gene stable ID": "gene_id_2",
            "Human paralogue last common ancestor with Human": "ancestor_clade",
            "Human paralogue homology type": "homology_type",
            "Paralogue %id. target Human gene identical to query gene": "perc_id_1",
            "Paralogue %id. query gene identical to target Human gene": "perc_id_2",
        }

        lf_raw = pl.scan_csv(
            raw_tsv,
            separator="\t",
            low_memory=True,
            infer_schema_length=10000,
            ignore_errors=True,
            schema_overrides={
                "Gene stable ID": pl.Utf8,
                "Human paralogue gene stable ID": pl.Utf8,
            },
        )
        existing_cols = lf_raw.collect_schema().names()
        rename_dict = {k: v for k, v in col_map.items() if k in existing_cols}

        lf_pc = pl.scan_parquet(self.paths.silver_ensembl_pc_genes).select([
            "gene_id", "gene_name", "chr", "start", "end", "strand"
        ])
        pc1 = lf_pc.rename({c: f"{c}_1" for c in ["gene_id", "gene_name", "chr", "start", "end", "strand"]})
        pc2 = lf_pc.rename({c: f"{c}_2" for c in ["gene_id", "gene_name", "chr", "start", "end", "strand"]})

        df_par = (
            lf_raw.rename(rename_dict)
            .filter(
                pl.col("gene_id_1").is_not_null()
                & pl.col("gene_id_2").is_not_null()
                & (pl.col("gene_id_1") != "")
                & (pl.col("gene_id_2") != "")
                & (pl.col("gene_id_1") != pl.col("gene_id_2"))
            )
            .with_columns(
                pl.when(pl.col("gene_id_1") < pl.col("gene_id_2"))
                .then(pl.concat_str([pl.col("gene_id_1"), pl.lit("_"), pl.col("gene_id_2")]))
                .otherwise(pl.concat_str([pl.col("gene_id_2"), pl.lit("_"), pl.col("gene_id_1")]))
                .alias("pair_key")
            )
            .unique(subset=["pair_key"])
            .join(pc1, on="gene_id_1", how="inner")
            .join(pc2, on="gene_id_2", how="inner")
            .collect()
        )

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

    def fetch_and_build_string_ppi(
        self,
        min_score: int = 700,
        force_rebuild: bool = False,
    ) -> Path:
        out_ppi = self.paths.silver_string_ppi
        if out_ppi.exists() and not force_rebuild:
            logger.info("Rete STRING PPI già presente in Silver. Skip.")
            return out_ppi

        info_gz = self.paths.raw_string_info
        links_gz = self.paths.raw_string_links

        if not info_gz.exists() or force_rebuild:
            self._download_file(STRING_INFO_URL, info_gz)
        if not links_gz.exists() or force_rebuild:
            self._download_file(STRING_LINKS_URL, links_gz)

        tmp_info = self._decompress_to_scratch(info_gz)
        tmp_links = self._decompress_to_scratch(links_gz)

        try:
            df_info = (
                pl.read_csv(
                    tmp_info,
                    separator="\t",
                    columns=[0, 1],
                    new_columns=["string_id", "preferred_name"],
                    schema_overrides={"string_id": pl.Utf8, "preferred_name": pl.Utf8},
                )
                .drop_nulls()
                .unique(subset=["string_id"])
            )

            lf_links = (
                pl.scan_csv(
                    tmp_links,
                    separator=" ",
                    low_memory=True,
                    schema_overrides={
                        "protein1": pl.Utf8,
                        "protein2": pl.Utf8,
                        "combined_score": pl.Int32,
                    },
                )
                .filter(pl.col("combined_score") >= min_score)
            )

            lf_i1 = df_info.lazy().rename({"string_id": "protein1", "preferred_name": "gene_1"})
            lf_i2 = df_info.lazy().rename({"string_id": "protein2", "preferred_name": "gene_2"})

            df_ppi = (
                lf_links
                .join(lf_i1, on="protein1", how="inner")
                .join(lf_i2, on="protein2", how="inner")
                .filter(pl.col("gene_1") != pl.col("gene_2"))
                .with_columns(
                    pl.when(pl.col("gene_1") < pl.col("gene_2"))
                    .then(pl.concat_str([pl.col("gene_1"), pl.lit("_"), pl.col("gene_2")]))
                    .otherwise(pl.concat_str([pl.col("gene_2"), pl.lit("_"), pl.col("gene_1")]))
                    .alias("pair_key")
                )
                .group_by("pair_key")
                .agg([
                    pl.min_horizontal("gene_1", "gene_2").first().alias("protein_1"),
                    pl.max_horizontal("gene_1", "gene_2").first().alias("protein_2"),
                    pl.col("combined_score").max().alias("combined_score"),
                ])
                .sort("combined_score", descending=True)
                .collect()
            )

            self.paths.save_parquet_safe(df_ppi, out_ppi)
            return out_ppi
        finally:
            if tmp_info.exists():
                tmp_info.unlink()
            if tmp_links.exists():
                tmp_links.unlink()

    def run_all(self, include_string: bool = True, force_rebuild: bool = False) -> dict[str, Path]:
        outputs: dict[str, Path] = {}

        if self.paths.raw_lncbook_gtf.exists():
            tx, gn = self.build_lncbook_master_tables(force_rebuild=force_rebuild)
            outputs["transcripts"] = tx
            outputs["genes"] = gn

        if self.paths.raw_lncbook_mirna_int.exists():
            outputs["mirna_interactions"] = self.build_lncbook_mirna_interactions(
                force_rebuild=force_rebuild
            )

        csv_jobs = [
            ("conservation", self.paths.raw_lncbook_cons, self.paths.silver_lncbook_cons),
            ("rbp_interactions", self.paths.raw_lncbook_rbp_int, self.paths.silver_lncbook_rbp_int),
            ("expression", self.paths.raw_lncbook_expr, self.paths.silver_lncbook_expr),
            ("sprotein", self.paths.raw_lncbook_sprotein, self.paths.silver_lncbook_sprotein),
            ("methylation", self.paths.raw_lncbook_methylation, self.paths.silver_lncbook_methylation),
        ]
        for key, raw_p, silver_p in csv_jobs:
            if raw_p.exists():
                outputs[key] = self.convert_csv_dataset_to_silver(
                    raw_p, silver_p, separator=",", force_rebuild=force_rebuild
                )

        outputs["ensembl_pc_genes"] = self.fetch_and_build_ensembl_pc_genes(
            force_rebuild=force_rebuild
        )

        if self.paths.raw_ohnologs_human_list.exists() and self.paths.raw_ohnologs_genes.exists():
            wgd_g, wgd_p = self.build_ohnologs_wgd_tables(force_rebuild=force_rebuild)
            outputs["wgd_genes"] = wgd_g
            outputs["wgd_pairs"] = wgd_p

        outputs["ssd_pairs"] = self.fetch_and_build_ensembl_ssd_pairs(
            force_rebuild=force_rebuild
        )

        if include_string:
            outputs["string_ppi"] = self.fetch_and_build_string_ppi(
                min_score=700, force_rebuild=force_rebuild
            )

        return outputs
