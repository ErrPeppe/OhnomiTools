# OhnomiTools/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import logging
import os
import shutil
import subprocess
import tempfile
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("OhnomiTools")

DEFAULT_DRIVE_SUBPATH = Path("UniTo/Tesi/Data")
DEFAULT_COLAB_REPO_DIR = Path("/content/OhnomiTools")


def resolve_default_data_root() -> Path:
    """
    Ordine di priorità per la cartella dati:
    1. Variabile d'ambiente `OHNOMI_DATA_DIR` (impostata da setup_colab o dall'utente)
    2. Google Drive montato su Colab (`/content/drive/MyDrive/UniTo/Tesi/Data`)
    3. Cartella locale `<repo_root>/data` (se eseguito su PC/workstation locale)
    """
    env_dir = os.environ.get("OHNOMI_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    colab_drive_mount = Path("/content/drive/MyDrive")
    if colab_drive_mount.exists():
        return colab_drive_mount / DEFAULT_DRIVE_SUBPATH

    return Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path = field(default_factory=resolve_default_data_root)
    scratch_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "ohnomi_scratch"
    )

    @property
    def bronze(self) -> Path:
        path = self.data_root / "bronze"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def silver(self) -> Path:
        path = self.data_root / "silver"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def gold(self) -> Path:
        path = self.data_root / "gold"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def scratch(self) -> Path:
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        return self.scratch_dir

    # --- File Bronze (Database sorgente grezzi su Google Drive) ---
    @property
    def raw_lncbook_gtf(self) -> Path:
        return self.bronze / "lncRNA_LncBookv2.0_GRCh38.gtf.gz"

    @property
    def raw_lncbook_cons(self) -> Path:
        return self.bronze / "conservation_LncBook2.0.csv.gz"

    @property
    def raw_lncbook_mirna_int(self) -> Path:
        return self.bronze / "lncrna_mirna_miRandaAndTargetScanAndRNAhybrid_LncBook2.0.csv.gz"

    @property
    def raw_lncbook_rbp_int(self) -> Path:
        return self.bronze / "lncrna_rbp_LncBook2.0.csv.gz"

    @property
    def raw_lncbook_expr(self) -> Path:
        return self.bronze / "expression_LncBook2.0.csv.gz"

    @property
    def raw_ohnologs_pairs(self) -> Path:
        return self.bronze / "ohnologs" / "homo_sapiens.tsv"

    @property
    def raw_ohnologs_genes(self) -> Path:
        return self.bronze / "database" / "genes.tsv"

    @property
    def raw_ensembl_paralogues(self) -> Path:
        return self.bronze / "ensembl_paralogues.tsv"

    # --- File Silver (Parquet normalizzati su Google Drive) ---
    @property
    def silver_lncrna_transcripts(self) -> Path:
        return self.silver / "lncbook_transcripts.parquet"

    @property
    def silver_lncrna_genes(self) -> Path:
        return self.silver / "lncbook_genes.parquet"

    @property
    def silver_lncbook_cons(self) -> Path:
        return self.silver / "lncbook_conservation.parquet"

    @property
    def silver_lncbook_mirna_int(self) -> Path:
        return self.silver / "lncbook_mirna_interactions.parquet"

    @property
    def silver_lncbook_rbp_int(self) -> Path:
        return self.silver / "lncbook_rbp_interactions.parquet"

    @property
    def silver_wgd_genes(self) -> Path:
        return self.silver / "ohnologs_wgd_genes.parquet"

    @property
    def silver_wgd_pairs(self) -> Path:
        return self.silver / "ohnologs_wgd_pairs.parquet"

    @property
    def silver_ssd_pairs(self) -> Path:
        return self.silver / "ensembl_ssd_pairs.parquet"

    def save_parquet_safe(self, df: pl.DataFrame, target_drive_path: Path) -> Path:
        """
        Scrive il DataFrame Polars sull'SSD locale di Colab (/tmp) compresso in zstd
        e poi sposta atomicamente il file nella cartella di Google Drive.
        """
        target_drive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.scratch / target_drive_path.name
        df.write_parquet(tmp_file, compression="zstd")
        shutil.move(str(tmp_file), str(target_drive_path))
        logger.info(f"Salvato: {target_drive_path.name} ({len(df):,} righe)")
        return target_drive_path

    def status_table(self) -> pl.DataFrame:
        """Restituisce lo stato di presenza e dimensione (MB) dei dataset Bronze e Silver."""
        checks = [
            ("Bronze", "LncBook v2.0 GTF", self.raw_lncbook_gtf),
            ("Bronze", "LncBook v2.0 Conservation", self.raw_lncbook_cons),
            ("Bronze", "LncBook v2.0 miRNA Interactions", self.raw_lncbook_mirna_int),
            ("Bronze", "LncBook v2.0 RBP Interactions", self.raw_lncbook_rbp_int),
            ("Bronze", "LncBook v2.0 Expression", self.raw_lncbook_expr),
            ("Bronze", "OHNOLOGS v2 Pairs (WGD)", self.raw_ohnologs_pairs),
            ("Bronze", "OHNOLOGS v2 Genes", self.raw_ohnologs_genes),
            ("Bronze", "Ensembl Paralogues (SSD)", self.raw_ensembl_paralogues),
            ("Silver", "Transcripts Parquet", self.silver_lncrna_transcripts),
            ("Silver", "Genes Parquet", self.silver_lncrna_genes),
            ("Silver", "Conservation Parquet", self.silver_lncbook_cons),
            ("Silver", "miRNA Interactions Parquet", self.silver_lncbook_mirna_int),
            ("Silver", "RBP Interactions Parquet", self.silver_lncbook_rbp_int),
            ("Silver", "WGD Genes Parquet", self.silver_wgd_genes),
            ("Silver", "WGD Pairs Parquet", self.silver_wgd_pairs),
            ("Silver", "SSD Pairs Parquet", self.silver_ssd_pairs),
        ]
        rows = []
        for layer, label, path in checks:
            exists = path.exists()
            size_mb = round(path.stat().st_size / (1024 * 1024), 2) if exists else 0.0
            rows.append({
                "Layer": layer,
                "Dataset": label,
                "Presente": "SI" if exists else "NO",
                "Size_MB": size_mb,
                "Path": str(path.relative_to(self.data_root)),
            })
        return pl.DataFrame(rows)


def setup_colab(
    drive_subpath: str | Path = DEFAULT_DRIVE_SUBPATH,
    mount_point: str = "/content/drive",
    auto_pull_repo: str | Path | None = DEFAULT_COLAB_REPO_DIR,
    show_status: bool = True,
) -> ProjectPaths:
    """
    Inizializza l'ambiente OhnomiTools su Google Colab o in locale:
    1. Monta Google Drive e punta a `/content/drive/MyDrive/UniTo/Tesi/Data` (o `drive_subpath`).
    2. Esegue `git pull` sul clone locale della libreria `/content/OhnomiTools` se presente.
    3. Attiva `%autoreload 2` nel kernel Jupyter/Colab.
    4. Crea le directory Medallion (`bronze/`, `silver/`, `gold/`) e verifica i dataset.
    """
    try:
        from google.colab import drive  # type: ignore
        mount_path = Path(mount_point)
        if not (mount_path / "MyDrive").exists():
            logger.info("Montaggio di Google Drive in corso...")
            drive.mount(mount_point)
        data_root = mount_path / "MyDrive" / Path(drive_subpath)
        os.environ["OHNOMI_DATA_DIR"] = str(data_root)
    except ImportError:
        logger.info("Esecuzione fuori da Google Colab: utilizzo storage locale.")

    if auto_pull_repo and Path(auto_pull_repo).exists():
        subprocess.run(["git", "-C", str(auto_pull_repo), "pull", "-q"], check=False)

    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("load_ext", "autoreload")
            ip.run_line_magic("autoreload", "2")
    except Exception:
        pass

    paths = ProjectPaths(data_root=resolve_default_data_root())
    _ = (paths.bronze, paths.silver, paths.gold, paths.scratch)

    logger.info(f"OhnomiTools Data Root attiva: {paths.data_root}")
    if show_status:
        print(paths.status_table())

    return paths


PATHS = ProjectPaths()

CANONICAL_CHROMOSOMES: tuple[str, ...] = tuple(
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
)
