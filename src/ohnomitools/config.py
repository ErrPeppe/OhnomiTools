# src/ohnomitools/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import logging
import os
import shutil
import subprocess
import tempfile
import time
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("OhnomiTools")

DEFAULT_DRIVE_SUBPATH = Path("UniTo/Tesi/Data")
DEFAULT_RCLONE_MOUNT = Path("/content/rclone_drive")
DEFAULT_RCLONE_REMOTE = "gdrive"
DEFAULT_RCLONE_SECRET = "RCLONE_CONF"
GITHUB_INSTALL_URL = "git+https://github.com/ErrPeppe/OhnomiTools.git"


def resolve_default_data_root() -> Path:
    env_dir = os.environ.get("OHNOMI_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    if os.path.ismount(DEFAULT_RCLONE_MOUNT):
        return DEFAULT_RCLONE_MOUNT

    colab_drive_mount = Path("/content/drive/MyDrive")
    if colab_drive_mount.exists():
        return colab_drive_mount / DEFAULT_DRIVE_SUBPATH

    return Path(__file__).resolve().parent.parent.parent / "data"


def _mount_via_rclone_secret(
    secret_name: str,
    remote_name: str,
    drive_subpath: str | Path,
    mount_dir: Path = DEFAULT_RCLONE_MOUNT,
) -> Path | None:
    try:
        from google.colab import userdata  # type: ignore
        rclone_conf = userdata.get(secret_name)
    except Exception:
        return None

    if not rclone_conf or not rclone_conf.strip():
        return None

    if os.path.ismount(mount_dir):
        logger.info(f"Cartella Drive già montata via rclone in {mount_dir}")
        return mount_dir

    if shutil.which("rclone") is None or (
        shutil.which("fusermount3") is None and shutil.which("fusermount") is None
    ):
        logger.info("Installazione di rclone e fuse3 sul runtime Colab...")
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "rclone", "fuse3"], check=True)

    conf_path = Path("/root/.config/rclone/rclone.conf")
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(rclone_conf.strip() + "\n", encoding="utf-8")
    os.chmod(conf_path, 0o600)

    subpath_str = Path(drive_subpath).as_posix().strip("/")
    remote_target = f"{remote_name}:{subpath_str}"

    mkdir_proc = subprocess.run(
        ["rclone", "mkdir", remote_target, "--config", str(conf_path)],
        capture_output=True,
        text=True,
    )
    if mkdir_proc.returncode != 0:
        logger.warning(
            f"Impossibile accedere a '{remote_target}' con rclone: {mkdir_proc.stderr.strip()}"
        )
        return None

    mount_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.gettempdir()) / "rclone_vfs_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    mount_cmd = [
        "rclone",
        "mount",
        remote_target,
        str(mount_dir),
        "--config", str(conf_path),
        "--vfs-cache-mode", "writes",
        "--vfs-write-back", "1s",
        "--cache-dir", str(cache_dir),
        "--daemon",
    ]
    mount_proc = subprocess.run(mount_cmd, capture_output=True, text=True)
    if mount_proc.returncode != 0:
        logger.warning(f"Mount rclone fallito: {mount_proc.stderr.strip()}")
        return None

    for _ in range(40):
        if os.path.ismount(mount_dir):
            logger.info(f"Drive montato senza popup via rclone: {remote_target} -> {mount_dir}")
            return mount_dir
        time.sleep(0.25)

    logger.warning("Timeout attesa mount rclone; passo al fallback drive.mount().")
    return None


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
    def raw_lncbook_fasta(self) -> Path:
        return self.bronze / "lncRNA_LncBookv2.0.fa.gz"

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
    def raw_lncbook_sprotein(self) -> Path:
        return self.bronze / "sprotein_LncBook2.0.csv.gz"

    @property
    def raw_lncbook_methylation(self) -> Path:
        return self.bronze / "methylation_LncBook2.0.csv.gz"

    @property
    def raw_ohnologs_human_list(self) -> Path:
        return self.bronze / "ohnologs" / "homo_sapiens.tsv"

    @property
    def raw_ohnologs_genes(self) -> Path:
        return self.bronze / "database" / "genes.tsv"

    @property
    def raw_ohnologs_pairs(self) -> Path:
        return self.bronze / "database" / "gene_ohnology.tsv"

    @property
    def raw_ensembl_genes(self) -> Path:
        return self.bronze / "ensembl_genes.tsv"

    @property
    def raw_ensembl_paralogues(self) -> Path:
        return self.bronze / "ensembl_paralogues.tsv"

    @property
    def raw_string_info(self) -> Path:
        return self.bronze / "string" / "9606.protein.info.v12.0.txt.gz"

    @property
    def raw_string_links(self) -> Path:
        return self.bronze / "string" / "9606.protein.links.v12.0.txt.gz"

    # --- File Silver Fase 1 (Parquet normalizzati) ---
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
    def silver_lncbook_expr(self) -> Path:
        return self.silver / "lncbook_expression.parquet"

    @property
    def silver_lncbook_sprotein(self) -> Path:
        return self.silver / "lncbook_sprotein.parquet"

    @property
    def silver_lncbook_methylation(self) -> Path:
        return self.silver / "lncbook_methylation.parquet"

    @property
    def silver_ensembl_pc_genes(self) -> Path:
        return self.silver / "ensembl_protein_coding_genes.parquet"

    @property
    def silver_wgd_genes(self) -> Path:
        return self.silver / "ohnologs_wgd_genes.parquet"

    @property
    def silver_wgd_pairs(self) -> Path:
        return self.silver / "ohnologs_wgd_pairs.parquet"

    @property
    def silver_ssd_pairs(self) -> Path:
        return self.silver / "ensembl_ssd_pairs.parquet"

    @property
    def silver_string_ppi(self) -> Path:
        return self.silver / "string_ppi_edges.parquet"

    # --- File Silver Fase 2 (Coorti delle 3 Vie) ---
    @property
    def silver_phylostratigraphy(self) -> Path:
        return self.silver / "lncbook_gene_phylostratigraphy.parquet"

    @property
    def silver_host_lncrna_map(self) -> Path:
        return self.silver / "intragenic_host_lncrna_map.parquet"

    @property
    def silver_lane1_wgd_pairs(self) -> Path:
        return self.silver / "wgd_intragenic_lncrna_pairs.parquet"

    @property
    def silver_lane1_ssd_pairs(self) -> Path:
        return self.silver / "ssd_intragenic_lncrna_pairs.parquet"

    @property
    def silver_lane2_wgd_pairs(self) -> Path:
        return self.silver / "lane2_ab_wgd_pairs.parquet"

    @property
    def silver_lane2_ssd_pairs(self) -> Path:
        return self.silver / "lane2_ab_ssd_pairs.parquet"

    @property
    def silver_lane3_euteleostomi(self) -> Path:
        return self.silver / "lane3_ramoB_euteleostomi_cohort.parquet"

    def save_parquet_safe(self, df: pl.DataFrame, target_drive_path: Path) -> Path:
        target_drive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.scratch / target_drive_path.name
        df.write_parquet(tmp_file, compression="zstd")
        shutil.move(str(tmp_file), str(target_drive_path))
        logger.info(f"Salvato: {target_drive_path.name} ({len(df):,} righe)")
        return target_drive_path

    def status_table(self) -> pl.DataFrame:
        checks = [
            ("Bronze", "LncBook v2.0 GTF", self.raw_lncbook_gtf),
            ("Bronze", "LncBook v2.0 FASTA", self.raw_lncbook_fasta),
            ("Bronze", "LncBook v2.0 Conservation", self.raw_lncbook_cons),
            ("Bronze", "LncBook v2.0 miRNA Interactions", self.raw_lncbook_mirna_int),
            ("Bronze", "LncBook v2.0 RBP Interactions", self.raw_lncbook_rbp_int),
            ("Bronze", "LncBook v2.0 Expression", self.raw_lncbook_expr),
            ("Bronze", "LncBook v2.0 Small Protein", self.raw_lncbook_sprotein),
            ("Bronze", "LncBook v2.0 Methylation", self.raw_lncbook_methylation),
            ("Bronze", "OHNOLOGS Human List", self.raw_ohnologs_human_list),
            ("Bronze", "OHNOLOGS Genes DB", self.raw_ohnologs_genes),
            ("Bronze", "OHNOLOGS Pairs DB", self.raw_ohnologs_pairs),
            ("Bronze", "Ensembl Genes (BioMart)", self.raw_ensembl_genes),
            ("Bronze", "Ensembl Paralogues (BioMart)", self.raw_ensembl_paralogues),
            ("Bronze", "STRING v12.0 PPI Links", self.raw_string_links),
            ("Silver-1", "Transcripts Parquet", self.silver_lncrna_transcripts),
            ("Silver-1", "Genes Parquet", self.silver_lncrna_genes),
            ("Silver-1", "Conservation Parquet", self.silver_lncbook_cons),
            ("Silver-1", "miRNA Interactions Parquet", self.silver_lncbook_mirna_int),
            ("Silver-1", "RBP Interactions Parquet", self.silver_lncbook_rbp_int),
            ("Silver-1", "Expression Parquet", self.silver_lncbook_expr),
            ("Silver-1", "Small Protein Parquet", self.silver_lncbook_sprotein),
            ("Silver-1", "Methylation Parquet", self.silver_lncbook_methylation),
            ("Silver-1", "Ensembl Protein-Coding Parquet", self.silver_ensembl_pc_genes),
            ("Silver-1", "WGD Genes Parquet", self.silver_wgd_genes),
            ("Silver-1", "WGD Pairs Parquet", self.silver_wgd_pairs),
            ("Silver-1", "SSD Pairs Parquet", self.silver_ssd_pairs),
            ("Silver-1", "STRING PPI Parquet", self.silver_string_ppi),
            ("Silver-2", "Via 1 WGD Pairs (Ramo A)", self.silver_lane1_wgd_pairs),
            ("Silver-2", "Via 2 WGD Pairs (A ∩ B)", self.silver_lane2_wgd_pairs),
            ("Silver-2", "Via 3 Cohort (Ramo B Eut.)", self.silver_lane3_euteleostomi),
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

    def print_status(self) -> None:
        with pl.Config(tbl_rows=40, tbl_cols=10, fmt_str_lengths=65):
            print(self.status_table())


def setup_colab(
    drive_subpath: str | Path = DEFAULT_DRIVE_SUBPATH,
    mount_point: str = "/content/drive",
    rclone_secret: str | None = DEFAULT_RCLONE_SECRET,
    rclone_remote: str = DEFAULT_RCLONE_REMOTE,
    update: bool = False,
    show_status: bool = True,
) -> ProjectPaths:
    if update:
        logger.info("Aggiornamento di OhnomiTools da GitHub (no-cache) in corso...")
        subprocess.run(
            [
                "pip", "install", "-q", "--no-cache-dir",
                "--no-deps", "--force-reinstall", GITHUB_INSTALL_URL,
            ],
            check=True,
        )

    try:
        from google.colab import drive  # type: ignore

        mounted_rclone: Path | None = None
        if rclone_secret:
            mounted_rclone = _mount_via_rclone_secret(
                secret_name=rclone_secret,
                remote_name=rclone_remote,
                drive_subpath=drive_subpath,
            )

        if mounted_rclone is not None:
            os.environ["OHNOMI_DATA_DIR"] = str(mounted_rclone)
        else:
            mount_path = Path(mount_point)
            if not (mount_path / "MyDrive").exists():
                logger.info("Secret rclone non trovato: montaggio classico di Google Drive...")
                drive.mount(mount_point)
            data_root = mount_path / "MyDrive" / Path(drive_subpath)
            os.environ["OHNOMI_DATA_DIR"] = str(data_root)
    except ImportError:
        logger.info("Esecuzione fuori da Google Colab: utilizzo storage locale.")

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
        paths.print_status()

    return paths


PATHS = ProjectPaths()

CANONICAL_CHROMOSOMES: tuple[str, ...] = tuple(
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
)
