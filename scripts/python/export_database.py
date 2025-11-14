import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SECRET = "CLkZf9fxk8p8zXWefb2wiXzmipKxps89MTHQjkZL"


def check_sudo_access() -> None:
    """Check if user has recently validated sudo access"""
    try:
        # -n: non-interactive, will fail if password is required
        # -v: validate sudo timestamp
        subprocess.run(["sudo", "-n", "-v"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Error: Please run 'sudo -v' first to validate your sudo access.", file=sys.stderr)
        print("Your sudo password may be required.", file=sys.stderr)
        sys.exit(1)


def get_postgres_info() -> tuple[str, str, str]:
    """Get postgres container, username and database from docker-compose.yml"""
    # Get container ID
    container_cmd = ["sudo", "docker", "compose", "ps"]
    container_output = subprocess.check_output(container_cmd).decode()
    container_id = None
    for line in container_output.splitlines():
        if "postgres" in line:
            container_id = line.split()[0]
            break
    if not container_id:
        raise RuntimeError("Postgres container not running!")

    # Get username and database from docker-compose.yml
    compose_file = Path("./docker-compose.yml")
    if not compose_file.exists():
        raise RuntimeError("docker-compose.yml not found")

    compose_content = compose_file.read_text()
    username = None
    database = None
    for line in compose_content.splitlines():
        if "POSTGRES_USER" in line:
            username = line.split(":", 1)[1].strip()
        elif "POSTGRES_DB" in line:
            database = line.split(":", 1)[1].strip()

    if not username or not database:
        raise RuntimeError("Could not find postgres username or database in docker-compose.yml")

    return container_id, username, database


def get_backup_filepath() -> Path:
    """Create backup directory and return timestamped backup filepath"""
    backup_dir = Path("./backups")
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    return backup_dir / f"dump_{timestamp}.sql.gz"


def export_database(outfile: Path) -> None:
    """Export database and compress using gzip"""
    container_id, username, database = get_postgres_info()

    # Run pg_dump in container and pipe to gzip
    dump_cmd = [
        "sudo",
        "docker",
        "exec",
        "-t",
        container_id,
        "pg_dump",
        "-d",
        database,
        "-U",
        username,
    ]

    gzip_cmd = ["gzip"]

    # Pipe pg_dump output through gzip
    with open(outfile, "wb") as f:
        dump_process = subprocess.Popen(dump_cmd, stdout=subprocess.PIPE)
        gzip_process = subprocess.Popen(gzip_cmd, stdin=dump_process.stdout, stdout=f)

        # Allow dump_process to receive a SIGPIPE if gzip_process exits
        if dump_process.stdout:
            dump_process.stdout.close()

        # Wait for both processes to complete
        gzip_process.communicate()
        dump_process.wait()

        if dump_process.returncode != 0 or gzip_process.returncode != 0:
            print("Error: Database export failed", file=sys.stderr)
            sys.exit(1)

    print(f"Created {outfile}")

    # Copy to latest.sql.gz
    latest_file = Path("./backups/latest.sql.gz")
    shutil.copy2(outfile, latest_file)
    print(f"Copied to {latest_file}")


def format_size(size_bytes: float) -> str:
    """Convert size in bytes to human readable string"""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def upload_to_r2() -> None:
    """Upload latest.sql.gz to R2 using rclone"""
    latest_file = Path("./backups/latest.sql.gz")
    r2_path = f"r2:/resources/{SECRET}"

    # Get and display file size
    file_size = latest_file.stat().st_size
    print(f"File size: {format_size(file_size)}")

    cmd = ["rclone", "copy", str(latest_file), r2_path]

    try:
        subprocess.run(cmd, check=True)
        print(f"Uploaded {latest_file} to {r2_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error uploading to R2: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    # Check sudo access before proceeding
    check_sudo_access()

    parser = argparse.ArgumentParser(description="Export database")
    parser.add_argument(
        "outfile",
        type=Path,
        nargs="?",
        default=None,
        help="Path to output SQL file (default: ./backups/dump_TIMESTAMP.sql.gz)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload latest.sql.gz to R2 after creation",
    )
    args = parser.parse_args()

    outfile = args.outfile if args.outfile else get_backup_filepath()
    export_database(outfile)

    if args.upload:
        print("Uploading to R2...")
        upload_to_r2()


if __name__ == "__main__":
    main()
