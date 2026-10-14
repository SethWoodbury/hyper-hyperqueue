#!/usr/bin/env bash
#SBATCH --job-name=hhq-dashboard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=hhq-dashboard-%j.log

# Supply the site's CPU partition/account and a private scratch log path to sbatch.
# 512 MiB is an initial declared request; measure a representative state directory.
set -euo pipefail
umask 077

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "Usage: sbatch [site options] dashboard.sbatch /absolute/bin/hhq /absolute/config.yaml [port]" >&2
    exit 2
fi

hhq_dashboard_bin="$1"
hhq_dashboard_config="$2"
hhq_dashboard_port="${3:-8765}"

if [[ "$hhq_dashboard_bin" != /* || ! -x "$hhq_dashboard_bin" ]]; then
    echo "The HHQ executable must be an existing absolute path; install before submission." >&2
    exit 2
fi
if [[ "$hhq_dashboard_config" != /* || ! -r "$hhq_dashboard_config" ]]; then
    echo "The selected configuration must be a readable absolute path on shared storage." >&2
    exit 2
fi
if [[ ! "$hhq_dashboard_port" =~ ^[0-9]{1,5}$ ]] ||
    (( 10#$hhq_dashboard_port < 1024 || 10#$hhq_dashboard_port > 65535 )); then
    echo "Port must be an integer from 1024 through 65535." >&2
    exit 2
fi
hhq_dashboard_port="$((10#$hhq_dashboard_port))"

printf 'Dashboard host: %s\n' "$(hostname -f)"
printf 'Dashboard port: %s (loopback only)\n' "$hhq_dashboard_port"
printf 'Selected configuration: %s\n' "$hhq_dashboard_config"
echo "Use SSH through your login host to this compute host; see docs/dashboard.md."
echo "This allocation hosts the viewer only. It does not start an HHQ controller or worker."

exec "$hhq_dashboard_bin" --config "$hhq_dashboard_config" ui \
    --host 127.0.0.1 --port "$hhq_dashboard_port" --read-only
