#!/usr/bin/env bash

set -euo pipefail

service_name="weatherlogger"
service_user="weatherlogger"
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$script_directory"

if [[ "$(id --user)" -ne 0 ]]; then
    printf '%s\n' 'Run this installer as root (for example: sudo ./install-systemd.sh).' >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    printf "%s\n" "uv is required but was not found in root's PATH." >&2
    printf '%s\n' 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi

uv_path="$(command -v uv)"
if [[ "$uv_path" != "/usr/local/bin/uv" ]]; then
    install --mode 0755 "$uv_path" /usr/local/bin/uv
fi

if ! id --user "$service_user" >/dev/null 2>&1; then
    useradd --system --create-home --home-dir "$project_directory" --shell /usr/sbin/nologin "$service_user"
fi

if getent group video >/dev/null 2>&1; then
    usermod --append --groups video "$service_user"
fi

install --directory --owner "$service_user" --group "$service_user" "$project_directory"
/usr/local/bin/uv sync --locked --project "$project_directory"
chown --recursive "$service_user:$service_user" "$project_directory/.venv"

sed "s|/opt/weatherlogger|$project_directory|g" \
    "${script_directory}/systemd/${service_name}.service" \
    > "/etc/systemd/system/${service_name}.service"
chmod 0644 "/etc/systemd/system/${service_name}.service"
systemctl daemon-reload
systemctl enable --now "${service_name}.service"

printf 'Installed and started %s.service.\n' "$service_name"