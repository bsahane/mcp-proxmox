# MCP Proxmox Server

Advanced Proxmox Model Context Protocol (MCP) server in Python exposing rich Proxmox utilities for discovery, lifecycle, networking, snapshots/backups, metrics, pools/permissions, and orchestration.

- Guide reference: [MCP Quickstart (Python)](https://modelcontextprotocol.io/quickstart/server#python)
- Structure mirrors: [`bsahane/mcp-ansible`](https://github.com/bsahane/mcp-ansible/tree/main)

## Quick start

```bash
git clone https://github.com/bsahane/mcp-proxmox.git
cd mcp-proxmox

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

# (Optional) install the package locally
pip install -e .
```

## .env configuration

Create a `.env` at project root (the server auto-loads it):

```bash
PROXMOX_API_URL="https://proxmox.example.com:8006"
PROXMOX_TOKEN_ID="root@pam!mcp-proxmox"
PROXMOX_TOKEN_SECRET="<secret>"
PROXMOX_VERIFY="true"
PROXMOX_DEFAULT_NODE="pve"
PROXMOX_DEFAULT_STORAGE="local-lvm"
PROXMOX_DEFAULT_BRIDGE="vmbr0"
```

Notes:
- Use an API token with appropriate ACLs; for discovery, `PVEAuditor` at `/` is sufficient; for lifecycle, grant narrower roles (e.g., `PVEVMAdmin`) on a pool.
- Using `.env` avoids zsh history expansion issues with `!` in token IDs.

## Run the MCP server (stdio)

Preferred (module form):

```bash
source .venv/bin/activate
python -m proxmox_mcp.server
```

Or installed console script:

```bash
source .venv/bin/activate
proxmox-mcp
```

## Configure in Cursor

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "proxmox-mcp": {
      "command": "/Users/bsahane/Developer/cursor/mcp-proxmox/.venv/bin/python",
      "args": ["-m", "proxmox_mcp.server"]
    }
  }
}
```

## Configure in Claude for Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "proxmox-mcp": {
      "command": "/Users/bsahane/Developer/cursor/mcp-proxmox/.venv/bin/python",
      "args": ["-m", "proxmox_mcp.server"]
    }
  }
}
```

## Tools reference

All tools are available via MCP. Destructive tools accept `confirm`, and most write operations support `dry_run`, `wait`, `timeout`, `poll_interval`.

Format below per tool:
- Description
- Example question → Possible answer (shape)

### Core discovery
- `proxmox-list-nodes`
  - List cluster nodes (name, status, CPU/RAM/disk summary)
  - Example: "List cluster nodes"
  - Answer: `[ { "node": "pve", "status": "online", ... } ]`
- `proxmox-node-status`
  - Detailed node health (load, uptime, versions)
  - Example: `{ "node": "pve" }`
  - Answer: `{ "kversion": "...", "uptime": 123456, ... }`
- `proxmox-list-vms`
  - List VMs (filter by node, status, name substring)
  - Example: `{ "node": "pve", "status": "running" }`
  - Answer: `[ { "vmid": 100, "name": "web01", ... } ]`
- `proxmox-vm-info`
  - Get VM details by `vmid` or `name` (+optional node), includes config
  - Example: `{ "name": "web01" }`
  - Answer: `{ "selector": {...}, "config": {...} }`
- `proxmox-list-lxc`
  - List LXC containers (filterable)
  - Example: `{ "node": "pve" }`
  - Answer: `[ { "vmid": 50001, "name": "ct01", ... } ]`
- `proxmox-lxc-info`
  - Get LXC details by `vmid` or `name` (+optional node)
  - Example: `{ "vmid": 50001 }`
  - Answer: `{ "selector": {...}, "config": {...} }`
- `proxmox-list-storage`
  - List storages (types, free/used)
  - Example: `{}`
  - Answer: `[ { "storage": "local-lvm", "type": "lvmthin", ... } ]`
- `proxmox-storage-content`
  - List storage content (ISOs, templates, images)
  - Example: `{ "node": "pve", "storage": "local" }`
  - Answer: `[ { "volid": "local:iso/foo.iso", ... } ]`
- `proxmox-list-bridges`
  - List node bridges (vmbr...)
  - Example: `{ "node": "pve" }`
  - Answer: `[ { "iface": "vmbr0", ... } ]`
- `proxmox-list-tasks`
  - Recent tasks (filter by node, user)
  - Example: `{ "node": "pve", "limit": 20 }`
  - Answer: `[ { "upid": "UPID:...", "status": "OK" }, ... ]`
- `proxmox-task-status`
  - Check a task status
  - Example: `{ "upid": "UPID:..." }`
  - Answer: `{ "status": "stopped", "exitstatus": "OK" }`

### VM lifecycle
- `proxmox-clone-vm`
  - Clone template VM to new VMID/name (supports target node, storage)
  - Example: `{ "source_vmid": 101, "new_vmid": 50009, "name": "web01", "storage": "local-lvm", "confirm": true, "wait": true }`
  - Answer: `{ "upid": "UPID:...", "status": {...} }`
- `proxmox-create-vm`
  - Create new VM from ISO/template (minimal config)
  - Example: `{ "node": "pve", "vmid": 200, "name": "web02", "iso": "debian.iso", "confirm": true }`
  - Answer: `{ "upid": "UPID:..." }`
- `proxmox-delete-vm`
  - Delete VM (confirm, purge)
  - Example: `{ "name": "web01", "purge": true, "confirm": true }`
  - Answer: `{ "upid": "UPID:..." }`
- `proxmox-start-vm` / `proxmox-stop-vm` / `proxmox-reboot-vm` / `proxmox-shutdown-vm`
  - Manage power state (stop supports hard and timeout)
  - Example: `{ "name": "web01", "wait": true }`
  - Answer: `{ "upid": "UPID:...", "status": {...} }`
- `proxmox-migrate-vm`
  - Live/offline migrate to another node
  - Example: `{ "name": "web01", "target_node": "pve2", "live": true }`
  - Answer: `{ "upid": "UPID:..." }`
- `proxmox-resize-vm-disk`
  - Grow disk (GB) on target disk (e.g., scsi0)
  - Example: `{ "name": "web01", "disk": "scsi0", "grow_gb": 10, "confirm": true, "wait": true }`
  - Answer: `{ "upid": "UPID:...", "status": {...} }`
- `proxmox-configure-vm`
  - Set whitelisted params (cores, memory, balloon, netX, agent, etc.)
  - Example: `{ "name": "web01", "params": { "memory": 4096, "cores": 4 }, "confirm": true }`
  - Answer: `{ "upid": "UPID:..." }` or `{ "result": null }`

### LXC lifecycle
- `proxmox-create-lxc`
  - Create container from template (CPU/mem, rootfs size, net, storage)
  - Example: `{ "node": "pve", "vmid": 50050, "hostname": "ct01", "ostemplate": "debian-12.tar.zst", "confirm": true }`
  - Answer: `{ "upid": "UPID:..." }`
- `proxmox-delete-lxc` / `proxmox-start-lxc` / `proxmox-stop-lxc` / `proxmox-configure-lxc`
  - Manage container lifecycle and config

### Cloud-init & networking
- `proxmox-cloudinit-set`
  - Set CI params (ipconfig0, sshkeys, ciuser/cipassword)
  - Example: `{ "name": "web01", "ipconfig0": "ip=192.168.1.50/24,gw=192.168.1.1", "confirm": true }`
  - Answer: `{ "upid": "UPID:..." }` or `{ "result": null }`
- `proxmox-vm-nic-add` / `proxmox-vm-nic-remove`
  - Add/remove NICs (bridge, model, VLAN)
- `proxmox-vm-firewall-get` / `proxmox-vm-firewall-set`
  - Get/set per-VM firewall state and rules

### Images, templates, snapshots, backups
- `proxmox-upload-iso` / `proxmox-upload-template`
  - Upload ISO or LXC template to storage
- `proxmox-template-vm`
  - Convert VM to template
- `proxmox-list-snapshots` / `proxmox-create-snapshot` / `proxmox-delete-snapshot` / `proxmox-rollback-snapshot`
  - Manage snapshots; rollback supports `wait`
- `proxmox-backup-vm` / `proxmox-restore-vm`
  - Run vzdump and restore archives

### Metrics and monitoring
- `proxmox-vm-metrics`
  - RRD metrics for VM (timeframe, cf)
- `proxmox-node-metrics`
  - RRD metrics for node

### Pools, users, permissions
- `proxmox-list-pools` / `proxmox-create-pool` / `proxmox-delete-pool` / `proxmox-pool-add` / `proxmox-pool-remove`
- `proxmox-list-users` / `proxmox-list-roles` / `proxmox-assign-permission`

### Orchestration helpers
- `proxmox-wait-task`
  - Poll a task until done/timeout
- `proxmox-register-vm-as-host`
  - Emit JSON/INI snippet for Ansible inventory (hostname, IP, SSH user/key)
- `proxmox-guest-exec` (optional)
  - Run a command via QEMU Guest Agent (requires agent in guest)

## Examples

- List nodes: `{}` for `proxmox-list-nodes`
- VMs on node `pve`: `{ "node": "pve" }` for `proxmox-list-vms`
- Clone a template: `{ "source_vmid": 101, "new_vmid": 50009, "name": "web01", "storage": "local-lvm", "confirm": true, "wait": true }`
- Configure Cloud-init IP: `{ "name": "web01", "ipconfig0": "ip=192.168.1.50/24,gw=192.168.1.1", "confirm": true }`

## Notes

- Server uses stdio transport; prints only MCP protocol to stdout. Logs go to stderr.
- Authentication uses your environment variables and/or `.env` file.
- Name collisions across nodes return clear errors unless you specify `node`.

## Development

```bash
# Lint/type-check as needed (not included by default)
```

## License

MIT
