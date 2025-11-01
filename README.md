# MCP Proxmox Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Model Context Protocol (MCP) server for Proxmox VE, providing advanced automation, orchestration, and management capabilities through a rich set of tools.

## 🚀 Features

### Core Capabilities
- **VM & LXC Management**: Full lifecycle management (create, clone, start, stop, migrate, delete)
- **Cloud-Init Integration**: Automated VM provisioning with network configuration
- **Snapshot & Backup**: Comprehensive backup and restore operations
- **Network Management**: VLAN configuration, firewall rules, and bridge management
- **Storage Operations**: Multi-storage support with content management
- **Monitoring & Metrics**: Real-time resource monitoring and performance tracking
- **User & Permissions**: Role-based access control and pool management

### Advanced Features
- **Multi-Cluster Support**: Manage multiple Proxmox clusters from a single interface
- **Security & Authentication**: MFA setup, certificate management, and secret storage
- **Infrastructure as Code**: Terraform and Ansible integration
- **AI-Powered Optimization**: Predictive scaling and anomaly detection
- **Enterprise Features**: Multi-tenancy, compliance scanning, and cost management
- **Notes Management**: HTML/Markdown formatted notes for VMs and LXCs
- **Specialized Deployments**: OpenShift, RHCOS, Windows, and Docker Swarm support

## 📋 Prerequisites

- Python 3.8 or higher
- Proxmox VE 7.0 or higher
- API token with appropriate permissions

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bsahane/mcp-proxmox.git
cd mcp-proxmox
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python -m pip install -U pip
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Package (Optional)

```bash
pip install -e .
```

## ⚙️ Configuration

### Environment Setup

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

### Required Environment Variables

```bash
# Proxmox API Configuration
PROXMOX_API_URL="https://proxmox.example.com:8006"
PROXMOX_TOKEN_ID="root@pam!mcp-proxmox"
PROXMOX_TOKEN_SECRET="your-secret-token"
PROXMOX_VERIFY="true"

# Default Settings
PROXMOX_DEFAULT_NODE="pve"
PROXMOX_DEFAULT_STORAGE="local-lvm"
PROXMOX_DEFAULT_BRIDGE="vmbr0"
```

### Multi-Cluster Configuration (Optional)

For managing multiple clusters, add cluster-specific configurations:

```bash
# Cluster 1
PROXMOX_CLUSTER_1_NAME="production"
PROXMOX_CLUSTER_1_API_URL="https://prod-proxmox.example.com:8006"
PROXMOX_CLUSTER_1_TOKEN_ID="root@pam!prod-token"
PROXMOX_CLUSTER_1_TOKEN_SECRET="prod-secret"

# Cluster 2
PROXMOX_CLUSTER_2_NAME="staging"
PROXMOX_CLUSTER_2_API_URL="https://staging-proxmox.example.com:8006"
PROXMOX_CLUSTER_2_TOKEN_ID="root@pam!staging-token"
PROXMOX_CLUSTER_2_TOKEN_SECRET="staging-secret"
```

### API Token Setup

1. Log in to your Proxmox web interface
2. Navigate to **Datacenter → Permissions → API Tokens**
3. Create a new token with appropriate privileges:
   - **Discovery/Read-only**: `PVEAuditor` role at `/`
   - **Full Management**: `PVEVMAdmin` role or higher

## 🎯 Usage

### Running the MCP Server

The server supports multiple transport modes for different use cases:

#### STDIO Mode (Default - Local Use)

For local MCP clients like Cursor and Claude Desktop:

```bash
source .venv/bin/activate
python -m proxmox_mcp.server
```

Or using the console script:

```bash
source .venv/bin/activate
proxmox-mcp
```

#### SSE Mode (Remote Access via Server-Sent Events)

For remote access with real-time streaming:

```bash
source .venv/bin/activate
python -m proxmox_mcp.server --transport sse --host 0.0.0.0 --port 8000
```

Access the server at: `http://your-server:8000`

#### HTTP Mode (Remote Access via HTTP API)

For remote access with HTTP REST API:

```bash
source .venv/bin/activate
python -m proxmox_mcp.server --transport http --host 0.0.0.0 --port 8000
```

**Available Endpoints:**
- `GET /` - Server information
- `GET /health` - Health check and Proxmox connectivity
- `GET /tools` - List all available MCP tools
- `POST /execute` - Execute an MCP tool
- `GET /stream` - SSE streaming endpoint (SSE mode only)
- `GET /docs` - Interactive API documentation

**Command-Line Options:**
- `--transport` - Transport mode: `stdio` (default), `sse`, or `http`
- `--host` - Host to bind to (default: `127.0.0.1`, use `0.0.0.0` for remote access)
- `--port` - Port to bind to (default: `8000`)
- `--reload` - Enable auto-reload on code changes (development only)

### Integration with Cursor

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "proxmox-mcp": {
      "command": "python",
      "args": ["-m", "proxmox_mcp.server"],
      "cwd": "/path/to/mcp-proxmox",
      "env": {
        "PYTHONPATH": "/path/to/mcp-proxmox/src"
      }
    }
  }
}
```

### Integration with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "proxmox-mcp": {
      "command": "python",
      "args": ["-m", "proxmox_mcp.server"],
      "cwd": "/path/to/mcp-proxmox",
      "env": {
        "PYTHONPATH": "/path/to/mcp-proxmox/src"
      }
    }
  }
}
```

## 🛠️ Available Tools

### Discovery & Monitoring
- `proxmox-list-nodes` - List all cluster nodes
- `proxmox-node-status` - Get detailed node information
- `proxmox-list-vms` - List virtual machines
- `proxmox-vm-info` - Get VM details
- `proxmox-list-lxc` - List LXC containers
- `proxmox-lxc-info` - Get container details
- `proxmox-list-storage` - List storage configurations
- `proxmox-storage-content` - View storage content
- `proxmox-list-bridges` - List network bridges
- `proxmox-vm-metrics` - Get VM performance metrics
- `proxmox-node-metrics` - Get node performance metrics

### VM Lifecycle Management
- `proxmox-create-vm` - Create new virtual machine
- `proxmox-clone-vm` - Clone existing VM
- `proxmox-delete-vm` - Delete virtual machine
- `proxmox-start-vm` - Start VM
- `proxmox-stop-vm` - Stop VM
- `proxmox-reboot-vm` - Reboot VM
- `proxmox-shutdown-vm` - Graceful shutdown
- `proxmox-migrate-vm` - Live/offline migration
- `proxmox-resize-vm-disk` - Expand VM disk
- `proxmox-configure-vm` - Update VM configuration
- `proxmox-template-vm` - Convert VM to template

### LXC Container Management
- `proxmox-create-lxc` - Create new container
- `proxmox-delete-lxc` - Delete container
- `proxmox-start-lxc` - Start container
- `proxmox-stop-lxc` - Stop container
- `proxmox-configure-lxc` - Update container configuration

### Cloud-Init & Networking
- `proxmox-cloudinit-set` - Configure cloud-init parameters
- `proxmox-vm-nic-add` - Add network interface
- `proxmox-vm-nic-remove` - Remove network interface
- `proxmox-vm-firewall-get` - Get firewall configuration
- `proxmox-vm-firewall-set` - Set firewall rules
- `proxmox-create-vlan` - Create VLAN configuration

### Snapshots & Backups
- `proxmox-list-snapshots` - List VM snapshots
- `proxmox-create-snapshot` - Create snapshot
- `proxmox-delete-snapshot` - Delete snapshot
- `proxmox-rollback-snapshot` - Rollback to snapshot
- `proxmox-backup-vm` - Backup virtual machine
- `proxmox-restore-vm` - Restore from backup

### Storage & Templates
- `proxmox-upload-iso` - Upload ISO image
- `proxmox-upload-template` - Upload container template

### Access Control
- `proxmox-list-pools` - List resource pools
- `proxmox-create-pool` - Create resource pool
- `proxmox-delete-pool` - Delete resource pool
- `proxmox-pool-add` - Add resources to pool
- `proxmox-pool-remove` - Remove resources from pool
- `proxmox-list-users` - List users
- `proxmox-list-roles` - List roles
- `proxmox-assign-permission` - Assign permissions

### Notes Management
- `proxmox-vm-notes-read` - Read VM notes (HTML/Markdown)
- `proxmox-vm-notes-update` - Update VM notes
- `proxmox-vm-notes-remove` - Remove VM notes
- `proxmox-lxc-notes-read` - Read LXC notes
- `proxmox-lxc-notes-update` - Update LXC notes
- `proxmox-lxc-notes-remove` - Remove LXC notes
- `proxmox-notes-template` - Generate notes template

### Multi-Cluster Operations
- `proxmox-list-all-clusters` - List configured clusters
- `proxmox-list-all-nodes-from-all-clusters` - List all nodes
- `proxmox-list-all-vms-from-all-clusters` - List all VMs
- `proxmox-get-all-cluster-status` - Get cluster health status

### Task Management
- `proxmox-list-tasks` - List recent tasks
- `proxmox-task-status` - Get task status
- `proxmox-wait-task` - Wait for task completion

### Orchestration
- `proxmox-register-vm-as-host` - Generate Ansible inventory
- `proxmox-guest-exec` - Execute commands via QEMU Guest Agent

## 📚 Documentation

Additional documentation is available in the `docs/` directory:

- **Multi-Cluster Setup**: `docs/MULTI_CLUSTER_QUICK_START.md`
- **OpenShift Deployment**: `docs/openshift_lan_exposure_guide.md`
- **Server Management**: `docs/MCP_SERVER_START_GUIDE.md`
- **Feature Specifications**: `docs/MULTI_CLUSTER_SPEC.md`
- **Notes Management**: `docs/NOTES_FEATURE_IMPLEMENTATION.md`

## 🔒 Security Best Practices

1. **Never store secrets in VM/LXC notes** - Use the `proxmox-secret-store` tool instead
2. **Use API tokens** instead of username/password authentication
3. **Apply least privilege** - Grant only necessary permissions
4. **Enable SSL verification** in production environments
5. **Rotate API tokens** regularly
6. **Use separate tokens** for different environments (dev/staging/prod)

## 🧪 Testing

Run the test suite:

```bash
source .venv/bin/activate
python tests/test_notes_feature.py
```

## 📁 Project Structure

```
mcp-proxmox/
├── src/
│   └── proxmox_mcp/
│       ├── __init__.py
│       ├── server.py              # Main MCP server
│       ├── client.py              # Proxmox API client
│       ├── utils.py               # Utility functions
│       ├── cloudinit.py           # Cloud-init support
│       ├── notes_manager.py       # Notes management
│       ├── rhcos.py               # Red Hat CoreOS support
│       ├── windows.py             # Windows VM support
│       └── docker_swarm.py        # Docker Swarm integration
├── tests/                         # Test files
├── scripts/                       # Utility scripts
├── docs/                          # Documentation
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Package configuration
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp)
- Proxmox API via [Proxmoxer](https://github.com/proxmoxer/proxmoxer)
- Inspired by [MCP Ansible](https://github.com/bsahane/mcp-ansible)

## 📞 Support

For issues, questions, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/bsahane/mcp-proxmox/issues)
- **Documentation**: Check the `docs/` directory
- **MCP Reference**: [Model Context Protocol](https://modelcontextprotocol.io/)

## 🔄 Version History

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

---

**Note**: This server uses stdio transport and prints only MCP protocol messages to stdout. All logs are sent to stderr for proper MCP operation.
