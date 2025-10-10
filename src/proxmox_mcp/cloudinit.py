"""CloudInit configuration and VM provisioning module."""

from __future__ import annotations

import base64
import os
import tempfile
import yaml
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from jsonschema import validate, ValidationError
import jinja2


class CloudInitConfig:
    """CloudInit configuration builder and validator."""
    
    # Common OS templates with their specific configurations
    OS_TEMPLATES = {
        "ubuntu-22.04": {
            "name": "Ubuntu 22.04 LTS",
            "image_url": "https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img",
            "default_user": "ubuntu",
            "package_manager": "apt",
            "default_packages": ["curl", "wget", "git", "htop", "vim", "openssh-server"]
        },
        "ubuntu-24.04": {
            "name": "Ubuntu 24.04 LTS", 
            "image_url": "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img",
            "default_user": "ubuntu",
            "package_manager": "apt",
            "default_packages": ["curl", "wget", "git", "htop", "vim", "openssh-server"]
        },
        "fedora-39": {
            "name": "Fedora 39",
            "image_url": "https://download.fedoraproject.org/pub/fedora/linux/releases/39/Cloud/x86_64/images/Fedora-Cloud-Base-39-1.5.x86_64.qcow2",
            "default_user": "fedora",
            "package_manager": "dnf",
            "default_packages": ["curl", "wget", "git", "htop", "vim", "openssh-server"]
        },
        "fedora-40": {
            "name": "Fedora 40",
            "image_url": "https://download.fedoraproject.org/pub/fedora/linux/releases/40/Cloud/x86_64/images/Fedora-Cloud-Base-40-1.14.x86_64.qcow2",
            "default_user": "fedora", 
            "package_manager": "dnf",
            "default_packages": ["curl", "wget", "git", "htop", "vim", "openssh-server"]
        },
        "rocky-9": {
            "name": "Rocky Linux 9",
            "image_url": "https://download.rockylinux.org/pub/rocky/9/images/x86_64/Rocky-9-GenericCloud-Base.latest.x86_64.qcow2",
            "default_user": "rocky",
            "package_manager": "dnf", 
            "default_packages": ["curl", "wget", "git", "htop", "vim", "openssh-server"]
        },
        "almalinux-9": {
            "name": "AlmaLinux 9",
            "image_url": "https://repo.almalinux.org/almalinux/9/cloud/x86_64/images/AlmaLinux-9-GenericCloud-latest.x86_64.qcow2",
            "default_user": "almalinux",
            "package_manager": "dnf",
            "default_packages": ["curl", "wget", "git", "htop", "vim", "openssh-server"]
        }
    }

    # CloudInit schema for validation
    CLOUDINIT_SCHEMA = {
        "type": "object",
        "properties": {
            "users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "sudo": {"type": ["string", "array"]},
                        "shell": {"type": "string"},
                        "ssh_authorized_keys": {"type": "array", "items": {"type": "string"}},
                        "passwd": {"type": "string"},
                        "lock_passwd": {"type": "boolean"}
                    },
                    "required": ["name"]
                }
            },
            "packages": {"type": "array", "items": {"type": "string"}},
            "package_update": {"type": "boolean"},
            "package_upgrade": {"type": "boolean"},
            "runcmd": {"type": "array", "items": {"type": ["string", "array"]}},
            "write_files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "permissions": {"type": "string"},
                        "owner": {"type": "string"},
                        "encoding": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            },
            "network": {
                "type": "object",
                "properties": {
                    "version": {"type": "integer"},
                    "ethernets": {"type": "object"}
                }
            },
            "hostname": {"type": "string"},
            "fqdn": {"type": "string"},
            "timezone": {"type": "string"},
            "ssh_pwauth": {"type": "boolean"},
            "disable_root": {"type": "boolean"}
        }
    }

    def __init__(self, template: str = "ubuntu-22.04"):
        """Initialize CloudInit configuration with OS template."""
        if template not in self.OS_TEMPLATES:
            raise ValueError(f"Unsupported template: {template}. Supported: {list(self.OS_TEMPLATES.keys())}")
        
        self.template = template
        self.template_info = self.OS_TEMPLATES[template]
        self.config: Dict[str, Any] = {
            "package_update": True,
            "package_upgrade": False,
            "ssh_pwauth": True,
            "disable_root": True,
        }

    def add_user(self, name: str, ssh_keys: List[str], sudo: Union[str, List[str]] = "ALL=(ALL) NOPASSWD:ALL", 
                 shell: str = "/bin/bash", passwd: Optional[str] = None) -> None:
        """Add user to CloudInit configuration."""
        if "users" not in self.config:
            self.config["users"] = []
        
        user_config = {
            "name": name,
            "sudo": sudo,
            "shell": shell,
            "ssh_authorized_keys": ssh_keys,
            "lock_passwd": passwd is None
        }
        
        if passwd:
            user_config["passwd"] = passwd
            
        self.config["users"].append(user_config)

    def add_packages(self, packages: List[str]) -> None:
        """Add packages to install."""
        if "packages" not in self.config:
            self.config["packages"] = []
        
        # Add default packages for the OS
        all_packages = list(set(self.template_info["default_packages"] + packages))
        self.config["packages"] = all_packages

    def add_commands(self, commands: List[Union[str, List[str]]]) -> None:
        """Add commands to run on first boot."""
        if "runcmd" not in self.config:
            self.config["runcmd"] = []
        
        self.config["runcmd"].extend(commands)

    def add_file(self, path: str, content: str, permissions: str = "0644", 
                 owner: str = "root:root", encoding: str = "text/plain") -> None:
        """Add file to write during cloud-init."""
        if "write_files" not in self.config:
            self.config["write_files"] = []
        
        self.config["write_files"].append({
            "path": path,
            "content": content,
            "permissions": permissions,
            "owner": owner,
            "encoding": encoding
        })

    def set_network_config(self, interface: str = "ens18", dhcp: bool = True, 
                          ip: Optional[str] = None, gateway: Optional[str] = None, 
                          nameservers: Optional[List[str]] = None) -> None:
        """Configure network settings."""
        network_config = {
            "version": 2,
            "ethernets": {
                interface: {}
            }
        }
        
        if dhcp:
            network_config["ethernets"][interface]["dhcp4"] = True
        else:
            if not ip:
                raise ValueError("IP address required when DHCP is disabled")
            
            network_config["ethernets"][interface]["addresses"] = [ip]
            if gateway:
                network_config["ethernets"][interface]["gateway4"] = gateway
            if nameservers:
                network_config["ethernets"][interface]["nameservers"] = {"addresses": nameservers}
        
        self.config["network"] = network_config

    def set_hostname(self, hostname: str, fqdn: Optional[str] = None) -> None:
        """Set hostname and FQDN."""
        self.config["hostname"] = hostname
        if fqdn:
            self.config["fqdn"] = fqdn

    def set_timezone(self, timezone: str = "UTC") -> None:
        """Set system timezone."""
        self.config["timezone"] = timezone

    def validate_config(self) -> bool:
        """Validate CloudInit configuration against schema."""
        try:
            validate(instance=self.config, schema=self.CLOUDINIT_SCHEMA)
            return True
        except ValidationError as e:
            raise ValueError(f"CloudInit configuration validation error: {e.message}")

    def to_yaml(self) -> str:
        """Convert configuration to CloudInit YAML format."""
        self.validate_config()
        
        # Add cloud-config header
        yaml_content = "#cloud-config\n"
        yaml_content += yaml.dump(self.config, default_flow_style=False, allow_unicode=True)
        
        return yaml_content

    def to_user_data(self) -> str:
        """Generate user-data for CloudInit."""
        return self.to_yaml()

    def create_iso(self, output_path: str, instance_id: str = "vm-instance", 
                   local_hostname: Optional[str] = None) -> str:
        """Create CloudInit NoCloud ISO with user-data and meta-data."""
        import subprocess
        
        # Create temporary directory for ISO content
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write user-data
            user_data_path = os.path.join(temp_dir, "user-data")
            with open(user_data_path, "w") as f:
                f.write(self.to_user_data())
            
            # Create meta-data
            meta_data = {
                "instance-id": instance_id,
                "local-hostname": local_hostname or instance_id
            }
            meta_data_path = os.path.join(temp_dir, "meta-data")
            with open(meta_data_path, "w") as f:
                yaml.dump(meta_data, f)
            
            # Create network-config if specified
            if "network" in self.config:
                network_config_path = os.path.join(temp_dir, "network-config")
                with open(network_config_path, "w") as f:
                    yaml.dump(self.config["network"], f)
            
            # Create ISO using genisoimage or mkisofs
            iso_cmd = [
                "genisoimage", "-output", output_path, "-volid", "cidata",
                "-joliet", "-rock", temp_dir
            ]
            
            try:
                subprocess.run(iso_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                # Fallback to mkisofs
                iso_cmd[0] = "mkisofs"
                subprocess.run(iso_cmd, check=True, capture_output=True)
            except FileNotFoundError:
                raise RuntimeError("Neither genisoimage nor mkisofs found. Please install one of them.")
        
        return output_path


class CloudInitProvisioner:
    """Provisions VMs with CloudInit configuration."""
    
    def __init__(self, proxmox_client):
        """Initialize with Proxmox client."""
        self.client = proxmox_client

    def create_vm_with_cloudinit(self, *, node: str, vmid: int, name: str, template: str,
                                cloudinit_config: CloudInitConfig, hardware: Dict[str, Any],
                                storage: Optional[str] = None, bridge: Optional[str] = None) -> str:
        """Create VM with CloudInit configuration."""
        storage_id = storage or self.client.default_storage
        bridge_id = bridge or self.client.default_bridge
        
        # Create the base VM first
        cores = hardware.get("cores", 2)
        memory_mb = hardware.get("memory_mb", 2048)
        disk_gb = hardware.get("disk_gb", 20)
        
        # Download and prepare OS template if needed
        template_info = CloudInitConfig.OS_TEMPLATES[template]
        
        # Create VM with cloud-init specific configuration
        vm_params = {
            "vmid": vmid,
            "name": name,
            "cores": cores,
            "memory": memory_mb,
            "scsihw": "virtio-scsi-pci",
            "agent": 1,
            "ostype": "l26",
            "serial0": "socket",
            "vga": "serial0",
            "scsi0": f"{storage_id}:{disk_gb}",
            "net0": f"virtio,bridge={bridge_id}",
            "ide2": f"{storage_id}:cloudinit",  # CloudInit drive
        }
        
        upid = self.client.api.nodes(node).qemu.post(**vm_params)
        
        # Create and attach CloudInit ISO
        iso_path = f"/tmp/cloudinit-{vmid}.iso"
        cloudinit_config.create_iso(iso_path, instance_id=f"vm-{vmid}", local_hostname=name)
        
        # Upload ISO to storage and attach to VM
        iso_upid = self.client.upload_iso(node, storage_id, iso_path)
        
        # Clean up temporary ISO
        os.unlink(iso_path)
        
        return upid

    def prompt_for_config(self, template: str) -> CloudInitConfig:
        """Interactive prompt for CloudInit configuration."""
        config = CloudInitConfig(template)
        
        print(f"Configuring {config.template_info['name']} CloudInit setup...")
        
        # Hostname
        hostname = input("Enter hostname: ").strip()
        if hostname:
            fqdn = input("Enter FQDN (optional): ").strip() or None
            config.set_hostname(hostname, fqdn)
        
        # User configuration
        print("\n--- User Configuration ---")
        username = input(f"Username (default: {config.template_info['default_user']}): ").strip()
        username = username or config.template_info['default_user']
        
        ssh_keys = []
        print("Enter SSH public keys (press Enter twice when done):")
        while True:
            key = input("SSH key: ").strip()
            if not key:
                break
            ssh_keys.append(key)
        
        if ssh_keys:
            config.add_user(username, ssh_keys)
        
        # Package configuration
        print("\n--- Package Configuration ---")
        additional_packages = input("Additional packages (comma-separated): ").strip()
        if additional_packages:
            packages = [pkg.strip() for pkg in additional_packages.split(",")]
            config.add_packages(packages)
        else:
            config.add_packages([])  # Just default packages
        
        # Network configuration
        print("\n--- Network Configuration ---")
        use_dhcp = input("Use DHCP? (y/N): ").strip().lower()
        if use_dhcp != 'y':
            ip = input("IP address (CIDR notation, e.g., 192.168.1.100/24): ").strip()
            gateway = input("Gateway: ").strip()
            nameservers = input("DNS servers (comma-separated): ").strip()
            
            if ip:
                ns_list = [ns.strip() for ns in nameservers.split(",")] if nameservers else None
                config.set_network_config(dhcp=False, ip=ip, gateway=gateway, nameservers=ns_list)
        
        # Timezone
        timezone = input("Timezone (default: UTC): ").strip() or "UTC"
        config.set_timezone(timezone)
        
        # Custom commands
        print("\n--- Custom Commands (optional) ---")
        commands = []
        print("Enter commands to run on first boot (press Enter twice when done):")
        while True:
            cmd = input("Command: ").strip()
            if not cmd:
                break
            commands.append(cmd)
        
        if commands:
            config.add_commands(commands)
        
        return config


# Template configurations for quick deployment
def get_ubuntu_web_server_config(hostname: str, ssh_keys: List[str], admin_user: str = "ubuntu") -> CloudInitConfig:
    """Pre-configured Ubuntu web server setup."""
    config = CloudInitConfig("ubuntu-22.04")
    config.set_hostname(hostname)
    config.add_user(admin_user, ssh_keys)
    config.add_packages(["nginx", "ufw", "certbot", "python3-certbot-nginx"])
    config.add_commands([
        "systemctl enable nginx",
        "systemctl start nginx", 
        "ufw allow 'Nginx Full'",
        "ufw allow ssh",
        "ufw --force enable"
    ])
    return config


def get_docker_host_config(hostname: str, ssh_keys: List[str], admin_user: str = "ubuntu") -> CloudInitConfig:
    """Pre-configured Docker host setup."""
    config = CloudInitConfig("ubuntu-22.04")
    config.set_hostname(hostname)
    config.add_user(admin_user, ssh_keys)
    config.add_packages(["docker.io", "docker-compose", "curl"])
    config.add_commands([
        f"usermod -aG docker {admin_user}",
        "systemctl enable docker",
        "systemctl start docker"
    ])
    return config


def get_development_config(hostname: str, ssh_keys: List[str], admin_user: str = "fedora") -> CloudInitConfig:
    """Pre-configured development environment."""
    config = CloudInitConfig("fedora-40")
    config.set_hostname(hostname)
    config.add_user(admin_user, ssh_keys)
    config.add_packages([
        "git", "vim", "tmux", "nodejs", "npm", "python3", "python3-pip",
        "gcc", "make", "golang", "docker", "podman"
    ])
    config.add_commands([
        f"usermod -aG docker {admin_user}",
        "systemctl enable docker",
        "systemctl start docker"
    ])
    return config
