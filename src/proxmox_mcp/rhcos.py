"""Red Hat Enterprise Linux CoreOS (RHCOS) and Ignition support module."""

from __future__ import annotations

import json
import base64
import hashlib
import os
import tempfile
import yaml
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from urllib.parse import urlparse

import requests
from jsonschema import validate, ValidationError


class IgnitionConfig:
    """Ignition configuration builder for RHCOS."""
    
    # RHCOS release streams and versions
    RHCOS_STREAMS = {
        "4.14": {
            "stream": "stable",
            "version": "414.92.202309112003-0",
            "base_url": "https://builds.coreos.fedoraproject.org/prod/streams/stable/builds",
            "description": "RHCOS 4.14 - Latest Stable for OpenShift 4.14"
        },
        "4.13": {
            "stream": "stable", 
            "version": "413.92.202309112003-0",
            "base_url": "https://builds.coreos.fedoraproject.org/prod/streams/stable/builds",
            "description": "RHCOS 4.13 - Stable for OpenShift 4.13"
        },
        "4.15": {
            "stream": "next",
            "version": "415.92.202311021637-0",
            "base_url": "https://builds.coreos.fedoraproject.org/prod/streams/next/builds",
            "description": "RHCOS 4.15 - Development for OpenShift 4.15"
        }
    }

    # Ignition v3.x specification schema (simplified)
    IGNITION_SCHEMA = {
        "type": "object",
        "properties": {
            "ignition": {
                "type": "object",
                "properties": {
                    "version": {"type": "string"},
                    "config": {
                        "type": "object",
                        "properties": {
                            "merge": {"type": "array"},
                            "replace": {"type": "object"}
                        }
                    }
                },
                "required": ["version"]
            },
            "passwd": {
                "type": "object", 
                "properties": {
                    "users": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "sshAuthorizedKeys": {"type": "array", "items": {"type": "string"}},
                                "groups": {"type": "array", "items": {"type": "string"}},
                                "shell": {"type": "string"},
                                "homeDir": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "storage": {
                "type": "object",
                "properties": {
                    "files": {"type": "array"},
                    "directories": {"type": "array"},
                    "links": {"type": "array"},
                    "filesystems": {"type": "array"}
                }
            },
            "systemd": {
                "type": "object",
                "properties": {
                    "units": {"type": "array"}
                }
            }
        },
        "required": ["ignition"]
    }

    def __init__(self, version: str = "3.4.0"):
        """Initialize Ignition configuration."""
        self.config = {
            "ignition": {
                "version": version
            }
        }
        self.version = version

    def add_user(self, name: str, ssh_keys: List[str], groups: Optional[List[str]] = None, 
                 shell: str = "/bin/bash", home_dir: Optional[str] = None) -> None:
        """Add user to Ignition configuration."""
        if "passwd" not in self.config:
            self.config["passwd"] = {"users": []}
        
        user_config = {
            "name": name,
            "sshAuthorizedKeys": ssh_keys
        }
        
        if groups:
            user_config["groups"] = groups
        if shell:
            user_config["shell"] = shell
        if home_dir:
            user_config["homeDir"] = home_dir
            
        self.config["passwd"]["users"].append(user_config)

    def add_file(self, path: str, content: str, mode: int = 0o644, 
                 user_id: int = 0, group_id: int = 0, overwrite: bool = True) -> None:
        """Add file to Ignition configuration."""
        if "storage" not in self.config:
            self.config["storage"] = {}
        if "files" not in self.config["storage"]:
            self.config["storage"]["files"] = []
        
        # Encode content as base64
        content_b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
        
        file_config = {
            "path": path,
            "mode": mode,
            "user": {"id": user_id},
            "group": {"id": group_id},
            "contents": {
                "source": f"data:text/plain;charset=utf-8;base64,{content_b64}"
            },
            "overwrite": overwrite
        }
        
        self.config["storage"]["files"].append(file_config)

    def add_systemd_unit(self, name: str, content: str, enabled: bool = True, 
                        mask: bool = False) -> None:
        """Add systemd unit to Ignition configuration."""
        if "systemd" not in self.config:
            self.config["systemd"] = {"units": []}
        
        unit_config = {
            "name": name,
            "enabled": enabled,
            "mask": mask
        }
        
        if content:
            unit_config["contents"] = content
            
        self.config["systemd"]["units"].append(unit_config)

    def add_directory(self, path: str, mode: int = 0o755, user_id: int = 0, 
                     group_id: int = 0) -> None:
        """Add directory to Ignition configuration."""
        if "storage" not in self.config:
            self.config["storage"] = {}
        if "directories" not in self.config["storage"]:
            self.config["storage"]["directories"] = []
        
        dir_config = {
            "path": path,
            "mode": mode,
            "user": {"id": user_id},
            "group": {"id": group_id}
        }
        
        self.config["storage"]["directories"].append(dir_config)

    def add_ca_certificate(self, cert_content: str) -> None:
        """Add CA certificate for secure communications."""
        cert_path = "/etc/pki/ca-trust/source/anchors/openshift-ca.crt"
        self.add_file(cert_path, cert_content, mode=0o644)
        
        # Add systemd unit to update CA trust
        update_ca_unit = """
[Unit]
Description=Update CA Trust
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/bin/update-ca-trust
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
        self.add_systemd_unit("update-ca-trust.service", update_ca_unit, enabled=True)

    def set_hostname(self, hostname: str) -> None:
        """Set system hostname via Ignition."""
        hostname_content = f"{hostname}\n"
        self.add_file("/etc/hostname", hostname_content, mode=0o644)

    def add_pull_secret(self, pull_secret: Dict[str, Any]) -> None:
        """Add container registry pull secret."""
        pull_secret_json = json.dumps(pull_secret, indent=2)
        self.add_file("/var/lib/kubelet/config.json", pull_secret_json, mode=0o600)

    def validate_config(self) -> bool:
        """Validate Ignition configuration against schema."""
        try:
            validate(instance=self.config, schema=self.IGNITION_SCHEMA)
            return True
        except ValidationError as e:
            raise ValueError(f"Ignition configuration validation error: {e.message}")

    def to_json(self) -> str:
        """Convert configuration to Ignition JSON format."""
        self.validate_config()
        return json.dumps(self.config, indent=2)

    def to_compact_json(self) -> str:
        """Convert configuration to compact JSON for URL embedding."""
        self.validate_config()
        return json.dumps(self.config, separators=(',', ':'))

    def create_iso(self, output_path: str, label: str = "ignition") -> str:
        """Create Ignition ISO for VM boot."""
        import subprocess
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write ignition configuration
            ignition_path = os.path.join(temp_dir, "ignition.json")
            with open(ignition_path, "w") as f:
                f.write(self.to_json())
            
            # Create ISO using genisoimage or mkisofs
            cmd = [
                "genisoimage", "-output", output_path, "-volid", label,
                "-joliet", "-rock", temp_dir
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to mkisofs
                cmd[0] = "mkisofs"
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    raise RuntimeError("Neither genisoimage nor mkisofs available for ISO creation")
        
        return output_path


class RHCOSProvisioner:
    """RHCOS VM provisioning with Ignition configuration."""
    
    def __init__(self, proxmox_client):
        """Initialize with Proxmox client."""
        self.client = proxmox_client

    def download_rhcos_image(self, version: str, node: str, storage: str) -> str:
        """Download RHCOS image from official sources."""
        if version not in IgnitionConfig.RHCOS_STREAMS:
            raise ValueError(f"Unsupported RHCOS version: {version}. Supported: {list(IgnitionConfig.RHCOS_STREAMS.keys())}")
        
        stream_info = IgnitionConfig.RHCOS_STREAMS[version]
        
        # Construct download URL for QEMU image
        image_url = f"{stream_info['base_url']}/{stream_info['version']}/x86_64/rhcos-{stream_info['version']}-qemu.x86_64.qcow2.gz"
        
        # Download and decompress image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".qcow2.gz") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            compressed_path = temp_file.name
        
        try:
            # Decompress the image
            import gzip
            import shutil
            
            decompressed_path = compressed_path.replace(".gz", "")
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Upload to Proxmox storage
            upid = self.client.upload_template(node, storage, decompressed_path)
            return upid
            
        finally:
            # Clean up temporary files
            if os.path.exists(compressed_path):
                os.unlink(compressed_path)
            if os.path.exists(decompressed_path):
                os.unlink(decompressed_path)

    def create_rhcos_vm(self, *, node: str, vmid: int, name: str, rhcos_version: str,
                       ignition_config: IgnitionConfig, hardware: Dict[str, Any],
                       storage: Optional[str] = None, bridge: Optional[str] = None) -> str:
        """Create RHCOS VM with Ignition configuration."""
        storage_id = storage or self.client.default_storage
        bridge_id = bridge or self.client.default_bridge
        
        cores = hardware.get("cores", 4)
        memory_mb = hardware.get("memory_mb", 8192)
        disk_gb = hardware.get("disk_gb", 50)
        
        # Create VM with RHCOS-specific configuration
        vm_params = {
            "vmid": vmid,
            "name": name,
            "cores": cores,
            "memory": memory_mb,
            "scsihw": "virtio-scsi-pci",
            "agent": 0,  # QEMU guest agent not typically used in RHCOS
            "ostype": "l26",
            "machine": "q35",
            "cpu": "host",
            "boot": "order=scsi0;net0",
            "scsi0": f"{storage_id}:{disk_gb},format=qcow2",
            "net0": f"virtio,bridge={bridge_id}",
            "serial0": "socket",
            "vga": "serial0",
        }
        
        upid = self.client.api.nodes(node).qemu.post(**vm_params)
        
        # Create and attach Ignition ISO
        iso_path = f"/tmp/ignition-{vmid}.iso"
        ignition_config.create_iso(iso_path, label="ignition")
        
        # Upload ISO and attach as IDE drive
        iso_upid = self.client.upload_iso(node, storage_id, iso_path)
        iso_volid = f"{storage_id}:iso/ignition-{vmid}.iso"
        
        # Attach Ignition ISO to IDE2
        self.client.api.nodes(node).qemu(vmid).config.put(ide2=f"{iso_volid},media=cdrom")
        
        # Clean up temporary ISO
        os.unlink(iso_path)
        
        return upid

    def create_bootstrap_config(self, cluster_name: str, cluster_domain: str, 
                               ssh_key: str, pull_secret: Dict[str, Any], 
                               ignition_endpoint: Optional[str] = None) -> IgnitionConfig:
        """Create Ignition configuration for OpenShift bootstrap node."""
        config = IgnitionConfig()
        
        # Add core user with SSH key
        config.add_user("core", [ssh_key], groups=["sudo", "docker"])
        
        # Set hostname
        config.set_hostname(f"bootstrap.{cluster_name}.{cluster_domain}")
        
        # Add pull secret
        config.add_pull_secret(pull_secret)
        
        # Add bootstrap-specific configuration
        bootstrap_service = """
[Unit]
Description=OpenShift Bootstrap
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/bootstrap.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
        config.add_systemd_unit("openshift-bootstrap.service", bootstrap_service, enabled=True)
        
        # Add bootstrap script
        bootstrap_script = f"""#!/bin/bash
set -euo pipefail

echo "Starting OpenShift bootstrap process..."

# Download and run bootstrap
curl -L -o /tmp/openshift-install https://mirror.openshift.com/pub/openshift-v4/clients/ocp/latest/openshift-install-linux.tar.gz
tar -xzf /tmp/openshift-install -C /tmp/
chmod +x /tmp/openshift-install

# Run bootstrap
/tmp/openshift-install bootstrap --dir=/opt/openshift
"""
        config.add_file("/usr/local/bin/bootstrap.sh", bootstrap_script, mode=0o755)
        
        return config

    def create_master_config(self, cluster_name: str, cluster_domain: str, node_index: int,
                            ssh_key: str, pull_secret: Dict[str, Any],
                            master_ignition_url: Optional[str] = None) -> IgnitionConfig:
        """Create Ignition configuration for OpenShift master node."""
        config = IgnitionConfig()
        
        # Add core user with SSH key
        config.add_user("core", [ssh_key], groups=["sudo", "docker"])
        
        # Set hostname
        config.set_hostname(f"master-{node_index}.{cluster_name}.{cluster_domain}")
        
        # Add pull secret
        config.add_pull_secret(pull_secret)
        
        # Add master-specific configuration
        if master_ignition_url:
            # Reference remote ignition config for masters
            config.config["ignition"]["config"] = {
                "merge": [{
                    "source": master_ignition_url
                }]
            }
        
        return config

    def create_worker_config(self, cluster_name: str, cluster_domain: str, node_index: int,
                            ssh_key: str, pull_secret: Dict[str, Any],
                            worker_ignition_url: Optional[str] = None) -> IgnitionConfig:
        """Create Ignition configuration for OpenShift worker node."""
        config = IgnitionConfig()
        
        # Add core user with SSH key
        config.add_user("core", [ssh_key], groups=["sudo", "docker"])
        
        # Set hostname
        config.set_hostname(f"worker-{node_index}.{cluster_name}.{cluster_domain}")
        
        # Add pull secret
        config.add_pull_secret(pull_secret)
        
        # Add worker-specific configuration
        if worker_ignition_url:
            # Reference remote ignition config for workers
            config.config["ignition"]["config"] = {
                "merge": [{
                    "source": worker_ignition_url
                }]
            }
        
        return config


class OpenShiftInstaller:
    """OpenShift cluster installation orchestrator."""
    
    def __init__(self, proxmox_client):
        """Initialize with Proxmox client."""
        self.client = proxmox_client
        self.rhcos_provisioner = RHCOSProvisioner(proxmox_client)

    def create_install_config(self, cluster_name: str, base_domain: str, 
                             ssh_key: str, pull_secret: Dict[str, Any],
                             master_count: int = 3, worker_count: int = 2,
                             network_cidr: str = "10.0.0.0/16",
                             service_cidr: str = "172.30.0.0/16") -> Dict[str, Any]:
        """Create OpenShift install-config.yaml content."""
        install_config = {
            "apiVersion": "v1",
            "baseDomain": base_domain,
            "metadata": {
                "name": cluster_name
            },
            "compute": [{
                "hyperthreading": "Enabled",
                "name": "worker",
                "replicas": worker_count,
                "platform": {}
            }],
            "controlPlane": {
                "hyperthreading": "Enabled", 
                "name": "master",
                "replicas": master_count,
                "platform": {}
            },
            "networking": {
                "clusterNetwork": [{
                    "cidr": network_cidr,
                    "hostPrefix": 23
                }],
                "networkType": "OVNKubernetes",
                "serviceNetwork": [service_cidr]
            },
            "platform": {
                "none": {}
            },
            "pullSecret": json.dumps(pull_secret),
            "sshKey": ssh_key.strip()
        }
        
        return install_config

    def deploy_cluster(self, cluster_config: Dict[str, Any], node: str, 
                      storage: str, bridge: str, base_vmid: int = 500) -> Dict[str, Any]:
        """Deploy complete OpenShift cluster."""
        cluster_name = cluster_config["cluster_name"]
        base_domain = cluster_config["base_domain"]
        ssh_key = cluster_config["ssh_key"]
        pull_secret = cluster_config["pull_secret"]
        rhcos_version = cluster_config.get("rhcos_version", "4.14")
        
        master_count = cluster_config.get("master_count", 3)
        worker_count = cluster_config.get("worker_count", 2)
        
        deployment_result = {
            "cluster_name": cluster_name,
            "base_domain": base_domain,
            "nodes": [],
            "upids": []
        }
        
        # Create bootstrap node
        bootstrap_config = self.rhcos_provisioner.create_bootstrap_config(
            cluster_name, base_domain, ssh_key, pull_secret
        )
        
        bootstrap_upid = self.rhcos_provisioner.create_rhcos_vm(
            node=node,
            vmid=base_vmid,
            name=f"{cluster_name}-bootstrap",
            rhcos_version=rhcos_version,
            ignition_config=bootstrap_config,
            hardware={"cores": 4, "memory_mb": 8192, "disk_gb": 50},
            storage=storage,
            bridge=bridge
        )
        
        deployment_result["nodes"].append({
            "type": "bootstrap",
            "vmid": base_vmid,
            "name": f"{cluster_name}-bootstrap",
            "upid": bootstrap_upid
        })
        deployment_result["upids"].append(bootstrap_upid)
        
        # Create master nodes
        for i in range(master_count):
            vmid = base_vmid + 1 + i
            master_config = self.rhcos_provisioner.create_master_config(
                cluster_name, base_domain, i, ssh_key, pull_secret
            )
            
            master_upid = self.rhcos_provisioner.create_rhcos_vm(
                node=node,
                vmid=vmid,
                name=f"{cluster_name}-master-{i}",
                rhcos_version=rhcos_version,
                ignition_config=master_config,
                hardware={"cores": 4, "memory_mb": 8192, "disk_gb": 50},
                storage=storage,
                bridge=bridge
            )
            
            deployment_result["nodes"].append({
                "type": "master",
                "vmid": vmid,
                "name": f"{cluster_name}-master-{i}",
                "upid": master_upid
            })
            deployment_result["upids"].append(master_upid)
        
        # Create worker nodes
        for i in range(worker_count):
            vmid = base_vmid + 1 + master_count + i
            worker_config = self.rhcos_provisioner.create_worker_config(
                cluster_name, base_domain, i, ssh_key, pull_secret
            )
            
            worker_upid = self.rhcos_provisioner.create_rhcos_vm(
                node=node,
                vmid=vmid,
                name=f"{cluster_name}-worker-{i}",
                rhcos_version=rhcos_version,
                ignition_config=worker_config,
                hardware={"cores": 2, "memory_mb": 4096, "disk_gb": 30},
                storage=storage,
                bridge=bridge
            )
            
            deployment_result["nodes"].append({
                "type": "worker",
                "vmid": vmid,
                "name": f"{cluster_name}-worker-{i}",
                "upid": worker_upid
            })
            deployment_result["upids"].append(worker_upid)
        
        return deployment_result

    def deploy_single_node_cluster(self, cluster_config: Dict[str, Any], node: str,
                                  storage: str, bridge: str, vmid: int = 600) -> Dict[str, Any]:
        """Deploy OpenShift Single Node Openshift (SNO) cluster."""
        cluster_name = cluster_config["cluster_name"]
        base_domain = cluster_config["base_domain"]
        ssh_key = cluster_config["ssh_key"]
        pull_secret = cluster_config["pull_secret"]
        rhcos_version = cluster_config.get("rhcos_version", "4.14")
        
        # Create SNO-specific Ignition config
        config = IgnitionConfig()
        config.add_user("core", [ssh_key], groups=["sudo", "docker"])
        config.set_hostname(f"sno.{cluster_name}.{base_domain}")
        config.add_pull_secret(pull_secret)
        
        # Add SNO-specific bootstrap
        sno_service = """
[Unit]
Description=OpenShift Single Node
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/sno-bootstrap.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
        config.add_systemd_unit("openshift-sno.service", sno_service, enabled=True)
        
        # Create VM
        upid = self.rhcos_provisioner.create_rhcos_vm(
            node=node,
            vmid=vmid,
            name=f"{cluster_name}-sno",
            rhcos_version=rhcos_version,
            ignition_config=config,
            hardware={"cores": 8, "memory_mb": 16384, "disk_gb": 100},
            storage=storage,
            bridge=bridge
        )
        
        return {
            "cluster_name": cluster_name,
            "deployment_type": "single-node",
            "vmid": vmid,
            "name": f"{cluster_name}-sno",
            "upid": upid,
            "console_url": f"https://console-openshift-console.apps.sno.{cluster_name}.{base_domain}"
        }
