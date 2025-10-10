"""Docker Swarm cluster management and orchestration module."""

from __future__ import annotations

import json
import time
import base64
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .cloudinit import CloudInitConfig


class DockerSwarmConfig:
    """Docker Swarm cluster configuration and management."""
    
    # Docker Swarm node roles
    NODE_ROLES = ["manager", "worker"]
    
    # Supported base OS templates for Docker nodes
    SUPPORTED_OS = {
        "ubuntu-22.04": {
            "name": "Ubuntu 22.04 LTS",
            "default_user": "ubuntu",
            "docker_install_script": "install-docker-ubuntu.sh",
            "package_manager": "apt"
        },
        "ubuntu-24.04": {
            "name": "Ubuntu 24.04 LTS", 
            "default_user": "ubuntu",
            "docker_install_script": "install-docker-ubuntu.sh",
            "package_manager": "apt"
        },
        "rocky-9": {
            "name": "Rocky Linux 9",
            "default_user": "rocky",
            "docker_install_script": "install-docker-rhel.sh",
            "package_manager": "dnf"
        },
        "almalinux-9": {
            "name": "AlmaLinux 9",
            "default_user": "almalinux",
            "docker_install_script": "install-docker-rhel.sh",
            "package_manager": "dnf"
        }
    }

    def __init__(self, cluster_name: str, base_os: str = "ubuntu-22.04"):
        """Initialize Docker Swarm configuration."""
        if base_os not in self.SUPPORTED_OS:
            raise ValueError(f"Unsupported OS: {base_os}. Supported: {list(self.SUPPORTED_OS.keys())}")
        
        self.cluster_name = cluster_name
        self.base_os = base_os
        self.os_info = self.SUPPORTED_OS[base_os]
        self.config = {
            "cluster_name": cluster_name,
            "base_os": base_os,
            "nodes": [],
            "networks": [],
            "services": [],
            "secrets": [],
            "configs": []
        }

    def add_node(self, name: str, role: str, vmid: int, ip: Optional[str] = None,
                 cores: int = 2, memory_mb: int = 2048, disk_gb: int = 30) -> None:
        """Add node to swarm configuration."""
        if role not in self.NODE_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of: {self.NODE_ROLES}")
        
        node = {
            "name": name,
            "role": role,
            "vmid": vmid,
            "ip": ip,
            "cores": cores,
            "memory_mb": memory_mb,
            "disk_gb": disk_gb,
            "status": "pending"
        }
        self.config["nodes"].append(node)

    def add_network(self, name: str, driver: str = "overlay", subnet: Optional[str] = None,
                   attachable: bool = False, encrypted: bool = False) -> None:
        """Add Docker network configuration."""
        network = {
            "name": name,
            "driver": driver,
            "subnet": subnet,
            "attachable": attachable,
            "encrypted": encrypted
        }
        self.config["networks"].append(network)

    def add_service(self, name: str, image: str, replicas: int = 1, 
                   ports: Optional[List[Dict[str, Any]]] = None,
                   environment: Optional[Dict[str, str]] = None,
                   networks: Optional[List[str]] = None,
                   constraints: Optional[List[str]] = None,
                   volumes: Optional[List[str]] = None) -> None:
        """Add Docker service configuration."""
        service = {
            "name": name,
            "image": image,
            "replicas": replicas,
            "ports": ports or [],
            "environment": environment or {},
            "networks": networks or [],
            "constraints": constraints or [],
            "volumes": volumes or []
        }
        self.config["services"].append(service)

    def add_secret(self, name: str, data: str) -> None:
        """Add Docker secret."""
        secret = {
            "name": name,
            "data": base64.b64encode(data.encode()).decode()
        }
        self.config["secrets"].append(secret)

    def get_manager_nodes(self) -> List[Dict[str, Any]]:
        """Get manager nodes."""
        return [node for node in self.config["nodes"] if node["role"] == "manager"]

    def get_worker_nodes(self) -> List[Dict[str, Any]]:
        """Get worker nodes."""
        return [node for node in self.config["nodes"] if node["role"] == "worker"]

    def get_primary_manager(self) -> Optional[Dict[str, Any]]:
        """Get primary manager node (first manager)."""
        managers = self.get_manager_nodes()
        return managers[0] if managers else None

    def generate_docker_install_script(self) -> str:
        """Generate Docker installation script for the base OS."""
        if self.os_info["package_manager"] == "apt":
            return self._generate_ubuntu_docker_script()
        else:
            return self._generate_rhel_docker_script()

    def _generate_ubuntu_docker_script(self) -> str:
        """Generate Docker installation script for Ubuntu."""
        return """#!/bin/bash
set -euo pipefail

# Update package index
apt-get update

# Install prerequisites
apt-get install -y \\
    ca-certificates \\
    curl \\
    gnupg \\
    lsb-release

# Add Docker's official GPG key
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \\
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \\
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index with new repository
apt-get update

# Install Docker Engine
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add current user to docker group
usermod -aG docker $USER

# Install Docker Compose standalone
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Configure Docker daemon for production
cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "userland-proxy": false,
  "experimental": false,
  "live-restore": true
}
EOF

# Restart Docker to apply configuration
systemctl restart docker

echo "Docker installation completed successfully"
"""

    def _generate_rhel_docker_script(self) -> str:
        """Generate Docker installation script for RHEL-based systems."""
        return """#!/bin/bash
set -euo pipefail

# Update system
dnf update -y

# Install prerequisites
dnf install -y dnf-plugins-core

# Add Docker repository
dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Install Docker Engine
dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add current user to docker group
usermod -aG docker $USER

# Install Docker Compose standalone
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Configure Docker daemon for production
cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "userland-proxy": false,
  "experimental": false,
  "live-restore": true
}
EOF

# Configure firewall for Docker Swarm
firewall-cmd --permanent --add-port=2376/tcp  # Docker daemon TLS port
firewall-cmd --permanent --add-port=2377/tcp  # Docker Swarm management port
firewall-cmd --permanent --add-port=7946/tcp  # Swarm communication (TCP)
firewall-cmd --permanent --add-port=7946/udp  # Swarm communication (UDP)
firewall-cmd --permanent --add-port=4789/udp  # Overlay network traffic
firewall-cmd --reload

# Restart Docker to apply configuration
systemctl restart docker

echo "Docker installation completed successfully"
"""

    def generate_swarm_init_script(self, advertise_ip: str) -> str:
        """Generate script to initialize Docker Swarm."""
        return f"""#!/bin/bash
set -euo pipefail

echo "Initializing Docker Swarm cluster..."

# Initialize Docker Swarm
docker swarm init --advertise-addr {advertise_ip}

# Get join tokens
MANAGER_TOKEN=$(docker swarm join-token manager -q)
WORKER_TOKEN=$(docker swarm join-token worker -q)

# Save tokens to files
echo "$MANAGER_TOKEN" > /tmp/manager-token
echo "$WORKER_TOKEN" > /tmp/worker-token

# Create join commands
echo "docker swarm join --token $MANAGER_TOKEN {advertise_ip}:2377" > /tmp/manager-join-command
echo "docker swarm join --token $WORKER_TOKEN {advertise_ip}:2377" > /tmp/worker-join-command

echo "Swarm initialization completed"
echo "Manager join token: $MANAGER_TOKEN"
echo "Worker join token: $WORKER_TOKEN"
"""

    def generate_swarm_join_script(self, role: str, manager_ip: str, token: str) -> str:
        """Generate script to join Docker Swarm."""
        return f"""#!/bin/bash
set -euo pipefail

echo "Joining Docker Swarm as {role}..."

# Join Docker Swarm
docker swarm join --token {token} {manager_ip}:2377

echo "Successfully joined swarm as {role}"
"""

    def generate_service_deployment_script(self) -> str:
        """Generate script to deploy services."""
        if not self.config["services"]:
            return "#!/bin/bash\necho 'No services to deploy'"
        
        script = "#!/bin/bash\nset -euo pipefail\n\n"
        script += "echo 'Deploying Docker Swarm services...'\n\n"
        
        # Create networks first
        for network in self.config["networks"]:
            script += f"# Create network: {network['name']}\n"
            script += f"docker network create --driver {network['driver']}"
            if network.get("subnet"):
                script += f" --subnet {network['subnet']}"
            if network.get("attachable"):
                script += " --attachable"
            if network.get("encrypted"):
                script += " --opt encrypted"
            script += f" {network['name']} || true\n\n"
        
        # Create secrets
        for secret in self.config["secrets"]:
            script += f"# Create secret: {secret['name']}\n"
            script += f"echo '{secret['data']}' | base64 -d | docker secret create {secret['name']} - || true\n\n"
        
        # Deploy services
        for service in self.config["services"]:
            script += f"# Deploy service: {service['name']}\n"
            script += f"docker service create --name {service['name']}"
            script += f" --replicas {service['replicas']}"
            
            # Add port mappings
            for port in service["ports"]:
                script += f" --publish {port['published']}:{port['target']}"
                if port.get("protocol"):
                    script += f"/{port['protocol']}"
            
            # Add environment variables
            for key, value in service["environment"].items():
                script += f" --env {key}='{value}'"
            
            # Add network attachments
            for network in service["networks"]:
                script += f" --network {network}"
            
            # Add constraints
            for constraint in service["constraints"]:
                script += f" --constraint '{constraint}'"
            
            # Add volume mounts
            for volume in service["volumes"]:
                script += f" --mount {volume}"
            
            script += f" {service['image']}\n\n"
        
        script += "echo 'Service deployment completed'\n"
        return script


class DockerSwarmProvisioner:
    """Docker Swarm cluster provisioning and management."""
    
    def __init__(self, proxmox_client):
        """Initialize with Proxmox client."""
        self.client = proxmox_client

    def create_swarm_cluster(self, swarm_config: DockerSwarmConfig, node: str, 
                           storage: str, bridge: str, ssh_keys: List[str],
                           base_vmid: int = 800) -> Dict[str, Any]:
        """Create complete Docker Swarm cluster."""
        cluster_name = swarm_config.cluster_name
        base_os = swarm_config.base_os
        
        deployment_result = {
            "cluster_name": cluster_name,
            "base_os": base_os,
            "nodes": [],
            "upids": [],
            "join_tokens": {}
        }
        
        # Create nodes
        node_configs = []
        for i, node_config in enumerate(swarm_config.config["nodes"]):
            vmid = base_vmid + i
            vm_name = f"{cluster_name}-{node_config['name']}"
            
            # Create CloudInit configuration for Docker node
            cloudinit_config = self._create_docker_node_config(
                swarm_config, node_config, ssh_keys
            )
            
            # Create VM
            from .cloudinit import CloudInitProvisioner
            provisioner = CloudInitProvisioner(self.client)
            upid = provisioner.create_vm_with_cloudinit(
                node=node,
                vmid=vmid,
                name=vm_name,
                template=base_os,
                cloudinit_config=cloudinit_config,
                hardware={
                    "cores": node_config["cores"],
                    "memory_mb": node_config["memory_mb"],
                    "disk_gb": node_config["disk_gb"]
                },
                storage=storage,
                bridge=bridge
            )
            
            node_result = {
                "name": vm_name,
                "vmid": vmid,
                "role": node_config["role"],
                "upid": upid,
                "ip": node_config.get("ip", "dhcp")
            }
            
            deployment_result["nodes"].append(node_result)
            deployment_result["upids"].append(upid)
            node_configs.append(node_result)
        
        return deployment_result

    def _create_docker_node_config(self, swarm_config: DockerSwarmConfig, 
                                  node_config: Dict[str, Any], 
                                  ssh_keys: List[str]) -> CloudInitConfig:
        """Create CloudInit configuration for Docker Swarm node."""
        config = CloudInitConfig(swarm_config.base_os)
        
        # Set hostname
        hostname = f"{swarm_config.cluster_name}-{node_config['name']}"
        config.set_hostname(hostname)
        
        # Add user with SSH keys
        default_user = swarm_config.os_info["default_user"]
        config.add_user(default_user, ssh_keys, sudo="ALL=(ALL) NOPASSWD:ALL")
        
        # Add Docker installation packages
        if swarm_config.os_info["package_manager"] == "apt":
            packages = ["curl", "gnupg", "lsb-release", "ca-certificates"]
        else:
            packages = ["curl", "dnf-plugins-core"]
        
        config.add_packages(packages)
        
        # Add Docker installation script
        docker_script = swarm_config.generate_docker_install_script()
        config.add_file(
            "/usr/local/bin/install-docker.sh",
            docker_script,
            permissions="0755"
        )
        
        # Add commands to install Docker and configure node
        commands = [
            "/usr/local/bin/install-docker.sh",
            f"echo 'Docker installed on {hostname}' >> /var/log/docker-install.log"
        ]
        
        # Add role-specific configuration
        if node_config["role"] == "manager":
            # Managers need additional ports open
            if swarm_config.os_info["package_manager"] == "dnf":
                commands.extend([
                    "firewall-cmd --permanent --add-port=2376/tcp",
                    "firewall-cmd --permanent --add-port=2377/tcp", 
                    "firewall-cmd --permanent --add-port=7946/tcp",
                    "firewall-cmd --permanent --add-port=7946/udp",
                    "firewall-cmd --permanent --add-port=4789/udp",
                    "firewall-cmd --reload"
                ])
        
        config.add_commands(commands)
        
        return config

    def initialize_swarm(self, cluster_name: str, manager_vmid: int, 
                        manager_node: str, advertise_ip: str) -> Dict[str, Any]:
        """Initialize Docker Swarm on primary manager."""
        try:
            # Generate swarm init script
            swarm_config = DockerSwarmConfig(cluster_name)
            init_script = swarm_config.generate_swarm_init_script(advertise_ip)
            
            # Execute swarm init on manager
            result = self.client.qga_exec(
                manager_node, manager_vmid,
                command="bash",
                args=["-c", init_script]
            )
            
            # Extract tokens from result (this would need proper parsing)
            return {
                "swarm_initialized": True,
                "manager_ip": advertise_ip,
                "result": result
            }
        except Exception as e:
            return {
                "swarm_initialized": False,
                "error": str(e)
            }

    def join_swarm_node(self, vmid: int, node: str, role: str, 
                       manager_ip: str, token: str) -> Dict[str, Any]:
        """Join node to existing Docker Swarm."""
        try:
            swarm_config = DockerSwarmConfig("temp")
            join_script = swarm_config.generate_swarm_join_script(role, manager_ip, token)
            
            result = self.client.qga_exec(
                node, vmid,
                command="bash",
                args=["-c", join_script]
            )
            
            return {
                "node_joined": True,
                "role": role,
                "manager_ip": manager_ip,
                "result": result
            }
        except Exception as e:
            return {
                "node_joined": False,
                "error": str(e)
            }

    def deploy_swarm_services(self, manager_vmid: int, manager_node: str,
                             swarm_config: DockerSwarmConfig) -> Dict[str, Any]:
        """Deploy services to Docker Swarm."""
        try:
            deploy_script = swarm_config.generate_service_deployment_script()
            
            result = self.client.qga_exec(
                manager_node, manager_vmid,
                command="bash",
                args=["-c", deploy_script]
            )
            
            return {
                "services_deployed": True,
                "services": [service["name"] for service in swarm_config.config["services"]],
                "result": result
            }
        except Exception as e:
            return {
                "services_deployed": False,
                "error": str(e)
            }

    def get_swarm_status(self, manager_vmid: int, manager_node: str) -> Dict[str, Any]:
        """Get Docker Swarm cluster status."""
        try:
            # Get node status
            nodes_script = "docker node ls --format '{{.ID}},{{.Hostname}},{{.Status}},{{.Availability}},{{.ManagerStatus}}'"
            nodes_result = self.client.qga_exec(
                manager_node, manager_vmid,
                command="bash",
                args=["-c", nodes_script]
            )
            
            # Get service status
            services_script = "docker service ls --format '{{.Name}},{{.Mode}},{{.Replicas}},{{.Image}}'"
            services_result = self.client.qga_exec(
                manager_node, manager_vmid,
                command="bash",
                args=["-c", services_script]
            )
            
            return {
                "status_retrieved": True,
                "nodes": nodes_result,
                "services": services_result
            }
        except Exception as e:
            return {
                "status_retrieved": False,
                "error": str(e)
            }

    def scale_service(self, manager_vmid: int, manager_node: str, 
                     service_name: str, replicas: int) -> Dict[str, Any]:
        """Scale Docker Swarm service."""
        try:
            scale_script = f"docker service scale {service_name}={replicas}"
            
            result = self.client.qga_exec(
                manager_node, manager_vmid,
                command="bash",
                args=["-c", scale_script]
            )
            
            return {
                "service_scaled": True,
                "service": service_name,
                "replicas": replicas,
                "result": result
            }
        except Exception as e:
            return {
                "service_scaled": False,
                "error": str(e)
            }

    def remove_swarm_service(self, manager_vmid: int, manager_node: str,
                           service_name: str) -> Dict[str, Any]:
        """Remove service from Docker Swarm."""
        try:
            remove_script = f"docker service rm {service_name}"
            
            result = self.client.qga_exec(
                manager_node, manager_vmid,
                command="bash",
                args=["-c", remove_script]
            )
            
            return {
                "service_removed": True,
                "service": service_name,
                "result": result
            }
        except Exception as e:
            return {
                "service_removed": False,
                "error": str(e)
            }


# Predefined Docker Swarm configurations
def get_web_cluster_config(cluster_name: str, manager_count: int = 1, 
                          worker_count: int = 2) -> DockerSwarmConfig:
    """Pre-configured web application cluster."""
    config = DockerSwarmConfig(cluster_name, "ubuntu-22.04")
    
    # Add manager nodes
    for i in range(manager_count):
        config.add_node(f"manager-{i}", "manager", 800 + i, cores=2, memory_mb=2048, disk_gb=30)
    
    # Add worker nodes
    for i in range(worker_count):
        config.add_node(f"worker-{i}", "worker", 800 + manager_count + i, cores=2, memory_mb=4096, disk_gb=30)
    
    # Add overlay network for web services
    config.add_network("web-network", "overlay", attachable=True)
    
    # Add nginx load balancer service
    config.add_service(
        "nginx-lb",
        "nginx:alpine",
        replicas=1,
        ports=[{"published": 80, "target": 80}, {"published": 443, "target": 443}],
        networks=["web-network"]
    )
    
    return config


def get_development_cluster_config(cluster_name: str) -> DockerSwarmConfig:
    """Pre-configured development cluster."""
    config = DockerSwarmConfig(cluster_name, "ubuntu-22.04")
    
    # Single manager for development
    config.add_node("manager", "manager", 800, cores=2, memory_mb=4096, disk_gb=40)
    config.add_node("worker", "worker", 801, cores=2, memory_mb=4096, disk_gb=40)
    
    # Development network
    config.add_network("dev-network", "overlay", attachable=True)
    
    # Development services
    config.add_service(
        "portainer",
        "portainer/portainer-ce:latest",
        replicas=1,
        ports=[{"published": 9000, "target": 9000}],
        volumes=["type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock"],
        constraints=["node.role==manager"]
    )
    
    config.add_service(
        "registry",
        "registry:2",
        replicas=1,
        ports=[{"published": 5000, "target": 5000}],
        networks=["dev-network"]
    )
    
    return config


def get_production_cluster_config(cluster_name: str) -> DockerSwarmConfig:
    """Pre-configured production cluster with HA."""
    config = DockerSwarmConfig(cluster_name, "ubuntu-22.04")
    
    # Three managers for HA
    for i in range(3):
        config.add_node(f"manager-{i}", "manager", 800 + i, cores=4, memory_mb=4096, disk_gb=50)
    
    # Five workers for workload distribution
    for i in range(5):
        config.add_node(f"worker-{i}", "worker", 803 + i, cores=4, memory_mb=8192, disk_gb=100)
    
    # Production networks
    config.add_network("frontend", "overlay")
    config.add_network("backend", "overlay", encrypted=True)
    
    # Monitoring stack
    config.add_service(
        "prometheus",
        "prom/prometheus:latest",
        replicas=1,
        ports=[{"published": 9090, "target": 9090}],
        networks=["backend"],
        constraints=["node.role==manager"]
    )
    
    config.add_service(
        "grafana",
        "grafana/grafana:latest",
        replicas=1,
        ports=[{"published": 3000, "target": 3000}],
        networks=["frontend", "backend"],
        environment={"GF_SECURITY_ADMIN_PASSWORD": "admin123"}
    )
    
    return config
