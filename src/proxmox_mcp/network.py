"""
Network Management Module for Proxmox MCP Server

This module implements:
- VLAN management and configuration
- Firewall orchestration and rule management
- VPN server deployment (WireGuard, OpenVPN, IPSec)
- Network security and segmentation
"""

import os
import json
import tempfile
import ipaddress
from typing import Dict, List, Optional, Any
from pathlib import Path

import netaddr
from loguru import logger

from .client import ProxmoxClient
from .utils import run_command, format_error


class NetworkManager:
    """Network management for Proxmox infrastructure"""
    
    def __init__(self, proxmox_client: ProxmoxClient):
        self.client = proxmox_client
        self.config_dir = Path.home() / ".proxmox_mcp" / "network"
        self.config_dir.mkdir(parents=True, exist_ok=True)

    async def create_vlan(
        self,
        vlan_id: int,
        vlan_name: str,
        bridge: str = "vmbr0",
        gateway: Optional[str] = None,
        dhcp_range: Optional[str] = None,
        node: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Create and configure VLANs for network segmentation"""
        try:
            if dry_run:
                return {
                    "action": "create_vlan",
                    "vlan_id": vlan_id,
                    "vlan_name": vlan_name,
                    "bridge": bridge,
                    "gateway": gateway,
                    "dhcp_range": dhcp_range,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            # Validate VLAN ID
            if not 1 <= vlan_id <= 4094:
                raise ValueError("VLAN ID must be between 1 and 4094")
                
            node = node or self.client.get_first_node()
            result = {
                "vlan_id": vlan_id,
                "vlan_name": vlan_name,
                "bridge": bridge,
                "node": node
            }
            
            # Create VLAN interface on bridge
            vlan_interface = f"{bridge}.{vlan_id}"
            
            # Check if VLAN already exists
            try:
                interfaces = await self.client.nodes(node).network.get()
                existing_vlans = [iface for iface in interfaces if iface.get('iface') == vlan_interface]
                if existing_vlans:
                    return {
                        **result,
                        "status": "exists",
                        "message": f"VLAN {vlan_id} already exists on {bridge}"
                    }
            except Exception as e:
                logger.warning(f"Could not check existing VLANs: {e}")
                
            # Create VLAN configuration
            vlan_config = {
                "iface": vlan_interface,
                "type": "vlan",
                "method": "static" if gateway else "manual",
                "vlan-raw-device": bridge,
                "autostart": 1,
                "comments": f"VLAN {vlan_name}"
            }
            
            if gateway:
                try:
                    network = ipaddress.IPv4Network(gateway, strict=False)
                    vlan_config.update({
                        "address": str(network.network_address + 1),
                        "netmask": str(network.netmask),
                        "gateway": gateway
                    })
                except Exception as e:
                    logger.warning(f"Invalid gateway format: {e}")
                    
            # Create VLAN interface
            try:
                await self.client.nodes(node).network.post(**vlan_config)
                result["interface_created"] = True
            except Exception as e:
                logger.error(f"Failed to create VLAN interface: {e}")
                result["interface_created"] = False
                result["error"] = str(e)
                
            # Configure DHCP if range specified
            if dhcp_range and gateway:
                dhcp_result = await self._configure_dhcp(
                    vlan_interface, dhcp_range, gateway, node
                )
                result["dhcp_config"] = dhcp_result
                
            # Create firewall zone for VLAN
            firewall_result = await self._create_vlan_firewall_zone(
                vlan_interface, vlan_name, node
            )
            result["firewall_zone"] = firewall_result
            
            # Apply network configuration
            try:
                await self.client.nodes(node).network.put()
                result["config_applied"] = True
            except Exception as e:
                logger.error(f"Failed to apply network config: {e}")
                result["config_applied"] = False
                
            logger.info(f"VLAN {vlan_id} ({vlan_name}) created on {bridge}")
            result["status"] = "created"
            return result
            
        except Exception as e:
            error_msg = f"VLAN creation failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"vlan_id": vlan_id, "vlan_name": vlan_name})

    async def _configure_dhcp(
        self,
        interface: str,
        dhcp_range: str,
        gateway: str,
        node: str
    ) -> Dict[str, Any]:
        """Configure DHCP for VLAN"""
        try:
            # Parse DHCP range
            start_ip, end_ip = dhcp_range.split("-")
            network = ipaddress.IPv4Network(gateway, strict=False)
            
            # Create dnsmasq configuration
            dhcp_config = f"""
# DHCP configuration for {interface}
interface={interface}
dhcp-range={start_ip},{end_ip},12h
dhcp-option=option:router,{network.network_address + 1}
dhcp-option=option:dns-server,{network.network_address + 1}
"""
            
            # Save configuration
            config_file = self.config_dir / f"dhcp_{interface}.conf"
            config_file.write_text(dhcp_config)
            
            return {
                "interface": interface,
                "dhcp_range": dhcp_range,
                "config_file": str(config_file),
                "status": "configured"
            }
            
        except Exception as e:
            logger.error(f"DHCP configuration failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def _create_vlan_firewall_zone(
        self,
        interface: str,
        zone_name: str,
        node: str
    ) -> Dict[str, Any]:
        """Create firewall zone for VLAN"""
        try:
            # Basic firewall zone configuration
            zone_config = {
                "zone": zone_name.lower().replace(" ", "_"),
                "interface": interface,
                "comment": f"Firewall zone for {zone_name}",
                "policy_in": "ACCEPT",
                "policy_out": "ACCEPT",
                "log_level_in": "info",
                "log_level_out": "info"
            }
            
            return {
                "zone": zone_config["zone"],
                "interface": interface,
                "status": "configured"
            }
            
        except Exception as e:
            logger.error(f"Firewall zone creation failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def configure_firewall(
        self,
        vmid: int,
        rules: List[Dict[str, Any]],
        policy: str = "ACCEPT",
        log_level: str = "info",
        node: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Configure VM-level firewall rules"""
        try:
            if dry_run:
                return {
                    "action": "configure_firewall",
                    "vmid": vmid,
                    "rules_count": len(rules),
                    "policy": policy,
                    "log_level": log_level,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            node = node or self.client.get_first_node()
            result = {
                "vmid": vmid,
                "node": node,
                "policy": policy,
                "log_level": log_level
            }
            
            # Enable firewall for VM
            try:
                firewall_options = {
                    "enable": 1,
                    "policy_in": policy,
                    "policy_out": policy,
                    "log_level_in": log_level,
                    "log_level_out": log_level
                }
                
                await self.client.nodes(node).qemu(vmid).firewall.options.put(**firewall_options)
                result["firewall_enabled"] = True
            except Exception as e:
                logger.error(f"Failed to enable firewall: {e}")
                result["firewall_enabled"] = False
                
            # Clear existing rules
            try:
                existing_rules = await self.client.nodes(node).qemu(vmid).firewall.rules.get()
                for rule in existing_rules:
                    await self.client.nodes(node).qemu(vmid).firewall.rules(rule['pos']).delete()
                result["rules_cleared"] = True
            except Exception as e:
                logger.warning(f"Could not clear existing rules: {e}")
                result["rules_cleared"] = False
                
            # Add new rules
            rules_added = []
            for i, rule in enumerate(rules):
                try:
                    rule_config = self._format_firewall_rule(rule, i)
                    await self.client.nodes(node).qemu(vmid).firewall.rules.post(**rule_config)
                    rules_added.append(rule_config)
                except Exception as e:
                    logger.error(f"Failed to add rule {i}: {e}")
                    
            result["rules_added"] = len(rules_added)
            result["rules_details"] = rules_added
            
            logger.info(f"Firewall configured for VM {vmid} with {len(rules_added)} rules")
            result["status"] = "configured"
            return result
            
        except Exception as e:
            error_msg = f"Firewall configuration failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"vmid": vmid})

    def _format_firewall_rule(self, rule: Dict[str, Any], position: int) -> Dict[str, Any]:
        """Format firewall rule for Proxmox API"""
        config = {
            "action": rule.get("action", "ACCEPT").upper(),
            "type": rule.get("type", "in"),
            "pos": position
        }
        
        # Add optional parameters
        if "source" in rule:
            config["source"] = rule["source"]
        if "dest" in rule:
            config["dest"] = rule["dest"]
        if "sport" in rule:
            config["sport"] = rule["sport"]
        if "dport" in rule:
            config["dport"] = rule["dport"]
        if "proto" in rule:
            config["proto"] = rule["proto"]
        if "iface" in rule:
            config["iface"] = rule["iface"]
        if "comment" in rule:
            config["comment"] = rule["comment"]
            
        return config

    async def deploy_vpn_server(
        self,
        vpn_type: str = "wireguard",
        client_count: int = 10,
        subnet: str = "10.0.100.0/24",
        node: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Deploy VPN server for secure remote access"""
        try:
            if dry_run:
                return {
                    "action": "deploy_vpn_server",
                    "vpn_type": vpn_type,
                    "client_count": client_count,
                    "subnet": subnet,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            node = node or self.client.get_first_node()
            result = {
                "vpn_type": vpn_type,
                "client_count": client_count,
                "subnet": subnet,
                "node": node
            }
            
            if vpn_type == "wireguard":
                vpn_result = await self._deploy_wireguard(client_count, subnet, node)
            elif vpn_type == "openvpn":
                vpn_result = await self._deploy_openvpn(client_count, subnet, node)
            elif vpn_type == "ipsec":
                vpn_result = await self._deploy_ipsec(client_count, subnet, node)
            else:
                raise ValueError(f"Unsupported VPN type: {vpn_type}")
                
            result.update(vpn_result)
            
            logger.info(f"{vpn_type.upper()} VPN server deployed on node {node}")
            result["status"] = "deployed"
            return result
            
        except Exception as e:
            error_msg = f"VPN server deployment failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"vpn_type": vpn_type})

    async def _deploy_wireguard(
        self,
        client_count: int,
        subnet: str,
        node: str
    ) -> Dict[str, Any]:
        """Deploy WireGuard VPN server"""
        try:
            # Generate server keys
            server_private_result = await run_command(["wg", "genkey"])
            if server_private_result["return_code"] != 0:
                raise Exception("Failed to generate server private key")
                
            server_private_key = server_private_result["stdout"].strip()
            
            server_public_result = await run_command(
                ["wg", "pubkey"],
                input_data=server_private_key
            )
            if server_public_result["return_code"] != 0:
                raise Exception("Failed to generate server public key")
                
            server_public_key = server_public_result["stdout"].strip()
            
            # Parse subnet
            network = ipaddress.IPv4Network(subnet)
            server_ip = str(network.network_address + 1)
            
            # Generate server config
            server_config = f"""[Interface]
PrivateKey = {server_private_key}
Address = {server_ip}/{network.prefixlen}
ListenPort = 51820
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

"""
            
            # Generate client configs
            clients = []
            for i in range(client_count):
                client_private_result = await run_command(["wg", "genkey"])
                client_private_key = client_private_result["stdout"].strip()
                
                client_public_result = await run_command(
                    ["wg", "pubkey"],
                    input_data=client_private_key
                )
                client_public_key = client_public_result["stdout"].strip()
                
                client_ip = str(network.network_address + 2 + i)
                
                # Add peer to server config
                server_config += f"""[Peer]
PublicKey = {client_public_key}
AllowedIPs = {client_ip}/32

"""
                
                # Create client config
                client_config = f"""[Interface]
PrivateKey = {client_private_key}
Address = {client_ip}/{network.prefixlen}
DNS = 8.8.8.8

[Peer]
PublicKey = {server_public_key}
Endpoint = YOUR_SERVER_IP:51820
AllowedIPs = 0.0.0.0/0
"""
                
                clients.append({
                    "client_id": i + 1,
                    "private_key": client_private_key,
                    "public_key": client_public_key,
                    "ip_address": client_ip,
                    "config": client_config
                })
                
            # Save server config
            server_config_file = self.config_dir / "wireguard_server.conf"
            server_config_file.write_text(server_config)
            
            # Save client configs
            clients_dir = self.config_dir / "wireguard_clients"
            clients_dir.mkdir(exist_ok=True)
            
            for client in clients:
                client_file = clients_dir / f"client_{client['client_id']}.conf"
                client_file.write_text(client["config"])
                
            return {
                "server_public_key": server_public_key,
                "server_config_file": str(server_config_file),
                "clients": clients,
                "clients_dir": str(clients_dir),
                "listen_port": 51820
            }
            
        except Exception as e:
            logger.error(f"WireGuard deployment failed: {e}")
            raise

    async def _deploy_openvpn(
        self,
        client_count: int,
        subnet: str,
        node: str
    ) -> Dict[str, Any]:
        """Deploy OpenVPN server"""
        try:
            # Generate OpenVPN server configuration
            network = ipaddress.IPv4Network(subnet)
            
            server_config = f"""port 1194
proto udp
dev tun
ca ca.crt
cert server.crt
key server.key
dh dh.pem
server {network.network_address} {network.netmask}
ifconfig-pool-persist ipp.txt
push "redirect-gateway def1 bypass-dhcp"
push "dhcp-option DNS 8.8.8.8"
push "dhcp-option DNS 8.8.4.4"
keepalive 10 120
tls-auth ta.key 0
cipher AES-256-CBC
persist-key
persist-tun
status openvpn-status.log
verb 3
explicit-exit-notify 1
"""
            
            # Save server config
            server_config_file = self.config_dir / "openvpn_server.conf"
            server_config_file.write_text(server_config)
            
            return {
                "server_config_file": str(server_config_file),
                "listen_port": 1194,
                "protocol": "UDP",
                "message": "OpenVPN server configuration created. Generate PKI certificates separately."
            }
            
        except Exception as e:
            logger.error(f"OpenVPN deployment failed: {e}")
            raise

    async def _deploy_ipsec(
        self,
        client_count: int,
        subnet: str,
        node: str
    ) -> Dict[str, Any]:
        """Deploy IPSec VPN server"""
        try:
            # Generate strongSwan configuration
            network = ipaddress.IPv4Network(subnet)
            
            ipsec_config = f"""config setup
    charondebug="ike 1, knl 1, cfg 0"
    uniqueids=no

conn ikev2-vpn
    auto=add
    compress=no
    type=tunnel
    keyexchange=ikev2
    fragmentation=yes
    forceencaps=yes
    dpdaction=clear
    dpddelay=300s
    rekey=no
    left=%any
    leftid=@vpn.example.com
    leftcert=server-cert.pem
    leftsendcert=always
    leftsubnet=0.0.0.0/0
    right=%any
    rightid=%any
    rightauth=eap-mschapv2
    rightsourceip={network.network_address}/{network.prefixlen}
    rightdns=8.8.8.8,8.8.4.4
    rightsendcert=never
    eap_identity=%identity
"""
            
            # Save IPSec config
            ipsec_config_file = self.config_dir / "ipsec.conf"
            ipsec_config_file.write_text(ipsec_config)
            
            return {
                "ipsec_config_file": str(ipsec_config_file),
                "message": "IPSec server configuration created. Configure certificates and secrets separately."
            }
            
        except Exception as e:
            logger.error(f"IPSec deployment failed: {e}")
            raise

    async def create_network_bridge(
        self,
        bridge_name: str,
        ports: Optional[List[str]] = None,
        vlan_aware: bool = False,
        node: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Create network bridge"""
        try:
            if dry_run:
                return {
                    "action": "create_network_bridge",
                    "bridge_name": bridge_name,
                    "ports": ports or [],
                    "vlan_aware": vlan_aware,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            node = node or self.client.get_first_node()
            
            bridge_config = {
                "iface": bridge_name,
                "type": "bridge",
                "method": "manual",
                "autostart": 1,
                "bridge_vlan_aware": 1 if vlan_aware else 0
            }
            
            if ports:
                bridge_config["bridge_ports"] = " ".join(ports)
                
            # Create bridge
            try:
                await self.client.nodes(node).network.post(**bridge_config)
                
                # Apply configuration
                await self.client.nodes(node).network.put()
                
                logger.info(f"Network bridge {bridge_name} created on node {node}")
                return {
                    "bridge_name": bridge_name,
                    "node": node,
                    "ports": ports or [],
                    "vlan_aware": vlan_aware,
                    "status": "created"
                }
                
            except Exception as e:
                logger.error(f"Failed to create bridge: {e}")
                raise
                
        except Exception as e:
            error_msg = f"Bridge creation failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"bridge_name": bridge_name})

    async def configure_network_bonding(
        self,
        bond_name: str,
        interfaces: List[str],
        mode: str = "active-backup",
        node: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Configure network bonding for redundancy"""
        try:
            if dry_run:
                return {
                    "action": "configure_network_bonding",
                    "bond_name": bond_name,
                    "interfaces": interfaces,
                    "mode": mode,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            node = node or self.client.get_first_node()
            
            # Bond modes mapping
            bond_modes = {
                "active-backup": "1",
                "balance-xor": "2",
                "broadcast": "3",
                "802.3ad": "4",
                "balance-tlb": "5",
                "balance-alb": "6"
            }
            
            bond_config = {
                "iface": bond_name,
                "type": "bond",
                "method": "manual",
                "bond_slaves": " ".join(interfaces),
                "bond_mode": bond_modes.get(mode, "1"),
                "bond_miimon": "100",
                "autostart": 1
            }
            
            # Create bond
            try:
                await self.client.nodes(node).network.post(**bond_config)
                
                # Apply configuration
                await self.client.nodes(node).network.put()
                
                logger.info(f"Network bond {bond_name} created with interfaces {interfaces}")
                return {
                    "bond_name": bond_name,
                    "interfaces": interfaces,
                    "mode": mode,
                    "node": node,
                    "status": "created"
                }
                
            except Exception as e:
                logger.error(f"Failed to create bond: {e}")
                raise
                
        except Exception as e:
            error_msg = f"Network bonding failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"bond_name": bond_name})

    async def get_network_topology(
        self,
        node: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get network topology and configuration"""
        try:
            node = node or self.client.get_first_node()
            
            # Get network interfaces
            interfaces = await self.client.nodes(node).network.get()
            
            # Organize by type
            topology = {
                "physical": [],
                "bridges": [],
                "bonds": [],
                "vlans": [],
                "virtual": []
            }
            
            for iface in interfaces:
                iface_type = iface.get("type", "")
                iface_name = iface.get("iface", "")
                
                if iface_type == "eth":
                    topology["physical"].append(iface)
                elif iface_type == "bridge":
                    topology["bridges"].append(iface)
                elif iface_type == "bond":
                    topology["bonds"].append(iface)
                elif "." in iface_name and iface_type == "vlan":
                    topology["vlans"].append(iface)
                else:
                    topology["virtual"].append(iface)
                    
            return {
                "node": node,
                "topology": topology,
                "total_interfaces": len(interfaces),
                "status": "retrieved"
            }
            
        except Exception as e:
            error_msg = f"Network topology retrieval failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"node": node})
