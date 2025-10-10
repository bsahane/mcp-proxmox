"""Windows VM provisioning and configuration module."""

from __future__ import annotations

import os
import base64
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import requests


class WindowsConfig:
    """Windows VM configuration and provisioning helper."""
    
    # Windows Server versions and ISO URLs
    WINDOWS_VERSIONS = {
        "server-2019": {
            "name": "Windows Server 2019",
            "iso_name": "windows-server-2019.iso",
            "virtio_version": "0.1.240",
            "default_edition": "ServerDatacenter",
            "architecture": "amd64"
        },
        "server-2022": {
            "name": "Windows Server 2022", 
            "iso_name": "windows-server-2022.iso",
            "virtio_version": "0.1.240",
            "default_edition": "ServerDatacenter",
            "architecture": "amd64"
        },
        "server-2025": {
            "name": "Windows Server 2025",
            "iso_name": "windows-server-2025.iso", 
            "virtio_version": "0.1.248",
            "default_edition": "ServerDatacenter",
            "architecture": "amd64"
        }
    }

    # VirtIO driver URLs
    VIRTIO_DRIVERS = {
        "0.1.240": "https://fedorapeople.org/groups/virt/virtio-win/direct-downloads/archive-virtio/virtio-win-0.1.240/virtio-win-0.1.240.iso",
        "0.1.248": "https://fedorapeople.org/groups/virt/virtio-win/direct-downloads/archive-virtio/virtio-win-0.1.248/virtio-win-0.1.248.iso"
    }

    def __init__(self, version: str = "server-2022"):
        """Initialize Windows configuration."""
        if version not in self.WINDOWS_VERSIONS:
            raise ValueError(f"Unsupported Windows version: {version}. Supported: {list(self.WINDOWS_VERSIONS.keys())}")
        
        self.version = version
        self.version_info = self.WINDOWS_VERSIONS[version]
        self.config = {
            "version": version,
            "admin_password": "",
            "computer_name": "",
            "domain": None,
            "domain_user": None,
            "domain_password": None,
            "timezone": "UTC",
            "locale": "en-US",
            "applications": [],
            "windows_features": [],
            "users": [],
            "firewall_rules": []
        }

    def set_admin_password(self, password: str) -> None:
        """Set administrator password."""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        self.config["admin_password"] = password

    def set_computer_name(self, name: str) -> None:
        """Set computer name."""
        if len(name) > 15:
            raise ValueError("Computer name must be 15 characters or less")
        self.config["computer_name"] = name

    def set_domain_config(self, domain: str, username: str, password: str, 
                         ou_path: Optional[str] = None) -> None:
        """Configure domain joining."""
        self.config["domain"] = domain
        self.config["domain_user"] = username
        self.config["domain_password"] = password
        if ou_path:
            self.config["domain_ou"] = ou_path

    def add_user(self, username: str, password: str, full_name: str = "", 
                description: str = "", admin: bool = False) -> None:
        """Add local user account."""
        user = {
            "username": username,
            "password": password,
            "full_name": full_name,
            "description": description,
            "admin": admin
        }
        self.config["users"].append(user)

    def add_application(self, name: str, installer_url: str, silent_args: str = "/S") -> None:
        """Add application to install."""
        app = {
            "name": name,
            "installer_url": installer_url,
            "silent_args": silent_args
        }
        self.config["applications"].append(app)

    def add_windows_feature(self, feature_name: str) -> None:
        """Add Windows feature to enable."""
        self.config["windows_features"].append(feature_name)

    def add_firewall_rule(self, name: str, port: int, protocol: str = "TCP", 
                         action: str = "Allow", direction: str = "Inbound") -> None:
        """Add firewall rule."""
        rule = {
            "name": name,
            "port": port,
            "protocol": protocol,
            "action": action,
            "direction": direction
        }
        self.config["firewall_rules"].append(rule)

    def generate_autounattend_xml(self) -> str:
        """Generate autounattend.xml for unattended Windows installation."""
        # Create root element
        root = ET.Element("unattend")
        root.set("xmlns", "urn:schemas-microsoft-com:unattend")
        
        # Windows PE pass
        winpe_settings = ET.SubElement(root, "settings")
        winpe_settings.set("pass", "windowsPE")
        
        # International settings
        intl_component = ET.SubElement(winpe_settings, "component")
        intl_component.set("name", "Microsoft-Windows-International-Core-WinPE")
        intl_component.set("processorArchitecture", self.version_info["architecture"])
        intl_component.set("publicKeyToken", "31bf3856ad364e35")
        intl_component.set("language", "neutral")
        intl_component.set("versionScope", "nonSxS")
        
        ET.SubElement(intl_component, "SetupUILanguage").text = "en-US"
        ET.SubElement(intl_component, "InputLocale").text = self.config["locale"]
        ET.SubElement(intl_component, "SystemLocale").text = self.config["locale"] 
        ET.SubElement(intl_component, "UILanguage").text = "en-US"
        ET.SubElement(intl_component, "UserLocale").text = self.config["locale"]
        
        # Windows Setup
        setup_component = ET.SubElement(winpe_settings, "component")
        setup_component.set("name", "Microsoft-Windows-Setup")
        setup_component.set("processorArchitecture", self.version_info["architecture"])
        setup_component.set("publicKeyToken", "31bf3856ad364e35")
        setup_component.set("language", "neutral")
        setup_component.set("versionScope", "nonSxS")
        
        # Disk configuration
        disk_config = ET.SubElement(setup_component, "DiskConfiguration")
        disk = ET.SubElement(disk_config, "Disk")
        disk.set("wcm:action", "add")
        ET.SubElement(disk, "DiskID").text = "0"
        ET.SubElement(disk, "WillWipeDisk").text = "true"
        
        # Create partitions
        create_partitions = ET.SubElement(disk, "CreatePartitions")
        
        # EFI partition
        efi_partition = ET.SubElement(create_partitions, "CreatePartition")
        efi_partition.set("wcm:action", "add")
        ET.SubElement(efi_partition, "Order").text = "1"
        ET.SubElement(efi_partition, "Type").text = "EFI"
        ET.SubElement(efi_partition, "Size").text = "100"
        
        # MSR partition
        msr_partition = ET.SubElement(create_partitions, "CreatePartition")
        msr_partition.set("wcm:action", "add")
        ET.SubElement(msr_partition, "Order").text = "2"
        ET.SubElement(msr_partition, "Type").text = "MSR"
        ET.SubElement(msr_partition, "Size").text = "16"
        
        # Windows partition
        win_partition = ET.SubElement(create_partitions, "CreatePartition")
        win_partition.set("wcm:action", "add")
        ET.SubElement(win_partition, "Order").text = "3"
        ET.SubElement(win_partition, "Type").text = "Primary"
        ET.SubElement(win_partition, "Extend").text = "true"
        
        # Modify partitions
        modify_partitions = ET.SubElement(disk, "ModifyPartitions")
        
        # EFI format
        efi_modify = ET.SubElement(modify_partitions, "ModifyPartition")
        efi_modify.set("wcm:action", "add")
        ET.SubElement(efi_modify, "Order").text = "1"
        ET.SubElement(efi_modify, "PartitionID").text = "1"
        ET.SubElement(efi_modify, "Label").text = "System"
        ET.SubElement(efi_modify, "Format").text = "FAT32"
        
        # Windows format
        win_modify = ET.SubElement(modify_partitions, "ModifyPartition")
        win_modify.set("wcm:action", "add")
        ET.SubElement(win_modify, "Order").text = "3"
        ET.SubElement(win_modify, "PartitionID").text = "3"
        ET.SubElement(win_modify, "Label").text = "Windows"
        ET.SubElement(win_modify, "Format").text = "NTFS"
        ET.SubElement(win_modify, "Letter").text = "C"
        
        # Image install
        image_install = ET.SubElement(setup_component, "ImageInstall")
        os_image = ET.SubElement(image_install, "OSImage")
        install_to = ET.SubElement(os_image, "InstallTo")
        ET.SubElement(install_to, "DiskID").text = "0"
        ET.SubElement(install_to, "PartitionID").text = "3"
        
        install_from = ET.SubElement(os_image, "InstallFrom")
        metadata = ET.SubElement(install_from, "MetaData")
        metadata.set("wcm:action", "add")
        ET.SubElement(metadata, "Key").text = "/IMAGE/NAME"
        ET.SubElement(metadata, "Value").text = self.version_info["default_edition"]
        
        # User data
        user_data = ET.SubElement(setup_component, "UserData")
        ET.SubElement(user_data, "AcceptEula").text = "true"
        
        # Product key (will be filled by user)
        product_key = ET.SubElement(user_data, "ProductKey")
        ET.SubElement(product_key, "Key").text = ""
        ET.SubElement(product_key, "WillShowUI").text = "OnError"
        
        # OOBE System pass
        oobe_settings = ET.SubElement(root, "settings")
        oobe_settings.set("pass", "oobeSystem")
        
        # OOBE component
        oobe_component = ET.SubElement(oobe_settings, "component")
        oobe_component.set("name", "Microsoft-Windows-Shell-Setup")
        oobe_component.set("processorArchitecture", self.version_info["architecture"])
        oobe_component.set("publicKeyToken", "31bf3856ad364e35")
        oobe_component.set("language", "neutral")
        oobe_component.set("versionScope", "nonSxS")
        
        # OOBE settings
        oobe = ET.SubElement(oobe_component, "OOBE")
        ET.SubElement(oobe, "HideEULAPage").text = "true"
        ET.SubElement(oobe, "HideOEMRegistrationScreen").text = "true"
        ET.SubElement(oobe, "HideOnlineAccountScreens").text = "true"
        ET.SubElement(oobe, "HideWirelessSetupInOOBE").text = "true"
        ET.SubElement(oobe, "NetworkLocation").text = "Work"
        ET.SubElement(oobe, "ProtectYourPC").text = "1"
        
        # User accounts
        user_accounts = ET.SubElement(oobe_component, "UserAccounts")
        admin_password = ET.SubElement(user_accounts, "AdministratorPassword")
        ET.SubElement(admin_password, "Value").text = self.config["admin_password"]
        ET.SubElement(admin_password, "PlainText").text = "true"
        
        # Computer name
        if self.config["computer_name"]:
            ET.SubElement(oobe_component, "ComputerName").text = self.config["computer_name"]
        
        # Time zone
        ET.SubElement(oobe_component, "TimeZone").text = self.config["timezone"]
        
        # First logon commands
        first_logon = ET.SubElement(oobe_component, "FirstLogonCommands")
        
        # Enable RDP
        rdp_command = ET.SubElement(first_logon, "SynchronousCommand")
        rdp_command.set("wcm:action", "add")
        ET.SubElement(rdp_command, "CommandLine").text = 'reg add "HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Terminal Server" /v fDenyTSConnections /t REG_DWORD /d 0 /f'
        ET.SubElement(rdp_command, "Order").text = "1"
        ET.SubElement(rdp_command, "Description").text = "Enable RDP"
        
        # Configure Windows firewall for RDP
        firewall_command = ET.SubElement(first_logon, "SynchronousCommand")
        firewall_command.set("wcm:action", "add")
        ET.SubElement(firewall_command, "CommandLine").text = 'netsh advfirewall firewall set rule group="remote desktop" new enable=Yes'
        ET.SubElement(firewall_command, "Order").text = "2"
        ET.SubElement(firewall_command, "Description").text = "Enable RDP Firewall"
        
        # Install VirtIO drivers
        virtio_command = ET.SubElement(first_logon, "SynchronousCommand")
        virtio_command.set("wcm:action", "add")
        ET.SubElement(virtio_command, "CommandLine").text = 'powershell.exe -ExecutionPolicy Bypass -File C:\\Windows\\Setup\\Scripts\\install-virtio.ps1'
        ET.SubElement(virtio_command, "Order").text = "3"
        ET.SubElement(virtio_command, "Description").text = "Install VirtIO drivers"
        
        # Generate XML string
        ET.register_namespace("wcm", "http://schemas.microsoft.com/WMIConfig/2002/State")
        root.set("xmlns:wcm", "http://schemas.microsoft.com/WMIConfig/2002/State")
        
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def generate_virtio_install_script(self) -> str:
        """Generate PowerShell script to install VirtIO drivers."""
        script = """
# VirtIO Driver Installation Script
Write-Host "Installing VirtIO drivers..."

# Mount VirtIO ISO
$virtio_drive = Get-Volume | Where-Object {$_.FileSystemLabel -eq "virtio-win"} | Select-Object -First 1
if ($virtio_drive) {
    $drive_letter = $virtio_drive.DriveLetter + ":"
    Write-Host "VirtIO ISO mounted at $drive_letter"
    
    # Install network driver
    $network_driver = "$drive_letter\\NetKVM\\w10\\amd64\\netkvm.inf"
    if (Test-Path $network_driver) {
        pnputil /add-driver $network_driver /install
        Write-Host "Network driver installed"
    }
    
    # Install balloon driver
    $balloon_driver = "$drive_letter\\Balloon\\w10\\amd64\\balloon.inf"
    if (Test-Path $balloon_driver) {
        pnputil /add-driver $balloon_driver /install
        Write-Host "Balloon driver installed"
    }
    
    # Install storage driver
    $storage_driver = "$drive_letter\\viostor\\w10\\amd64\\viostor.inf"
    if (Test-Path $storage_driver) {
        pnputil /add-driver $storage_driver /install
        Write-Host "Storage driver installed"
    }
    
    # Install QEMU guest agent
    $guest_agent = "$drive_letter\\guest-agent\\qemu-ga-x86_64.msi"
    if (Test-Path $guest_agent) {
        Start-Process msiexec.exe -ArgumentList "/i $guest_agent /quiet" -Wait
        Write-Host "QEMU Guest Agent installed"
    }
} else {
    Write-Host "VirtIO ISO not found"
}

# Enable PowerShell remoting
Enable-PSRemoting -Force -SkipNetworkProfileCheck
Set-NetFirewallRule -DisplayName "Windows Remote Management (HTTP-In)" -Enabled True

Write-Host "VirtIO driver installation completed"
"""
        return script

    def generate_domain_join_script(self) -> str:
        """Generate PowerShell script for domain joining."""
        if not self.config["domain"]:
            return ""
        
        script = f"""
# Domain Join Script
Write-Host "Joining domain {self.config['domain']}..."

try {{
    $secpasswd = ConvertTo-SecureString "{self.config['domain_password']}" -AsPlainText -Force
    $credential = New-Object System.Management.Automation.PSCredential("{self.config['domain_user']}", $secpasswd)
    
"""
        
        if self.config.get("domain_ou"):
            script += f'    Add-Computer -DomainName "{self.config["domain"]}" -OUPath "{self.config["domain_ou"]}" -Credential $credential -Force\n'
        else:
            script += f'    Add-Computer -DomainName "{self.config["domain"]}" -Credential $credential -Force\n'
        
        script += """
    Write-Host "Successfully joined domain"
    Restart-Computer -Force
} catch {
    Write-Host "Failed to join domain: $($_.Exception.Message)"
}
"""
        return script

    def generate_app_install_script(self) -> str:
        """Generate PowerShell script for application installation."""
        if not self.config["applications"]:
            return ""
        
        script = "# Application Installation Script\n"
        script += "Write-Host 'Installing applications...'\n\n"
        
        for app in self.config["applications"]:
            script += f"""
# Install {app['name']}
Write-Host "Installing {app['name']}..."
try {{
    $temp_path = "$env:TEMP\\{app['name']}-installer.exe"
    Invoke-WebRequest -Uri "{app['installer_url']}" -OutFile $temp_path
    Start-Process $temp_path -ArgumentList "{app['silent_args']}" -Wait
    Remove-Item $temp_path -Force
    Write-Host "{app['name']} installed successfully"
}} catch {{
    Write-Host "Failed to install {app['name']}: $($_.Exception.Message)"
}}

"""
        
        return script

    def create_setup_iso(self, output_path: str, license_key: Optional[str] = None) -> str:
        """Create Windows setup ISO with autounattend.xml and scripts."""
        import subprocess
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate autounattend.xml
            autounattend_xml = self.generate_autounattend_xml()
            if license_key:
                # Replace empty product key with actual key
                autounattend_xml = autounattend_xml.replace("<Key></Key>", f"<Key>{license_key}</Key>")
            
            autounattend_path = os.path.join(temp_dir, "autounattend.xml")
            with open(autounattend_path, "w", encoding="utf-8") as f:
                f.write(autounattend_xml)
            
            # Create scripts directory
            scripts_dir = os.path.join(temp_dir, "Scripts")
            os.makedirs(scripts_dir)
            
            # Generate VirtIO install script
            virtio_script = self.generate_virtio_install_script()
            with open(os.path.join(scripts_dir, "install-virtio.ps1"), "w") as f:
                f.write(virtio_script)
            
            # Generate domain join script
            if self.config["domain"]:
                domain_script = self.generate_domain_join_script()
                with open(os.path.join(scripts_dir, "join-domain.ps1"), "w") as f:
                    f.write(domain_script)
            
            # Generate app install script
            if self.config["applications"]:
                app_script = self.generate_app_install_script()
                with open(os.path.join(scripts_dir, "install-apps.ps1"), "w") as f:
                    f.write(app_script)
            
            # Create ISO
            cmd = [
                "genisoimage", "-output", output_path, "-volid", "WINSETUP",
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


class WindowsProvisioner:
    """Windows VM provisioning with automated installation."""
    
    def __init__(self, proxmox_client):
        """Initialize with Proxmox client."""
        self.client = proxmox_client

    def download_virtio_drivers(self, version: str, node: str, storage: str) -> str:
        """Download VirtIO drivers ISO."""
        if version not in WindowsConfig.VIRTIO_DRIVERS:
            raise ValueError(f"Unsupported VirtIO version: {version}")
        
        virtio_url = WindowsConfig.VIRTIO_DRIVERS[version]
        
        # Download VirtIO ISO
        response = requests.get(virtio_url, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".iso") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        try:
            # Upload to Proxmox storage
            upid = self.client.upload_iso(node, storage, temp_path)
            return upid
        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    def create_windows_vm(self, *, node: str, vmid: int, name: str, windows_version: str,
                         windows_config: WindowsConfig, hardware: Dict[str, Any],
                         storage: Optional[str] = None, bridge: Optional[str] = None,
                         license_key: Optional[str] = None) -> str:
        """Create Windows VM with automated installation."""
        storage_id = storage or self.client.default_storage
        bridge_id = bridge or self.client.default_bridge
        
        cores = hardware.get("cores", 4)
        memory_mb = hardware.get("memory_mb", 4096)
        disk_gb = hardware.get("disk_gb", 60)
        
        # Create VM with Windows-optimized configuration
        vm_params = {
            "vmid": vmid,
            "name": name,
            "cores": cores,
            "memory": memory_mb,
            "scsihw": "virtio-scsi-pci",
            "agent": 1,
            "ostype": "win10",
            "machine": "pc-q35-6.2",
            "cpu": "host",
            "bios": "ovmf",  # UEFI BIOS for modern Windows
            "boot": "order=scsi0;ide2;net0",
            "scsi0": f"{storage_id}:{disk_gb},format=qcow2,cache=writeback",
            "net0": f"virtio,bridge={bridge_id}",
            "vga": "qxl",
            "tablet": 1,
            "usb": "nec-xhci,u2=1,u3=1",
            # Add EFI disk for UEFI boot
            "efidisk0": f"{storage_id}:1,format=qcow2,efitype=4m,pre-enrolled-keys=1",
            # Add TPM for Windows 11 compatibility
            "tpmstate0": f"{storage_id}:1,version=v2.0",
        }
        
        upid = self.client.api.nodes(node).qemu.post(**vm_params)
        
        # Create and attach Windows setup ISO
        setup_iso_path = f"/tmp/windows-setup-{vmid}.iso"
        windows_config.create_setup_iso(setup_iso_path, license_key)
        
        # Upload setup ISO
        setup_upid = self.client.upload_iso(node, storage_id, setup_iso_path)
        setup_volid = f"{storage_id}:iso/windows-setup-{vmid}.iso"
        
        # Attach setup ISO to IDE2
        self.client.api.nodes(node).qemu(vmid).config.put(ide2=f"{setup_volid},media=cdrom")
        
        # Download and attach VirtIO drivers if needed
        virtio_version = WindowsConfig.WINDOWS_VERSIONS[windows_version]["virtio_version"]
        try:
            # Check if VirtIO ISO already exists
            storage_content = self.client.storage_content(node, storage_id)
            virtio_exists = any(
                f"virtio-win-{virtio_version}.iso" in item.get("volid", "")
                for item in storage_content
            )
            
            if not virtio_exists:
                self.download_virtio_drivers(virtio_version, node, storage_id)
            
            # Attach VirtIO ISO to IDE3
            virtio_volid = f"{storage_id}:iso/virtio-win-{virtio_version}.iso"
            self.client.api.nodes(node).qemu(vmid).config.put(ide3=f"{virtio_volid},media=cdrom")
            
        except Exception as e:
            print(f"Warning: Could not attach VirtIO drivers: {e}")
        
        # Clean up temporary ISO
        os.unlink(setup_iso_path)
        
        return upid

    def join_domain(self, node: str, vmid: int, domain: str, username: str, 
                   password: str, ou_path: Optional[str] = None) -> Dict[str, Any]:
        """Join Windows VM to Active Directory domain."""
        # Create domain join script
        config = WindowsConfig()
        config.set_domain_config(domain, username, password, ou_path)
        domain_script = config.generate_domain_join_script()
        
        # Execute via guest agent (requires QEMU guest agent)
        try:
            result = self.client.qga_exec(
                node, vmid,
                command="powershell.exe",
                args=["-ExecutionPolicy", "Bypass", "-Command", domain_script]
            )
            
            return {
                "domain_joined": True,
                "domain": domain,
                "result": result
            }
        except Exception as e:
            return {
                "domain_joined": False,
                "error": str(e)
            }

    def install_applications(self, node: str, vmid: int, applications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Install applications on Windows VM."""
        config = WindowsConfig()
        config.config["applications"] = applications
        app_script = config.generate_app_install_script()
        
        try:
            result = self.client.qga_exec(
                node, vmid,
                command="powershell.exe",
                args=["-ExecutionPolicy", "Bypass", "-Command", app_script]
            )
            
            return {
                "applications_installed": True,
                "applications": [app["name"] for app in applications],
                "result": result
            }
        except Exception as e:
            return {
                "applications_installed": False,
                "error": str(e)
            }

    def configure_windows_features(self, node: str, vmid: int, features: List[str]) -> Dict[str, Any]:
        """Enable Windows features."""
        if not features:
            return {"features_configured": True, "features": []}
        
        # Create script to enable features
        script = "# Enable Windows Features\n"
        script += "Write-Host 'Enabling Windows features...'\n\n"
        
        for feature in features:
            script += f"""
Write-Host "Enabling feature: {feature}"
try {{
    Enable-WindowsOptionalFeature -Online -FeatureName {feature} -All -NoRestart
    Write-Host "Feature {feature} enabled successfully"
}} catch {{
    Write-Host "Failed to enable feature {feature}: $($_.Exception.Message)"
}}

"""
        
        try:
            result = self.client.qga_exec(
                node, vmid,
                command="powershell.exe",
                args=["-ExecutionPolicy", "Bypass", "-Command", script]
            )
            
            return {
                "features_configured": True,
                "features": features,
                "result": result
            }
        except Exception as e:
            return {
                "features_configured": False,
                "error": str(e)
            }


# Predefined Windows configurations
def get_windows_web_server_config(computer_name: str, admin_password: str, 
                                 domain: Optional[str] = None) -> WindowsConfig:
    """Pre-configured Windows web server setup."""
    config = WindowsConfig("server-2022")
    config.set_computer_name(computer_name)
    config.set_admin_password(admin_password)
    
    if domain:
        config.config["domain"] = domain
    
    # Add IIS and related features
    config.add_windows_feature("IIS-WebServerRole")
    config.add_windows_feature("IIS-WebServer")
    config.add_windows_feature("IIS-CommonHttpFeatures")
    config.add_windows_feature("IIS-HttpErrors")
    config.add_windows_feature("IIS-HttpRedirect")
    config.add_windows_feature("IIS-ApplicationDevelopment")
    config.add_windows_feature("IIS-NetFx45")
    config.add_windows_feature("IIS-NetFxExtensibility45")
    config.add_windows_feature("IIS-ISAPIExtensions")
    config.add_windows_feature("IIS-ISAPIFilter")
    config.add_windows_feature("IIS-ASPNET45")
    
    # Add common applications
    config.add_application(
        "Chrome",
        "https://dl.google.com/chrome/install/latest/chrome_installer.exe",
        "/silent /install"
    )
    
    # Add firewall rules for web server
    config.add_firewall_rule("HTTP", 80, "TCP", "Allow", "Inbound")
    config.add_firewall_rule("HTTPS", 443, "TCP", "Allow", "Inbound")
    
    return config


def get_windows_domain_controller_config(computer_name: str, admin_password: str, 
                                        domain_name: str) -> WindowsConfig:
    """Pre-configured Windows domain controller setup."""
    config = WindowsConfig("server-2022")
    config.set_computer_name(computer_name)
    config.set_admin_password(admin_password)
    
    # Add AD DS features
    config.add_windows_feature("AD-Domain-Services")
    config.add_windows_feature("RSAT-AD-PowerShell")
    config.add_windows_feature("RSAT-ADDS")
    config.add_windows_feature("RSAT-AD-AdminCenter")
    
    # Add DNS feature
    config.add_windows_feature("DNS")
    
    # Add firewall rules for domain controller
    config.add_firewall_rule("DNS", 53, "UDP", "Allow", "Inbound")
    config.add_firewall_rule("DNS-TCP", 53, "TCP", "Allow", "Inbound")
    config.add_firewall_rule("LDAP", 389, "TCP", "Allow", "Inbound")
    config.add_firewall_rule("LDAPS", 636, "TCP", "Allow", "Inbound")
    config.add_firewall_rule("Kerberos", 88, "TCP", "Allow", "Inbound")
    
    return config
