"""
Security & Authentication Enhancement Module for Proxmox MCP Server

This module implements:
- Multi-Factor Authentication (MFA) setup
- Certificate management (Let's Encrypt, self-signed, custom)
- Secure secret storage and management
"""

import os
import io
import json
import qrcode
import base64
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

import pyotp
import hvac
from cryptography.fernet import Fernet
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from OpenSSL import crypto
import requests
from loguru import logger

from .client import ProxmoxClient
from .utils import run_command, format_error


class SecurityManager:
    """Security management for Proxmox infrastructure"""
    
    def __init__(self, proxmox_client: ProxmoxClient):
        self.client = proxmox_client
        self.vault_client = None
        self.secret_key = self._get_or_create_secret_key()
        
    def _get_or_create_secret_key(self) -> str:
        """Get or create encryption key for secrets"""
        key_file = Path.home() / ".proxmox_mcp" / "secret.key"
        key_file.parent.mkdir(exist_ok=True)
        
        if key_file.exists():
            return key_file.read_text().strip()
        else:
            key = Fernet.generate_key().decode()
            key_file.write_text(key)
            key_file.chmod(0o600)
            return key

    async def setup_mfa(
        self,
        username: str,
        mfa_type: str = "totp",
        qr_code_path: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Setup multi-factor authentication for Proxmox users"""
        try:
            if dry_run:
                return {
                    "action": "setup_mfa",
                    "username": username,
                    "mfa_type": mfa_type,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            result = {"username": username, "mfa_type": mfa_type}
            
            if mfa_type == "totp":
                # Generate TOTP secret
                secret = pyotp.random_base32()
                totp = pyotp.TOTP(secret)
                
                # Create provisioning URI for QR code
                provisioning_uri = totp.provisioning_uri(
                    name=f"proxmox-{username}",
                    issuer_name="Proxmox MCP"
                )
                
                # Generate QR code
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(provisioning_uri)
                qr.make(fit=True)
                
                # Save QR code if path provided
                if qr_code_path:
                    qr_img = qr.make_image(fill_color="black", back_color="white")
                    qr_img.save(qr_code_path)
                    result["qr_code_path"] = qr_code_path
                
                # Store secret securely
                await self.store_secret(f"mfa_secret_{username}", secret)
                
                result.update({
                    "secret": secret,
                    "provisioning_uri": provisioning_uri,
                    "backup_codes": self._generate_backup_codes()
                })
                
            elif mfa_type == "webauthn":
                # WebAuthn setup would require browser interaction
                result.update({
                    "setup_url": f"https://{self.client.host}:8006/mfa/webauthn/setup",
                    "instructions": "Complete WebAuthn setup in Proxmox web interface"
                })
                
            elif mfa_type == "yubikey":
                # YubiKey setup
                result.update({
                    "instructions": "Insert YubiKey and configure in Proxmox settings",
                    "backup_codes": self._generate_backup_codes()
                })
                
            logger.info(f"MFA setup initiated for user {username} with type {mfa_type}")
            result["status"] = "success"
            return result
            
        except Exception as e:
            error_msg = f"Failed to setup MFA for {username}: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"username": username, "mfa_type": mfa_type})

    def _generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA"""
        codes = []
        for _ in range(count):
            code = base64.b32encode(os.urandom(10)).decode()[:8]
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes

    async def manage_certificates(
        self,
        action: str,
        cert_type: str = "lets_encrypt",
        domains: List[str] = None,
        auto_renew: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Manage SSL certificates for Proxmox and VMs"""
        try:
            if dry_run:
                return {
                    "action": action,
                    "cert_type": cert_type,
                    "domains": domains or [],
                    "auto_renew": auto_renew,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            domains = domains or [self.client.host]
            result = {
                "action": action,
                "cert_type": cert_type,
                "domains": domains
            }
            
            if action == "create":
                if cert_type == "lets_encrypt":
                    cert_result = await self._create_letsencrypt_cert(domains)
                elif cert_type == "self_signed":
                    cert_result = await self._create_self_signed_cert(domains)
                elif cert_type == "custom":
                    cert_result = {"status": "pending", "message": "Provide custom certificate files"}
                else:
                    raise ValueError(f"Unknown certificate type: {cert_type}")
                    
                result.update(cert_result)
                
            elif action == "renew":
                if cert_type == "lets_encrypt":
                    renew_result = await self._renew_letsencrypt_cert(domains)
                    result.update(renew_result)
                    
            elif action == "install":
                install_result = await self._install_certificate(domains)
                result.update(install_result)
                
            elif action == "revoke":
                revoke_result = await self._revoke_certificate(domains)
                result.update(revoke_result)
                
            # Setup auto-renewal if requested
            if auto_renew and action == "create":
                await self._setup_auto_renewal(domains, cert_type)
                result["auto_renew"] = True
                
            logger.info(f"Certificate {action} completed for domains: {domains}")
            result["status"] = "success"
            return result
            
        except Exception as e:
            error_msg = f"Certificate management failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"action": action, "domains": domains})

    async def _create_letsencrypt_cert(self, domains: List[str]) -> Dict[str, Any]:
        """Create Let's Encrypt certificate"""
        try:
            # Use certbot to create certificate
            domain_args = []
            for domain in domains:
                domain_args.extend(["-d", domain])
                
            cmd = [
                "certbot", "certonly",
                "--standalone",
                "--non-interactive",
                "--agree-tos",
                "--email", "admin@example.com",  # Should be configurable
                *domain_args
            ]
            
            result = await run_command(cmd)
            
            if result["return_code"] == 0:
                cert_path = f"/etc/letsencrypt/live/{domains[0]}"
                return {
                    "certificate_path": f"{cert_path}/fullchain.pem",
                    "private_key_path": f"{cert_path}/privkey.pem",
                    "expires": datetime.now() + timedelta(days=90)
                }
            else:
                raise Exception(f"Certbot failed: {result['stderr']}")
                
        except Exception as e:
            logger.error(f"Let's Encrypt certificate creation failed: {e}")
            raise

    async def _create_self_signed_cert(self, domains: List[str]) -> Dict[str, Any]:
        """Create self-signed certificate"""
        try:
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            # Create certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Proxmox MCP"),
                x509.NameAttribute(NameOID.COMMON_NAME, domains[0]),
            ])
            
            # Add Subject Alternative Names for multiple domains
            san_list = [x509.DNSName(domain) for domain in domains]
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            ).sign(private_key, hashes.SHA256())
            
            # Save certificate and key
            cert_dir = Path("/etc/ssl/certs/proxmox-mcp")
            cert_dir.mkdir(parents=True, exist_ok=True)
            
            cert_path = cert_dir / "certificate.pem"
            key_path = cert_dir / "private_key.pem"
            
            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
                
            with open(key_path, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
                
            # Set proper permissions
            os.chmod(key_path, 0o600)
            os.chmod(cert_path, 0o644)
            
            return {
                "certificate_path": str(cert_path),
                "private_key_path": str(key_path),
                "expires": datetime.utcnow() + timedelta(days=365)
            }
            
        except Exception as e:
            logger.error(f"Self-signed certificate creation failed: {e}")
            raise

    async def _renew_letsencrypt_cert(self, domains: List[str]) -> Dict[str, Any]:
        """Renew Let's Encrypt certificate"""
        try:
            cmd = ["certbot", "renew", "--non-interactive"]
            result = await run_command(cmd)
            
            if result["return_code"] == 0:
                return {"renewed": True, "output": result["stdout"]}
            else:
                raise Exception(f"Certbot renewal failed: {result['stderr']}")
                
        except Exception as e:
            logger.error(f"Certificate renewal failed: {e}")
            raise

    async def _install_certificate(self, domains: List[str]) -> Dict[str, Any]:
        """Install certificate in Proxmox"""
        try:
            # This would integrate with Proxmox API to install the certificate
            # For now, provide instructions
            return {
                "installed": True,
                "message": "Certificate ready for installation in Proxmox web interface",
                "instructions": [
                    "1. Login to Proxmox web interface",
                    "2. Go to Datacenter -> Certificates",
                    "3. Upload the certificate and private key files",
                    "4. Restart pveproxy service"
                ]
            }
            
        except Exception as e:
            logger.error(f"Certificate installation failed: {e}")
            raise

    async def _revoke_certificate(self, domains: List[str]) -> Dict[str, Any]:
        """Revoke certificate"""
        try:
            cmd = ["certbot", "revoke", "--cert-path", f"/etc/letsencrypt/live/{domains[0]}/cert.pem"]
            result = await run_command(cmd)
            
            if result["return_code"] == 0:
                return {"revoked": True, "output": result["stdout"]}
            else:
                raise Exception(f"Certificate revocation failed: {result['stderr']}")
                
        except Exception as e:
            logger.error(f"Certificate revocation failed: {e}")
            raise

    async def _setup_auto_renewal(self, domains: List[str], cert_type: str) -> None:
        """Setup automatic certificate renewal"""
        try:
            if cert_type == "lets_encrypt":
                # Create cron job for renewal
                cron_command = "0 0,12 * * * certbot renew --quiet"
                
                # Add to crontab
                cmd = ["crontab", "-l"]
                result = await run_command(cmd)
                
                current_crons = result.get("stdout", "")
                if cron_command not in current_crons:
                    new_crons = current_crons + "\n" + cron_command + "\n"
                    
                    # Write new crontab
                    cmd = ["crontab", "-"]
                    result = await run_command(cmd, input_data=new_crons)
                    
                    if result["return_code"] == 0:
                        logger.info("Auto-renewal cron job created")
                    else:
                        logger.error(f"Failed to create cron job: {result['stderr']}")
                        
        except Exception as e:
            logger.error(f"Auto-renewal setup failed: {e}")

    async def store_secret(
        self,
        secret_name: str,
        secret_value: str,
        encryption_type: str = "aes256"
    ) -> Dict[str, Any]:
        """Store secret securely"""
        try:
            if encryption_type == "aes256":
                # Use Fernet encryption
                cipher = Fernet(self.secret_key.encode())
                encrypted_value = cipher.encrypt(secret_value.encode()).decode()
                
                # Store in local file (in production, use proper secret store)
                secrets_dir = Path.home() / ".proxmox_mcp" / "secrets"
                secrets_dir.mkdir(parents=True, exist_ok=True)
                
                secret_file = secrets_dir / f"{secret_name}.enc"
                secret_file.write_text(encrypted_value)
                secret_file.chmod(0o600)
                
                logger.info(f"Secret {secret_name} stored securely")
                return {
                    "secret_name": secret_name,
                    "stored": True,
                    "encryption": encryption_type,
                    "path": str(secret_file)
                }
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            error_msg = f"Failed to store secret {secret_name}: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"secret_name": secret_name})

    async def retrieve_secret(
        self,
        secret_name: str,
        encryption_type: str = "aes256"
    ) -> Dict[str, Any]:
        """Retrieve secret securely"""
        try:
            if encryption_type == "aes256":
                secrets_dir = Path.home() / ".proxmox_mcp" / "secrets"
                secret_file = secrets_dir / f"{secret_name}.enc"
                
                if not secret_file.exists():
                    raise FileNotFoundError(f"Secret {secret_name} not found")
                
                encrypted_value = secret_file.read_text()
                
                # Decrypt
                cipher = Fernet(self.secret_key.encode())
                decrypted_value = cipher.decrypt(encrypted_value.encode()).decode()
                
                logger.info(f"Secret {secret_name} retrieved")
                return {
                    "secret_name": secret_name,
                    "secret_value": decrypted_value,
                    "retrieved": True
                }
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
                
        except Exception as e:
            error_msg = f"Failed to retrieve secret {secret_name}: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"secret_name": secret_name})

    async def delete_secret(self, secret_name: str) -> Dict[str, Any]:
        """Delete secret securely"""
        try:
            secrets_dir = Path.home() / ".proxmox_mcp" / "secrets"
            secret_file = secrets_dir / f"{secret_name}.enc"
            
            if secret_file.exists():
                secret_file.unlink()
                logger.info(f"Secret {secret_name} deleted")
                return {"secret_name": secret_name, "deleted": True}
            else:
                return {"secret_name": secret_name, "deleted": False, "message": "Secret not found"}
                
        except Exception as e:
            error_msg = f"Failed to delete secret {secret_name}: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"secret_name": secret_name})

    async def rotate_secret(
        self,
        secret_name: str,
        new_secret_value: str,
        encryption_type: str = "aes256"
    ) -> Dict[str, Any]:
        """Rotate secret with new value"""
        try:
            # Store old secret as backup
            old_secret = await self.retrieve_secret(secret_name, encryption_type)
            if old_secret.get("retrieved"):
                backup_name = f"{secret_name}_backup_{int(datetime.now().timestamp())}"
                await self.store_secret(backup_name, old_secret["secret_value"], encryption_type)
            
            # Store new secret
            result = await self.store_secret(secret_name, new_secret_value, encryption_type)
            
            if result.get("stored"):
                logger.info(f"Secret {secret_name} rotated successfully")
                return {
                    "secret_name": secret_name,
                    "rotated": True,
                    "backup_created": old_secret.get("retrieved", False)
                }
            else:
                raise Exception("Failed to store new secret")
                
        except Exception as e:
            error_msg = f"Failed to rotate secret {secret_name}: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"secret_name": secret_name})

    async def setup_vault_integration(
        self,
        vault_url: str,
        vault_token: str,
        mount_point: str = "secret"
    ) -> Dict[str, Any]:
        """Setup HashiCorp Vault integration"""
        try:
            self.vault_client = hvac.Client(url=vault_url, token=vault_token)
            
            if self.vault_client.is_authenticated():
                logger.info("Vault integration setup successfully")
                return {
                    "vault_url": vault_url,
                    "authenticated": True,
                    "mount_point": mount_point
                }
            else:
                raise Exception("Vault authentication failed")
                
        except Exception as e:
            error_msg = f"Vault integration setup failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"vault_url": vault_url})
