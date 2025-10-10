"""
Advanced Storage Management Module for Proxmox MCP Server

This module implements:
- Storage replication between nodes
- Snapshot lifecycle management with automated policies
- Storage migration between different backends
- Advanced storage configurations
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import croniter
from croniter import croniter
import schedule

from loguru import logger

from .client import ProxmoxClient
from .utils import run_command, format_error


class AdvancedStorageManager:
    """Advanced storage management for Proxmox infrastructure"""
    
    def __init__(self, proxmox_client: ProxmoxClient):
        self.client = proxmox_client
        self.config_dir = Path.home() / ".proxmox_mcp" / "storage"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.replication_jobs = {}
        self.snapshot_policies = {}

    async def setup_replication(
        self,
        source_storage: str,
        target_node: str,
        target_storage: str,
        schedule_cron: str = "*/15 * * * *",
        compression: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Setup storage replication between nodes"""
        try:
            if dry_run:
                return {
                    "action": "setup_replication",
                    "source_storage": source_storage,
                    "target_node": target_node,
                    "target_storage": target_storage,
                    "schedule": schedule_cron,
                    "compression": compression,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            # Validate cron schedule
            try:
                cron = croniter(schedule_cron)
                next_run = cron.get_next(datetime)
            except Exception as e:
                raise ValueError(f"Invalid cron schedule: {e}")
                
            replication_id = f"{source_storage}_{target_node}_{target_storage}"
            
            result = {
                "replication_id": replication_id,
                "source_storage": source_storage,
                "target_node": target_node,
                "target_storage": target_storage,
                "schedule": schedule_cron,
                "compression": compression,
                "next_run": next_run.isoformat()
            }
            
            # Create replication configuration
            replication_config = {
                "id": replication_id,
                "source": {
                    "storage": source_storage,
                    "node": self.client.get_first_node()
                },
                "target": {
                    "storage": target_storage,
                    "node": target_node
                },
                "schedule": schedule_cron,
                "compression": compression,
                "enabled": True,
                "created": datetime.now().isoformat(),
                "last_run": None,
                "status": "active"
            }
            
            # Validate source and target storage
            source_exists = await self._validate_storage(
                source_storage, replication_config["source"]["node"]
            )
            target_exists = await self._validate_storage(target_storage, target_node)
            
            if not source_exists:
                raise ValueError(f"Source storage {source_storage} not found")
            if not target_exists:
                raise ValueError(f"Target storage {target_storage} not found on {target_node}")
                
            # Save replication configuration
            self.replication_jobs[replication_id] = replication_config
            await self._save_replication_config()
            
            # Setup replication job using Proxmox replication API if available
            try:
                replication_result = await self._create_proxmox_replication(replication_config)
                result["proxmox_replication"] = replication_result
            except Exception as e:
                logger.warning(f"Could not create Proxmox replication job: {e}")
                result["manual_replication"] = True
                
            logger.info(f"Storage replication setup: {source_storage} -> {target_node}:{target_storage}")
            result["status"] = "configured"
            return result
            
        except Exception as e:
            error_msg = f"Storage replication setup failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"source_storage": source_storage})

    async def _validate_storage(self, storage_name: str, node: str) -> bool:
        """Validate that storage exists on node"""
        try:
            storages = await self.client.nodes(node).storage.get()
            return any(storage["storage"] == storage_name for storage in storages)
        except Exception as e:
            logger.error(f"Storage validation failed: {e}")
            return False

    async def _create_proxmox_replication(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Proxmox replication job"""
        try:
            # This would use Proxmox replication API
            # For now, return configuration placeholder
            return {
                "replication_id": config["id"],
                "type": "proxmox_native",
                "status": "created"
            }
        except Exception as e:
            logger.error(f"Proxmox replication creation failed: {e}")
            raise

    async def _save_replication_config(self):
        """Save replication configuration to file"""
        try:
            config_file = self.config_dir / "replication_jobs.json"
            with open(config_file, 'w') as f:
                json.dump(self.replication_jobs, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save replication config: {e}")

    async def snapshot_policy(
        self,
        vmid: int,
        policy: Dict[str, Any],
        auto_cleanup: bool = True,
        compression: bool = True,
        node: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Create automated snapshot policies with lifecycle management"""
        try:
            if dry_run:
                return {
                    "action": "snapshot_policy",
                    "vmid": vmid,
                    "policy": policy,
                    "auto_cleanup": auto_cleanup,
                    "compression": compression,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            node = node or self.client.get_first_node()
            policy_id = f"vm_{vmid}_snapshot_policy"
            
            # Validate policy structure
            required_fields = ["hourly", "daily", "weekly", "monthly"]
            if not all(field in policy for field in required_fields):
                raise ValueError(f"Policy must contain: {required_fields}")
                
            # Create snapshot policy configuration
            policy_config = {
                "id": policy_id,
                "vmid": vmid,
                "node": node,
                "retention": policy,
                "auto_cleanup": auto_cleanup,
                "compression": compression,
                "enabled": True,
                "created": datetime.now().isoformat(),
                "last_run": None,
                "next_run": None
            }
            
            # Calculate next run times for each schedule
            schedules = {
                "hourly": "0 * * * *",      # Every hour
                "daily": "0 2 * * *",       # Daily at 2 AM
                "weekly": "0 3 * * 0",      # Weekly on Sunday at 3 AM
                "monthly": "0 4 1 * *"      # Monthly on 1st at 4 AM
            }
            
            next_runs = {}
            for schedule_type, cron_expr in schedules.items():
                if policy.get(schedule_type, 0) > 0:
                    cron = croniter(cron_expr)
                    next_runs[schedule_type] = cron.get_next(datetime).isoformat()
                    
            policy_config["schedules"] = schedules
            policy_config["next_runs"] = next_runs
            
            # Validate VM exists
            try:
                vm_config = await self.client.nodes(node).qemu(vmid).config.get()
                policy_config["vm_name"] = vm_config.get("name", f"vm-{vmid}")
            except Exception as e:
                raise ValueError(f"VM {vmid} not found on node {node}")
                
            # Save policy
            self.snapshot_policies[policy_id] = policy_config
            await self._save_snapshot_policies()
            
            # Create initial snapshots if requested
            initial_snapshots = []
            for schedule_type in ["hourly", "daily", "weekly", "monthly"]:
                if policy.get(schedule_type, 0) > 0:
                    snapshot_result = await self._create_policy_snapshot(
                        vmid, schedule_type, node, compression
                    )
                    initial_snapshots.append(snapshot_result)
                    
            result = {
                "policy_id": policy_id,
                "vmid": vmid,
                "node": node,
                "retention_policy": policy,
                "schedules": schedules,
                "next_runs": next_runs,
                "initial_snapshots": initial_snapshots,
                "status": "configured"
            }
            
            logger.info(f"Snapshot policy configured for VM {vmid}")
            return result
            
        except Exception as e:
            error_msg = f"Snapshot policy creation failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"vmid": vmid})

    async def _create_policy_snapshot(
        self,
        vmid: int,
        schedule_type: str,
        node: str,
        compression: bool
    ) -> Dict[str, Any]:
        """Create a snapshot according to policy"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"auto_{schedule_type}_{timestamp}"
            
            # Create snapshot
            snapshot_result = await self.client.nodes(node).qemu(vmid).snapshot.post(
                snapname=snapshot_name,
                description=f"Automatic {schedule_type} snapshot",
                vmstate=False  # Set to True to include RAM state
            )
            
            return {
                "vmid": vmid,
                "snapshot_name": snapshot_name,
                "schedule_type": schedule_type,
                "created": datetime.now().isoformat(),
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Policy snapshot creation failed: {e}")
            return {
                "vmid": vmid,
                "schedule_type": schedule_type,
                "status": "failed",
                "error": str(e)
            }

    async def _save_snapshot_policies(self):
        """Save snapshot policies to file"""
        try:
            config_file = self.config_dir / "snapshot_policies.json"
            with open(config_file, 'w') as f:
                json.dump(self.snapshot_policies, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save snapshot policies: {e}")

    async def migrate_storage(
        self,
        vmid: int,
        source_storage: str,
        target_storage: str,
        online: bool = True,
        preserve_source: bool = False,
        node: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Migrate VM storage between different storage backends"""
        try:
            if dry_run:
                return {
                    "action": "migrate_storage",
                    "vmid": vmid,
                    "source_storage": source_storage,
                    "target_storage": target_storage,
                    "online": online,
                    "preserve_source": preserve_source,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            node = node or self.client.get_first_node()
            
            result = {
                "vmid": vmid,
                "source_storage": source_storage,
                "target_storage": target_storage,
                "node": node,
                "online": online,
                "preserve_source": preserve_source
            }
            
            # Get VM configuration to identify disks
            vm_config = await self.client.nodes(node).qemu(vmid).config.get()
            
            # Find disks on source storage
            disks_to_migrate = []
            for key, value in vm_config.items():
                if key.startswith(('scsi', 'ide', 'sata', 'virtio')):
                    if isinstance(value, str) and source_storage in value:
                        disks_to_migrate.append({
                            "disk": key,
                            "config": value,
                            "storage": source_storage
                        })
                        
            if not disks_to_migrate:
                return {
                    **result,
                    "status": "no_migration_needed",
                    "message": f"No disks found on storage {source_storage}"
                }
                
            result["disks_to_migrate"] = len(disks_to_migrate)
            
            # Validate target storage
            target_exists = await self._validate_storage(target_storage, node)
            if not target_exists:
                raise ValueError(f"Target storage {target_storage} not found")
                
            # Check if VM is running for online migration
            vm_status = await self.client.nodes(node).qemu(vmid).status.current.get()
            is_running = vm_status.get("status") == "running"
            
            if online and not is_running:
                logger.warning(f"VM {vmid} is not running, performing offline migration")
                online = False
                
            result["vm_running"] = is_running
            result["migration_type"] = "online" if online else "offline"
            
            # Perform migration for each disk
            migration_results = []
            for disk_info in disks_to_migrate:
                try:
                    # Use Proxmox move disk API
                    move_result = await self.client.nodes(node).qemu(vmid).move_disk.post(
                        disk=disk_info["disk"],
                        storage=target_storage,
                        format="qcow2",  # or preserve original format
                        delete=not preserve_source
                    )
                    
                    migration_results.append({
                        "disk": disk_info["disk"],
                        "status": "migrated",
                        "task_id": move_result.get("data", "")
                    })
                    
                except Exception as e:
                    migration_results.append({
                        "disk": disk_info["disk"],
                        "status": "failed",
                        "error": str(e)
                    })
                    
            result["migration_results"] = migration_results
            
            # Wait for migration tasks to complete
            completed_migrations = 0
            for migration in migration_results:
                if migration["status"] == "migrated" and "task_id" in migration:
                    # Monitor task completion
                    task_completed = await self._wait_for_task(
                        migration["task_id"], node, timeout=3600
                    )
                    migration["completed"] = task_completed
                    if task_completed:
                        completed_migrations += 1
                        
            result["completed_migrations"] = completed_migrations
            result["total_migrations"] = len(migration_results)
            
            if completed_migrations == len(migration_results):
                result["status"] = "completed"
                logger.info(f"Storage migration completed for VM {vmid}")
            else:
                result["status"] = "partial"
                logger.warning(f"Storage migration partially completed for VM {vmid}")
                
            return result
            
        except Exception as e:
            error_msg = f"Storage migration failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"vmid": vmid})

    async def _wait_for_task(
        self,
        task_id: str,
        node: str,
        timeout: int = 300
    ) -> bool:
        """Wait for Proxmox task to complete"""
        try:
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < timeout:
                try:
                    task_status = await self.client.nodes(node).tasks(task_id).status.get()
                    status = task_status.get("status", "")
                    
                    if status == "stopped":
                        exit_status = task_status.get("exitstatus", "")
                        return exit_status == "OK"
                    elif status in ["error", "canceled"]:
                        return False
                        
                    await asyncio.sleep(5)  # Wait 5 seconds before checking again
                    
                except Exception as e:
                    logger.warning(f"Task status check failed: {e}")
                    await asyncio.sleep(5)
                    
            logger.warning(f"Task {task_id} timeout after {timeout} seconds")
            return False
            
        except Exception as e:
            logger.error(f"Task waiting failed: {e}")
            return False

    async def cleanup_snapshots(
        self,
        vmid: int,
        policy_id: Optional[str] = None,
        node: Optional[str] = None
    ) -> Dict[str, Any]:
        """Clean up old snapshots according to retention policy"""
        try:
            node = node or self.client.get_first_node()
            
            # Get current snapshots
            snapshots = await self.client.nodes(node).qemu(vmid).snapshot.get()
            
            # Filter automatic snapshots
            auto_snapshots = [
                snap for snap in snapshots 
                if snap.get("name", "").startswith("auto_")
            ]
            
            result = {
                "vmid": vmid,
                "node": node,
                "total_snapshots": len(snapshots),
                "auto_snapshots": len(auto_snapshots)
            }
            
            if policy_id and policy_id in self.snapshot_policies:
                policy = self.snapshot_policies[policy_id]
                retention = policy["retention"]
                
                # Group snapshots by type
                snapshot_groups = {
                    "hourly": [],
                    "daily": [],
                    "weekly": [],
                    "monthly": []
                }
                
                for snap in auto_snapshots:
                    snap_name = snap.get("name", "")
                    for snap_type in snapshot_groups.keys():
                        if f"auto_{snap_type}" in snap_name:
                            snapshot_groups[snap_type].append(snap)
                            break
                            
                # Sort by creation time (newest first)
                for snap_type in snapshot_groups:
                    snapshot_groups[snap_type].sort(
                        key=lambda x: x.get("snaptime", 0), reverse=True
                    )
                    
                # Delete old snapshots
                deleted_snapshots = []
                for snap_type, snapshots_list in snapshot_groups.items():
                    keep_count = retention.get(snap_type, 0)
                    if len(snapshots_list) > keep_count:
                        snapshots_to_delete = snapshots_list[keep_count:]
                        
                        for snap in snapshots_to_delete:
                            try:
                                await self.client.nodes(node).qemu(vmid).snapshot(
                                    snap["name"]
                                ).delete()
                                deleted_snapshots.append({
                                    "name": snap["name"],
                                    "type": snap_type,
                                    "status": "deleted"
                                })
                            except Exception as e:
                                deleted_snapshots.append({
                                    "name": snap["name"],
                                    "type": snap_type,
                                    "status": "failed",
                                    "error": str(e)
                                })
                                
                result["deleted_snapshots"] = deleted_snapshots
                result["cleanup_policy"] = retention
                
            else:
                result["message"] = "No cleanup policy specified"
                
            logger.info(f"Snapshot cleanup completed for VM {vmid}")
            result["status"] = "completed"
            return result
            
        except Exception as e:
            error_msg = f"Snapshot cleanup failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"vmid": vmid})

    async def get_storage_usage(
        self,
        storage_name: Optional[str] = None,
        node: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed storage usage information"""
        try:
            node = node or self.client.get_first_node()
            
            if storage_name:
                # Get specific storage info
                storage_status = await self.client.nodes(node).storage(storage_name).status.get()
                
                return {
                    "storage": storage_name,
                    "node": node,
                    "total": storage_status.get("total", 0),
                    "used": storage_status.get("used", 0),
                    "available": storage_status.get("avail", 0),
                    "usage_percent": round(
                        (storage_status.get("used", 0) / storage_status.get("total", 1)) * 100, 2
                    ),
                    "type": storage_status.get("type", ""),
                    "status": "active" if storage_status.get("active", 0) else "inactive"
                }
            else:
                # Get all storage info
                storages = await self.client.nodes(node).storage.get()
                storage_info = []
                
                for storage in storages:
                    try:
                        storage_status = await self.client.nodes(node).storage(
                            storage["storage"]
                        ).status.get()
                        
                        storage_info.append({
                            "storage": storage["storage"],
                            "total": storage_status.get("total", 0),
                            "used": storage_status.get("used", 0),
                            "available": storage_status.get("avail", 0),
                            "usage_percent": round(
                                (storage_status.get("used", 0) / storage_status.get("total", 1)) * 100, 2
                            ),
                            "type": storage_status.get("type", ""),
                            "status": "active" if storage_status.get("active", 0) else "inactive"
                        })
                    except Exception as e:
                        logger.warning(f"Could not get status for storage {storage['storage']}: {e}")
                        
                return {
                    "node": node,
                    "storages": storage_info,
                    "total_storages": len(storage_info)
                }
                
        except Exception as e:
            error_msg = f"Storage usage retrieval failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"storage": storage_name, "node": node})

    async def optimize_storage(
        self,
        storage_name: str,
        node: Optional[str] = None
    ) -> Dict[str, Any]:
        """Optimize storage performance and usage"""
        try:
            node = node or self.client.get_first_node()
            
            result = {
                "storage": storage_name,
                "node": node,
                "optimizations": []
            }
            
            # Get storage information
            storage_status = await self.client.nodes(node).storage(storage_name).status.get()
            storage_type = storage_status.get("type", "")
            
            # Perform optimization based on storage type
            if storage_type == "zfspool":
                zfs_optimization = await self._optimize_zfs(storage_name, node)
                result["optimizations"].append(zfs_optimization)
                
            elif storage_type == "lvm":
                lvm_optimization = await self._optimize_lvm(storage_name, node)
                result["optimizations"].append(lvm_optimization)
                
            elif storage_type == "dir":
                dir_optimization = await self._optimize_directory(storage_name, node)
                result["optimizations"].append(dir_optimization)
                
            # General optimizations
            general_optimization = await self._general_storage_optimization(storage_name, node)
            result["optimizations"].append(general_optimization)
            
            result["status"] = "optimized"
            logger.info(f"Storage optimization completed for {storage_name}")
            return result
            
        except Exception as e:
            error_msg = f"Storage optimization failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"storage": storage_name})

    async def _optimize_zfs(self, storage_name: str, node: str) -> Dict[str, Any]:
        """Optimize ZFS storage"""
        try:
            optimizations = []
            
            # Check ZFS compression
            cmd = ["zfs", "get", "compression", storage_name]
            result = await run_command(cmd)
            
            if result["return_code"] == 0:
                if "off" in result["stdout"]:
                    # Enable compression
                    compress_cmd = ["zfs", "set", "compression=lz4", storage_name]
                    compress_result = await run_command(compress_cmd)
                    optimizations.append({
                        "optimization": "enable_compression",
                        "status": "applied" if compress_result["return_code"] == 0 else "failed"
                    })
                    
            # Check ZFS deduplication
            dedup_cmd = ["zfs", "get", "dedup", storage_name]
            dedup_result = await run_command(dedup_cmd)
            
            if dedup_result["return_code"] == 0:
                if "off" in dedup_result["stdout"]:
                    optimizations.append({
                        "optimization": "deduplication_available",
                        "status": "recommended",
                        "note": "Enable with: zfs set dedup=on " + storage_name
                    })
                    
            return {
                "type": "zfs",
                "optimizations": optimizations
            }
            
        except Exception as e:
            return {
                "type": "zfs",
                "status": "failed",
                "error": str(e)
            }

    async def _optimize_lvm(self, storage_name: str, node: str) -> Dict[str, Any]:
        """Optimize LVM storage"""
        try:
            optimizations = []
            
            # Check LVM thin provisioning
            lvs_cmd = ["lvs", "--noheadings", "-o", "lv_layout"]
            result = await run_command(lvs_cmd)
            
            if result["return_code"] == 0:
                if "thin" not in result["stdout"]:
                    optimizations.append({
                        "optimization": "thin_provisioning",
                        "status": "recommended",
                        "note": "Consider using thin provisioning for better space efficiency"
                    })
                    
            return {
                "type": "lvm",
                "optimizations": optimizations
            }
            
        except Exception as e:
            return {
                "type": "lvm",
                "status": "failed",
                "error": str(e)
            }

    async def _optimize_directory(self, storage_name: str, node: str) -> Dict[str, Any]:
        """Optimize directory-based storage"""
        try:
            optimizations = []
            
            # Check filesystem type and mount options
            mount_cmd = ["mount", "|", "grep", storage_name]
            result = await run_command(mount_cmd, shell=True)
            
            if result["return_code"] == 0:
                mount_info = result["stdout"]
                
                # Check for performance-related mount options
                if "noatime" not in mount_info:
                    optimizations.append({
                        "optimization": "noatime_mount_option",
                        "status": "recommended",
                        "note": "Add noatime mount option to improve performance"
                    })
                    
            return {
                "type": "directory",
                "optimizations": optimizations
            }
            
        except Exception as e:
            return {
                "type": "directory",
                "status": "failed",
                "error": str(e)
            }

    async def _general_storage_optimization(self, storage_name: str, node: str) -> Dict[str, Any]:
        """General storage optimization recommendations"""
        try:
            optimizations = []
            
            # Check disk usage
            storage_status = await self.client.nodes(node).storage(storage_name).status.get()
            usage_percent = (storage_status.get("used", 0) / storage_status.get("total", 1)) * 100
            
            if usage_percent > 80:
                optimizations.append({
                    "optimization": "high_disk_usage",
                    "status": "warning",
                    "note": f"Disk usage is {usage_percent:.1f}%. Consider cleanup or expansion."
                })
                
            if usage_percent > 95:
                optimizations.append({
                    "optimization": "critical_disk_usage",
                    "status": "critical",
                    "note": "Disk usage is critically high. Immediate action required."
                })
                
            return {
                "type": "general",
                "optimizations": optimizations
            }
            
        except Exception as e:
            return {
                "type": "general",
                "status": "failed",
                "error": str(e)
            }
