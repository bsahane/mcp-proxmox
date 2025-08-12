from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from proxmoxer import ProxmoxAPI

from .utils import parse_api_url, read_env, split_token_id


class ProxmoxClient:
    """Wrapper around proxmoxer.ProxmoxAPI with helper methods and sane defaults."""

    def __init__(
        self,
        *,
        base_url: str,
        token_id: str,
        token_secret: str,
        verify: bool,
        default_node: Optional[str] = None,
        default_storage: Optional[str] = None,
        default_bridge: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url
        self.token_id = token_id
        self.token_secret = token_secret
        self.verify = verify
        self.default_node = default_node
        self.default_storage = default_storage
        self.default_bridge = default_bridge
        self.timeout = timeout

        url = parse_api_url(base_url)
        token_parts = split_token_id(token_id)
        self._api = ProxmoxAPI(
            url["host"],
            port=url["port"],
            user=token_parts["user"],
            token_name=token_parts["token_name"],
            token_value=token_secret,
            verify_ssl=verify,
            timeout=timeout,
        )

    @classmethod
    def from_env(cls) -> "ProxmoxClient":
        env = read_env()
        return cls(
            base_url=env.base_url,
            token_id=env.token_id,
            token_secret=env.token_secret,
            verify=env.verify,
            default_node=env.default_node,
            default_storage=env.default_storage,
            default_bridge=env.default_bridge,
        )

    # Low-level accessor
    @property
    def api(self) -> ProxmoxAPI:
        return self._api

    # -------- Core discovery --------
    def list_nodes(self) -> List[Dict[str, Any]]:
        return self._api.nodes.get()

    def get_node_status(self, node: str) -> Dict[str, Any]:
        return self._api.nodes(node).status.get()

    def list_vms(self, node: Optional[str] = None, status: Optional[str] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
        vms = self._api.cluster.resources.get(type="vm")
        if node:
            vms = [v for v in vms if v.get("node") == node]
        if status:
            vms = [v for v in vms if v.get("status") == status]
        if search:
            s = search.lower()
            vms = [v for v in vms if s in str(v.get("name", "")).lower()]
        return vms

    def list_lxc(self, node: Optional[str] = None, status: Optional[str] = None, search: Optional[str] = None) -> List[Dict[str, Any]]:
        lxcs = self._api.cluster.resources.get(type="lxc")
        if node:
            lxcs = [c for c in lxcs if c.get("node") == node]
        if status:
            lxcs = [c for c in lxcs if c.get("status") == status]
        if search:
            s = search.lower()
            lxcs = [c for c in lxcs if s in str(c.get("name", "")).lower()]
        return lxcs

    def resolve_vm(self, vmid: Optional[int] = None, name: Optional[str] = None, node: Optional[str] = None) -> Tuple[int, str, Dict[str, Any]]:
        resources = self._api.cluster.resources.get(type="vm")
        candidates: List[Dict[str, Any]] = []
        if vmid is not None:
            candidates = [r for r in resources if r.get("vmid") == vmid]
        elif name is not None:
            candidates = [r for r in resources if r.get("name") == name]
        else:
            raise ValueError("Provide either vmid or name")

        if node:
            candidates = [r for r in candidates if r.get("node") == node]

        if not candidates:
            raise ValueError("VM not found with given selector")
        if len(candidates) > 1 and not node:
            raise ValueError("Multiple VMs match name; specify node")

        vm = candidates[0]
        return int(vm["vmid"]), str(vm["node"]), vm

    def resolve_lxc(self, vmid: Optional[int] = None, name: Optional[str] = None, node: Optional[str] = None) -> Tuple[int, str, Dict[str, Any]]:
        resources = self._api.cluster.resources.get(type="lxc")
        candidates: List[Dict[str, Any]] = []
        if vmid is not None:
            candidates = [r for r in resources if r.get("vmid") == vmid]
        elif name is not None:
            candidates = [r for r in resources if r.get("name") == name]
        else:
            raise ValueError("Provide either vmid or name")

        if node:
            candidates = [r for r in candidates if r.get("node") == node]

        if not candidates:
            raise ValueError("LXC not found with given selector")
        if len(candidates) > 1 and not node:
            raise ValueError("Multiple LXCs match name; specify node")

        ct = candidates[0]
        return int(ct["vmid"]), str(ct["node"]), ct

    def vm_config(self, node: str, vmid: int) -> Dict[str, Any]:
        return self._api.nodes(node).qemu(vmid).config.get()

    def lxc_config(self, node: str, vmid: int) -> Dict[str, Any]:
        return self._api.nodes(node).lxc(vmid).config.get()

    def list_storage(self) -> List[Dict[str, Any]]:
        return self._api.storage.get()

    def storage_status(self, node: str, storage: str) -> Dict[str, Any]:
        return self._api.nodes(node).storage(storage).status.get()

    def storage_content(self, node: str, storage: str) -> List[Dict[str, Any]]:
        return self._api.nodes(node).storage(storage).content.get()

    def list_bridges(self, node: str) -> List[Dict[str, Any]]:
        nets = self._api.nodes(node).network.get()
        return [n for n in nets if n.get("type") == "bridge" or str(n.get("iface", "")).startswith("vmbr")]

    def list_tasks(self, node: Optional[str] = None, user: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        tasks = self._api.cluster.tasks.get()
        if node:
            tasks = [t for t in tasks if t.get("node") == node]
        if user:
            tasks = [t for t in tasks if t.get("user") == user]
        return tasks[:limit]

    def task_status(self, upid: str, node: Optional[str] = None) -> Dict[str, Any]:
        # If node is unknown, try cluster lookup then fall back to nodes
        try:
            return self._api.cluster.tasks(upid).status.get()
        except Exception:
            if not node:
                raise
            return self._api.nodes(node).tasks(upid).status.get()

    # -------- VM lifecycle --------
    def clone_vm(
        self,
        *,
        source_node: str,
        source_vmid: int,
        target_node: Optional[str],
        new_vmid: int,
        name: Optional[str] = None,
        full: bool = True,
        storage: Optional[str] = None,
    ) -> str:
        params: Dict[str, Any] = {"newid": new_vmid, "full": int(full)}
        if name:
            params["name"] = name
        if target_node:
            params["target"] = target_node
        if storage:
            params["storage"] = storage
        return self._api.nodes(source_node).qemu(source_vmid).clone.post(**params)  # returns upid

    def create_vm(
        self,
        *,
        node: str,
        vmid: int,
        name: str,
        cores: int = 2,
        memory_mb: int = 2048,
        disk_gb: int = 20,
        storage: Optional[str] = None,
        bridge: Optional[str] = None,
        iso: Optional[str] = None,
        scsihw: str = "virtio-scsi-pci",
        agent: bool = True,
        ostype: str = "l26",
    ) -> str:
        storage_id = storage or self.default_storage or "local-lvm"
        bridge_id = bridge or self.default_bridge or "vmbr0"
        scsi0 = f"{storage_id}:{max(disk_gb, 1)}"
        params: Dict[str, Any] = {
            "vmid": vmid,
            "name": name,
            "cores": cores,
            "memory": memory_mb,
            "scsihw": scsihw,
            "agent": int(agent),
            "ostype": ostype,
            "scsi0": scsi0,
            "net0": f"virtio,bridge={bridge_id}",
        }
        if iso:
            # ide2 expects format storage:iso/filename.iso,media=cdrom
            params["ide2"] = iso if ":" in iso else f"{storage_id}:iso/{iso}"
            params["boot"] = "order=scsi0;ide2;net0"
        return self._api.nodes(node).qemu.post(**params)

    def delete_vm(self, node: str, vmid: int, purge: bool = True) -> str:
        return self._api.nodes(node).qemu(vmid).delete.post(purge=int(purge))

    def start_vm(self, node: str, vmid: int) -> str:
        return self._api.nodes(node).qemu(vmid).status.start.post()

    def stop_vm(self, node: str, vmid: int, force: bool = False, timeout: Optional[int] = None) -> str:
        params: Dict[str, Any] = {}
        if force:
            params["forceStop"] = 1
        if timeout is not None:
            params["timeout"] = int(timeout)
        return self._api.nodes(node).qemu(vmid).status.stop.post(**params)

    def reboot_vm(self, node: str, vmid: int) -> str:
        return self._api.nodes(node).qemu(vmid).status.reboot.post()

    def shutdown_vm(self, node: str, vmid: int, timeout: Optional[int] = None) -> str:
        params: Dict[str, Any] = {}
        if timeout is not None:
            params["timeout"] = int(timeout)
        return self._api.nodes(node).qemu(vmid).status.shutdown.post(**params)

    def migrate_vm(self, node: str, vmid: int, target_node: str, online: bool = True) -> str:
        return self._api.nodes(node).qemu(vmid).migrate.post(target=target_node, online=int(online))

    def resize_vm_disk(self, node: str, vmid: int, disk: str, size_gb: int) -> str:
        # size format like +10G to grow
        return self._api.nodes(node).qemu(vmid).resize.put(disk=disk, size=f"+{size_gb}G")

    def configure_vm(self, node: str, vmid: int, params: Dict[str, Any]) -> Dict[str, Any]:
        # Returns a task upid for most changes; some return nothing. Normalize to dict
        upid = self._api.nodes(node).qemu(vmid).config.put(**params)
        return {"upid": upid} if isinstance(upid, str) else {"result": upid}

    # -------- LXC lifecycle --------
    def create_lxc(
        self,
        *,
        node: str,
        vmid: int,
        hostname: str,
        ostemplate: str,
        cores: int = 2,
        memory_mb: int = 1024,
        rootfs_gb: int = 8,
        storage: Optional[str] = None,
        bridge: Optional[str] = None,
        net_ip: Optional[str] = None,  # e.g. "dhcp" or "192.168.1.50/24,gw=192.168.1.1"
    ) -> str:
        storage_id = storage or self.default_storage or "local-lvm"
        bridge_id = bridge or self.default_bridge or "vmbr0"
        rootfs = f"{storage_id}:{max(rootfs_gb,1)}"
        net0 = f"name=eth0,bridge={bridge_id},ip={net_ip or 'dhcp'}"
        params: Dict[str, Any] = {
            "vmid": vmid,
            "hostname": hostname,
            "cores": cores,
            "memory": memory_mb,
            "ostemplate": ostemplate if ":" in ostemplate else f"{storage_id}:vztmpl/{ostemplate}",
            "rootfs": rootfs,
            "net0": net0,
            "password": os.environ.get("PROXMOX_DEFAULT_LXC_PASSWORD", "changeMe123!"),
        }
        return self._api.nodes(node).lxc.post(**params)

    def delete_lxc(self, node: str, vmid: int, purge: bool = True) -> str:
        return self._api.nodes(node).lxc(vmid).delete.post(purge=int(purge))

    def start_lxc(self, node: str, vmid: int) -> str:
        return self._api.nodes(node).lxc(vmid).status.start.post()

    def stop_lxc(self, node: str, vmid: int, timeout: Optional[int] = None) -> str:
        params: Dict[str, Any] = {}
        if timeout is not None:
            params["timeout"] = int(timeout)
        return self._api.nodes(node).lxc(vmid).status.stop.post(**params)

    def configure_lxc(self, node: str, vmid: int, params: Dict[str, Any]) -> Dict[str, Any]:
        upid = self._api.nodes(node).lxc(vmid).config.put(**params)
        return {"upid": upid} if isinstance(upid, str) else {"result": upid}

    # -------- Cloud-init & networking --------
    def cloudinit_set(self, node: str, vmid: int, params: Dict[str, Any]) -> Dict[str, Any]:
        upid = self._api.nodes(node).qemu(vmid).config.put(**params)
        return {"upid": upid} if isinstance(upid, str) else {"result": upid}

    def vm_nic_add(self, node: str, vmid: int, bridge: str, model: str = "virtio", vlan: Optional[int] = None) -> Dict[str, Any]:
        cfg = self.vm_config(node, vmid)
        used = sorted(int(k.replace("net", "")) for k in cfg.keys() if k.startswith("net"))
        idx = 0
        while idx in used:
            idx += 1
        parts = [model]
        parts.append(f"bridge={bridge}")
        if vlan is not None:
            parts.append(f"tag={vlan}")
        net_val = ",".join(parts)
        upid = self._api.nodes(node).qemu(vmid).config.put(**{f"net{idx}": net_val})
        return {"upid": upid, "added": f"net{idx}"}

    def vm_nic_remove(self, node: str, vmid: int, slot: int) -> Dict[str, Any]:
        upid = self._api.nodes(node).qemu(vmid).config.put(delete=f"net{slot}")
        return {"upid": upid, "removed": f"net{slot}"}

    def vm_firewall_get(self, node: str, vmid: int) -> Dict[str, Any]:
        opts = self._api.nodes(node).qemu(vmid).firewall.options.get()
        rules = self._api.nodes(node).qemu(vmid).firewall.rules.get()
        return {"options": opts, "rules": rules}

    def vm_firewall_set(self, node: str, vmid: int, enable: Optional[bool] = None, rules: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if enable is not None:
            upid = self._api.nodes(node).qemu(vmid).firewall.options.put(enable=int(enable))
            result["options_upid"] = upid
        if rules:
            # Very simple approach: append new rules at the end
            for rule in rules:
                self._api.nodes(node).qemu(vmid).firewall.rules.post(**rule)
            result["rules_added"] = len(rules)
        return result

    # -------- Images, templates, snapshots, backups --------
    def upload_iso(self, node: str, storage: str, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return self._api.nodes(node).storage(storage).upload.post(content="iso", filename=os.path.basename(file_path), file=f)

    def upload_template(self, node: str, storage: str, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return self._api.nodes(node).storage(storage).upload.post(content="vztmpl", filename=os.path.basename(file_path), file=f)

    def template_vm(self, node: str, vmid: int) -> str:
        return self._api.nodes(node).qemu(vmid).template.post()

    def list_snapshots(self, node: str, vmid: int) -> List[Dict[str, Any]]:
        return self._api.nodes(node).qemu(vmid).snapshot.get()

    def create_snapshot(self, node: str, vmid: int, name: str, description: Optional[str] = None, vmstate: bool = False) -> str:
        params: Dict[str, Any] = {"snapname": name, "vmstate": int(vmstate)}
        if description:
            params["description"] = description
        return self._api.nodes(node).qemu(vmid).snapshot.post(**params)

    def delete_snapshot(self, node: str, vmid: int, name: str) -> str:
        return self._api.nodes(node).qemu(vmid).snapshot(name).delete.post()

    def rollback_snapshot(self, node: str, vmid: int, name: str) -> str:
        return self._api.nodes(node).qemu(vmid).snapshot(name).rollback.post()

    def backup_vm(self, node: str, vmid: int, mode: str = "snapshot", compress: str = "zstd", storage: Optional[str] = None) -> str:
        params: Dict[str, Any] = {"vmid": vmid, "mode": mode, "compress": compress}
        if storage:
            params["storage"] = storage
        return self._api.nodes(node).vzdump.post(**params)

    def restore_vm(self, node: str, vmid: int, archive: str, storage: Optional[str] = None, force: bool = False) -> str:
        params: Dict[str, Any] = {"vmid": vmid, "archive": archive, "force": int(force)}
        if storage:
            params["storage"] = storage
        return self._api.nodes(node).qemu.restore.post(**params)

    # -------- Metrics --------
    def vm_metrics(self, node: str, vmid: int, timeframe: str = "hour", cf: str = "AVERAGE") -> List[Dict[str, Any]]:
        return self._api.nodes(node).qemu(vmid).rrddata.get(timeframe=timeframe, cf=cf)

    def node_metrics(self, node: str, timeframe: str = "hour", cf: str = "AVERAGE") -> List[Dict[str, Any]]:
        return self._api.nodes(node).rrddata.get(timeframe=timeframe, cf=cf)

    # -------- Pools / permissions --------
    def list_pools(self) -> List[Dict[str, Any]]:
        return self._api.pools.get()

    def create_pool(self, poolid: str, comment: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {"poolid": poolid}
        if comment:
            params["comment"] = comment
        return self._api.pools.post(**params)

    def delete_pool(self, poolid: str) -> Any:
        return self._api.pools(poolid).delete()

    def pool_add(self, poolid: str, vmid: int, node: str, type_: str = "qemu") -> Any:
        # Using set on the resource is more reliable
        if type_ == "qemu":
            return self._api.nodes(node).qemu(vmid).config.put(pool=poolid)
        else:
            return self._api.nodes(node).lxc(vmid).config.put(pool=poolid)

    def pool_remove(self, poolid: str, vmid: int, node: str, type_: str = "qemu") -> Any:
        if type_ == "qemu":
            return self._api.nodes(node).qemu(vmid).config.put(pool="")
        else:
            return self._api.nodes(node).lxc(vmid).config.put(pool="")

    def list_users(self) -> List[Dict[str, Any]]:
        return self._api.access.users.get()

    def list_roles(self) -> List[Dict[str, Any]]:
        return self._api.access.roles.get()

    def assign_permission(self, path: str, roles: str, users: Optional[str] = None, groups: Optional[str] = None, propagate: bool = True) -> Any:
        params: Dict[str, Any] = {"path": path, "roles": roles, "propagate": int(propagate)}
        if users:
            params["users"] = users
        if groups:
            params["groups"] = groups
        return self._api.access.acl.put(**params)

    # -------- Tasks/wait helpers --------
    def wait_task(self, upid: str, node: Optional[str] = None, timeout: int = 600, poll_interval: float = 2.0) -> Dict[str, Any]:
        start = time.time()
        while True:
            status = self.task_status(upid, node=node)
            if status.get("status") == "stopped":
                return status
            if (time.time() - start) > timeout:
                raise TimeoutError(f"Task {upid} did not complete within {timeout}s")
            time.sleep(poll_interval)

    def qga_exec(self, node: str, vmid: int, command: str, args: Optional[List[str]] = None, input_data: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"command": command}
        if args:
            payload["args"] = args
        if input_data is not None:
            payload["input-data"] = input_data
        return self._api.nodes(node).qemu(vmid).agent.exec.post(**payload)

    def qga_network_get_interfaces(self, node: str, vmid: int) -> Dict[str, Any]:
        return self._api.nodes(node).qemu(vmid).agent["network-get-interfaces"].get()
