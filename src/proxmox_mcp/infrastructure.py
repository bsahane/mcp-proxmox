"""
Infrastructure Automation Module for Proxmox MCP Server

This module implements:
- Terraform integration for Infrastructure as Code
- Ansible integration for configuration management  
- GitOps integration for declarative infrastructure
"""

import os
import json
import tempfile
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
import yaml

import git
from git import Repo
from python_terraform import Terraform
import ansible_runner
from loguru import logger

from .client import ProxmoxClient
from .utils import run_command, format_error


class InfrastructureManager:
    """Infrastructure automation and management"""
    
    def __init__(self, proxmox_client: ProxmoxClient):
        self.client = proxmox_client
        self.terraform = None
        self.working_dir = Path.home() / ".proxmox_mcp" / "infrastructure"
        self.working_dir.mkdir(parents=True, exist_ok=True)

    async def terraform_plan(
        self,
        config_path: str,
        workspace: Optional[str] = None,
        auto_approve: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Execute Terraform plans for infrastructure as code"""
        try:
            if dry_run:
                return {
                    "action": "terraform_plan",
                    "config_path": config_path,
                    "workspace": workspace,
                    "auto_approve": auto_approve,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Terraform config not found: {config_path}")
                
            # Initialize Terraform
            self.terraform = Terraform(working_dir=str(config_path))
            
            result = {"config_path": str(config_path)}
            
            # Initialize terraform
            init_result = self.terraform.init()
            if init_result[0] != 0:
                raise Exception(f"Terraform init failed: {init_result[1]}")
                
            # Select workspace if specified
            if workspace:
                workspace_result = self.terraform.workspace('select', workspace)
                if workspace_result[0] != 0:
                    # Try to create workspace if it doesn't exist
                    create_result = self.terraform.workspace('new', workspace)
                    if create_result[0] != 0:
                        raise Exception(f"Failed to create/select workspace {workspace}")
                result["workspace"] = workspace
                
            # Create terraform plan
            plan_result = self.terraform.plan(
                capture_output=True,
                out="tfplan"
            )
            
            if plan_result[0] != 0:
                raise Exception(f"Terraform plan failed: {plan_result[1]}")
                
            result.update({
                "plan_output": plan_result[1],
                "plan_file": "tfplan",
                "status": "planned"
            })
            
            # Apply if auto_approve is True
            if auto_approve:
                apply_result = self.terraform.apply(
                    "tfplan",
                    capture_output=True,
                    auto_approve=True
                )
                
                if apply_result[0] != 0:
                    raise Exception(f"Terraform apply failed: {apply_result[1]}")
                    
                result.update({
                    "apply_output": apply_result[1],
                    "status": "applied"
                })
                
            logger.info(f"Terraform plan executed for {config_path}")
            return result
            
        except Exception as e:
            error_msg = f"Terraform plan failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"config_path": config_path})

    async def terraform_apply(
        self,
        config_path: str,
        plan_file: Optional[str] = None,
        auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Apply Terraform configuration"""
        try:
            config_path = Path(config_path)
            self.terraform = Terraform(working_dir=str(config_path))
            
            if plan_file:
                apply_result = self.terraform.apply(
                    plan_file,
                    capture_output=True,
                    auto_approve=auto_approve
                )
            else:
                apply_result = self.terraform.apply(
                    capture_output=True,
                    auto_approve=auto_approve
                )
                
            if apply_result[0] != 0:
                raise Exception(f"Terraform apply failed: {apply_result[1]}")
                
            return {
                "config_path": str(config_path),
                "apply_output": apply_result[1],
                "status": "applied"
            }
            
        except Exception as e:
            error_msg = f"Terraform apply failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"config_path": config_path})

    async def terraform_destroy(
        self,
        config_path: str,
        auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Destroy Terraform-managed infrastructure"""
        try:
            config_path = Path(config_path)
            self.terraform = Terraform(working_dir=str(config_path))
            
            destroy_result = self.terraform.destroy(
                capture_output=True,
                auto_approve=auto_approve
            )
            
            if destroy_result[0] != 0:
                raise Exception(f"Terraform destroy failed: {destroy_result[1]}")
                
            return {
                "config_path": str(config_path),
                "destroy_output": destroy_result[1],
                "status": "destroyed"
            }
            
        except Exception as e:
            error_msg = f"Terraform destroy failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"config_path": config_path})

    async def ansible_playbook(
        self,
        playbook_path: str,
        inventory: Optional[str] = None,
        extra_vars: Optional[Dict[str, Any]] = None,
        limit: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Execute Ansible playbooks against Proxmox VMs"""
        try:
            if dry_run:
                return {
                    "action": "ansible_playbook",
                    "playbook_path": playbook_path,
                    "inventory": inventory,
                    "extra_vars": extra_vars or {},
                    "limit": limit,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            playbook_path = Path(playbook_path)
            if not playbook_path.exists():
                raise FileNotFoundError(f"Playbook not found: {playbook_path}")
                
            # Prepare runner parameters
            runner_params = {
                "playbook": str(playbook_path),
                "private_data_dir": str(self.working_dir),
                "verbosity": 2
            }
            
            if inventory:
                if Path(inventory).exists():
                    runner_params["inventory"] = inventory
                else:
                    # Create temporary inventory file
                    inv_file = self.working_dir / "inventory.yml"
                    with open(inv_file, 'w') as f:
                        f.write(inventory)
                    runner_params["inventory"] = str(inv_file)
                    
            if extra_vars:
                runner_params["extravars"] = extra_vars
                
            if limit:
                runner_params["limit"] = limit
                
            # Run playbook
            result = ansible_runner.run(**runner_params)
            
            # Process results
            playbook_result = {
                "playbook_path": str(playbook_path),
                "status": result.status,
                "return_code": result.rc,
                "stats": result.stats if hasattr(result, 'stats') else {}
            }
            
            # Get detailed results
            if hasattr(result, 'events'):
                events = []
                for event in result.events:
                    if event.get('event') in ['runner_on_ok', 'runner_on_failed', 'runner_on_unreachable']:
                        events.append({
                            "event": event.get('event'),
                            "host": event.get('event_data', {}).get('host'),
                            "task": event.get('event_data', {}).get('task'),
                            "result": event.get('event_data', {}).get('res', {})
                        })
                playbook_result["events"] = events[-10:]  # Last 10 events
                
            if result.status == "successful":
                logger.info(f"Ansible playbook {playbook_path} executed successfully")
            else:
                logger.error(f"Ansible playbook {playbook_path} failed")
                
            return playbook_result
            
        except Exception as e:
            error_msg = f"Ansible playbook execution failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"playbook_path": playbook_path})

    async def ansible_adhoc(
        self,
        module: str,
        hosts: str = "all",
        args: Optional[str] = None,
        inventory: Optional[str] = None,
        extra_vars: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute Ansible ad-hoc commands"""
        try:
            runner_params = {
                "module": module,
                "host_pattern": hosts,
                "private_data_dir": str(self.working_dir)
            }
            
            if args:
                runner_params["module_args"] = args
                
            if inventory:
                runner_params["inventory"] = inventory
                
            if extra_vars:
                runner_params["extravars"] = extra_vars
                
            result = ansible_runner.run(**runner_params)
            
            return {
                "module": module,
                "hosts": hosts,
                "status": result.status,
                "return_code": result.rc,
                "stats": result.stats if hasattr(result, 'stats') else {}
            }
            
        except Exception as e:
            error_msg = f"Ansible ad-hoc command failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"module": module, "hosts": hosts})

    async def gitops_sync(
        self,
        repo_url: str,
        branch: str = "main",
        config_path: str = "./infrastructure",
        auto_deploy: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Sync infrastructure state with Git repository"""
        try:
            if dry_run:
                return {
                    "action": "gitops_sync",
                    "repo_url": repo_url,
                    "branch": branch,
                    "config_path": config_path,
                    "auto_deploy": auto_deploy,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            # Create temporary directory for repository
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_dir = Path(temp_dir) / "repo"
                
                # Clone or update repository
                if repo_dir.exists():
                    repo = Repo(repo_dir)
                    origin = repo.remotes.origin
                    origin.pull()
                else:
                    repo = Repo.clone_from(repo_url, repo_dir, branch=branch)
                    
                # Check for changes
                repo.git.checkout(branch)
                latest_commit = repo.head.commit
                
                result = {
                    "repo_url": repo_url,
                    "branch": branch,
                    "latest_commit": latest_commit.hexsha,
                    "commit_message": latest_commit.message.strip(),
                    "author": latest_commit.author.name,
                    "timestamp": latest_commit.committed_datetime.isoformat()
                }
                
                # Look for infrastructure configurations
                infra_path = repo_dir / config_path.lstrip('./')
                if not infra_path.exists():
                    raise FileNotFoundError(f"Infrastructure path not found: {config_path}")
                    
                # Detect configuration type and process
                configs_found = []
                
                # Check for Terraform configurations
                tf_files = list(infra_path.glob("*.tf"))
                if tf_files:
                    configs_found.append("terraform")
                    if auto_deploy:
                        tf_result = await self.terraform_plan(
                            str(infra_path),
                            auto_approve=True
                        )
                        result["terraform_result"] = tf_result
                        
                # Check for Ansible playbooks
                ansible_files = list(infra_path.glob("*.yml")) + list(infra_path.glob("*.yaml"))
                playbooks = [f for f in ansible_files if 'playbook' in f.name.lower() or 'site' in f.name.lower()]
                if playbooks:
                    configs_found.append("ansible")
                    if auto_deploy:
                        for playbook in playbooks:
                            ansible_result = await self.ansible_playbook(str(playbook))
                            result.setdefault("ansible_results", []).append(ansible_result)
                            
                # Check for Kubernetes manifests
                k8s_files = list(infra_path.glob("*.yaml")) + list(infra_path.glob("*.yml"))
                k8s_configs = []
                for f in k8s_files:
                    with open(f) as file:
                        content = yaml.safe_load(file)
                        if isinstance(content, dict) and content.get("apiVersion"):
                            k8s_configs.append(f)
                            
                if k8s_configs:
                    configs_found.append("kubernetes")
                    result["kubernetes_manifests"] = [str(f) for f in k8s_configs]
                    
                result["configurations_found"] = configs_found
                result["status"] = "synced"
                
                logger.info(f"GitOps sync completed for {repo_url}")
                return result
                
        except Exception as e:
            error_msg = f"GitOps sync failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"repo_url": repo_url})

    async def create_terraform_config(
        self,
        config_name: str,
        resources: List[Dict[str, Any]],
        variables: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create Terraform configuration for Proxmox resources"""
        try:
            config_dir = self.working_dir / "terraform" / config_name
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate main.tf
            main_tf_content = self._generate_terraform_main(resources)
            (config_dir / "main.tf").write_text(main_tf_content)
            
            # Generate variables.tf
            if variables:
                variables_tf_content = self._generate_terraform_variables(variables)
                (config_dir / "variables.tf").write_text(variables_tf_content)
                
            # Generate outputs.tf
            if outputs:
                outputs_tf_content = self._generate_terraform_outputs(outputs)
                (config_dir / "outputs.tf").write_text(outputs_tf_content)
                
            # Generate provider.tf
            provider_tf_content = self._generate_terraform_provider()
            (config_dir / "provider.tf").write_text(provider_tf_content)
            
            logger.info(f"Terraform configuration created: {config_dir}")
            return {
                "config_name": config_name,
                "config_dir": str(config_dir),
                "files_created": ["main.tf", "provider.tf"] + 
                               (["variables.tf"] if variables else []) +
                               (["outputs.tf"] if outputs else []),
                "status": "created"
            }
            
        except Exception as e:
            error_msg = f"Terraform config creation failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"config_name": config_name})

    def _generate_terraform_main(self, resources: List[Dict[str, Any]]) -> str:
        """Generate main.tf content"""
        content = []
        
        for resource in resources:
            resource_type = resource.get("type", "proxmox_vm_qemu")
            resource_name = resource.get("name", "vm")
            
            content.append(f'resource "{resource_type}" "{resource_name}" {{')
            
            for key, value in resource.get("config", {}).items():
                if isinstance(value, str):
                    content.append(f'  {key} = "{value}"')
                elif isinstance(value, bool):
                    content.append(f'  {key} = {str(value).lower()}')
                elif isinstance(value, (int, float)):
                    content.append(f'  {key} = {value}')
                elif isinstance(value, list):
                    content.append(f'  {key} = {json.dumps(value)}')
                elif isinstance(value, dict):
                    content.append(f'  {key} = {{')
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str):
                            content.append(f'    {subkey} = "{subvalue}"')
                        else:
                            content.append(f'    {subkey} = {subvalue}')
                    content.append('  }')
                    
            content.append('}')
            content.append('')
            
        return '\n'.join(content)

    def _generate_terraform_variables(self, variables: Dict[str, Any]) -> str:
        """Generate variables.tf content"""
        content = []
        
        for name, config in variables.items():
            content.append(f'variable "{name}" {{')
            if "description" in config:
                content.append(f'  description = "{config["description"]}"')
            if "type" in config:
                content.append(f'  type = {config["type"]}')
            if "default" in config:
                if isinstance(config["default"], str):
                    content.append(f'  default = "{config["default"]}"')
                else:
                    content.append(f'  default = {config["default"]}')
            content.append('}')
            content.append('')
            
        return '\n'.join(content)

    def _generate_terraform_outputs(self, outputs: Dict[str, Any]) -> str:
        """Generate outputs.tf content"""
        content = []
        
        for name, config in outputs.items():
            content.append(f'output "{name}" {{')
            content.append(f'  value = {config["value"]}')
            if "description" in config:
                content.append(f'  description = "{config["description"]}"')
            content.append('}')
            content.append('')
            
        return '\n'.join(content)

    def _generate_terraform_provider(self) -> str:
        """Generate provider.tf content"""
        return f'''terraform {{
  required_providers {{
    proxmox = {{
      source = "telmate/proxmox"
      version = "2.9.14"
    }}
  }}
}}

provider "proxmox" {{
  pm_api_url = "https://{self.client.host}:8006/api2/json"
  pm_user = var.proxmox_user
  pm_password = var.proxmox_password
  pm_tls_insecure = true
}}

variable "proxmox_user" {{
  description = "Proxmox username"
  type = string
}}

variable "proxmox_password" {{
  description = "Proxmox password"
  type = string
  sensitive = true
}}
'''

    async def create_ansible_inventory(
        self,
        inventory_name: str,
        hosts: List[Dict[str, Any]],
        groups: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """Create Ansible inventory for Proxmox VMs"""
        try:
            inventory_dir = self.working_dir / "ansible" / "inventories"
            inventory_dir.mkdir(parents=True, exist_ok=True)
            
            inventory_file = inventory_dir / f"{inventory_name}.yml"
            
            inventory_data = {"all": {"hosts": {}}}
            
            # Add hosts
            for host in hosts:
                host_name = host["name"]
                host_vars = host.get("vars", {})
                inventory_data["all"]["hosts"][host_name] = host_vars
                
            # Add groups
            if groups:
                for group_name, host_list in groups.items():
                    inventory_data[group_name] = {"hosts": {}}
                    for host_name in host_list:
                        inventory_data[group_name]["hosts"][host_name] = {}
                        
            # Write inventory file
            with open(inventory_file, 'w') as f:
                yaml.dump(inventory_data, f, default_flow_style=False)
                
            logger.info(f"Ansible inventory created: {inventory_file}")
            return {
                "inventory_name": inventory_name,
                "inventory_file": str(inventory_file),
                "hosts_count": len(hosts),
                "groups_count": len(groups) if groups else 0,
                "status": "created"
            }
            
        except Exception as e:
            error_msg = f"Ansible inventory creation failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"inventory_name": inventory_name})
