"""
Integration & API Enhancement Module for Proxmox MCP Server

This module implements:
- Webhook system for event-driven automation
- REST API gateway for enhanced API management
- Third-party service integrations (Slack, Teams, PagerDuty, etc.)
- Event management and notification system
"""

import os
import json
import asyncio
import hmac
import hashlib
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
import uuid

import httpx
import websockets
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import jwt
from loguru import logger

from .client import ProxmoxClient
from .utils import run_command, format_error


class IntegrationManager:
    """Integration and API management for Proxmox infrastructure"""
    
    def __init__(self, proxmox_client: ProxmoxClient):
        self.client = proxmox_client
        self.config_dir = Path.home() / ".proxmox_mcp" / "integrations"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Webhook management
        self.webhooks = {}
        self.event_handlers = {}
        
        # API Gateway
        self.api_app = None
        self.security = HTTPBearer()
        
        # Third-party integrations
        self.integrations = {}
        
        # Event queue
        self.event_queue = asyncio.Queue()
        self.event_processor_task = None

    async def setup_webhooks(
        self,
        webhook_url: str,
        events: List[str] = None,
        secret_token: Optional[str] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Setup webhooks for event-driven automation"""
        try:
            if dry_run:
                return {
                    "action": "setup_webhooks",
                    "webhook_url": webhook_url,
                    "events": events or [],
                    "secret_token": "***" if secret_token else None,
                    "retry_policy": retry_policy or {},
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            if events is None:
                events = ["vm_start", "vm_stop", "backup_complete", "node_status_change"]
                
            if retry_policy is None:
                retry_policy = {
                    "max_retries": 3,
                    "retry_delay": 5,
                    "backoff_multiplier": 2
                }
                
            webhook_id = str(uuid.uuid4())
            
            webhook_config = {
                "id": webhook_id,
                "url": webhook_url,
                "events": events,
                "secret_token": secret_token,
                "retry_policy": retry_policy,
                "created_at": datetime.now().isoformat(),
                "enabled": True,
                "stats": {
                    "total_sent": 0,
                    "successful": 0,
                    "failed": 0,
                    "last_sent": None
                }
            }
            
            # Validate webhook URL
            try:
                async with httpx.AsyncClient() as client:
                    test_payload = {
                        "event": "webhook_test",
                        "timestamp": datetime.now().isoformat(),
                        "webhook_id": webhook_id
                    }
                    
                    headers = {"Content-Type": "application/json"}
                    if secret_token:
                        signature = self._generate_webhook_signature(
                            json.dumps(test_payload), secret_token
                        )
                        headers["X-Webhook-Signature"] = signature
                        
                    response = await client.post(
                        webhook_url,
                        json=test_payload,
                        headers=headers,
                        timeout=10
                    )
                    
                    webhook_config["test_response"] = {
                        "status_code": response.status_code,
                        "success": response.status_code < 400
                    }
                    
            except Exception as e:
                logger.warning(f"Webhook test failed: {e}")
                webhook_config["test_response"] = {
                    "error": str(e),
                    "success": False
                }
                
            # Store webhook configuration
            self.webhooks[webhook_id] = webhook_config
            await self._save_webhook_config()
            
            # Start event processor if not running
            if not self.event_processor_task:
                self.event_processor_task = asyncio.create_task(self._process_events())
                
            result = {
                "webhook_id": webhook_id,
                "webhook_url": webhook_url,
                "events": events,
                "retry_policy": retry_policy,
                "test_result": webhook_config["test_response"],
                "status": "configured"
            }
            
            logger.info(f"Webhook configured: {webhook_id} -> {webhook_url}")
            return result
            
        except Exception as e:
            error_msg = f"Webhook setup failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"webhook_url": webhook_url})

    def _generate_webhook_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook security"""
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    async def _save_webhook_config(self):
        """Save webhook configuration to file"""
        try:
            config_file = self.config_dir / "webhooks.json"
            with open(config_file, 'w') as f:
                json.dump(self.webhooks, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save webhook config: {e}")

    async def _process_events(self):
        """Process events from the event queue"""
        try:
            while True:
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    await self._handle_event(event)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Event processing failed: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Event processor stopped")

    async def _handle_event(self, event: Dict[str, Any]):
        """Handle a single event by sending to configured webhooks"""
        try:
            event_type = event.get("type", "")
            
            # Find webhooks that should receive this event
            relevant_webhooks = [
                webhook for webhook in self.webhooks.values()
                if webhook.get("enabled", False) and event_type in webhook.get("events", [])
            ]
            
            if not relevant_webhooks:
                logger.debug(f"No webhooks configured for event type: {event_type}")
                return
                
            # Send event to each relevant webhook
            for webhook in relevant_webhooks:
                await self._send_webhook(webhook, event)
                
        except Exception as e:
            logger.error(f"Event handling failed: {e}")

    async def _send_webhook(self, webhook: Dict[str, Any], event: Dict[str, Any]):
        """Send event to a specific webhook"""
        try:
            webhook_id = webhook["id"]
            url = webhook["url"]
            secret_token = webhook.get("secret_token")
            retry_policy = webhook.get("retry_policy", {})
            
            # Prepare payload
            payload = {
                "webhook_id": webhook_id,
                "event": event,
                "timestamp": datetime.now().isoformat()
            }
            
            payload_json = json.dumps(payload, default=str)
            headers = {"Content-Type": "application/json"}
            
            if secret_token:
                signature = self._generate_webhook_signature(payload_json, secret_token)
                headers["X-Webhook-Signature"] = signature
                
            # Send with retry logic
            max_retries = retry_policy.get("max_retries", 3)
            retry_delay = retry_policy.get("retry_delay", 5)
            backoff_multiplier = retry_policy.get("backoff_multiplier", 2)
            
            for attempt in range(max_retries + 1):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            url,
                            content=payload_json,
                            headers=headers,
                            timeout=30
                        )
                        
                    # Update statistics
                    webhook["stats"]["total_sent"] += 1
                    webhook["stats"]["last_sent"] = datetime.now().isoformat()
                    
                    if response.status_code < 400:
                        webhook["stats"]["successful"] += 1
                        logger.debug(f"Webhook sent successfully: {webhook_id}")
                        break
                    else:
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}",
                            request=response.request,
                            response=response
                        )
                        
                except Exception as e:
                    if attempt < max_retries:
                        wait_time = retry_delay * (backoff_multiplier ** attempt)
                        logger.warning(f"Webhook attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        webhook["stats"]["failed"] += 1
                        logger.error(f"Webhook failed after {max_retries} attempts: {e}")
                        
            # Save updated statistics
            await self._save_webhook_config()
            
        except Exception as e:
            logger.error(f"Webhook sending failed: {e}")

    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to the event queue"""
        try:
            event = {
                "type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "id": str(uuid.uuid4())
            }
            
            await self.event_queue.put(event)
            logger.debug(f"Event emitted: {event_type}")
            
        except Exception as e:
            logger.error(f"Event emission failed: {e}")

    async def api_gateway(
        self,
        enable_rate_limiting: bool = True,
        auth_providers: List[str] = None,
        cors_enabled: bool = True,
        api_versioning: bool = True,
        port: int = 8000,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Deploy API gateway for enhanced API management"""
        try:
            if dry_run:
                return {
                    "action": "api_gateway",
                    "enable_rate_limiting": enable_rate_limiting,
                    "auth_providers": auth_providers or [],
                    "cors_enabled": cors_enabled,
                    "api_versioning": api_versioning,
                    "port": port,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            if auth_providers is None:
                auth_providers = ["oauth2", "jwt"]
                
            # Create FastAPI application
            self.api_app = FastAPI(
                title="Proxmox MCP API Gateway",
                description="Enhanced API gateway for Proxmox MCP Server",
                version="1.0.0" if api_versioning else None
            )
            
            # Configure CORS
            if cors_enabled:
                self.api_app.add_middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"]
                )
                
            # Setup authentication
            if "jwt" in auth_providers:
                await self._setup_jwt_auth()
                
            # Setup rate limiting
            if enable_rate_limiting:
                await self._setup_rate_limiting()
                
            # Add API routes
            await self._setup_api_routes()
            
            result = {
                "api_gateway_url": f"http://localhost:{port}",
                "features": {
                    "rate_limiting": enable_rate_limiting,
                    "cors": cors_enabled,
                    "versioning": api_versioning,
                    "auth_providers": auth_providers
                },
                "endpoints": [
                    "/docs",  # Swagger documentation
                    "/health",  # Health check
                    "/metrics",  # Prometheus metrics
                    "/webhooks",  # Webhook management
                    "/events"  # Event stream
                ]
            }
            
            # Start API server (in production, use proper ASGI server)
            logger.info(f"API Gateway configured on port {port}")
            result["status"] = "configured"
            return result
            
        except Exception as e:
            error_msg = f"API gateway setup failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"port": port})

    async def _setup_jwt_auth(self):
        """Setup JWT authentication for API gateway"""
        try:
            # JWT secret key (in production, use proper key management)
            self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-here")
            
            async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
                try:
                    payload = jwt.decode(
                        credentials.credentials,
                        self.jwt_secret,
                        algorithms=["HS256"]
                    )
                    return payload
                except jwt.ExpiredSignatureError:
                    raise HTTPException(status_code=401, detail="Token expired")
                except jwt.InvalidTokenError:
                    raise HTTPException(status_code=401, detail="Invalid token")
                    
            self.verify_token = verify_token
            
        except Exception as e:
            logger.error(f"JWT auth setup failed: {e}")

    async def _setup_rate_limiting(self):
        """Setup rate limiting for API gateway"""
        try:
            # Simple in-memory rate limiting
            self.rate_limit_store = {}
            
            async def rate_limit_middleware(request: Request, call_next):
                client_ip = request.client.host
                current_time = datetime.now().timestamp()
                
                # Clean old entries
                cutoff_time = current_time - 60  # 1 minute window
                self.rate_limit_store = {
                    ip: requests for ip, requests in self.rate_limit_store.items()
                    if any(req_time > cutoff_time for req_time in requests)
                }
                
                # Check rate limit (100 requests per minute per IP)
                client_requests = self.rate_limit_store.get(client_ip, [])
                client_requests = [t for t in client_requests if t > cutoff_time]
                
                if len(client_requests) >= 100:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                    
                client_requests.append(current_time)
                self.rate_limit_store[client_ip] = client_requests
                
                response = await call_next(request)
                return response
                
            self.api_app.middleware("http")(rate_limit_middleware)
            
        except Exception as e:
            logger.error(f"Rate limiting setup failed: {e}")

    async def _setup_api_routes(self):
        """Setup API routes for the gateway"""
        try:
            @self.api_app.get("/health")
            async def health_check():
                return {"status": "healthy", "timestamp": datetime.now().isoformat()}
                
            @self.api_app.get("/metrics")
            async def get_metrics():
                # Return Prometheus metrics
                from .monitoring import MonitoringManager
                monitoring = MonitoringManager(self.client)
                return await monitoring.get_prometheus_metrics()
                
            @self.api_app.get("/webhooks")
            async def list_webhooks():
                return {"webhooks": list(self.webhooks.values())}
                
            @self.api_app.post("/webhooks")
            async def create_webhook(webhook_data: dict):
                return await self.setup_webhooks(**webhook_data)
                
            @self.api_app.delete("/webhooks/{webhook_id}")
            async def delete_webhook(webhook_id: str):
                if webhook_id in self.webhooks:
                    del self.webhooks[webhook_id]
                    await self._save_webhook_config()
                    return {"deleted": True}
                return {"deleted": False, "error": "Webhook not found"}
                
            @self.api_app.get("/events")
            async def get_events():
                # Return recent events (implement event history storage)
                return {"events": [], "message": "Event history not implemented"}
                
        except Exception as e:
            logger.error(f"API routes setup failed: {e}")

    async def integrate_service(
        self,
        service_type: str,
        credentials: Dict[str, str],
        notification_types: List[str] = None,
        webhook_url: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Integrate with external services for notifications and automation"""
        try:
            if dry_run:
                return {
                    "action": "integrate_service",
                    "service_type": service_type,
                    "credentials": {k: "***" for k in credentials.keys()},
                    "notification_types": notification_types or [],
                    "webhook_url": webhook_url,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            if notification_types is None:
                notification_types = ["alerts", "deployments"]
                
            integration_id = f"{service_type}_{int(datetime.now().timestamp())}"
            
            result = {
                "integration_id": integration_id,
                "service_type": service_type,
                "notification_types": notification_types
            }
            
            if service_type == "slack":
                integration_result = await self._integrate_slack(
                    credentials, notification_types, webhook_url
                )
            elif service_type == "teams":
                integration_result = await self._integrate_teams(
                    credentials, notification_types, webhook_url
                )
            elif service_type == "pagerduty":
                integration_result = await self._integrate_pagerduty(
                    credentials, notification_types
                )
            elif service_type == "jira":
                integration_result = await self._integrate_jira(
                    credentials, notification_types
                )
            elif service_type == "github":
                integration_result = await self._integrate_github(
                    credentials, notification_types, webhook_url
                )
            else:
                raise ValueError(f"Unsupported service type: {service_type}")
                
            # Store integration configuration
            integration_config = {
                "id": integration_id,
                "service_type": service_type,
                "credentials": credentials,  # In production, encrypt these
                "notification_types": notification_types,
                "webhook_url": webhook_url,
                "created_at": datetime.now().isoformat(),
                "enabled": True,
                **integration_result
            }
            
            self.integrations[integration_id] = integration_config
            await self._save_integrations_config()
            
            result.update(integration_result)
            result["status"] = "integrated"
            
            logger.info(f"Service integration completed: {service_type}")
            return result
            
        except Exception as e:
            error_msg = f"Service integration failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"service_type": service_type})

    async def _integrate_slack(
        self,
        credentials: Dict[str, str],
        notification_types: List[str],
        webhook_url: Optional[str]
    ) -> Dict[str, Any]:
        """Integrate with Slack"""
        try:
            webhook_url = webhook_url or credentials.get("webhook_url")
            if not webhook_url:
                raise ValueError("Slack webhook URL is required")
                
            # Test Slack integration
            test_message = {
                "text": "Proxmox MCP integration test",
                "attachments": [
                    {
                        "color": "good",
                        "fields": [
                            {
                                "title": "Status",
                                "value": "Integration successful",
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": datetime.now().isoformat(),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=test_message,
                    timeout=10
                )
                
            return {
                "webhook_url": webhook_url,
                "test_successful": response.status_code == 200,
                "supported_notifications": notification_types
            }
            
        except Exception as e:
            logger.error(f"Slack integration failed: {e}")
            raise

    async def _integrate_teams(
        self,
        credentials: Dict[str, str],
        notification_types: List[str],
        webhook_url: Optional[str]
    ) -> Dict[str, Any]:
        """Integrate with Microsoft Teams"""
        try:
            webhook_url = webhook_url or credentials.get("webhook_url")
            if not webhook_url:
                raise ValueError("Teams webhook URL is required")
                
            # Test Teams integration
            test_message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "0076D7",
                "summary": "Proxmox MCP Integration Test",
                "sections": [
                    {
                        "activityTitle": "Proxmox MCP",
                        "activitySubtitle": "Integration Test",
                        "activityImage": "https://teamsnodesample.azurewebsites.net/static/img/image5.png",
                        "facts": [
                            {
                                "name": "Status",
                                "value": "Integration successful"
                            },
                            {
                                "name": "Timestamp",
                                "value": datetime.now().isoformat()
                            }
                        ],
                        "markdown": True
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=test_message,
                    timeout=10
                )
                
            return {
                "webhook_url": webhook_url,
                "test_successful": response.status_code == 200,
                "supported_notifications": notification_types
            }
            
        except Exception as e:
            logger.error(f"Teams integration failed: {e}")
            raise

    async def _integrate_pagerduty(
        self,
        credentials: Dict[str, str],
        notification_types: List[str]
    ) -> Dict[str, Any]:
        """Integrate with PagerDuty"""
        try:
            integration_key = credentials.get("integration_key")
            if not integration_key:
                raise ValueError("PagerDuty integration key is required")
                
            # Test PagerDuty integration
            test_event = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "payload": {
                    "summary": "Proxmox MCP Integration Test",
                    "source": "proxmox-mcp",
                    "severity": "info",
                    "custom_details": {
                        "timestamp": datetime.now().isoformat(),
                        "status": "Integration test"
                    }
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=test_event,
                    timeout=10
                )
                
            return {
                "integration_key": integration_key,
                "test_successful": response.status_code == 202,
                "supported_notifications": notification_types
            }
            
        except Exception as e:
            logger.error(f"PagerDuty integration failed: {e}")
            raise

    async def _integrate_jira(
        self,
        credentials: Dict[str, str],
        notification_types: List[str]
    ) -> Dict[str, Any]:
        """Integrate with Jira"""
        try:
            base_url = credentials.get("base_url")
            username = credentials.get("username")
            api_token = credentials.get("api_token")
            
            if not all([base_url, username, api_token]):
                raise ValueError("Jira base_url, username, and api_token are required")
                
            # Test Jira integration
            auth = (username, api_token)
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/rest/api/2/myself",
                    auth=auth,
                    timeout=10
                )
                
            return {
                "base_url": base_url,
                "username": username,
                "test_successful": response.status_code == 200,
                "supported_notifications": notification_types
            }
            
        except Exception as e:
            logger.error(f"Jira integration failed: {e}")
            raise

    async def _integrate_github(
        self,
        credentials: Dict[str, str],
        notification_types: List[str],
        webhook_url: Optional[str]
    ) -> Dict[str, Any]:
        """Integrate with GitHub"""
        try:
            token = credentials.get("token")
            if not token:
                raise ValueError("GitHub token is required")
                
            # Test GitHub integration
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.github.com/user",
                    headers=headers,
                    timeout=10
                )
                
            user_info = response.json() if response.status_code == 200 else {}
            
            return {
                "token": token,
                "webhook_url": webhook_url,
                "test_successful": response.status_code == 200,
                "user": user_info.get("login", "unknown"),
                "supported_notifications": notification_types
            }
            
        except Exception as e:
            logger.error(f"GitHub integration failed: {e}")
            raise

    async def _save_integrations_config(self):
        """Save integrations configuration to file"""
        try:
            config_file = self.config_dir / "integrations.json"
            with open(config_file, 'w') as f:
                json.dump(self.integrations, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save integrations config: {e}")

    async def send_notification(
        self,
        message: str,
        severity: str = "info",
        service_types: Optional[List[str]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send notification through configured integrations"""
        try:
            if service_types is None:
                service_types = list(set(
                    integration["service_type"] 
                    for integration in self.integrations.values()
                    if integration.get("enabled", False)
                ))
                
            results = []
            
            for integration in self.integrations.values():
                if (integration.get("enabled", False) and 
                    integration["service_type"] in service_types):
                    
                    try:
                        result = await self._send_service_notification(
                            integration, message, severity, additional_data
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Notification failed for {integration['service_type']}: {e}")
                        results.append({
                            "service_type": integration["service_type"],
                            "status": "failed",
                            "error": str(e)
                        })
                        
            return {
                "message": message,
                "severity": severity,
                "service_types": service_types,
                "results": results,
                "total_sent": len([r for r in results if r.get("status") == "sent"])
            }
            
        except Exception as e:
            error_msg = f"Notification sending failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"message": message})

    async def _send_service_notification(
        self,
        integration: Dict[str, Any],
        message: str,
        severity: str,
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Send notification to a specific service"""
        try:
            service_type = integration["service_type"]
            
            if service_type == "slack":
                return await self._send_slack_notification(
                    integration, message, severity, additional_data
                )
            elif service_type == "teams":
                return await self._send_teams_notification(
                    integration, message, severity, additional_data
                )
            elif service_type == "pagerduty":
                return await self._send_pagerduty_notification(
                    integration, message, severity, additional_data
                )
            else:
                return {
                    "service_type": service_type,
                    "status": "not_implemented"
                }
                
        except Exception as e:
            logger.error(f"Service notification failed: {e}")
            raise

    async def _send_slack_notification(
        self,
        integration: Dict[str, Any],
        message: str,
        severity: str,
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Send Slack notification"""
        try:
            webhook_url = integration.get("webhook_url")
            
            color_map = {
                "info": "good",
                "warning": "warning", 
                "error": "danger",
                "critical": "danger"
            }
            
            slack_message = {
                "text": message,
                "attachments": [
                    {
                        "color": color_map.get(severity, "good"),
                        "fields": [
                            {
                                "title": "Severity",
                                "value": severity.upper(),
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": datetime.now().isoformat(),
                                "short": True
                            }
                        ]
                    }
                ]
            }
            
            if additional_data:
                for key, value in additional_data.items():
                    slack_message["attachments"][0]["fields"].append({
                        "title": key.title(),
                        "value": str(value),
                        "short": True
                    })
                    
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=slack_message,
                    timeout=10
                )
                
            return {
                "service_type": "slack",
                "status": "sent" if response.status_code == 200 else "failed",
                "response_code": response.status_code
            }
            
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            raise

    async def _send_teams_notification(
        self,
        integration: Dict[str, Any],
        message: str,
        severity: str,
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Send Teams notification"""
        try:
            webhook_url = integration.get("webhook_url")
            
            color_map = {
                "info": "0076D7",
                "warning": "FF8C00",
                "error": "FF0000",
                "critical": "8B0000"
            }
            
            facts = [
                {
                    "name": "Severity",
                    "value": severity.upper()
                },
                {
                    "name": "Timestamp", 
                    "value": datetime.now().isoformat()
                }
            ]
            
            if additional_data:
                for key, value in additional_data.items():
                    facts.append({
                        "name": key.title(),
                        "value": str(value)
                    })
                    
            teams_message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color_map.get(severity, "0076D7"),
                "summary": message,
                "sections": [
                    {
                        "activityTitle": "Proxmox MCP Notification",
                        "activitySubtitle": message,
                        "facts": facts,
                        "markdown": True
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=teams_message,
                    timeout=10
                )
                
            return {
                "service_type": "teams",
                "status": "sent" if response.status_code == 200 else "failed",
                "response_code": response.status_code
            }
            
        except Exception as e:
            logger.error(f"Teams notification failed: {e}")
            raise

    async def _send_pagerduty_notification(
        self,
        integration: Dict[str, Any],
        message: str,
        severity: str,
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Send PagerDuty notification"""
        try:
            integration_key = integration.get("integration_key")
            
            # Only send to PagerDuty for warnings and errors
            if severity not in ["warning", "error", "critical"]:
                return {
                    "service_type": "pagerduty",
                    "status": "skipped",
                    "reason": "Severity too low for PagerDuty"
                }
                
            pagerduty_event = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "payload": {
                    "summary": message,
                    "source": "proxmox-mcp",
                    "severity": severity,
                    "custom_details": additional_data or {}
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=pagerduty_event,
                    timeout=10
                )
                
            return {
                "service_type": "pagerduty",
                "status": "sent" if response.status_code == 202 else "failed",
                "response_code": response.status_code
            }
            
        except Exception as e:
            logger.error(f"PagerDuty notification failed: {e}")
            raise

    async def start_api_server(self, port: int = 8000):
        """Start the API gateway server"""
        try:
            if self.api_app:
                config = uvicorn.Config(
                    self.api_app,
                    host="0.0.0.0",
                    port=port,
                    log_level="info"
                )
                server = uvicorn.Server(config)
                await server.serve()
            else:
                logger.warning("API gateway not configured")
                
        except Exception as e:
            logger.error(f"API server start failed: {e}")

    async def stop_integrations(self):
        """Stop all integrations and cleanup"""
        try:
            # Stop event processor
            if self.event_processor_task:
                self.event_processor_task.cancel()
                try:
                    await self.event_processor_task
                except asyncio.CancelledError:
                    pass
                    
            logger.info("Integrations stopped")
            
        except Exception as e:
            logger.error(f"Integration cleanup failed: {e}")
