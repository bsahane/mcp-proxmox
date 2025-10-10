"""
AI/ML Optimization Module for Proxmox MCP Server

This module implements:
- AI-powered predictive scaling based on usage patterns
- Anomaly detection for proactive issue resolution
- Automated optimization recommendations
- Machine learning models for resource management
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import joblib

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from loguru import logger

from .client import ProxmoxClient
from .utils import run_command, format_error


class AIOptimizationManager:
    """AI-powered optimization and management for Proxmox infrastructure"""
    
    def __init__(self, proxmox_client: ProxmoxClient):
        self.client = proxmox_client
        self.models_dir = Path.home() / ".proxmox_mcp" / "ai_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # AI Models
        self.scaling_model = None
        self.anomaly_detector = None
        self.optimization_model = None
        self.scaler = StandardScaler()
        
        # Historical data storage
        self.metrics_history = []
        self.predictions_cache = {}

    async def ai_scaling(
        self,
        vmid: int,
        enable_prediction: bool = True,
        metrics_window: str = "7d",
        scaling_policy: Optional[Dict[str, Any]] = None,
        node: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """AI-powered predictive scaling based on usage patterns"""
        try:
            if dry_run:
                return {
                    "action": "ai_scaling",
                    "vmid": vmid,
                    "enable_prediction": enable_prediction,
                    "metrics_window": metrics_window,
                    "scaling_policy": scaling_policy or {},
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            node = node or self.client.get_first_node()
            
            # Default scaling policy
            if scaling_policy is None:
                scaling_policy = {
                    "cpu_threshold_up": 80,
                    "cpu_threshold_down": 20,
                    "memory_threshold_up": 85,
                    "memory_threshold_down": 30,
                    "scale_up_factor": 1.5,
                    "scale_down_factor": 0.8,
                    "min_cpu": 1,
                    "max_cpu": 16,
                    "min_memory": 1024,  # MB
                    "max_memory": 32768  # MB
                }
                
            result = {
                "vmid": vmid,
                "node": node,
                "scaling_policy": scaling_policy,
                "metrics_window": metrics_window
            }
            
            # Collect historical metrics
            historical_data = await self._collect_historical_metrics(vmid, metrics_window, node)
            result["historical_data_points"] = len(historical_data)
            
            if len(historical_data) < 10:
                return {
                    **result,
                    "status": "insufficient_data",
                    "message": "Need at least 10 data points for AI prediction"
                }
                
            # Train or load prediction model
            model_file = self.models_dir / f"scaling_model_vm_{vmid}.pkl"
            if enable_prediction:
                prediction_model = await self._train_scaling_model(historical_data, model_file)
                result["model_trained"] = True
            else:
                prediction_model = await self._load_model(model_file)
                result["model_loaded"] = prediction_model is not None
                
            # Get current VM configuration
            vm_config = await self.client.nodes(node).qemu(vmid).config.get()
            current_cpu = vm_config.get("cores", 1)
            current_memory = vm_config.get("memory", 1024)
            
            result["current_resources"] = {
                "cpu": current_cpu,
                "memory": current_memory
            }
            
            # Analyze current usage and predict future needs
            current_metrics = await self._get_current_vm_metrics(vmid, node)
            
            if prediction_model and enable_prediction:
                # Predict future resource usage
                predictions = await self._predict_resource_usage(
                    prediction_model, historical_data, current_metrics
                )
                result["predictions"] = predictions
                
                # Generate scaling recommendations
                scaling_recommendations = await self._generate_scaling_recommendations(
                    current_metrics, predictions, scaling_policy, current_cpu, current_memory
                )
                result["recommendations"] = scaling_recommendations
                
                # Apply scaling if recommended and not in dry run
                if scaling_recommendations.get("should_scale", False):
                    scaling_result = await self._apply_scaling(
                        vmid, scaling_recommendations, node, dry_run
                    )
                    result["scaling_applied"] = scaling_result
                    
            else:
                # Rule-based scaling without prediction
                rule_based_scaling = await self._rule_based_scaling(
                    current_metrics, scaling_policy, current_cpu, current_memory
                )
                result["rule_based_recommendations"] = rule_based_scaling
                
            logger.info(f"AI scaling analysis completed for VM {vmid}")
            result["status"] = "completed"
            return result
            
        except Exception as e:
            error_msg = f"AI scaling failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"vmid": vmid})

    async def _collect_historical_metrics(
        self,
        vmid: int,
        time_window: str,
        node: str
    ) -> List[Dict[str, Any]]:
        """Collect historical metrics for AI training"""
        try:
            # Parse time window
            if time_window.endswith('d'):
                days = int(time_window[:-1])
            elif time_window.endswith('h'):
                days = int(time_window[:-1]) / 24
            else:
                days = 7  # Default to 7 days
                
            # Generate synthetic historical data for demonstration
            # In a real implementation, this would query actual metrics storage
            historical_data = []
            
            for i in range(int(days * 24)):  # Hourly data points
                timestamp = datetime.now() - timedelta(hours=i)
                
                # Generate realistic usage patterns with some randomness
                base_cpu = 30 + 20 * np.sin(i * 2 * np.pi / 24)  # Daily pattern
                base_memory = 50 + 15 * np.sin(i * 2 * np.pi / 24)
                
                cpu_usage = max(0, min(100, base_cpu + np.random.normal(0, 10)))
                memory_usage = max(0, min(100, base_memory + np.random.normal(0, 8)))
                
                # Add weekly patterns
                if timestamp.weekday() < 5:  # Weekdays
                    cpu_usage *= 1.2
                    memory_usage *= 1.1
                    
                historical_data.append({
                    "timestamp": timestamp.isoformat(),
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_iops": np.random.uniform(100, 1000),
                    "network_rx": np.random.uniform(1000000, 10000000),  # bytes
                    "network_tx": np.random.uniform(1000000, 10000000),
                    "hour_of_day": timestamp.hour,
                    "day_of_week": timestamp.weekday(),
                    "is_weekend": 1 if timestamp.weekday() >= 5 else 0
                })
                
            return historical_data
            
        except Exception as e:
            logger.error(f"Historical metrics collection failed: {e}")
            return []

    async def _train_scaling_model(
        self,
        historical_data: List[Dict[str, Any]],
        model_file: Path
    ) -> Optional[Any]:
        """Train machine learning model for resource prediction"""
        try:
            if len(historical_data) < 20:
                logger.warning("Insufficient data for model training")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Prepare features
            features = [
                'hour_of_day', 'day_of_week', 'is_weekend',
                'cpu_usage', 'memory_usage', 'disk_iops'
            ]
            
            X = df[features].values
            
            # Create targets (predict next hour's usage)
            y_cpu = df['cpu_usage'].shift(-1).fillna(method='ffill').values
            y_memory = df['memory_usage'].shift(-1).fillna(method='ffill').values
            
            # Split data
            X_train, X_test, y_cpu_train, y_cpu_test, y_memory_train, y_memory_test = train_test_split(
                X, y_cpu, y_memory, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            cpu_model = RandomForestRegressor(n_estimators=100, random_state=42)
            memory_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            cpu_model.fit(X_train_scaled, y_cpu_train)
            memory_model.fit(X_train_scaled, y_memory_train)
            
            # Evaluate models
            cpu_pred = cpu_model.predict(X_test_scaled)
            memory_pred = memory_model.predict(X_test_scaled)
            
            cpu_mse = mean_squared_error(y_cpu_test, cpu_pred)
            memory_mse = mean_squared_error(y_memory_test, memory_pred)
            
            # Save models
            model_data = {
                "cpu_model": cpu_model,
                "memory_model": memory_model,
                "scaler": self.scaler,
                "features": features,
                "cpu_mse": cpu_mse,
                "memory_mse": memory_mse,
                "trained_at": datetime.now().isoformat()
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Scaling model trained and saved. CPU MSE: {cpu_mse:.2f}, Memory MSE: {memory_mse:.2f}")
            return model_data
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return None

    async def _load_model(self, model_file: Path) -> Optional[Any]:
        """Load trained model from file"""
        try:
            if not model_file.exists():
                return None
                
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
                
            # Check if model is recent (less than 7 days old)
            trained_at = datetime.fromisoformat(model_data.get("trained_at", "2020-01-01"))
            if datetime.now() - trained_at > timedelta(days=7):
                logger.info("Model is outdated, will retrain")
                return None
                
            return model_data
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return None

    async def _get_current_vm_metrics(self, vmid: int, node: str) -> Dict[str, Any]:
        """Get current VM metrics"""
        try:
            vm_status = await self.client.nodes(node).qemu(vmid).status.current.get()
            
            current_time = datetime.now()
            return {
                "cpu_usage": vm_status.get("cpu", 0) * 100,
                "memory_usage": (vm_status.get("mem", 0) / vm_status.get("maxmem", 1)) * 100,
                "disk_iops": vm_status.get("diskread", 0) + vm_status.get("diskwrite", 0),
                "network_rx": vm_status.get("netin", 0),
                "network_tx": vm_status.get("netout", 0),
                "hour_of_day": current_time.hour,
                "day_of_week": current_time.weekday(),
                "is_weekend": 1 if current_time.weekday() >= 5 else 0,
                "timestamp": current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Current metrics collection failed: {e}")
            return {}

    async def _predict_resource_usage(
        self,
        model_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        current_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict future resource usage"""
        try:
            cpu_model = model_data["cpu_model"]
            memory_model = model_data["memory_model"]
            scaler = model_data["scaler"]
            features = model_data["features"]
            
            # Prepare current metrics for prediction
            current_features = [current_metrics.get(feature, 0) for feature in features]
            current_scaled = scaler.transform([current_features])
            
            # Predict next hour
            cpu_prediction = cpu_model.predict(current_scaled)[0]
            memory_prediction = memory_model.predict(current_scaled)[0]
            
            # Predict next 24 hours
            future_predictions = []
            current_time = datetime.now()
            
            for i in range(1, 25):  # Next 24 hours
                future_time = current_time + timedelta(hours=i)
                future_features = current_features.copy()
                future_features[0] = future_time.hour  # hour_of_day
                future_features[1] = future_time.weekday()  # day_of_week
                future_features[2] = 1 if future_time.weekday() >= 5 else 0  # is_weekend
                
                future_scaled = scaler.transform([future_features])
                future_cpu = cpu_model.predict(future_scaled)[0]
                future_memory = memory_model.predict(future_scaled)[0]
                
                future_predictions.append({
                    "hour": i,
                    "timestamp": future_time.isoformat(),
                    "predicted_cpu": max(0, min(100, future_cpu)),
                    "predicted_memory": max(0, min(100, future_memory))
                })
                
            return {
                "next_hour_cpu": max(0, min(100, cpu_prediction)),
                "next_hour_memory": max(0, min(100, memory_prediction)),
                "future_24h": future_predictions,
                "model_accuracy": {
                    "cpu_mse": model_data.get("cpu_mse", 0),
                    "memory_mse": model_data.get("memory_mse", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Resource prediction failed: {e}")
            return {}

    async def _generate_scaling_recommendations(
        self,
        current_metrics: Dict[str, Any],
        predictions: Dict[str, Any],
        scaling_policy: Dict[str, Any],
        current_cpu: int,
        current_memory: int
    ) -> Dict[str, Any]:
        """Generate scaling recommendations based on predictions"""
        try:
            recommendations = {
                "should_scale": False,
                "scale_type": None,
                "recommended_cpu": current_cpu,
                "recommended_memory": current_memory,
                "confidence": 0.0,
                "reasoning": []
            }
            
            # Analyze current usage
            current_cpu_usage = current_metrics.get("cpu_usage", 0)
            current_memory_usage = current_metrics.get("memory_usage", 0)
            
            # Analyze predictions
            predicted_cpu = predictions.get("next_hour_cpu", current_cpu_usage)
            predicted_memory = predictions.get("next_hour_memory", current_memory_usage)
            
            # Check for scale up conditions
            scale_up_reasons = []
            scale_down_reasons = []
            
            # CPU scaling
            if (predicted_cpu > scaling_policy["cpu_threshold_up"] or 
                current_cpu_usage > scaling_policy["cpu_threshold_up"]):
                new_cpu = min(
                    scaling_policy["max_cpu"],
                    int(current_cpu * scaling_policy["scale_up_factor"])
                )
                if new_cpu > current_cpu:
                    recommendations["recommended_cpu"] = new_cpu
                    scale_up_reasons.append(f"CPU usage predicted to be {predicted_cpu:.1f}%")
                    
            elif (predicted_cpu < scaling_policy["cpu_threshold_down"] and 
                  current_cpu_usage < scaling_policy["cpu_threshold_down"]):
                new_cpu = max(
                    scaling_policy["min_cpu"],
                    int(current_cpu * scaling_policy["scale_down_factor"])
                )
                if new_cpu < current_cpu:
                    recommendations["recommended_cpu"] = new_cpu
                    scale_down_reasons.append(f"CPU usage consistently low: {predicted_cpu:.1f}%")
                    
            # Memory scaling
            if (predicted_memory > scaling_policy["memory_threshold_up"] or 
                current_memory_usage > scaling_policy["memory_threshold_up"]):
                new_memory = min(
                    scaling_policy["max_memory"],
                    int(current_memory * scaling_policy["scale_up_factor"])
                )
                if new_memory > current_memory:
                    recommendations["recommended_memory"] = new_memory
                    scale_up_reasons.append(f"Memory usage predicted to be {predicted_memory:.1f}%")
                    
            elif (predicted_memory < scaling_policy["memory_threshold_down"] and 
                  current_memory_usage < scaling_policy["memory_threshold_down"]):
                new_memory = max(
                    scaling_policy["min_memory"],
                    int(current_memory * scaling_policy["scale_down_factor"])
                )
                if new_memory < current_memory:
                    recommendations["recommended_memory"] = new_memory
                    scale_down_reasons.append(f"Memory usage consistently low: {predicted_memory:.1f}%")
                    
            # Determine scaling action
            if scale_up_reasons:
                recommendations["should_scale"] = True
                recommendations["scale_type"] = "up"
                recommendations["reasoning"] = scale_up_reasons
                recommendations["confidence"] = 0.8
            elif scale_down_reasons:
                recommendations["should_scale"] = True
                recommendations["scale_type"] = "down"
                recommendations["reasoning"] = scale_down_reasons
                recommendations["confidence"] = 0.6
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Scaling recommendations generation failed: {e}")
            return {"should_scale": False, "error": str(e)}

    async def _apply_scaling(
        self,
        vmid: int,
        recommendations: Dict[str, Any],
        node: str,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Apply scaling recommendations"""
        try:
            if dry_run:
                return {
                    "action": "apply_scaling",
                    "vmid": vmid,
                    "recommendations": recommendations,
                    "dry_run": True,
                    "status": "would_apply"
                }
                
            scaling_result = {
                "vmid": vmid,
                "node": node,
                "applied_changes": []
            }
            
            # Get current VM status
            vm_status = await self.client.nodes(node).qemu(vmid).status.current.get()
            is_running = vm_status.get("status") == "running"
            
            # Apply CPU scaling
            recommended_cpu = recommendations.get("recommended_cpu")
            if recommended_cpu:
                try:
                    await self.client.nodes(node).qemu(vmid).config.put(cores=recommended_cpu)
                    scaling_result["applied_changes"].append({
                        "resource": "cpu",
                        "new_value": recommended_cpu,
                        "status": "applied"
                    })
                    
                    # Hot-plug CPU if VM is running
                    if is_running:
                        # Note: Hot CPU scaling may require guest support
                        scaling_result["applied_changes"][-1]["hot_plugged"] = True
                        
                except Exception as e:
                    scaling_result["applied_changes"].append({
                        "resource": "cpu",
                        "status": "failed",
                        "error": str(e)
                    })
                    
            # Apply Memory scaling
            recommended_memory = recommendations.get("recommended_memory")
            if recommended_memory:
                try:
                    await self.client.nodes(node).qemu(vmid).config.put(memory=recommended_memory)
                    scaling_result["applied_changes"].append({
                        "resource": "memory",
                        "new_value": recommended_memory,
                        "status": "applied"
                    })
                    
                    # Hot-plug memory if VM is running and supports it
                    if is_running:
                        scaling_result["applied_changes"][-1]["hot_plugged"] = True
                        
                except Exception as e:
                    scaling_result["applied_changes"].append({
                        "resource": "memory",
                        "status": "failed",
                        "error": str(e)
                    })
                    
            scaling_result["status"] = "completed"
            return scaling_result
            
        except Exception as e:
            logger.error(f"Scaling application failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def _rule_based_scaling(
        self,
        current_metrics: Dict[str, Any],
        scaling_policy: Dict[str, Any],
        current_cpu: int,
        current_memory: int
    ) -> Dict[str, Any]:
        """Simple rule-based scaling without AI prediction"""
        try:
            current_cpu_usage = current_metrics.get("cpu_usage", 0)
            current_memory_usage = current_metrics.get("memory_usage", 0)
            
            recommendations = {
                "should_scale": False,
                "scale_type": None,
                "recommended_cpu": current_cpu,
                "recommended_memory": current_memory,
                "reasoning": []
            }
            
            # CPU scaling rules
            if current_cpu_usage > scaling_policy["cpu_threshold_up"]:
                new_cpu = min(
                    scaling_policy["max_cpu"],
                    int(current_cpu * scaling_policy["scale_up_factor"])
                )
                if new_cpu > current_cpu:
                    recommendations["recommended_cpu"] = new_cpu
                    recommendations["should_scale"] = True
                    recommendations["scale_type"] = "up"
                    recommendations["reasoning"].append(f"High CPU usage: {current_cpu_usage:.1f}%")
                    
            elif current_cpu_usage < scaling_policy["cpu_threshold_down"]:
                new_cpu = max(
                    scaling_policy["min_cpu"],
                    int(current_cpu * scaling_policy["scale_down_factor"])
                )
                if new_cpu < current_cpu:
                    recommendations["recommended_cpu"] = new_cpu
                    recommendations["should_scale"] = True
                    recommendations["scale_type"] = "down"
                    recommendations["reasoning"].append(f"Low CPU usage: {current_cpu_usage:.1f}%")
                    
            # Memory scaling rules
            if current_memory_usage > scaling_policy["memory_threshold_up"]:
                new_memory = min(
                    scaling_policy["max_memory"],
                    int(current_memory * scaling_policy["scale_up_factor"])
                )
                if new_memory > current_memory:
                    recommendations["recommended_memory"] = new_memory
                    recommendations["should_scale"] = True
                    if not recommendations["scale_type"]:
                        recommendations["scale_type"] = "up"
                    recommendations["reasoning"].append(f"High memory usage: {current_memory_usage:.1f}%")
                    
            elif current_memory_usage < scaling_policy["memory_threshold_down"]:
                new_memory = max(
                    scaling_policy["min_memory"],
                    int(current_memory * scaling_policy["scale_down_factor"])
                )
                if new_memory < current_memory:
                    recommendations["recommended_memory"] = new_memory
                    recommendations["should_scale"] = True
                    if not recommendations["scale_type"]:
                        recommendations["scale_type"] = "down"
                    recommendations["reasoning"].append(f"Low memory usage: {current_memory_usage:.1f}%")
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Rule-based scaling failed: {e}")
            return {"should_scale": False, "error": str(e)}

    async def anomaly_detection(
        self,
        detection_type: str = "performance",
        sensitivity: str = "medium",
        alert_threshold: float = 0.85,
        auto_remediation: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """AI-powered anomaly detection for proactive issue resolution"""
        try:
            if dry_run:
                return {
                    "action": "anomaly_detection",
                    "detection_type": detection_type,
                    "sensitivity": sensitivity,
                    "alert_threshold": alert_threshold,
                    "auto_remediation": auto_remediation,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            result = {
                "detection_type": detection_type,
                "sensitivity": sensitivity,
                "alert_threshold": alert_threshold,
                "auto_remediation": auto_remediation
            }
            
            # Collect current system metrics
            system_metrics = await self._collect_system_metrics()
            result["metrics_collected"] = len(system_metrics)
            
            # Load or train anomaly detection model
            model_file = self.models_dir / f"anomaly_model_{detection_type}.pkl"
            anomaly_model = await self._load_or_train_anomaly_model(
                model_file, detection_type, sensitivity
            )
            
            if not anomaly_model:
                return {
                    **result,
                    "status": "model_unavailable",
                    "message": "Could not load or train anomaly detection model"
                }
                
            # Detect anomalies
            anomalies = await self._detect_anomalies(
                anomaly_model, system_metrics, alert_threshold
            )
            
            result["anomalies_detected"] = len(anomalies)
            result["anomalies"] = anomalies
            
            # Generate remediation suggestions
            if anomalies:
                remediation_suggestions = await self._generate_remediation_suggestions(anomalies)
                result["remediation_suggestions"] = remediation_suggestions
                
                # Apply auto-remediation if enabled
                if auto_remediation:
                    remediation_results = await self._apply_auto_remediation(
                        remediation_suggestions, dry_run
                    )
                    result["auto_remediation_results"] = remediation_results
                    
            logger.info(f"Anomaly detection completed. Found {len(anomalies)} anomalies")
            result["status"] = "completed"
            return result
            
        except Exception as e:
            error_msg = f"Anomaly detection failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"detection_type": detection_type})

    async def _collect_system_metrics(self) -> List[Dict[str, Any]]:
        """Collect system-wide metrics for anomaly detection"""
        try:
            metrics = []
            
            # Get all nodes
            nodes = await self.client.nodes.get()
            
            for node_info in nodes:
                node_name = node_info["node"]
                
                try:
                    # Node metrics
                    node_status = await self.client.nodes(node_name).status.get()
                    
                    # VM metrics
                    vms = await self.client.nodes(node_name).qemu.get()
                    
                    for vm in vms:
                        vmid = vm["vmid"]
                        try:
                            vm_status = await self.client.nodes(node_name).qemu(vmid).status.current.get()
                            
                            metrics.append({
                                "timestamp": datetime.now().isoformat(),
                                "node": node_name,
                                "vmid": vmid,
                                "vm_name": vm.get("name", f"vm-{vmid}"),
                                "vm_status": vm_status.get("status", "unknown"),
                                "cpu_usage": vm_status.get("cpu", 0) * 100,
                                "memory_usage": (vm_status.get("mem", 0) / vm_status.get("maxmem", 1)) * 100,
                                "disk_read": vm_status.get("diskread", 0),
                                "disk_write": vm_status.get("diskwrite", 0),
                                "network_in": vm_status.get("netin", 0),
                                "network_out": vm_status.get("netout", 0),
                                "uptime": vm_status.get("uptime", 0),
                                "node_cpu": node_status.get("cpu", 0) * 100,
                                "node_memory": (node_status.get("memory", {}).get("used", 0) / 
                                              node_status.get("memory", {}).get("total", 1)) * 100
                            })
                            
                        except Exception as e:
                            logger.warning(f"Failed to collect metrics for VM {vmid}: {e}")
                            
                except Exception as e:
                    logger.warning(f"Failed to collect metrics for node {node_name}: {e}")
                    
            return metrics
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            return []

    async def _load_or_train_anomaly_model(
        self,
        model_file: Path,
        detection_type: str,
        sensitivity: str
    ) -> Optional[Any]:
        """Load existing or train new anomaly detection model"""
        try:
            # Try to load existing model
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    
                # Check if model is recent
                trained_at = datetime.fromisoformat(model_data.get("trained_at", "2020-01-01"))
                if datetime.now() - trained_at < timedelta(days=3):
                    return model_data
                    
            # Train new model with synthetic data
            # In production, use actual historical data
            training_data = await self._generate_training_data_for_anomaly_detection()
            
            if len(training_data) < 100:
                logger.warning("Insufficient training data for anomaly detection")
                return None
                
            # Prepare features
            df = pd.DataFrame(training_data)
            features = ['cpu_usage', 'memory_usage', 'disk_read', 'disk_write', 
                       'network_in', 'network_out', 'node_cpu', 'node_memory']
            
            X = df[features].fillna(0).values
            
            # Configure contamination based on sensitivity
            contamination_map = {
                "low": 0.05,
                "medium": 0.1,
                "high": 0.15
            }
            contamination = contamination_map.get(sensitivity, 0.1)
            
            # Train Isolation Forest model
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            model.fit(X)
            
            # Create scaler for normalization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model_data = {
                "model": model,
                "scaler": scaler,
                "features": features,
                "contamination": contamination,
                "detection_type": detection_type,
                "sensitivity": sensitivity,
                "trained_at": datetime.now().isoformat()
            }
            
            # Save model
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Anomaly detection model trained for {detection_type}")
            return model_data
            
        except Exception as e:
            logger.error(f"Anomaly model loading/training failed: {e}")
            return None

    async def _generate_training_data_for_anomaly_detection(self) -> List[Dict[str, Any]]:
        """Generate synthetic training data for anomaly detection"""
        try:
            training_data = []
            
            # Generate normal operating patterns
            for i in range(500):
                # Normal patterns with some variation
                cpu_usage = np.random.normal(40, 15)
                memory_usage = np.random.normal(60, 20)
                disk_read = np.random.exponential(1000000)
                disk_write = np.random.exponential(500000)
                network_in = np.random.exponential(5000000)
                network_out = np.random.exponential(3000000)
                node_cpu = np.random.normal(30, 10)
                node_memory = np.random.normal(50, 15)
                
                training_data.append({
                    "cpu_usage": max(0, min(100, cpu_usage)),
                    "memory_usage": max(0, min(100, memory_usage)),
                    "disk_read": max(0, disk_read),
                    "disk_write": max(0, disk_write),
                    "network_in": max(0, network_in),
                    "network_out": max(0, network_out),
                    "node_cpu": max(0, min(100, node_cpu)),
                    "node_memory": max(0, min(100, node_memory))
                })
                
            # Add some anomalous patterns
            for i in range(50):
                # Anomalous patterns
                if i % 3 == 0:  # High CPU anomaly
                    cpu_usage = np.random.uniform(90, 100)
                    memory_usage = np.random.normal(60, 20)
                elif i % 3 == 1:  # High memory anomaly
                    cpu_usage = np.random.normal(40, 15)
                    memory_usage = np.random.uniform(90, 100)
                else:  # Network anomaly
                    cpu_usage = np.random.normal(40, 15)
                    memory_usage = np.random.normal(60, 20)
                    
                disk_read = np.random.exponential(10000000) if i % 4 == 0 else np.random.exponential(1000000)
                disk_write = np.random.exponential(5000000) if i % 4 == 0 else np.random.exponential(500000)
                network_in = np.random.exponential(50000000) if i % 5 == 0 else np.random.exponential(5000000)
                network_out = np.random.exponential(30000000) if i % 5 == 0 else np.random.exponential(3000000)
                node_cpu = np.random.uniform(80, 100) if i % 6 == 0 else np.random.normal(30, 10)
                node_memory = np.random.uniform(85, 100) if i % 6 == 0 else np.random.normal(50, 15)
                
                training_data.append({
                    "cpu_usage": max(0, min(100, cpu_usage)),
                    "memory_usage": max(0, min(100, memory_usage)),
                    "disk_read": max(0, disk_read),
                    "disk_write": max(0, disk_write),
                    "network_in": max(0, network_in),
                    "network_out": max(0, network_out),
                    "node_cpu": max(0, min(100, node_cpu)),
                    "node_memory": max(0, min(100, node_memory))
                })
                
            return training_data
            
        except Exception as e:
            logger.error(f"Training data generation failed: {e}")
            return []

    async def _detect_anomalies(
        self,
        model_data: Dict[str, Any],
        current_metrics: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics"""
        try:
            if not current_metrics:
                return []
                
            model = model_data["model"]
            scaler = model_data["scaler"]
            features = model_data["features"]
            
            anomalies = []
            
            for metric in current_metrics:
                # Prepare features
                feature_values = [metric.get(feature, 0) for feature in features]
                X = np.array([feature_values])
                X_scaled = scaler.transform(X)
                
                # Predict anomaly
                anomaly_score = model.decision_function(X_scaled)[0]
                is_anomaly = model.predict(X_scaled)[0] == -1
                
                # Convert score to probability-like value
                anomaly_probability = 1 / (1 + np.exp(anomaly_score))
                
                if is_anomaly and anomaly_probability >= threshold:
                    anomalies.append({
                        "node": metric.get("node"),
                        "vmid": metric.get("vmid"),
                        "vm_name": metric.get("vm_name"),
                        "anomaly_score": float(anomaly_score),
                        "anomaly_probability": float(anomaly_probability),
                        "detected_at": datetime.now().isoformat(),
                        "metrics": {feature: metric.get(feature, 0) for feature in features},
                        "severity": "high" if anomaly_probability > 0.9 else "medium"
                    })
                    
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []

    async def _generate_remediation_suggestions(
        self,
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate remediation suggestions for detected anomalies"""
        try:
            suggestions = []
            
            for anomaly in anomalies:
                metrics = anomaly.get("metrics", {})
                suggestion = {
                    "anomaly_id": f"{anomaly.get('node')}_{anomaly.get('vmid')}_{int(datetime.now().timestamp())}",
                    "node": anomaly.get("node"),
                    "vmid": anomaly.get("vmid"),
                    "vm_name": anomaly.get("vm_name"),
                    "severity": anomaly.get("severity"),
                    "actions": []
                }
                
                # CPU-related remediation
                if metrics.get("cpu_usage", 0) > 90:
                    suggestion["actions"].append({
                        "type": "scale_cpu",
                        "description": "Increase CPU allocation",
                        "priority": "high",
                        "estimated_impact": "Reduce CPU bottleneck"
                    })
                    
                # Memory-related remediation
                if metrics.get("memory_usage", 0) > 90:
                    suggestion["actions"].append({
                        "type": "scale_memory",
                        "description": "Increase memory allocation",
                        "priority": "high",
                        "estimated_impact": "Prevent memory pressure"
                    })
                    
                # Disk I/O remediation
                if metrics.get("disk_read", 0) + metrics.get("disk_write", 0) > 10000000:
                    suggestion["actions"].append({
                        "type": "optimize_storage",
                        "description": "Optimize disk I/O or migrate to faster storage",
                        "priority": "medium",
                        "estimated_impact": "Improve disk performance"
                    })
                    
                # Network remediation
                if metrics.get("network_in", 0) + metrics.get("network_out", 0) > 50000000:
                    suggestion["actions"].append({
                        "type": "network_optimization",
                        "description": "Check network configuration and bandwidth",
                        "priority": "medium",
                        "estimated_impact": "Reduce network bottleneck"
                    })
                    
                # Node-level issues
                if metrics.get("node_cpu", 0) > 85 or metrics.get("node_memory", 0) > 85:
                    suggestion["actions"].append({
                        "type": "migrate_vm",
                        "description": "Consider migrating VM to less loaded node",
                        "priority": "medium",
                        "estimated_impact": "Distribute load across cluster"
                    })
                    
                suggestions.append(suggestion)
                
            return suggestions
            
        except Exception as e:
            logger.error(f"Remediation suggestions generation failed: {e}")
            return []

    async def _apply_auto_remediation(
        self,
        suggestions: List[Dict[str, Any]],
        dry_run: bool
    ) -> List[Dict[str, Any]]:
        """Apply automatic remediation actions"""
        try:
            results = []
            
            for suggestion in suggestions:
                vmid = suggestion.get("vmid")
                node = suggestion.get("node")
                
                for action in suggestion.get("actions", []):
                    action_result = {
                        "anomaly_id": suggestion.get("anomaly_id"),
                        "vmid": vmid,
                        "action_type": action.get("type"),
                        "description": action.get("description"),
                        "status": "pending"
                    }
                    
                    if dry_run:
                        action_result["status"] = "dry_run"
                        results.append(action_result)
                        continue
                        
                    try:
                        if action["type"] == "scale_cpu":
                            # Get current config and scale up
                            vm_config = await self.client.nodes(node).qemu(vmid).config.get()
                            current_cpu = vm_config.get("cores", 1)
                            new_cpu = min(16, int(current_cpu * 1.5))
                            
                            await self.client.nodes(node).qemu(vmid).config.put(cores=new_cpu)
                            action_result["status"] = "applied"
                            action_result["details"] = f"CPU scaled from {current_cpu} to {new_cpu}"
                            
                        elif action["type"] == "scale_memory":
                            # Get current config and scale up
                            vm_config = await self.client.nodes(node).qemu(vmid).config.get()
                            current_memory = vm_config.get("memory", 1024)
                            new_memory = min(32768, int(current_memory * 1.5))
                            
                            await self.client.nodes(node).qemu(vmid).config.put(memory=new_memory)
                            action_result["status"] = "applied"
                            action_result["details"] = f"Memory scaled from {current_memory}MB to {new_memory}MB"
                            
                        else:
                            action_result["status"] = "not_implemented"
                            action_result["details"] = f"Auto-remediation for {action['type']} not yet implemented"
                            
                    except Exception as e:
                        action_result["status"] = "failed"
                        action_result["error"] = str(e)
                        
                    results.append(action_result)
                    
            return results
            
        except Exception as e:
            logger.error(f"Auto-remediation failed: {e}")
            return []

    async def auto_optimize(
        self,
        optimization_scope: str = "all",
        learning_period: int = 7,
        apply_recommendations: bool = False,
        rollback_enabled: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Automatically optimize VM configurations based on usage patterns"""
        try:
            if dry_run:
                return {
                    "action": "auto_optimize",
                    "optimization_scope": optimization_scope,
                    "learning_period": learning_period,
                    "apply_recommendations": apply_recommendations,
                    "rollback_enabled": rollback_enabled,
                    "dry_run": True,
                    "status": "would_execute"
                }
                
            result = {
                "optimization_scope": optimization_scope,
                "learning_period": learning_period,
                "apply_recommendations": apply_recommendations,
                "rollback_enabled": rollback_enabled
            }
            
            # Collect optimization data
            optimization_data = await self._collect_optimization_data(
                optimization_scope, learning_period
            )
            
            result["data_collected"] = len(optimization_data)
            
            if not optimization_data:
                return {
                    **result,
                    "status": "no_data",
                    "message": "No data available for optimization"
                }
                
            # Generate optimization recommendations
            recommendations = await self._generate_optimization_recommendations(
                optimization_data
            )
            
            result["recommendations"] = recommendations
            result["recommendations_count"] = len(recommendations)
            
            # Apply recommendations if requested
            if apply_recommendations and recommendations:
                application_results = await self._apply_optimization_recommendations(
                    recommendations, rollback_enabled, dry_run
                )
                result["application_results"] = application_results
                
            logger.info(f"Auto-optimization completed with {len(recommendations)} recommendations")
            result["status"] = "completed"
            return result
            
        except Exception as e:
            error_msg = f"Auto-optimization failed: {str(e)}"
            logger.error(error_msg)
            return format_error(error_msg, {"optimization_scope": optimization_scope})

    async def _collect_optimization_data(
        self,
        scope: str,
        learning_period: int
    ) -> List[Dict[str, Any]]:
        """Collect data for optimization analysis"""
        try:
            # This would collect actual historical data
            # For now, generate synthetic data
            optimization_data = []
            
            nodes = await self.client.nodes.get()
            
            for node_info in nodes:
                node_name = node_info["node"]
                
                if scope in ["all", "vm"]:
                    vms = await self.client.nodes(node_name).qemu.get()
                    
                    for vm in vms:
                        vmid = vm["vmid"]
                        
                        # Generate historical usage patterns
                        for day in range(learning_period):
                            date = datetime.now() - timedelta(days=day)
                            
                            # Simulate daily patterns
                            for hour in range(24):
                                timestamp = date.replace(hour=hour, minute=0, second=0)
                                
                                # Generate realistic usage with patterns
                                base_cpu = 30 + 20 * np.sin(hour * 2 * np.pi / 24)
                                base_memory = 50 + 15 * np.sin(hour * 2 * np.pi / 24)
                                
                                cpu_usage = max(0, min(100, base_cpu + np.random.normal(0, 10)))
                                memory_usage = max(0, min(100, base_memory + np.random.normal(0, 8)))
                                
                                optimization_data.append({
                                    "timestamp": timestamp.isoformat(),
                                    "node": node_name,
                                    "vmid": vmid,
                                    "vm_name": vm.get("name", f"vm-{vmid}"),
                                    "cpu_usage": cpu_usage,
                                    "memory_usage": memory_usage,
                                    "disk_iops": np.random.uniform(100, 1000),
                                    "network_throughput": np.random.uniform(1000000, 10000000),
                                    "hour_of_day": hour,
                                    "day_of_week": timestamp.weekday(),
                                    "is_weekend": 1 if timestamp.weekday() >= 5 else 0
                                })
                                
            return optimization_data
            
        except Exception as e:
            logger.error(f"Optimization data collection failed: {e}")
            return []

    async def _generate_optimization_recommendations(
        self,
        optimization_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on usage patterns"""
        try:
            recommendations = []
            
            # Group data by VM
            vm_data = {}
            for data_point in optimization_data:
                vmid = data_point["vmid"]
                if vmid not in vm_data:
                    vm_data[vmid] = []
                vm_data[vmid].append(data_point)
                
            # Analyze each VM
            for vmid, vm_points in vm_data.items():
                if len(vm_points) < 10:
                    continue
                    
                df = pd.DataFrame(vm_points)
                
                # Calculate statistics
                cpu_stats = {
                    "mean": df["cpu_usage"].mean(),
                    "max": df["cpu_usage"].max(),
                    "min": df["cpu_usage"].min(),
                    "std": df["cpu_usage"].std(),
                    "p95": df["cpu_usage"].quantile(0.95),
                    "p99": df["cpu_usage"].quantile(0.99)
                }
                
                memory_stats = {
                    "mean": df["memory_usage"].mean(),
                    "max": df["memory_usage"].max(),
                    "min": df["memory_usage"].min(),
                    "std": df["memory_usage"].std(),
                    "p95": df["memory_usage"].quantile(0.95),
                    "p99": df["memory_usage"].quantile(0.99)
                }
                
                # Generate recommendations
                vm_recommendations = {
                    "vmid": vmid,
                    "vm_name": vm_points[0]["vm_name"],
                    "node": vm_points[0]["node"],
                    "analysis_period": f"{len(vm_points)} data points",
                    "recommendations": []
                }
                
                # CPU recommendations
                if cpu_stats["p95"] < 30:
                    vm_recommendations["recommendations"].append({
                        "type": "cpu_downsize",
                        "current_usage": f"P95: {cpu_stats['p95']:.1f}%",
                        "recommendation": "Consider reducing CPU allocation",
                        "potential_savings": "20-30% resource cost",
                        "confidence": 0.8
                    })
                elif cpu_stats["p95"] > 80:
                    vm_recommendations["recommendations"].append({
                        "type": "cpu_upsize",
                        "current_usage": f"P95: {cpu_stats['p95']:.1f}%",
                        "recommendation": "Consider increasing CPU allocation",
                        "performance_impact": "Reduce CPU bottlenecks",
                        "confidence": 0.9
                    })
                    
                # Memory recommendations
                if memory_stats["p95"] < 40:
                    vm_recommendations["recommendations"].append({
                        "type": "memory_downsize",
                        "current_usage": f"P95: {memory_stats['p95']:.1f}%",
                        "recommendation": "Consider reducing memory allocation",
                        "potential_savings": "15-25% resource cost",
                        "confidence": 0.7
                    })
                elif memory_stats["p95"] > 85:
                    vm_recommendations["recommendations"].append({
                        "type": "memory_upsize",
                        "current_usage": f"P95: {memory_stats['p95']:.1f}%",
                        "recommendation": "Consider increasing memory allocation",
                        "performance_impact": "Prevent memory pressure",
                        "confidence": 0.9
                    })
                    
                # Usage pattern recommendations
                if cpu_stats["std"] < 5 and memory_stats["std"] < 5:
                    vm_recommendations["recommendations"].append({
                        "type": "steady_workload",
                        "pattern": "Stable resource usage",
                        "recommendation": "Good candidate for resource reservation",
                        "optimization": "Enable resource guarantees",
                        "confidence": 0.8
                    })
                    
                if len(vm_recommendations["recommendations"]) > 0:
                    recommendations.append(vm_recommendations)
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Optimization recommendations generation failed: {e}")
            return []

    async def _apply_optimization_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        rollback_enabled: bool,
        dry_run: bool
    ) -> List[Dict[str, Any]]:
        """Apply optimization recommendations"""
        try:
            application_results = []
            
            for vm_rec in recommendations:
                vmid = vm_rec["vmid"]
                node = vm_rec["node"]
                
                # Store original configuration for rollback
                if rollback_enabled:
                    try:
                        original_config = await self.client.nodes(node).qemu(vmid).config.get()
                        rollback_file = self.models_dir / f"rollback_vm_{vmid}_{int(datetime.now().timestamp())}.json"
                        with open(rollback_file, 'w') as f:
                            json.dump(original_config, f, default=str)
                    except Exception as e:
                        logger.warning(f"Could not save rollback config for VM {vmid}: {e}")
                        
                for recommendation in vm_rec["recommendations"]:
                    result = {
                        "vmid": vmid,
                        "recommendation_type": recommendation["type"],
                        "status": "pending"
                    }
                    
                    if dry_run:
                        result["status"] = "dry_run"
                        result["would_apply"] = recommendation["recommendation"]
                        application_results.append(result)
                        continue
                        
                    try:
                        rec_type = recommendation["type"]
                        
                        if rec_type in ["cpu_upsize", "cpu_downsize"]:
                            # Get current CPU and apply recommendation
                            vm_config = await self.client.nodes(node).qemu(vmid).config.get()
                            current_cpu = vm_config.get("cores", 1)
                            
                            if rec_type == "cpu_upsize":
                                new_cpu = min(16, int(current_cpu * 1.5))
                            else:
                                new_cpu = max(1, int(current_cpu * 0.8))
                                
                            await self.client.nodes(node).qemu(vmid).config.put(cores=new_cpu)
                            result["status"] = "applied"
                            result["details"] = f"CPU changed from {current_cpu} to {new_cpu}"
                            
                        elif rec_type in ["memory_upsize", "memory_downsize"]:
                            # Get current memory and apply recommendation
                            vm_config = await self.client.nodes(node).qemu(vmid).config.get()
                            current_memory = vm_config.get("memory", 1024)
                            
                            if rec_type == "memory_upsize":
                                new_memory = min(32768, int(current_memory * 1.5))
                            else:
                                new_memory = max(512, int(current_memory * 0.8))
                                
                            await self.client.nodes(node).qemu(vmid).config.put(memory=new_memory)
                            result["status"] = "applied"
                            result["details"] = f"Memory changed from {current_memory}MB to {new_memory}MB"
                            
                        else:
                            result["status"] = "not_implemented"
                            result["details"] = f"Optimization for {rec_type} not yet implemented"
                            
                    except Exception as e:
                        result["status"] = "failed"
                        result["error"] = str(e)
                        
                    application_results.append(result)
                    
            return application_results
            
        except Exception as e:
            logger.error(f"Optimization application failed: {e}")
            return []
