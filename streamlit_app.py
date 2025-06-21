import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import hashlib
from typing import Dict, List, Tuple, Optional
import uuid
import time
import base64
from io import BytesIO

# For PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Optional: Import for real Claude AI integration
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Enterprise AWS Migration Strategy Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


class EnterpriseCalculator:
    """Enterprise-grade calculator for AWS migration planning"""
    
    def __init__(self):
        self.file_size_multipliers = {
            "< 1MB (Many small files)": 0.25,
            "1-10MB (Small files)": 0.45,
            "10-100MB (Medium files)": 0.70,
            "100MB-1GB (Large files)": 0.90,
            "> 1GB (Very large files)": 0.95
        }
        
        self.instance_performance = {
            "m5.large": {"cpu": 2, "memory": 8, "network": 750, "baseline_throughput": 150, "cost_hour": 0.096},
            "m5.xlarge": {"cpu": 4, "memory": 16, "network": 750, "baseline_throughput": 250, "cost_hour": 0.192},
            "m5.2xlarge": {"cpu": 8, "memory": 32, "network": 1000, "baseline_throughput": 400, "cost_hour": 0.384},
            "m5.4xlarge": {"cpu": 16, "memory": 64, "network": 2000, "baseline_throughput": 600, "cost_hour": 0.768},
            "m5.8xlarge": {"cpu": 32, "memory": 128, "network": 4000, "baseline_throughput": 1000, "cost_hour": 1.536},
            "c5.2xlarge": {"cpu": 8, "memory": 16, "network": 2000, "baseline_throughput": 500, "cost_hour": 0.34},
            "c5.4xlarge": {"cpu": 16, "memory": 32, "network": 4000, "baseline_throughput": 800, "cost_hour": 0.68},
            "c5.9xlarge": {"cpu": 36, "memory": 72, "network": 10000, "baseline_throughput": 1500, "cost_hour": 1.53},
            "r5.2xlarge": {"cpu": 8, "memory": 64, "network": 2000, "baseline_throughput": 450, "cost_hour": 0.504},
            "r5.4xlarge": {"cpu": 16, "memory": 128, "network": 4000, "baseline_throughput": 700, "cost_hour": 1.008}
        }
        
        self.compliance_requirements = {
            "SOX": {"encryption_required": True, "audit_trail": True, "data_retention": 7},
            "GDPR": {"encryption_required": True, "data_residency": True, "right_to_delete": True},
            "HIPAA": {"encryption_required": True, "access_logging": True, "data_residency": True},
            "PCI-DSS": {"encryption_required": True, "network_segmentation": True, "access_control": True},
            "SOC2": {"encryption_required": True, "monitoring": True, "access_control": True},
            "ISO27001": {"risk_assessment": True, "documentation": True, "continuous_monitoring": True},
            "FedRAMP": {"encryption_required": True, "continuous_monitoring": True, "incident_response": True},
            "FISMA": {"encryption_required": True, "access_control": True, "audit_trail": True}
        }
        
        # Geographic latency matrix (ms)
        self.geographic_latency = {
            "San Jose, CA": {"us-west-1": 15, "us-west-2": 25, "us-east-1": 70, "us-east-2": 65},
            "San Antonio, TX": {"us-west-1": 45, "us-west-2": 50, "us-east-1": 35, "us-east-2": 30},
            "New York, NY": {"us-west-1": 75, "us-west-2": 80, "us-east-1": 10, "us-east-2": 15},
            "Chicago, IL": {"us-west-1": 60, "us-west-2": 65, "us-east-1": 25, "us-east-2": 20},
            "Dallas, TX": {"us-west-1": 40, "us-west-2": 45, "us-east-1": 35, "us-east-2": 30},
            "Los Angeles, CA": {"us-west-1": 20, "us-west-2": 15, "us-east-1": 75, "us-east-2": 70},
            "Atlanta, GA": {"us-west-1": 65, "us-west-2": 70, "us-east-1": 15, "us-east-2": 20},
            "London, UK": {"us-west-1": 150, "us-west-2": 155, "us-east-1": 80, "us-east-2": 85},
            "Frankfurt, DE": {"us-west-1": 160, "us-west-2": 165, "us-east-1": 90, "us-east-2": 95},
            "Tokyo, JP": {"us-west-1": 120, "us-west-2": 115, "us-east-1": 180, "us-east-2": 185},
            "Sydney, AU": {"us-west-1": 170, "us-west-2": 165, "us-east-1": 220, "us-east-2": 225}
        }
        
        # Database migration tools
        self.db_migration_tools = {
            "DMS": {
                "name": "Database Migration Service",
                "best_for": ["Homogeneous", "Heterogeneous", "Continuous Replication"],
                "data_size_limit": "Large (TB scale)",
                "downtime": "Minimal",
                "cost_factor": 1.0,
                "complexity": "Medium"
            },
            "DataSync": {
                "name": "AWS DataSync",
                "best_for": ["File Systems", "Object Storage", "Large Files"],
                "data_size_limit": "Very Large (PB scale)",
                "downtime": "None",
                "cost_factor": 0.8,
                "complexity": "Low"
            },
            "DMS+DataSync": {
                "name": "Hybrid DMS + DataSync",
                "best_for": ["Complex Workloads", "Mixed Data Types"],
                "data_size_limit": "Very Large",
                "downtime": "Low",
                "cost_factor": 1.3,
                "complexity": "High"
            },
            "Parallel Copy": {
                "name": "AWS Parallel Copy",
                "best_for": ["Time-Critical", "High Throughput"],
                "data_size_limit": "Large",
                "downtime": "Low",
                "cost_factor": 1.5,
                "complexity": "Medium"
            },
            "Snowball Edge": {
                "name": "AWS Snowball Edge",
                "best_for": ["Limited Bandwidth", "Large Datasets"],
                "data_size_limit": "Very Large (100TB per device)",
                "downtime": "Medium",
                "cost_factor": 0.6,
                "complexity": "Low"
            },
            "Storage Gateway": {
                "name": "AWS Storage Gateway",
                "best_for": ["Hybrid Cloud", "Gradual Migration"],
                "data_size_limit": "Large",
                "downtime": "None",
                "cost_factor": 1.2,
                "complexity": "Medium"
            }
        }
    
    def calculate_enterprise_throughput(self, instance_type, num_agents, file_size_category, 
                                        network_bw_mbps, latency, jitter, packet_loss, qos_enabled, 
                                        dedicated_bandwidth, real_world_mode=True):
        """Calculate optimized throughput considering all network factors including real-world limitations"""
        try:
            base_performance = self.instance_performance[instance_type]["baseline_throughput"]
            file_efficiency = self.file_size_multipliers[file_size_category]
            
            # Network impact calculations
            latency_factor = max(0.4, 1 - (latency - 5) / 500)
            jitter_factor = max(0.8, 1 - jitter / 100)
            packet_loss_factor = max(0.6, 1 - packet_loss / 10)
            qos_factor = 1.2 if qos_enabled else 1.0
            
            network_efficiency = latency_factor * jitter_factor * packet_loss_factor * qos_factor
            
            # Real-world efficiency factors (based on actual field testing)
            if real_world_mode:
                # DataSync specific overhead
                datasync_overhead = 0.75  # DataSync protocol overhead, checksums, metadata
                
                # Storage I/O limitations (major factor often overlooked)
                storage_io_factor = 0.6  # Source storage IOPS limitations, especially for spinning disks
                
                # TCP window scaling and buffer limitations
                tcp_efficiency = 0.8  # Real TCP performance vs theoretical
                
                # AWS API rate limiting (S3 PUT/GET limits)
                s3_api_efficiency = 0.85  # S3 request rate limits and throttling
                
                # File system overhead
                filesystem_overhead = 0.9  # File system metadata, fragmentation
                
                # Instance resource constraints
                if instance_type == "m5.large":
                    cpu_memory_factor = 0.7  # m5.large CPU/memory constraints for large files
                elif instance_type in ["m5.xlarge", "m5.2xlarge"]:
                    cpu_memory_factor = 0.8
                else:
                    cpu_memory_factor = 0.9
                
                # Concurrent workload impact
                concurrent_workload_factor = 0.85  # Other applications sharing resources
                
                # Time-of-day variations (AWS regional load)
                peak_hour_factor = 0.9  # Performance degradation during peak hours
                
                # Error handling and retransmissions
                error_handling_overhead = 0.95  # Retry logic, error correction
                
                # Combined real-world efficiency
                real_world_efficiency = (datasync_overhead * storage_io_factor * tcp_efficiency * 
                                       s3_api_efficiency * filesystem_overhead * cpu_memory_factor * 
                                       concurrent_workload_factor * peak_hour_factor * error_handling_overhead)
            else:
                # Laboratory/theoretical conditions
                real_world_efficiency = 0.95  # Only minor protocol overhead
            
            # Multi-agent scaling with diminishing returns
            total_throughput = 0
            for i in range(num_agents):
                agent_efficiency = max(0.4, 1 - (i * 0.05))
                agent_throughput = (base_performance * file_efficiency * network_efficiency * 
                                  real_world_efficiency * agent_efficiency)
                total_throughput += agent_throughput
            
            # Apply bandwidth limitation
            max_available_bandwidth = network_bw_mbps * (dedicated_bandwidth / 100)
            effective_throughput = min(total_throughput, max_available_bandwidth)
            
            # Return both theoretical and real-world calculations
            theoretical_throughput = min(base_performance * file_efficiency * network_efficiency * num_agents, 
                                       max_available_bandwidth)
            
            return effective_throughput, network_efficiency, theoretical_throughput, real_world_efficiency
        
        except Exception as e:
            # Return safe defaults
            return 100.0, 0.7, 150.0, 0.7
    
    def calculate_enterprise_costs(self, data_size_gb, transfer_days, instance_type, num_agents, 
                                   compliance_frameworks, s3_storage_class):
        """Calculate comprehensive migration costs"""
        try:
            instance_cost_hour = self.instance_performance[instance_type]["cost_hour"]
            datasync_compute_cost = instance_cost_hour * num_agents * 24 * transfer_days
            
            # Data transfer costs (AWS pricing)
            data_transfer_cost = data_size_gb * 0.09
            
            # S3 storage costs per GB
            s3_costs = {
                "Standard": 0.023,
                "Standard-IA": 0.0125,
                "One Zone-IA": 0.01,
                "Glacier Instant Retrieval": 0.004,
                "Glacier Flexible Retrieval": 0.0036,
                "Glacier Deep Archive": 0.00099
            }
            s3_storage_cost = data_size_gb * s3_costs.get(s3_storage_class, 0.023)
            
            # Additional enterprise costs
            compliance_cost = len(compliance_frameworks) * 500  # Compliance tooling per framework
            monitoring_cost = 200 * transfer_days  # Enhanced monitoring per day
            
            total_cost = datasync_compute_cost + data_transfer_cost + s3_storage_cost + compliance_cost + monitoring_cost
            
            return {
                "compute": datasync_compute_cost,
                "transfer": data_transfer_cost,
                "storage": s3_storage_cost,
                "compliance": compliance_cost,
                "monitoring": monitoring_cost,
                "total": total_cost
            }
        except Exception as e:
            # Return safe defaults
            return {
                "compute": 1000.0,
                "transfer": 500.0,
                "storage": 200.0,
                "compliance": 100.0,
                "monitoring": 50.0,
                "total": 1850.0
            }
    
    def assess_compliance_requirements(self, frameworks, data_classification, data_residency):
        """Assess compliance requirements and identify risks"""
        try:
            requirements = set()
            risks = []
            
            for framework in frameworks:
                if framework in self.compliance_requirements:
                    reqs = self.compliance_requirements[framework]
                    requirements.update(reqs.keys())
                    
                    # Check for compliance conflicts
                    if framework == "GDPR" and data_residency == "No restrictions":
                        risks.append("GDPR requires data residency controls")
                    
                    if framework in ["HIPAA", "PCI-DSS"] and data_classification == "Public":
                        risks.append(f"{framework} incompatible with Public data classification")
            
            return list(requirements), risks
        except Exception as e:
            return [], []
    
    def calculate_business_impact(self, transfer_days, data_types):
        """Calculate business impact score based on data types"""
        try:
            impact_weights = {
                "Customer Data": 0.9,
                "Financial Records": 0.95,
                "Employee Data": 0.7,
                "Intellectual Property": 0.85,
                "System Logs": 0.3,
                "Application Data": 0.8,
                "Database Backups": 0.6,
                "Media Files": 0.4,
                "Documents": 0.5
            }
            
            if not data_types:
                return {"score": 0.5, "level": "Medium", "recommendation": "Standard migration approach"}
            
            avg_impact = sum(impact_weights.get(dt, 0.5) for dt in data_types) / len(data_types)
            
            if avg_impact >= 0.8:
                level = "Critical"
                recommendation = "Phased migration with extensive testing"
            elif avg_impact >= 0.6:
                level = "High"
                recommendation = "Careful planning with pilot phase"
            elif avg_impact >= 0.4:
                level = "Medium"
                recommendation = "Standard migration approach"
            else:
                level = "Low"
                recommendation = "Direct migration acceptable"
            
            return {"score": avg_impact, "level": level, "recommendation": recommendation}
        except Exception as e:
            return {"score": 0.5, "level": "Medium", "recommendation": "Standard migration approach"}
    
    def get_optimal_networking_architecture(self, source_location, target_region, data_size_gb, 
                                          dx_bandwidth_mbps, database_types, data_types, config=None):
        """AI-powered networking architecture recommendations with real-time metrics"""
        try:
            # Get latency for the route
            estimated_latency = self.geographic_latency.get(source_location, {}).get(target_region, 50)
            
            # Analyze data characteristics
            has_databases = len(database_types) > 0
            has_large_files = any("Large" in str(dt) or "Media" in str(dt) for dt in data_types)
            data_size_tb = data_size_gb / 1024
            
            recommendations = {
                "primary_method": "",
                "secondary_method": "",
                "networking_option": "",
                "db_migration_tool": "",
                "rationale": "",
                "estimated_performance": {},
                "cost_efficiency": "",
                "risk_level": "",
                "ai_analysis": ""
            }
            
            # Try to get real AI analysis if enabled
            if config and config.get('enable_real_ai') and config.get('claude_api_key'):
                real_ai_analysis = self.get_real_ai_analysis(config, config['claude_api_key'], config.get('ai_model'))
                if real_ai_analysis:
                    recommendations["ai_analysis"] = real_ai_analysis
            
            # Network architecture decision logic (fallback built-in AI)
            if dx_bandwidth_mbps >= 1000 and estimated_latency < 50:
                recommendations["networking_option"] = "Direct Connect (Primary)"
                network_score = 9
            elif dx_bandwidth_mbps >= 500:
                recommendations["networking_option"] = "Direct Connect with Internet Backup"
                network_score = 7
            else:
                recommendations["networking_option"] = "Internet with VPN"
                network_score = 5
            
            # Database migration tool selection
            if has_databases and data_size_tb > 10:
                if len(database_types) > 2:
                    recommendations["db_migration_tool"] = "DMS+DataSync"
                else:
                    recommendations["db_migration_tool"] = "DMS"
            elif has_large_files and data_size_tb > 50:
                if dx_bandwidth_mbps < 1000:
                    recommendations["db_migration_tool"] = "Snowball Edge"
                else:
                    recommendations["db_migration_tool"] = "DataSync"
            elif data_size_tb > 100:
                recommendations["db_migration_tool"] = "Parallel Copy"
            else:
                recommendations["db_migration_tool"] = "DataSync"
            
            # Primary method selection
            if data_size_tb > 50 and dx_bandwidth_mbps < 1000:
                recommendations["primary_method"] = "Snowball Edge"
                recommendations["secondary_method"] = "DataSync (for ongoing sync)"
            elif has_databases:
                recommendations["primary_method"] = recommendations["db_migration_tool"]
                recommendations["secondary_method"] = "Storage Gateway (for hybrid)"
            else:
                recommendations["primary_method"] = "DataSync"
                recommendations["secondary_method"] = "S3 Transfer Acceleration"
            
            # Generate built-in AI rationale
            recommendations["rationale"] = self._generate_ai_rationale(
                source_location, target_region, data_size_tb, dx_bandwidth_mbps, 
                has_databases, has_large_files, estimated_latency, network_score
            )
            
            # Calculate performance using the same method as main metrics
            if config:
                try:
                    actual_throughput_result = self.calculate_enterprise_throughput(
                        config.get('datasync_instance_type', 'm5.large'), 
                        config.get('num_datasync_agents', 1), 
                        config.get('avg_file_size', '10-100MB (Medium files)'), 
                        dx_bandwidth_mbps, 
                        config.get('network_latency', 25), 
                        config.get('network_jitter', 5), 
                        config.get('packet_loss', 0.1), 
                        config.get('qos_enabled', True), 
                        config.get('dedicated_bandwidth', 60), 
                        config.get('real_world_mode', True)
                    )
                    
                    if len(actual_throughput_result) >= 4:
                        actual_throughput, network_efficiency, theoretical_throughput, real_world_efficiency = actual_throughput_result
                    else:
                        actual_throughput, network_efficiency = actual_throughput_result[:2]
                        theoretical_throughput = actual_throughput * 1.5
                    
                    # Apply network optimizations
                    tcp_efficiency = {"Default": 1.0, "64KB": 1.05, "128KB": 1.1, "256KB": 1.15, "512KB": 1.2, "1MB": 1.25, "2MB": 1.3}
                    mtu_efficiency = {"1500 (Standard)": 1.0, "9000 (Jumbo Frames)": 1.15, "Custom": 1.1}
                    congestion_efficiency = {"Cubic (Default)": 1.0, "BBR": 1.2, "Reno": 0.95, "Vegas": 1.05}
                    
                    tcp_factor = tcp_efficiency.get(config.get('tcp_window_size', 'Default'), 1.0)
                    mtu_factor = mtu_efficiency.get(config.get('mtu_size', '1500 (Standard)'), 1.0)
                    congestion_factor = congestion_efficiency.get(config.get('network_congestion_control', 'Cubic (Default)'), 1.0)
                    wan_factor = 1.3 if config.get('wan_optimization', False) else 1.0
                    
                    optimized_ai_throughput = actual_throughput * tcp_factor * mtu_factor * congestion_factor * wan_factor
                    optimized_ai_throughput = min(optimized_ai_throughput, dx_bandwidth_mbps * (config.get('dedicated_bandwidth', 60) / 100))
                    optimized_ai_throughput = max(1, optimized_ai_throughput)
                    
                    # Calculate timing with real configuration
                    effective_data_gb = data_size_gb * 0.85
                    available_hours_per_day = 16 if config.get('business_hours_restriction', True) else 24
                    estimated_days = (effective_data_gb * 8) / (optimized_ai_throughput * available_hours_per_day * 3600) / 1000
                    estimated_days = max(0.1, estimated_days)
                    
                    recommendations["estimated_performance"] = {
                        "throughput_mbps": optimized_ai_throughput,
                        "estimated_days": estimated_days,
                        "network_efficiency": network_efficiency,
                        "agents_used": config.get('num_datasync_agents', 1),
                        "instance_type": config.get('datasync_instance_type', 'm5.large'),
                        "optimization_factors": {
                            "tcp_factor": tcp_factor,
                            "mtu_factor": mtu_factor,
                            "congestion_factor": congestion_factor,
                            "wan_factor": wan_factor
                        }
                    }
                except Exception as e:
                    # Fallback calculation
                    recommendations["estimated_performance"] = {
                        "throughput_mbps": min(dx_bandwidth_mbps * 0.6, 1000),
                        "estimated_days": (data_size_gb * 8) / (min(dx_bandwidth_mbps * 0.6, 1000) * 16 * 3600) / 1000,
                        "network_efficiency": 0.7,
                        "agents_used": 1,
                        "instance_type": "m5.large",
                        "optimization_factors": {
                            "tcp_factor": 1.0,
                            "mtu_factor": 1.0,
                            "congestion_factor": 1.0,
                            "wan_factor": 1.0
                        }
                    }
            else:
                # Fallback calculation
                if recommendations["networking_option"] == "Direct Connect (Primary)":
                    base_throughput = min(dx_bandwidth_mbps * 0.8, 2000)
                elif "Direct Connect" in recommendations["networking_option"]:
                    base_throughput = min(dx_bandwidth_mbps * 0.6, 1500)
                else:
                    base_throughput = min(500, dx_bandwidth_mbps * 0.4)
                
                recommendations["estimated_performance"] = {
                    "throughput_mbps": base_throughput,
                    "estimated_days": (data_size_gb * 8) / (base_throughput * 86400) / 1000,
                    "network_efficiency": network_score / 10,
                    "agents_used": 1,
                    "instance_type": "m5.large",
                    "optimization_factors": {
                        "tcp_factor": 1.0,
                        "mtu_factor": 1.0,
                        "congestion_factor": 1.0,
                        "wan_factor": 1.0
                    }
                }
            
            # Cost and risk assessment
            if data_size_tb > 100 and dx_bandwidth_mbps < 1000:
                recommendations["cost_efficiency"] = "High (Physical transfer)"
                recommendations["risk_level"] = "Medium"
            elif dx_bandwidth_mbps >= 1000:
                recommendations["cost_efficiency"] = "Medium (Network transfer)"
                recommendations["risk_level"] = "Low"
            else:
                recommendations["cost_efficiency"] = "Medium"
                recommendations["risk_level"] = "Medium"
            
            return recommendations
        
        except Exception as e:
            # Return safe defaults
            return {
                "primary_method": "DataSync",
                "secondary_method": "S3 Transfer Acceleration",
                "networking_option": "Direct Connect",
                "db_migration_tool": "DMS",
                "rationale": "Default configuration recommendation",
                "estimated_performance": {
                    "throughput_mbps": 100,
                    "estimated_days": 10,
                    "network_efficiency": 0.7,
                    "agents_used": 1,
                    "instance_type": "m5.large",
                    "optimization_factors": {
                        "tcp_factor": 1.0,
                        "mtu_factor": 1.0,
                        "congestion_factor": 1.0,
                        "wan_factor": 1.0
                    }
                },
                "cost_efficiency": "Medium",
                "risk_level": "Low",
                "ai_analysis": ""
            }
    
    def _generate_ai_rationale(self, source, target, data_size_tb, bandwidth, has_db, has_large_files, latency, network_score):
        """Generate intelligent rationale for recommendations"""
        try:
            rationale_parts = []
            
            # Geographic analysis
            if latency < 30:
                rationale_parts.append(f"Excellent geographic proximity between {source} and {target} (≈{latency}ms latency)")
            elif latency < 80:
                rationale_parts.append(f"Good connectivity between {source} and {target} (≈{latency}ms latency)")
            else:
                rationale_parts.append(f"Significant distance between {source} and {target} (≈{latency}ms latency) - consider regional optimization")
            
            # Bandwidth analysis
            if bandwidth >= 10000:
                rationale_parts.append("High-bandwidth Direct Connect enables optimal network transfer performance")
            elif bandwidth >= 1000:
                rationale_parts.append("Adequate Direct Connect bandwidth supports efficient network-based migration")
            else:
                rationale_parts.append("Limited bandwidth suggests physical transfer methods for large datasets")
            
            # Data characteristics
            if data_size_tb > 100:
                rationale_parts.append(f"Large dataset ({data_size_tb:.1f}TB) requires high-throughput migration strategy")
            
            if has_db:
                rationale_parts.append("Database workloads require specialized migration tools with minimal downtime capabilities")
            
            if has_large_files:
                rationale_parts.append("Large file presence optimizes for high-throughput, parallel transfer methods")
            
            # Performance prediction
            if network_score >= 8:
                rationale_parts.append("Network conditions are optimal for direct cloud migration")
            elif network_score >= 6:
                rationale_parts.append("Network conditions support cloud migration with some optimization needed")
            else:
                rationale_parts.append("Network limitations suggest hybrid or physical transfer approaches")
            
            return ". ".join(rationale_parts) + "."
        
        except Exception as e:
            return "Standard migration approach recommended based on current configuration."
    
    def get_real_ai_analysis(self, config, api_key, model="claude-sonnet-4-20250514"):
        """Get real Claude AI analysis using Anthropic API"""
        if not ANTHROPIC_AVAILABLE or not api_key:
            return None
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            
            # Prepare context for Claude
            context = f"""
            You are an expert AWS migration architect. Analyze this migration scenario and provide recommendations:
            
            Project: {config.get('project_name', 'N/A')}
            Data Size: {config.get('data_size_gb', 0)} GB
            Source: {config.get('source_location', 'N/A')}
            Target: {config.get('target_aws_region', 'N/A')}
            Network: {config.get('dx_bandwidth_mbps', 0)} Mbps Direct Connect
            Databases: {', '.join(config.get('database_types', []))}
            Data Types: {', '.join(config.get('data_types', []))}
            Compliance: {', '.join(config.get('compliance_frameworks', []))}
            Data Classification: {config.get('data_classification', 'N/A')}
            
            Provide specific recommendations for:
            1. Best migration method and tools
            2. Network architecture approach
            3. Performance optimization strategies
            4. Risk mitigation approaches
            5. Cost optimization suggestions
            
            Be concise but specific. Focus on AWS best practices.
            """
            
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                temperature=0.3,
                messages=[{"role": "user", "content": context}]
            )
            
            return response.content[0].text if response.content else None
            
        except Exception as e:
            st.error(f"Claude AI API Error: {str(e)}")
            return None
    
    def get_intelligent_datasync_recommendations(self, config, metrics):
        """Get intelligent, dynamic DataSync optimization recommendations based on workload analysis"""
        try:
            current_instance = config['datasync_instance_type']
            current_agents = config['num_datasync_agents']
            data_size_gb = config['data_size_gb']
            data_size_tb = data_size_gb / 1024
            
            # Analyze current instance characteristics
            current_specs = self.instance_performance[current_instance]
            current_cpu = current_specs['cpu']
            current_memory = current_specs['memory']
            current_network = current_specs['network']
            current_cost_hour = current_specs['cost_hour']
            
            # Workload analysis factors
            has_databases = len(config.get('database_types', [])) > 0
            has_large_files = config.get('avg_file_size', '') in ['100MB-1GB (Large files)', '> 1GB (Very large files)']
            has_many_small_files = config.get('avg_file_size', '') in ['< 1MB (Many small files)', '1-10MB (Small files)']
            high_bandwidth = config.get('dx_bandwidth_mbps', 0) >= 10000
            network_latency = config.get('network_latency', 25)
            
            # Calculate workload complexity score
            complexity_score = 0
            if has_databases: 
                complexity_score += 3
            if has_large_files: 
                complexity_score += 2
            if has_many_small_files: 
                complexity_score += 1
            if data_size_tb > 10: 
                complexity_score += 2
            if data_size_tb > 100: 
                complexity_score += 2
            if high_bandwidth: 
                complexity_score += 1
            if network_latency < 30: 
                complexity_score += 1
            
            # Determine optimal instance type based on workload
            def get_optimal_instance_type():
                # For database-heavy workloads, prioritize memory
                if has_databases and data_size_tb > 5:
                    if complexity_score >= 8:
                        return "r5.4xlarge"  # High memory for large database workloads
                    elif complexity_score >= 6:
                        return "r5.2xlarge"  # Medium memory
                    else:
                        return "m5.2xlarge"  # Balanced
                
                # For large file workloads, prioritize CPU and network
                elif has_large_files:
                    if data_size_tb > 50 and high_bandwidth:
                        return "c5.9xlarge"  # Maximum CPU for large files + high bandwidth
                    elif data_size_tb > 20:
                        return "c5.4xlarge"  # High CPU
                    elif data_size_tb > 5:
                        return "c5.2xlarge"  # Medium CPU
                    else:
                        return "m5.xlarge"   # Balanced for smaller datasets
                
                # For many small files, prioritize CPU for metadata processing
                elif has_many_small_files:
                    if data_size_tb > 20:
                        return "c5.4xlarge"  # High CPU for metadata overhead
                    elif data_size_tb > 5:
                        return "c5.2xlarge"  # Medium CPU
                    else:
                        return "m5.xlarge"   # Balanced
                
                # General workloads - balanced approach
                else:
                    if data_size_tb > 50:
                        return "m5.8xlarge"  # Large balanced
                    elif data_size_tb > 20:
                        return "m5.4xlarge"  # Medium-large balanced
                    elif data_size_tb > 5:
                        return "m5.2xlarge"  # Medium balanced
                    else:
                        return "m5.xlarge"   # Small-medium balanced
            
            # Determine optimal number of agents with diminishing returns
            def get_optimal_agent_count():
                base_agents = max(1, min(10, int(data_size_tb / 10)))  # Base: 1 agent per 10TB
                
                # Adjust based on file characteristics
                if has_databases:
                    # Databases benefit from fewer, more powerful agents to avoid lock contention
                    agent_multiplier = 0.8
                elif has_large_files:
                    # Large files can use more agents efficiently
                    agent_multiplier = 1.3
                elif has_many_small_files:
                    # Small files need more agents for parallelism
                    agent_multiplier = 1.5
                else:
                    agent_multiplier = 1.0
                
                # Network bandwidth consideration
                if high_bandwidth:
                    agent_multiplier *= 1.2
                
                # Calculate optimal with practical limits
                optimal_agents = max(1, min(20, int(base_agents * agent_multiplier)))
                
                # Ensure we don't recommend the same number if current is already reasonable
                if abs(optimal_agents - current_agents) <= 1 and current_agents <= 10:
                    optimal_agents = current_agents
                
                return optimal_agents
            
            # Get recommendations
            recommended_instance = get_optimal_instance_type()
            recommended_agents = get_optimal_agent_count()
            
            # Calculate performance and cost impacts
            rec_specs = self.instance_performance[recommended_instance]
            
            # Performance calculation
            current_total_cpu = current_cpu * current_agents
            recommended_total_cpu = rec_specs['cpu'] * recommended_agents
            cpu_performance_change = (recommended_total_cpu - current_total_cpu) / current_total_cpu
            
            current_total_memory = current_memory * current_agents
            recommended_total_memory = rec_specs['memory'] * recommended_agents
            memory_performance_change = (recommended_total_memory - current_total_memory) / current_total_memory
            
            # Network throughput consideration
            current_baseline_throughput = current_specs['baseline_throughput'] * current_agents
            recommended_baseline_throughput = rec_specs['baseline_throughput'] * recommended_agents
            throughput_change = (recommended_baseline_throughput - current_baseline_throughput) / current_baseline_throughput
            
            # Overall performance change (weighted average)
            if has_databases:
                # Databases are more memory-sensitive
                performance_change_percent = (cpu_performance_change * 0.3 + memory_performance_change * 0.5 + throughput_change * 0.2) * 100
            elif has_large_files:
                # Large files are more CPU and throughput sensitive
                performance_change_percent = (cpu_performance_change * 0.5 + memory_performance_change * 0.2 + throughput_change * 0.3) * 100
            else:
                # Balanced workload
                performance_change_percent = (cpu_performance_change * 0.4 + memory_performance_change * 0.3 + throughput_change * 0.3) * 100
            
            # Cost calculation
            current_hourly_cost = current_cost_hour * current_agents
            recommended_hourly_cost = rec_specs['cost_hour'] * recommended_agents
            cost_impact_percent = ((recommended_hourly_cost - current_hourly_cost) / current_hourly_cost) * 100
            
            # Scaling effectiveness analysis
            def analyze_scaling_effectiveness():
                if current_agents <= 2:
                    return {"scaling_rating": "Under-scaled", "efficiency": 0.6}
                elif current_agents <= 5:
                    return {"scaling_rating": "Well-scaled", "efficiency": 0.85}
                elif current_agents <= 10:
                    return {"scaling_rating": "Optimally-scaled", "efficiency": 0.95}
                else:
                    return {"scaling_rating": "Over-scaled", "efficiency": 0.75}
            
            scaling_analysis = analyze_scaling_effectiveness()
            
            # Current efficiency assessment
            current_efficiency = min(100, max(20, 85 - abs(current_agents - recommended_agents) * 5 + scaling_analysis["efficiency"] * 15))
            
            # Performance rating
            if current_efficiency >= 85:
                performance_rating = "Optimal"
            elif current_efficiency >= 70:
                performance_rating = "Good"
            elif current_efficiency >= 55:
                performance_rating = "Fair"
            else:
                performance_rating = "Poor"
            
            # Generate intelligent recommendations
            instance_upgrade_needed = recommended_instance != current_instance
            agent_change_needed = recommended_agents - current_agents
            
            # Instance recommendation reasoning
            if instance_upgrade_needed:
                if has_databases and "r5" in recommended_instance:
                    instance_reason = f"Database workloads benefit from memory-optimized {recommended_instance} instances"
                elif has_large_files and "c5" in recommended_instance:
                    instance_reason = f"Large file transfers are CPU-intensive, {recommended_instance} provides optimal processing power"
                elif has_many_small_files and "c5" in recommended_instance:
                    instance_reason = f"Small file overhead requires CPU optimization, {recommended_instance} handles metadata efficiently"
                else:
                    instance_reason = f"Workload characteristics suggest {recommended_instance} for balanced performance"
            else:
                instance_reason = f"Current {current_instance} instance type is optimal for your workload"
            
            # Agent recommendation reasoning
            if agent_change_needed > 0:
                agent_reasoning = f"Scale up to {recommended_agents} agents for improved parallelism and {abs(performance_change_percent):.1f}% performance gain"
            elif agent_change_needed < 0:
                agent_reasoning = f"Scale down to {recommended_agents} agents to reduce costs by {abs(cost_impact_percent):.1f}% while maintaining performance"
            else:
                agent_reasoning = f"Current {current_agents} agents is optimal for your {data_size_tb:.1f}TB dataset"
            
            # Bottleneck analysis
            bottlenecks = []
            bottleneck_recommendations = []
            
            if current_instance == "m5.large" and data_size_tb > 5:
                bottlenecks.append("Instance CPU/Memory constraints for large dataset")
                bottleneck_recommendations.append("Upgrade to m5.2xlarge or larger for better resource allocation")
            
            if current_agents < 3 and data_size_tb > 10:
                bottlenecks.append("Insufficient parallelism for large dataset")
                bottleneck_recommendations.append("Increase agent count to improve concurrent transfer capability")
            
            if has_databases and current_agents > 8:
                bottlenecks.append("Too many agents may cause database lock contention")
                bottleneck_recommendations.append("Reduce agents to 4-6 for database workloads to avoid locking issues")
            
            if current_specs['network'] < 2000 and config.get('dx_bandwidth_mbps', 0) > 5000:
                bottlenecks.append("Instance network performance limiting high-bandwidth utilization")
                bottleneck_recommendations.append("Use network-optimized instances for high-bandwidth scenarios")
            
            # Cost-performance analysis
            current_cost_efficiency = current_hourly_cost / max(1, metrics.get('optimized_throughput', 100))
            
            # Rank current configuration against alternatives
            efficiency_ranking = 1
            for instance_type in self.instance_performance.keys():
                for agent_count in range(1, 11):
                    test_cost = self.instance_performance[instance_type]['cost_hour'] * agent_count
                    test_throughput = self.instance_performance[instance_type]['baseline_throughput'] * agent_count * 0.8  # Realistic throughput
                    test_efficiency = test_cost / max(1, test_throughput)
                    if test_efficiency < current_cost_efficiency:
                        efficiency_ranking += 1
            
            # Alternative configurations
            alternatives = []
            if instance_upgrade_needed or abs(agent_change_needed) > 0:
                alternatives.append({
                    "name": "AI Recommended",
                    "instance": recommended_instance,
                    "agents": recommended_agents,
                    "description": f"Optimized for your {complexity_score}-point complexity workload"
                })
            
            if has_databases:
                alternatives.append({
                    "name": "Database Optimized",
                    "instance": "r5.2xlarge",
                    "agents": min(6, max(2, recommended_agents)),
                    "description": "Memory-optimized for database migrations"
                })
            
            if data_size_tb > 50:
                alternatives.append({
                    "name": "High Volume",
                    "instance": "c5.9xlarge",
                    "agents": min(8, max(4, recommended_agents)),
                    "description": "Maximum throughput for large datasets"
                })
            
            return {
                "current_analysis": {
                    "current_efficiency": current_efficiency,
                    "performance_rating": performance_rating,
                    "scaling_effectiveness": scaling_analysis,
                    "workload_complexity": complexity_score
                },
                "recommended_instance": {
                    "recommended_instance": recommended_instance,
                    "upgrade_needed": instance_upgrade_needed,
                    "reason": instance_reason,
                    "expected_performance_gain": abs(performance_change_percent),
                    "cost_impact_percent": cost_impact_percent
                },
                "recommended_agents": {
                    "recommended_agents": recommended_agents,
                    "change_needed": agent_change_needed,
                    "reasoning": agent_reasoning,
                    "performance_change_percent": performance_change_percent,
                    "cost_change_percent": cost_impact_percent
                },
                "bottleneck_analysis": (bottlenecks, bottleneck_recommendations),
                "cost_performance_analysis": {
                    "current_cost_efficiency": current_cost_efficiency,
                    "efficiency_ranking": efficiency_ranking
                },
                "alternative_configurations": alternatives
            }
        
        except Exception as e:
            # Return safe defaults if analysis fails
            return {
                "current_analysis": {
                    "current_efficiency": 75,
                    "performance_rating": "Good",
                    "scaling_effectiveness": {"scaling_rating": "Well-scaled", "efficiency": 0.8},
                    "workload_complexity": 5
                },
                "recommended_instance": {
                    "recommended_instance": config.get('datasync_instance_type', 'm5.large'),
                    "upgrade_needed": False,
                    "reason": "Current configuration is adequate",
                    "expected_performance_gain": 0,
                    "cost_impact_percent": 0
                },
                "recommended_agents": {
                    "recommended_agents": config.get('num_datasync_agents', 1),
                    "change_needed": 0,
                    "reasoning": "Current agent count is suitable",
                    "performance_change_percent": 0,
                    "cost_change_percent": 0
                },
                "bottleneck_analysis": ([], []),
                "cost_performance_analysis": {
                    "current_cost_efficiency": 0.1,
                    "efficiency_ranking": 10
                },
                "alternative_configurations": []
            }


class PDFReportGenerator:
    """Generate comprehensive PDF reports for migration analysis"""
    
    def __init__(self):
        if not PDF_AVAILABLE:
            raise ImportError("PDF generation requires reportlab library")
        
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center alignment
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue,
            leftIndent=0
        )
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.darkgreen,
            leftIndent=20
        )
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leftIndent=20,
            rightIndent=20
        )
        self.highlight_style = ParagraphStyle(
            'Highlight',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            backColor=colors.lightblue,
            borderColor=colors.blue,
            borderWidth=1,
            borderPadding=5,
            leftIndent=20,
            rightIndent=20
        )
    
    def generate_conclusion_report(self, config, metrics, recommendations):
        """Generate comprehensive conclusion report"""
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
            
            # Calculate recommendation scores
            performance_score = min(100, (metrics['optimized_throughput'] / 1000) * 50)
            cost_score = min(50, max(0, 50 - (metrics['cost_breakdown']['total'] / config['budget_allocated'] - 1) * 100))
            timeline_score = min(30, max(0, 30 - (metrics['transfer_days'] / config['max_transfer_days'] - 1) * 100))
            risk_score = {"Low": 20, "Medium": 15, "High": 10, "Critical": 5}.get(recommendations['risk_level'], 15)
            overall_score = performance_score + cost_score + timeline_score + risk_score
            
            # Determine strategy status
            if overall_score >= 140:
                strategy_status = "RECOMMENDED"
                strategy_action = "PROCEED"
            elif overall_score >= 120:
                strategy_status = "CONDITIONAL"
                strategy_action = "PROCEED WITH OPTIMIZATIONS"
            elif overall_score >= 100:
                strategy_status = "REQUIRES MODIFICATION"
                strategy_action = "REVISE CONFIGURATION"
            else:
                strategy_status = "NOT RECOMMENDED"
                strategy_action = "RECONSIDER APPROACH"
            
            story = []
            
            # Title Page
            story.append(Paragraph("Enterprise AWS Migration Strategy", self.title_style))
            story.append(Paragraph("Comprehensive Analysis & Strategic Recommendation", self.styles['Heading2']))
            story.append(Spacer(1, 30))
            
            # Executive Summary Box
            exec_summary = f"""
            <b>Project:</b> {config['project_name']}<br/>
            <b>Data Volume:</b> {metrics['data_size_tb']:.1f} TB ({config['data_size_gb']:,} GB)<br/>
            <b>Strategic Recommendation:</b> {strategy_status}<br/>
            <b>Action Required:</b> {strategy_action}<br/>
            <b>Overall Score:</b> {overall_score:.0f}/150<br/>
            <b>Success Probability:</b> {85 + (overall_score - 100) * 0.3:.0f}%
            """
            story.append(Paragraph(exec_summary, self.highlight_style))
            story.append(Spacer(1, 20))
            
            # Key Metrics Table
            story.append(Paragraph("Key Performance Metrics", self.heading_style))
            key_metrics_data = [
                ['Metric', 'Value', 'Status'],
                ['Expected Throughput', f"{recommendations['estimated_performance']['throughput_mbps']:.0f} Mbps", 'Optimal'],
                ['Estimated Timeline', f"{metrics['transfer_days']:.1f} days", 'On Track' if metrics['transfer_days'] <= config['max_transfer_days'] else 'At Risk'],
                ['Total Investment', f"${metrics['cost_breakdown']['total']:,.0f}", 'Within Budget' if metrics['cost_breakdown']['total'] <= config['budget_allocated'] else 'Over Budget'],
                ['Risk Assessment', recommendations['risk_level'], 'Acceptable'],
                ['Network Efficiency', f"{recommendations['estimated_performance']['network_efficiency']:.1%}", 'Good']
            ]
            
            key_metrics_table = Table(key_metrics_data, colWidths=[2*inch, 2*inch, 1.5*inch])
            key_metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(key_metrics_table)
            story.append(Spacer(1, 20))
            
            # Final Recommendation
            final_recommendation = f"""
            <b>FINAL STRATEGIC DIRECTION:</b> {strategy_status}<br/>
            <b>AI CONFIDENCE:</b> {85 + (overall_score - 100) * 0.3:.0f}% based on comprehensive analysis<br/>
            <b>RECOMMENDED ACTION:</b> {strategy_action}<br/>
            <br/>
            <b>Next Steps:</b><br/>
            1. Executive Approval: Present this analysis to stakeholders<br/>
            2. Resource Allocation: Secure budget and technical resources<br/>
            3. Team Formation: Assemble migration team with defined roles<br/>
            4. Infrastructure Setup: Begin Phase 1 preparation activities<br/>
            5. Communication Plan: Notify affected users and departments
            """
            story.append(Paragraph("Final Strategic Recommendation", self.heading_style))
            story.append(Paragraph(final_recommendation, self.highlight_style))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer
        
        except Exception as e:
            st.error(f"Error generating conclusion report: {str(e)}")
            return None
    
    def generate_cost_analysis_report(self, config, metrics):
        """Generate detailed cost analysis report"""
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
            
            story = []
            
            # Title
            story.append(Paragraph("AWS Migration Cost Analysis", self.title_style))
            story.append(Paragraph(f"Project: {config['project_name']}", self.styles['Heading2']))
            story.append(Spacer(1, 30))
            
            # Cost Summary
            story.append(Paragraph("Executive Cost Summary", self.heading_style))
            cost_summary = f"""
            <b>Total Migration Cost:</b> ${metrics['cost_breakdown']['total']:,.2f}<br/>
            <b>Cost per TB:</b> ${metrics['cost_breakdown']['total']/metrics['data_size_tb']:.2f}<br/>
            <b>Budget Allocation:</b> ${config['budget_allocated']:,.0f}<br/>
            <b>Budget Status:</b> {'Within Budget' if metrics['cost_breakdown']['total'] <= config['budget_allocated'] else 'Over Budget'}<br/>
            <b>Variance:</b> ${metrics['cost_breakdown']['total'] - config['budget_allocated']:+,.0f}
            """
            story.append(Paragraph(cost_summary, self.highlight_style))
            story.append(Spacer(1, 20))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer
        
        except Exception as e:
            st.error(f"Error generating cost analysis report: {str(e)}")
            return None


class MigrationPlatform:
    """Main application class for the Enterprise AWS Migration Platform"""
    
    def __init__(self):
        self.calculator = EnterpriseCalculator()
        self.pdf_generator = PDFReportGenerator() if PDF_AVAILABLE else None
        self.initialize_session_state()
        self.setup_custom_css()
        
        # Add real-time tracking
        self.last_update_time = datetime.now()
        self.auto_refresh_interval = 30  # seconds
    
    def initialize_session_state(self):
        """Initialize session state variables with real-time tracking"""
        if 'migration_projects' not in st.session_state:
            st.session_state.migration_projects = {}
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {
                'role': 'Network Architect',
                'organization': 'Enterprise Corp',
                'security_clearance': 'Standard'
            }
        if 'audit_log' not in st.session_state:
            st.session_state.audit_log = []
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "dashboard"
        if 'last_config_hash' not in st.session_state:
            st.session_state.last_config_hash = None
        if 'config_change_count' not in st.session_state:
            st.session_state.config_change_count = 0
    
    def setup_custom_css(self):
        """Setup enhanced custom CSS styling with professional design"""
        st.markdown("""
        <style>
            /* Main container styling */
            .main-header {
                background: linear-gradient(135deg, #FF9900 0%, #232F3E 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            
            /* Enhanced tab container */
            .tab-container {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 16px rgba(0,0,0,0.1);
                border: 1px solid #dee2e6;
            }
            
            /* Standardized section headers */
            .section-header {
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 8px;
                margin: 1.5rem 0 1rem 0;
                font-size: 1.2rem;
                font-weight: bold;
                box-shadow: 0 2px 8px rgba(0,123,255,0.3);
            }
            
            /* Enhanced metric cards */
            .metric-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 5px solid #FF9900;
                margin: 0.75rem 0;
                transition: all 0.3s ease;
                box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                border: 1px solid #e9ecef;
            }
            
            .metric-card:hover {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            }
            
            /* Professional recommendation boxes */
            .recommendation-box {
                background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 5px solid #007bff;
                margin: 1rem 0;
                box-shadow: 0 3px 15px rgba(0,123,255,0.1);
                border: 1px solid #b8daff;
            }
            
            /* Enhanced AI insight boxes */
            .ai-insight {
                background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
                padding: 1.25rem;
                border-radius: 10px;
                border-left: 4px solid #007bff;
                margin: 1rem 0;
                font-style: italic;
                box-shadow: 0 2px 10px rgba(0,123,255,0.1);
                border: 1px solid #cce7ff;
            }
            
            /* Executive summary styling */
            .executive-summary {
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                margin: 1.5rem 0;
                box-shadow: 0 6px 24px rgba(40,167,69,0.2);
                text-align: center;
            }
            
            /* Real-time indicators */
            .real-time-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                border-radius: 50%;
                animation: pulse 2s infinite;
                margin-right: 8px;
                box-shadow: 0 0 8px rgba(40,167,69,0.5);
            }
            
            @keyframes pulse {
                0% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.7; transform: scale(1.1); }
                100% { opacity: 1; transform: scale(1); }
            }
        </style>
        """, unsafe_allow_html=True)
    
    def safe_dataframe_display(self, df, use_container_width=True, hide_index=True, **kwargs):
        """Safely display a DataFrame by ensuring all values are strings to prevent type mixing"""
        try:
            # Convert all values to strings to prevent type mixing issues
            df_safe = df.astype(str)
            st.dataframe(df_safe, use_container_width=use_container_width, hide_index=hide_index, **kwargs)
        except Exception as e:
            st.error(f"Error displaying table: {str(e)}")
            st.write("Raw data:")
            st.write(df)
    
    def render_header(self):
        """Render the enhanced main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏢 Enterprise AWS Migration Strategy Platform</h1>
            <p style="font-size: 1.1rem; margin-top: 0.5rem;">AI-Powered Migration Planning • Security-First • Compliance-Ready • Enterprise-Scale</p>
            <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">Comprehensive Analysis • Real-time Optimization • Professional Reporting</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_navigation(self):
        """Render enhanced navigation bar with consistent styling"""
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2, 2, 2])
        
        with col1:
            if st.button("🏠 Dashboard", key="nav_dashboard"):
                st.session_state.active_tab = "dashboard"
        with col2:
            if st.button("🌐 Network Analysis", key="nav_network"):
                st.session_state.active_tab = "network"
        with col3:
            if st.button("📊 Migration Planner", key="nav_planner"):
                st.session_state.active_tab = "planner"
        with col4:
            if st.button("⚡ Performance", key="nav_performance"):
                st.session_state.active_tab = "performance"
        with col5:
            if st.button("🔒 Security", key="nav_security"):
                st.session_state.active_tab = "security"
        with col6:
            if st.button("📈 Analytics", key="nav_analytics"):
                st.session_state.active_tab = "analytics"
        with col7:
            if st.button("🎯 Conclusion", key="nav_conclusion"):
                st.session_state.active_tab = "conclusion"
        
        st.markdown('</div>', unsafe_allow_html=True)

    def render_sidebar_controls(self):
        """Render sidebar configuration controls"""
        st.sidebar.header("🏢 Enterprise Controls")
        
        # Project management section
        st.sidebar.subheader("📁 Project Management")
        project_name = st.sidebar.text_input("Project Name", value="Migration-2025-Q1")
        business_unit = st.sidebar.selectbox("Business Unit", 
            ["Corporate IT", "Finance", "HR", "Operations", "R&D", "Sales & Marketing"])
        project_priority = st.sidebar.selectbox("Project Priority", 
            ["Critical", "High", "Medium", "Low"])
        migration_wave = st.sidebar.selectbox("Migration Wave", 
            ["Wave 1 (Pilot)", "Wave 2 (Core Systems)", "Wave 3 (Secondary)", "Wave 4 (Archive)"])
        
        # Security and compliance section
        st.sidebar.subheader("🔒 Security & Compliance")
        data_classification = st.sidebar.selectbox("Data Classification", 
            ["Public", "Internal", "Confidential", "Restricted", "Top Secret"])
        compliance_frameworks = st.sidebar.multiselect("Compliance Requirements", 
            ["SOX", "GDPR", "HIPAA", "PCI-DSS", "SOC2", "ISO27001", "FedRAMP", "FISMA"])
        encryption_in_transit = st.sidebar.checkbox("Encryption in Transit", value=True)
        encryption_at_rest = st.sidebar.checkbox("Encryption at Rest", value=True)
        data_residency = st.sidebar.selectbox("Data Residency Requirements", 
            ["No restrictions", "US only", "EU only", "Specific region", "On-premises only"])
        
        # Enterprise parameters section
        st.sidebar.subheader("🎯 Enterprise Parameters")
        sla_requirements = st.sidebar.selectbox("SLA Requirements", 
            ["99.9% availability", "99.95% availability", "99.99% availability", "99.999% availability"])
        rto_hours = st.sidebar.number_input("Recovery Time Objective (hours)", min_value=1, max_value=168, value=4)
        rpo_hours = st.sidebar.number_input("Recovery Point Objective (hours)", min_value=0, max_value=24, value=1)
        max_transfer_days = st.sidebar.number_input("Maximum Transfer Days", min_value=1, max_value=90, value=30)
        
        # Budget section
        budget_allocated = st.sidebar.number_input("Allocated Budget ($)", min_value=1000, max_value=10000000, value=100000, step=1000)
        approval_required = st.sidebar.checkbox("Executive Approval Required", value=True)
        
        # Data characteristics section
        st.sidebar.subheader("📊 Data Profile")
        data_size_gb = st.sidebar.number_input("Total Data Size (GB)", min_value=1, max_value=1000000, value=10000, step=100)
        data_types = st.sidebar.multiselect("Data Types", 
            ["Customer Data", "Financial Records", "Employee Data", "Intellectual Property", 
             "System Logs", "Application Data", "Database Backups", "Media Files", "Documents"])
        database_types = st.sidebar.multiselect("Database Systems", 
            ["Oracle", "SQL Server", "MySQL", "PostgreSQL", "MongoDB", "Cassandra", "Redis", "Elasticsearch"])
        avg_file_size = st.sidebar.selectbox("Average File Size",
            ["< 1MB (Many small files)", "1-10MB (Small files)", "10-100MB (Medium files)", 
             "100MB-1GB (Large files)", "> 1GB (Very large files)"])
        data_growth_rate = st.sidebar.slider("Annual Data Growth Rate (%)", min_value=0, max_value=100, value=20)
        data_volatility = st.sidebar.selectbox("Data Change Frequency", 
            ["Static (rarely changes)", "Low (daily changes)", "Medium (hourly changes)", "High (real-time)"])
        
        # Network infrastructure section
        st.sidebar.subheader("🌐 Network Configuration")
        network_topology = st.sidebar.selectbox("Network Topology", 
            ["Single DX", "Redundant DX", "Hybrid (DX + VPN)", "Multi-region", "SD-WAN"])
        dx_bandwidth_mbps = st.sidebar.number_input("Primary DX Bandwidth (Mbps)", min_value=50, max_value=100000, value=10000, step=100)
        dx_redundant = st.sidebar.checkbox("Redundant DX Connection", value=True)
        if dx_redundant:
            dx_secondary_mbps = st.sidebar.number_input("Secondary DX Bandwidth (Mbps)", min_value=50, max_value=100000, value=10000, step=100)
        else:
            dx_secondary_mbps = 0
        
        network_latency = st.sidebar.slider("Network Latency to AWS (ms)", min_value=1, max_value=500, value=25)
        network_jitter = st.sidebar.slider("Network Jitter (ms)", min_value=0, max_value=50, value=5)
        packet_loss = st.sidebar.slider("Packet Loss (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.1)
        qos_enabled = st.sidebar.checkbox("QoS Enabled", value=True)
        dedicated_bandwidth = st.sidebar.slider("Dedicated Migration Bandwidth (%)", min_value=10, max_value=90, value=60)
        business_hours_restriction = st.sidebar.checkbox("Restrict to Off-Business Hours", value=True)
        
        # Transfer configuration section
        st.sidebar.subheader("🚀 Transfer Configuration")
        num_datasync_agents = st.sidebar.number_input("DataSync Agents", min_value=1, max_value=50, value=5)
        datasync_instance_type = st.sidebar.selectbox("DataSync Instance Type",
            ["m5.large", "m5.xlarge", "m5.2xlarge", "m5.4xlarge", "m5.8xlarge", 
             "c5.2xlarge", "c5.4xlarge", "c5.9xlarge", "r5.2xlarge", "r5.4xlarge"])
        
        # Real-world performance modeling
        st.sidebar.subheader("📊 Performance Modeling")
        real_world_mode = st.sidebar.checkbox("Real-world Performance Mode", value=True, 
            help="Include real-world factors like storage I/O, DataSync overhead, and AWS API limits")
        
        # Network optimization section
        st.sidebar.subheader("🌐 Network Optimization")
        tcp_window_size = st.sidebar.selectbox("TCP Window Size", 
            ["Default", "64KB", "128KB", "256KB", "512KB", "1MB", "2MB"])
        mtu_size = st.sidebar.selectbox("MTU Size", 
            ["1500 (Standard)", "9000 (Jumbo Frames)", "Custom"])
        
        network_congestion_control = st.sidebar.selectbox("Congestion Control Algorithm",
            ["Cubic (Default)", "BBR", "Reno", "Vegas"])
        wan_optimization = st.sidebar.checkbox("WAN Optimization", value=False)
        parallel_streams = st.sidebar.slider("Parallel Streams per Agent", min_value=1, max_value=100, value=20)
        use_transfer_acceleration = st.sidebar.checkbox("S3 Transfer Acceleration", value=True)
        
        # Storage configuration section
        st.sidebar.subheader("💾 Storage Strategy")
        s3_storage_class = st.sidebar.selectbox("Primary S3 Storage Class",
            ["Standard", "Standard-IA", "One Zone-IA", "Glacier Instant Retrieval", 
             "Glacier Flexible Retrieval", "Glacier Deep Archive"])
        enable_versioning = st.sidebar.checkbox("Enable S3 Versioning", value=True)
        enable_lifecycle = st.sidebar.checkbox("Lifecycle Policies", value=True)
        cross_region_replication = st.sidebar.checkbox("Cross-Region Replication", value=False)
        
        # Geographic configuration section
        st.sidebar.subheader("🗺️ Geographic Settings")
        source_location = st.sidebar.selectbox("Source Data Center Location",
            ["San Jose, CA", "San Antonio, TX", "New York, NY", "Chicago, IL", "Dallas, TX", 
             "Los Angeles, CA", "Atlanta, GA", "London, UK", "Frankfurt, DE", "Tokyo, JP", "Sydney, AU", "Other"])
        target_aws_region = st.sidebar.selectbox("Target AWS Region",
            ["us-east-1 (N. Virginia)", "us-east-2 (Ohio)", "us-west-1 (N. California)", 
             "us-west-2 (Oregon)", "eu-west-1 (Ireland)", "eu-central-1 (Frankfurt)",
             "ap-southeast-1 (Singapore)", "ap-northeast-1 (Tokyo)"])
        
        # AI Configuration section
        st.sidebar.subheader("🤖 AI Configuration")
        enable_real_ai = st.sidebar.checkbox("Enable Real Claude AI API", value=False)
        
        if enable_real_ai:
            if ANTHROPIC_AVAILABLE:
                claude_api_key = st.sidebar.text_input(
                    "Claude API Key", 
                    type="password", 
                    help="Enter your Anthropic Claude API key for enhanced AI analysis"
                )
                ai_model = st.sidebar.selectbox(
                    "AI Model", 
                    ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-7-sonnet-20250219", 
                     "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
                    help="Select Claude model for analysis"
                )
            else:
                st.sidebar.error("Anthropic library not installed. Run: pip install anthropic")
                claude_api_key = ""
                ai_model = "claude-3-sonnet-20240229"
        else:
            claude_api_key = ""
            ai_model = "claude-sonnet-4-20250514"
            st.sidebar.info("Using built-in AI simulation")
        
        return {
            'project_name': project_name,
            'business_unit': business_unit,
            'project_priority': project_priority,
            'migration_wave': migration_wave,
            'data_classification': data_classification,
            'compliance_frameworks': compliance_frameworks,
            'encryption_in_transit': encryption_in_transit,
            'encryption_at_rest': encryption_at_rest,
            'data_residency': data_residency,
            'sla_requirements': sla_requirements,
            'rto_hours': rto_hours,
            'rpo_hours': rpo_hours,
            'max_transfer_days': max_transfer_days,
            'budget_allocated': budget_allocated,
            'approval_required': approval_required,
            'data_size_gb': data_size_gb,
            'data_types': data_types,
            'database_types': database_types,
            'avg_file_size': avg_file_size,
            'data_growth_rate': data_growth_rate,
            'data_volatility': data_volatility,
            'network_topology': network_topology,
            'dx_bandwidth_mbps': dx_bandwidth_mbps,
            'dx_redundant': dx_redundant,
            'dx_secondary_mbps': dx_secondary_mbps,
            'network_latency': network_latency,
            'network_jitter': network_jitter,
            'packet_loss': packet_loss,
            'qos_enabled': qos_enabled,
            'dedicated_bandwidth': dedicated_bandwidth,
            'business_hours_restriction': business_hours_restriction,
            'num_datasync_agents': num_datasync_agents,
            'datasync_instance_type': datasync_instance_type,
            'tcp_window_size': tcp_window_size,
            'mtu_size': mtu_size,
            'network_congestion_control': network_congestion_control,
            'wan_optimization': wan_optimization,
            'parallel_streams': parallel_streams,
            'use_transfer_acceleration': use_transfer_acceleration,
            's3_storage_class': s3_storage_class,
            'enable_versioning': enable_versioning,
            'enable_lifecycle': enable_lifecycle,
            'cross_region_replication': cross_region_replication,
            'source_location': source_location,
            'target_aws_region': target_aws_region,
            'enable_real_ai': enable_real_ai,
            'claude_api_key': claude_api_key,
            'ai_model': ai_model,
            'real_world_mode': real_world_mode
        }
    
    def detect_configuration_changes(self, config):
        """Detect when configuration changes and log them"""
        try:
            # Create a hash of the current configuration
            config_str = json.dumps(config, sort_keys=True, default=str)
            current_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            # Check if configuration changed
            if st.session_state.last_config_hash != current_hash:
                if st.session_state.last_config_hash is not None:  # Not the first load
                    st.session_state.config_change_count += 1
                    # Log configuration change
                    self.log_audit_event("CONFIG_CHANGED", f"Configuration updated - Change #{st.session_state.config_change_count}")
                
                st.session_state.last_config_hash = current_hash
                return True
            return False
        except Exception as e:
            return False
    
    def calculate_migration_metrics(self, config):
        """Calculate all migration metrics with error handling"""
        try:
            # Basic calculations
            data_size_tb = max(0.1, config['data_size_gb'] / 1024)  # Ensure minimum size
            effective_data_gb = config['data_size_gb'] * 0.85  # Account for compression/deduplication
            
            # Calculate throughput with optimizations
            throughput_result = self.calculator.calculate_enterprise_throughput(
                config['datasync_instance_type'], config['num_datasync_agents'], config['avg_file_size'], 
                config['dx_bandwidth_mbps'], config['network_latency'], config['network_jitter'], 
                config['packet_loss'], config['qos_enabled'], config['dedicated_bandwidth'], 
                config.get('real_world_mode', True)
            )
            
            if len(throughput_result) == 4:
                datasync_throughput, network_efficiency, theoretical_throughput, real_world_efficiency = throughput_result
            else:
                # Fallback for backward compatibility
                datasync_throughput, network_efficiency = throughput_result[:2]
                theoretical_throughput = datasync_throughput * 1.5
                real_world_efficiency = 0.7
            
            # Ensure valid throughput values
            datasync_throughput = max(1, datasync_throughput)  # Minimum 1 Mbps
            network_efficiency = max(0.1, min(1.0, network_efficiency))  # Between 10% and 100%
            
            # Apply network optimizations
            tcp_efficiency = {"Default": 1.0, "64KB": 1.05, "128KB": 1.1, "256KB": 1.15, "512KB": 1.2, "1MB": 1.25, "2MB": 1.3}
            mtu_efficiency = {"1500 (Standard)": 1.0, "9000 (Jumbo Frames)": 1.15, "Custom": 1.1}
            congestion_efficiency = {"Cubic (Default)": 1.0, "BBR": 1.2, "Reno": 0.95, "Vegas": 1.05}
            
            tcp_factor = tcp_efficiency.get(config['tcp_window_size'], 1.0)
            mtu_factor = mtu_efficiency.get(config['mtu_size'], 1.0)
            congestion_factor = congestion_efficiency.get(config['network_congestion_control'], 1.0)
            wan_factor = 1.3 if config['wan_optimization'] else 1.0
            
            optimized_throughput = datasync_throughput * tcp_factor * mtu_factor * congestion_factor * wan_factor
            optimized_throughput = min(optimized_throughput, config['dx_bandwidth_mbps'] * (config['dedicated_bandwidth'] / 100))
            optimized_throughput = max(1, optimized_throughput)  # Ensure minimum throughput
            
            # Calculate timing
            available_hours_per_day = 16 if config['business_hours_restriction'] else 24
            transfer_days = (effective_data_gb * 8) / (optimized_throughput * available_hours_per_day * 3600) / 1000
            transfer_days = max(0.1, transfer_days)  # Ensure minimum transfer time
            
            # Calculate costs
            cost_breakdown = self.calculator.calculate_enterprise_costs(
                config['data_size_gb'], transfer_days, config['datasync_instance_type'], 
                config['num_datasync_agents'], config['compliance_frameworks'], config['s3_storage_class']
            )
            
            # Ensure all cost values are valid
            for key in cost_breakdown:
                cost_breakdown[key] = max(0, cost_breakdown[key])
            
            # Compliance and business impact
            compliance_reqs, compliance_risks = self.calculator.assess_compliance_requirements(
                config['compliance_frameworks'], config['data_classification'], config['data_residency']
            )
            business_impact = self.calculator.calculate_business_impact(transfer_days, config['data_types'])
            
            # Get AI-powered networking recommendations
            target_region_short = config['target_aws_region'].split()[0]  # Extract region code
            networking_recommendations = self.calculator.get_optimal_networking_architecture(
                config['source_location'], target_region_short, config['data_size_gb'],
                config['dx_bandwidth_mbps'], config['database_types'], config['data_types'], config
            )
            
            return {
                'data_size_tb': data_size_tb,
                'effective_data_gb': effective_data_gb,
                'datasync_throughput': datasync_throughput,
                'theoretical_throughput': theoretical_throughput,
                'real_world_efficiency': real_world_efficiency,
                'optimized_throughput': optimized_throughput,
                'network_efficiency': network_efficiency,
                'transfer_days': transfer_days,
                'cost_breakdown': cost_breakdown,
                'compliance_reqs': compliance_reqs,
                'compliance_risks': compliance_risks,
                'business_impact': business_impact,
                'available_hours_per_day': available_hours_per_day,
                'networking_recommendations': networking_recommendations
            }
            
        except Exception as e:
            # Return default metrics if calculation fails
            st.error(f"Error in calculation: {str(e)}")
            return {
                'data_size_tb': 1.0,
                'effective_data_gb': 1000,
                'datasync_throughput': 100,
                'theoretical_throughput': 150,
                'real_world_efficiency': 0.7,
                'optimized_throughput': 100,
                'network_efficiency': 0.7,
                'transfer_days': 10,
                'cost_breakdown': {'compute': 1000, 'transfer': 500, 'storage': 200, 'compliance': 100, 'monitoring': 50, 'total': 1850},
                'compliance_reqs': [],
                'compliance_risks': [],
                'business_impact': {'score': 0.5, 'level': 'Medium', 'recommendation': 'Standard approach'},
                'available_hours_per_day': 24,
                'networking_recommendations': {
                    'primary_method': 'DataSync',
                    'secondary_method': 'S3 Transfer Acceleration',
                    'networking_option': 'Direct Connect',
                    'db_migration_tool': 'DMS',
                    'rationale': 'Default configuration recommendation',
                    'estimated_performance': {'throughput_mbps': 100, 'estimated_days': 10, 'network_efficiency': 0.7},
                    'cost_efficiency': 'Medium',
                    'risk_level': 'Low'
                }
            }
    
    def render_dashboard_tab(self, config, metrics):
        """Render the dashboard tab with enhanced styling"""
        st.markdown('<div class="section-header">🏠 Enterprise Migration Dashboard</div>', unsafe_allow_html=True)
        
        # Calculate dynamic executive summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Dynamic calculation for active projects
        active_projects = len(st.session_state.migration_projects) + 1  # +1 for current project
        project_change = "+1" if active_projects > 1 else "New"
        
        # Dynamic calculation for total data migrated
        total_data_tb = metrics['data_size_tb']
        for project_data in st.session_state.migration_projects.values():
            if 'performance_metrics' in project_data:
                total_data_tb += project_data.get('data_size_gb', 0) / 1024
        data_change = f"+{metrics['data_size_tb']:.1f} TB"
        
        # Dynamic migration success rate
        base_success_rate = 85
        network_efficiency_bonus = metrics['network_efficiency'] * 15
        compliance_bonus = len(config['compliance_frameworks']) * 2
        risk_penalty = {"Low": 0, "Medium": -3, "High": -8, "Critical": -15}
        risk_adjustment = risk_penalty.get(metrics['networking_recommendations']['risk_level'], 0)
        
        calculated_success_rate = min(99, base_success_rate + network_efficiency_bonus + compliance_bonus + risk_adjustment)
        success_change = f"+{calculated_success_rate - 85:.0f}%" if calculated_success_rate > 85 else f"{calculated_success_rate - 85:.0f}%"
        
        # Dynamic cost savings calculation
        on_premises_cost = metrics['data_size_tb'] * 1000 * 12
        aws_annual_cost = metrics['cost_breakdown']['storage'] * 12 + metrics['cost_breakdown']['total']
        annual_savings = max(0, on_premises_cost - aws_annual_cost)
        savings_change = f"+${annual_savings/1000:.0f}K"
        
        # Dynamic compliance score
        compliance_score = min(100, len(config['compliance_frameworks']) * 20 + 
                             (20 if config['encryption_in_transit'] and config['encryption_at_rest'] else 10))
        compliance_change = f"+{compliance_score - 80:.0f}%" if compliance_score > 80 else f"{compliance_score - 80:.0f}%"
        
        with col1:
            st.metric("Active Projects", str(active_projects), project_change)
        with col2:
            st.metric("Total Data Volume", f"{total_data_tb:.1f} TB", data_change)
        with col3:
            st.metric("Migration Success Rate", f"{calculated_success_rate:.0f}%", success_change)
        with col4:
            st.metric("Projected Annual Savings", f"${annual_savings/1000:.0f}K", savings_change)
        with col5:
            st.metric("Compliance Score", f"{compliance_score:.0f}%", compliance_change)
        
        # Current project overview
        st.markdown('<div class="section-header">📊 Current Project Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💾 Data Volume", f"{metrics['data_size_tb']:.1f} TB", f"{config['data_size_gb']:,.0f} GB")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            performance_mode = "Real-world" if config.get('real_world_mode', True) else "Theoretical"
            if 'theoretical_throughput' in metrics:
                efficiency_pct = f"{(metrics['optimized_throughput']/metrics['theoretical_throughput'])*100:.0f}%"
                delta_text = f"{efficiency_pct} of theoretical ({performance_mode})"
            else:
                delta_text = f"{metrics['network_efficiency']:.1%} efficiency ({performance_mode})"
            st.metric("⚡ Throughput", f"{metrics['optimized_throughput']:.0f} Mbps", delta_text)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            timeline_status = "On Track" if metrics['transfer_days'] <= config['max_transfer_days'] else "At Risk"
            timeline_delta = f"{metrics['transfer_days']*24:.0f} hours ({timeline_status})"
            st.metric("📅 Duration", f"{metrics['transfer_days']:.1f} days", timeline_delta)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            budget_status = "Under Budget" if metrics['cost_breakdown']['total'] <= config['budget_allocated'] else "Over Budget"
            budget_delta = f"${metrics['cost_breakdown']['total']/metrics['data_size_tb']:.0f}/TB ({budget_status})"
            st.metric("💰 Total Cost", f"${metrics['cost_breakdown']['total']:,.0f}", budget_delta)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # AI-Powered Recommendations Section
        st.markdown('<div class="section-header">🤖 AI-Powered Recommendations</div>', unsafe_allow_html=True)
        recommendations = metrics['networking_recommendations']
        
        ai_type = "Real-time Claude AI" if config.get('enable_real_ai') and config.get('claude_api_key') else "Built-in AI"
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if config.get('real_world_mode', True):
                theoretical_max = metrics.get('theoretical_throughput', metrics['optimized_throughput'] * 1.5)
                efficiency_ratio = metrics['optimized_throughput'] / theoretical_max
                performance_gap = (1 - efficiency_ratio) * 100
                
                if efficiency_ratio > 0.8:
                    performance_analysis = f"🟢 Excellent performance! Your configuration achieves {efficiency_ratio*100:.0f}% of theoretical maximum."
                elif efficiency_ratio > 0.6:
                    performance_analysis = f"🟡 Good performance with {performance_gap:.0f}% optimization potential remaining."
                else:
                    performance_analysis = f"🔴 Significant optimization opportunity! {performance_gap:.0f}% performance gap identified."
            else:
                performance_analysis = "🧪 Theoretical mode shows maximum possible performance."
            
            st.markdown(f"""
            <div class="ai-insight">
                <strong>🧠 {ai_type} Analysis:</strong> {recommendations['rationale']}
                <br><br>
                <strong>🔍 Performance Analysis:</strong> {performance_analysis}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🎯 AI Recommendations**")
            st.write(f"**Method:** {recommendations['primary_method']}")
            st.write(f"**Network:** {recommendations['networking_option']}")
            st.write(f"**DB Tool:** {recommendations['db_migration_tool']}")
            st.write(f"**Risk Level:** {recommendations['risk_level']}")
            st.write(f"**Cost Efficiency:** {recommendations['cost_efficiency']}")
        
        with col3:
            st.markdown("**⚡ AI Expected Performance**")
            ai_perf = recommendations['estimated_performance']
            st.write(f"**Throughput:** {ai_perf['throughput_mbps']:.0f} Mbps")
            st.write(f"**Duration:** {ai_perf['estimated_days']:.1f} days")
            st.write(f"**Agents:** {ai_perf.get('agents_used', 1)}x {ai_perf.get('instance_type', 'Unknown')}")
            st.write(f"**Network Eff:** {ai_perf['network_efficiency']:.1%}")
        
        # Performance comparison table
        st.markdown('<div class="section-header">📊 Performance Comparison</div>', unsafe_allow_html=True)
        
        comparison_data = pd.DataFrame({
            "Metric": ["Throughput (Mbps)", "Duration (Days)", "Efficiency (%)", "Agents Used", "Instance Type"],
            "Theoretical": [
                f"{metrics.get('theoretical_throughput', 0):.0f}",
                f"{(metrics['effective_data_gb'] * 8) / (metrics.get('theoretical_throughput', 1) * 24 * 3600) / 1000:.1f}",
                "95%",
                str(config['num_datasync_agents']),
                str(config['datasync_instance_type'])
            ],
            "Your Config": [
                f"{metrics['optimized_throughput']:.0f}",
                f"{metrics['transfer_days']:.1f}",
                f"{(metrics['optimized_throughput']/metrics.get('theoretical_throughput', metrics['optimized_throughput']*1.2))*100:.0f}%",
                str(config['num_datasync_agents']),
                str(config['datasync_instance_type'])
            ],
            "AI Recommendation": [
                f"{recommendations['estimated_performance']['throughput_mbps']:.0f}",
                f"{recommendations['estimated_performance']['estimated_days']:.1f}",
                f"{recommendations['estimated_performance']['network_efficiency']*100:.0f}%",
                str(recommendations['estimated_performance'].get('agents_used', 1)),
                str(recommendations['estimated_performance'].get('instance_type', 'Unknown'))
            ]
        })
        
        # Display the comparison table
        self.safe_dataframe_display(comparison_data)
        
        # Show real AI analysis if available
        if recommendations.get('ai_analysis'):
            st.markdown(f"""
            <div class="ai-insight">
                <strong>🔮 Advanced Claude AI Insights:</strong><br>
                {recommendations['ai_analysis'].replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        # Configuration change tracker
        if st.session_state.config_change_count > 0:
            st.markdown(f"""
            <div style="background: #e8f5e8; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;">
                <strong>🔄 Real-time Updates:</strong> Configuration has been updated {st.session_state.config_change_count} time(s). 
                AI recommendations automatically refreshed.
            </div>
            """, unsafe_allow_html=True)
    
    def render_network_tab(self, config, metrics):
        """Render the network analysis tab"""
        st.markdown('<div class="section-header">🌐 Network Analysis & Architecture Optimization</div>', unsafe_allow_html=True)
        
        # Network performance dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            utilization_pct = (metrics['optimized_throughput'] / config['dx_bandwidth_mbps']) * 100
            st.metric("Network Utilization", f"{utilization_pct:.1f}%", f"{metrics['optimized_throughput']:.0f} Mbps")
        
        with col2:
            if 'theoretical_throughput' in metrics:
                efficiency_vs_theoretical = (metrics['optimized_throughput'] / metrics['theoretical_throughput']) * 100
                st.metric("Real-world Efficiency", f"{efficiency_vs_theoretical:.1f}%", "vs theoretical")
            else:
                st.metric("Network Efficiency", f"{metrics['network_efficiency']:.1%}", "Current")
        
        with col3:
            st.metric("Network Latency", f"{config['network_latency']} ms", "RTT to AWS")
        
        with col4:
            st.metric("Packet Loss", f"{config['packet_loss']}%", "Quality indicator")
        
        # AI-Powered Network Architecture Recommendations
        st.markdown('<div class="section-header">🤖 AI-Powered Network Architecture</div>', unsafe_allow_html=True)
        
        recommendations = metrics['networking_recommendations']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>🎯 Recommended Configuration</h4>
                <p><strong>Primary Method:</strong> {recommendations['primary_method']}</p>
                <p><strong>Secondary Method:</strong> {recommendations['secondary_method']}</p>
                <p><strong>Network Option:</strong> {recommendations['networking_option']}</p>
                <p><strong>Database Tool:</strong> {recommendations['db_migration_tool']}</p>
                <p><strong>Cost Efficiency:</strong> {recommendations['cost_efficiency']}</p>
                <p><strong>Risk Level:</strong> {recommendations['risk_level']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>📊 Expected Performance</h4>
                <p><strong>Throughput:</strong> {recommendations['estimated_performance']['throughput_mbps']:.0f} Mbps</p>
                <p><strong>Estimated Duration:</strong> {recommendations['estimated_performance']['estimated_days']:.1f} days</p>
                <p><strong>Network Efficiency:</strong> {recommendations['estimated_performance']['network_efficiency']:.1%}</p>
                <p><strong>Route:</strong> {config['source_location']} → {config['target_aws_region']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Claude AI Rationale
        st.markdown(f"""
        <div class="ai-insight">
            <strong>🧠 Claude AI Analysis:</strong> {recommendations['rationale']}
        </div>
        """, unsafe_allow_html=True)
    
    def render_planner_tab(self, config, metrics):
        """Render the migration planner tab"""
        st.markdown('<div class="section-header">📊 Migration Planning & Strategy</div>', unsafe_allow_html=True)
        
        recommendations = metrics['networking_recommendations']
        
        # AI Recommendations
        st.markdown(f"""
        <div class="ai-insight">
            <strong>🧠 Claude AI Recommendation:</strong> Based on your data profile ({metrics['data_size_tb']:.1f}TB), 
            network configuration ({config['dx_bandwidth_mbps']} Mbps), and geographic location 
            ({config['source_location']} → {config['target_aws_region']}), the optimal approach is 
            <strong>{recommendations['primary_method']}</strong> with <strong>{recommendations['networking_option']}</strong>.
        </div>
        """, unsafe_allow_html=True)
        
        # Business impact assessment
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Impact Level", metrics['business_impact']['level'])
        
        with col2:
            st.metric("Impact Score", f"{metrics['business_impact']['score']:.2f}")
        
        with col3:
            timeline_status = "✅ On Track" if metrics['transfer_days'] <= config['max_transfer_days'] else "⚠️ At Risk"
            st.metric("Timeline Status", timeline_status)
        
        st.markdown(f"""
        <div class="recommendation-box">
            <strong>📋 Migration Recommendation:</strong> {metrics['business_impact']['recommendation']}
            <br><strong>🤖 AI Analysis:</strong> {recommendations['rationale']}
        </div>
        """, unsafe_allow_html=True)
    
    def render_performance_tab(self, config, metrics):
        """Render the performance optimization tab"""
        st.markdown('<div class="section-header">⚡ Performance Optimization</div>', unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Optimized Throughput", f"{metrics['optimized_throughput']:.0f} Mbps")
        
        with col2:
            st.metric("Network Efficiency", f"{metrics['network_efficiency']:.1%}")
        
        with col3:
            st.metric("Transfer Time", f"{metrics['transfer_days']:.1f} days")
        
        with col4:
            st.metric("Cost per TB", f"${metrics['cost_breakdown']['total']/metrics['data_size_tb']:.0f}")
        
        # AI-Powered Optimization Recommendations
        recommendations = metrics['networking_recommendations']
        
        st.markdown(f"""
        <div class="ai-insight">
            <strong>🧠 Claude AI Performance Analysis:</strong> Your current configuration achieves {metrics['network_efficiency']:.1%} efficiency. 
            The recommended {recommendations['primary_method']} with {recommendations['networking_option']} can deliver 
            {recommendations['estimated_performance']['throughput_mbps']:.0f} Mbps throughput.
        </div>
        """, unsafe_allow_html=True)
    
    def render_security_tab(self, config, metrics):
        """Render the security and compliance tab"""
        st.markdown('<div class="section-header">🔒 Security & Compliance Management</div>', unsafe_allow_html=True)
        
        # Security dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            security_score = 85 + (10 if config['encryption_in_transit'] else 0) + (5 if len(config['compliance_frameworks']) > 0 else 0)
            st.metric("Security Score", f"{security_score}/100")
        
        with col2:
            compliance_score = min(100, len(config['compliance_frameworks']) * 15)
            st.metric("Compliance Coverage", f"{compliance_score}%")
        
        with col3:
            data_risk_level = {"Public": "Low", "Internal": "Medium", "Confidential": "High", "Restricted": "Very High", "Top Secret": "Critical"}
            st.metric("Data Risk Level", data_risk_level.get(config['data_classification'], "Medium"))
        
        with col4:
            st.metric("Audit Events", len(st.session_state.audit_log))
        
        # AI Security Analysis
        recommendations = metrics['networking_recommendations']
        
        security_analysis = f"""
        Based on your data classification ({config['data_classification']}) and compliance requirements 
        ({', '.join(config['compliance_frameworks']) if config['compliance_frameworks'] else 'None specified'}), 
        the recommended {recommendations['primary_method']} provides appropriate security controls. 
        Risk level is assessed as {recommendations['risk_level']}.
        """
        
        st.markdown(f"""
        <div class="ai-insight">
            <strong>🧠 Claude AI Security Assessment:</strong> {security_analysis}
        </div>
        """, unsafe_allow_html=True)
        
        # Compliance frameworks
        if config['compliance_frameworks']:
            st.markdown("**🏛️ Active Compliance Frameworks:**")
            for framework in config['compliance_frameworks']:
                st.write(f"• {framework}")
        
        # Compliance risks
        if metrics['compliance_risks']:
            st.markdown("**⚠️ Compliance Risks:**")
            for risk in metrics['compliance_risks']:
                st.warning(risk)
    
    def render_analytics_tab(self, config, metrics):
        """Render the analytics and reporting tab"""
        st.markdown('<div class="section-header">📈 Analytics & Reporting</div>', unsafe_allow_html=True)
        
        # AI-Generated Executive Summary
        recommendations = metrics['networking_recommendations']
        
        executive_summary = f"""
        **Migration Project:** {config['project_name']} | **Data Volume:** {metrics['data_size_tb']:.1f}TB | 
        **Estimated Duration:** {metrics['transfer_days']:.1f} days | **Total Cost:** ${metrics['cost_breakdown']['total']:,.0f}
        
        **AI Recommendation:** Implement {recommendations['primary_method']} with {recommendations['networking_option']} 
        for optimal performance and cost efficiency. Expected throughput: {recommendations['estimated_performance']['throughput_mbps']:.0f} Mbps.
        
        **Risk Assessment:** {recommendations['risk_level']} risk level with {recommendations['cost_efficiency']} cost efficiency.
        """
        
        st.markdown(f"""
        <div class="ai-insight">
            {executive_summary}
        </div>
        """, unsafe_allow_html=True)
        
        # Cost breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            cost_data = []
            for key, value in metrics['cost_breakdown'].items():
                if key != 'total':
                    cost_data.append({
                        "Cost Category": key.replace('_', ' ').title(),
                        "Amount ($)": f"${value:,.2f}",
                        "Percentage": f"{(value/metrics['cost_breakdown']['total'])*100:.1f}%"
                    })
            
            cost_df = pd.DataFrame(cost_data)
            self.safe_dataframe_display(cost_df)
        
        with col2:
            # Cost breakdown pie chart
            cost_labels = list(metrics['cost_breakdown'].keys())[:-1]  # Exclude 'total'
            cost_values = [metrics['cost_breakdown'][key] for key in cost_labels]
            
            fig_costs = go.Figure(data=[go.Pie(
                labels=[label.replace('_', ' ').title() for label in cost_labels],
                values=cost_values,
                hole=0.3
            )])
            
            fig_costs.update_layout(title="Migration Cost Distribution", height=400)
            st.plotly_chart(fig_costs, use_container_width=True)
    
    def render_conclusion_tab(self, config, metrics):
        """Render the conclusion tab with professional formatting"""
        st.markdown('<div class="section-header">🎯 Final Strategic Recommendation</div>', unsafe_allow_html=True)
        
        recommendations = metrics['networking_recommendations']
        
        # Overall recommendation score calculation
        performance_score = min(100, (metrics['optimized_throughput'] / 1000) * 50)
        cost_score = min(50, max(0, 50 - (metrics['cost_breakdown']['total'] / config['budget_allocated'] - 1) * 100))
        timeline_score = min(30, max(0, 30 - (metrics['transfer_days'] / config['max_transfer_days'] - 1) * 100))
        risk_score = {"Low": 20, "Medium": 15, "High": 10, "Critical": 5}.get(recommendations['risk_level'], 15)
        
        overall_score = performance_score + cost_score + timeline_score + risk_score
        
        # Strategic recommendation
        if overall_score >= 140:
            strategy_status = "✅ RECOMMENDED"
            strategy_action = "PROCEED"
            banner_color = "#28a745"
        elif overall_score >= 120:
            strategy_status = "⚠️ CONDITIONAL"
            strategy_action = "PROCEED WITH OPTIMIZATIONS"
            banner_color = "#ffc107"
        else:
            strategy_status = "❌ REQUIRES MODIFICATION"
            strategy_action = "REVISE CONFIGURATION"
            banner_color = "#dc3545"
        
        # Executive Summary Banner
        st.markdown(f"""
        <div class="executive-summary" style="background: {banner_color};">
            <h1>🎯 STRATEGIC RECOMMENDATION: {strategy_status}</h1>
            <h2>Action Required: {strategy_action}</h2>
            <p><strong>Overall Strategy Score: {overall_score:.0f}/150</strong></p>
            <p><strong>Success Probability: {85 + (overall_score - 100) * 0.3:.0f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # PDF Download Section
        if self.pdf_generator and PDF_AVAILABLE:
            st.markdown('<div class="section-header">📥 Download Professional Reports</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Download Executive Summary", type="primary"):
                    try:
                        pdf_buffer = self.pdf_generator.generate_conclusion_report(config, metrics, recommendations)
                        if pdf_buffer:
                            pdf_bytes = pdf_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download Executive Summary PDF",
                                data=pdf_bytes,
                                file_name=f"{config['project_name']}_Executive_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                            st.success("✅ Executive summary generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
            
            with col2:
                if st.button("💰 Download Cost Analysis", type="primary"):
                    try:
                        pdf_buffer = self.pdf_generator.generate_cost_analysis_report(config, metrics)
                        if pdf_buffer:
                            pdf_bytes = pdf_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download Cost Analysis PDF",
                                data=pdf_bytes,
                                file_name=f"{config['project_name']}_Cost_Analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                            st.success("✅ Cost analysis generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Error generating PDF: {str(e)}")
        else:
            st.warning("📋 PDF generation requires reportlab library. Install with: pip install reportlab")
        
        # Key metrics and final recommendation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="recommendation-box">
                <h3>🚀 Strategic Migration Plan</h3>
                <p><strong>Primary Method:</strong> {recommendations['primary_method']}</p>
                <p><strong>Network Architecture:</strong> {recommendations['networking_option']}</p>
                <p><strong>Expected Performance:</strong> {recommendations['estimated_performance']['throughput_mbps']:.0f} Mbps</p>
                <p><strong>Estimated Timeline:</strong> {metrics['transfer_days']:.1f} days</p>
                <p><strong>Total Investment:</strong> ${metrics['cost_breakdown']['total']:,.0f}</p>
                <p><strong>Risk Assessment:</strong> {recommendations['risk_level']} risk level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="recommendation-box">
                <h3>📊 Success Criteria</h3>
                <p><strong>Performance:</strong> {performance_score:.0f}/50 points</p>
                <p><strong>Cost:</strong> {cost_score:.0f}/50 points</p>
                <p><strong>Timeline:</strong> {timeline_score:.0f}/30 points</p>
                <p><strong>Risk:</strong> {risk_score}/20 points</p>
                <p><strong>Total Score:</strong> {overall_score:.0f}/150</p>
                <p><strong>AI Confidence:</strong> {85 + (overall_score - 100) * 0.3:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Final recommendation
        st.markdown(f"""
        <div class="ai-insight">
            <strong>🎯 FINAL RECOMMENDATION:</strong> {strategy_action}<br>
            <strong>AI CONFIDENCE:</strong> {85 + (overall_score - 100) * 0.3:.0f}% based on comprehensive analysis<br>
            <strong>NEXT STEPS:</strong> Present this analysis to stakeholders for approval and begin implementation planning.
        </div>
        """, unsafe_allow_html=True)
    
    def log_audit_event(self, event_type, details):
        """Log audit events"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details,
            "user": st.session_state.user_profile["role"]
        }
        st.session_state.audit_log.append(event)
    
    def run(self):
        """Main application entry point"""
        # Render header and navigation
        self.render_header()
        self.render_navigation()
        
        # Get configuration from sidebar
        config = self.render_sidebar_controls()
        
        # Detect configuration changes
        config_changed = self.detect_configuration_changes(config)
        
        # Calculate migration metrics
        metrics = self.calculate_migration_metrics(config)
        
        # Show real-time update indicator
        if config_changed:
            st.success("🔄 Configuration updated - Dashboard refreshed with new calculations")
        
        # Render appropriate tab
        if st.session_state.active_tab == "dashboard":
            self.render_dashboard_tab(config, metrics)
        elif st.session_state.active_tab == "network":
            self.render_network_tab(config, metrics)
        elif st.session_state.active_tab == "planner":
            self.render_planner_tab(config, metrics)
        elif st.session_state.active_tab == "performance":
            self.render_performance_tab(config, metrics)
        elif st.session_state.active_tab == "security":
            self.render_security_tab(config, metrics)
        elif st.session_state.active_tab == "analytics":
            self.render_analytics_tab(config, metrics)
        elif st.session_state.active_tab == "conclusion":
            self.render_conclusion_tab(config, metrics)
        
        # Footer
        st.markdown("---")
        st.markdown("**🏢 Enterprise AWS Migration Platform v2.0** - *AI-Powered • Security-First • Compliance-Ready*")


def main():
    """Main function to run the Enterprise AWS Migration Platform"""
    try:
        platform = MigrationPlatform()
        platform.run()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your configuration and try again.")


if __name__ == "__main__":
    main()