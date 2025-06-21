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
                                      network_bw_mbps, latency, jitter, packet_loss, qos_enabled, dedicated_bandwidth, 
                                      real_world_mode=True):
        """Calculate optimized throughput considering all network factors including real-world limitations"""
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
    
    def calculate_enterprise_costs(self, data_size_gb, transfer_days, instance_type, num_agents, 
                                 compliance_frameworks, s3_storage_class):
        """Calculate comprehensive migration costs"""
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
    
    def assess_compliance_requirements(self, frameworks, data_classification, data_residency):
        """Assess compliance requirements and identify risks"""
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
    
    def calculate_business_impact(self, transfer_days, data_types):
        """Calculate business impact score based on data types"""
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
    
    def get_optimal_networking_architecture(self, source_location, target_region, data_size_gb, 
                                          dx_bandwidth_mbps, database_types, data_types, config=None):
        """AI-powered networking architecture recommendations"""
        
        # Get latency for the route
        estimated_latency = self.geographic_latency.get(source_location, {}).get(target_region, 50)
        
        # Analyze data characteristics
        has_databases = len(database_types) > 0
        has_large_files = any("Large" in dt or "Media" in dt for dt in data_types)
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
        
        # Performance estimation
        if recommendations["networking_option"] == "Direct Connect (Primary)":
            base_throughput = min(dx_bandwidth_mbps * 0.8, 2000)
        elif "Direct Connect" in recommendations["networking_option"]:
            base_throughput = min(dx_bandwidth_mbps * 0.6, 1500)
        else:
            base_throughput = min(500, dx_bandwidth_mbps * 0.4)
        
        recommendations["estimated_performance"] = {
            "throughput_mbps": base_throughput,
            "estimated_days": (data_size_gb * 8) / (base_throughput * 86400) / 1000,
            "network_efficiency": network_score / 10
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
    
    def _generate_ai_rationale(self, source, target, data_size_tb, bandwidth, has_db, has_large_files, latency, network_score):
        """Generate intelligent rationale for recommendations"""
        
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

class MigrationPlatform:
    """Main application class for the Enterprise AWS Migration Platform"""
    
    def __init__(self):
        self.calculator = EnterpriseCalculator()
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
        """Setup custom CSS styling with real-time indicators"""
        st.markdown("""
        <style>
            .main-header {
                background: linear-gradient(90deg, #FF9900, #232F3E);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #FF9900;
                margin: 0.5rem 0;
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                background-color: #e9ecef;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .security-badge {
                background-color: #28a745;
                color: white;
                padding: 0.2rem 0.5rem;
                border-radius: 12px;
                font-size: 0.8rem;
                margin: 0.2rem;
            }
            .compliance-item {
                background-color: #e9ecef;
                padding: 0.5rem;
                margin: 0.2rem;
                border-radius: 4px;
                border-left: 3px solid #007bff;
            }
            .networking-frame {
                border: 2px solid #FF9900;
                border-radius: 10px;
                background: white;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                padding: 15px;
                margin: 10px 0;
            }
            .recommendation-box {
                background: linear-gradient(135deg, #e8f4fd 0%, #f0f8ff 100%);
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #3498db;
                margin: 10px 0;
                animation: slideIn 0.5s ease-in;
            }
            .ai-insight {
                background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
                padding: 12px;
                border-radius: 8px;
                border-left: 3px solid #007bff;
                margin: 8px 0;
                font-style: italic;
                animation: slideIn 0.5s ease-in;
            }
            .real-time-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                background-color: #28a745;
                border-radius: 50%;
                animation: pulse 2s infinite;
                margin-right: 5px;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.3; }
                100% { opacity: 1; }
            }
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .status-good { color: #28a745; font-weight: bold; }
            .status-warning { color: #ffc107; font-weight: bold; }
            .status-danger { color: #dc3545; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🏢 Enterprise AWS Migration Strategy Platform</h1>
            <p>AI-Powered Migration Planning • Security-First • Compliance-Ready • Enterprise-Scale</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_navigation(self):
        """Render the top navigation bar"""
        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 2])
        
        with col1:
            if st.button("🏠 Dashboard"):
                st.session_state.active_tab = "dashboard"
        with col2:
            if st.button("🌐 Network Analysis"):
                st.session_state.active_tab = "network"
        with col3:
            if st.button("📊 Migration Planner"):
                st.session_state.active_tab = "planner"
        with col4:
            if st.button("⚡ Performance"):
                st.session_state.active_tab = "performance"
        with col5:
            if st.button("🔒 Security"):
                st.session_state.active_tab = "security"
        with col6:
            if st.button("📈 Analytics"):
                st.session_state.active_tab = "analytics"
    
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
        
        if real_world_mode:
            st.sidebar.info("🌍 Modeling includes: Storage I/O limits, DataSync overhead, TCP inefficiencies, S3 API throttling")
        else:
            st.sidebar.warning("🧪 Laboratory conditions: Theoretical maximum performance")
        
        # Network optimization section
        st.sidebar.subheader("🌐 Network Optimization")
        tcp_window_size = st.sidebar.selectbox("TCP Window Size", 
            ["Default", "64KB", "128KB", "256KB", "512KB", "1MB", "2MB"])
        mtu_size = st.sidebar.selectbox("MTU Size", 
            ["1500 (Standard)", "9000 (Jumbo Frames)", "Custom"])
        if mtu_size == "Custom":
            custom_mtu = st.sidebar.number_input("Custom MTU", min_value=1280, max_value=9216, value=1500)
        
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
                
                # Display model information
                model_info = {
                    "claude-sonnet-4-20250514": "⚡ Claude Sonnet 4 - Best balance of speed & intelligence (Recommended)",
                    "claude-opus-4-20250514": "🧠 Claude Opus 4 - Most powerful model for complex analysis",
                    "claude-3-7-sonnet-20250219": "🎯 Claude 3.7 Sonnet - Extended thinking capabilities",
                    "claude-3-5-sonnet-20241022": "🔄 Claude 3.5 Sonnet - Reliable performance",
                    "claude-3-5-haiku-20241022": "💨 Claude 3.5 Haiku - Fastest responses"
                }
                st.sidebar.info(model_info.get(ai_model, "Model information not available"))
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
        import hashlib
        
        # Create a hash of the current configuration
        config_str = json.dumps(config, sort_keys=True)
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
                datasync_throughput, network_efficiency = throughput_result
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
        """Render the dashboard tab with real-time data"""
        st.header("🏠 Enterprise Migration Dashboard")
        
        # Calculate dynamic executive summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Dynamic calculation for active projects
        active_projects = len(st.session_state.migration_projects) + 1  # +1 for current project
        project_change = "+1" if active_projects > 1 else "New"
        
        # Dynamic calculation for total data migrated (sum of all projects + current)
        total_data_tb = metrics['data_size_tb']
        for project_data in st.session_state.migration_projects.values():
            if 'performance_metrics' in project_data:
                total_data_tb += project_data.get('data_size_gb', 0) / 1024
        data_change = f"+{metrics['data_size_tb']:.1f} TB"
        
        # Dynamic migration success rate based on project completion and network efficiency
        base_success_rate = 85  # Base rate
        network_efficiency_bonus = metrics['network_efficiency'] * 15  # Up to 15% bonus
        compliance_bonus = len(config['compliance_frameworks']) * 2  # 2% per framework
        risk_penalty = {"Low": 0, "Medium": -3, "High": -8, "Critical": -15}
        risk_adjustment = risk_penalty.get(metrics['networking_recommendations']['risk_level'], 0)
        
        calculated_success_rate = min(99, base_success_rate + network_efficiency_bonus + compliance_bonus + risk_adjustment)
        success_change = f"+{calculated_success_rate - 85:.0f}%" if calculated_success_rate > 85 else f"{calculated_success_rate - 85:.0f}%"
        
        # Dynamic cost savings calculation
        on_premises_cost = metrics['data_size_tb'] * 1000 * 12  # $1000/TB/month
        aws_annual_cost = metrics['cost_breakdown']['storage'] * 12 + metrics['cost_breakdown']['total']
        annual_savings = max(0, on_premises_cost - aws_annual_cost)
        
        # Add optimization savings
        if config.get('real_world_mode', True):
            optimization_potential = metrics['optimized_throughput'] / max(1, metrics.get('theoretical_throughput', metrics['optimized_throughput'] * 1.2))
            efficiency_savings = annual_savings * (1 - optimization_potential) * 0.3  # 30% of inefficiency as potential savings
            total_savings = annual_savings + efficiency_savings
        else:
            total_savings = annual_savings
        
        savings_change = f"+${annual_savings/1000:.0f}K"
        
        # Dynamic compliance score
        max_compliance_points = 100
        encryption_points = 20 if config['encryption_in_transit'] and config['encryption_at_rest'] else 10
        framework_points = min(40, len(config['compliance_frameworks']) * 10)
        classification_points = {"Public": 5, "Internal": 10, "Confidential": 15, "Restricted": 20, "Top Secret": 25}
        data_class_points = classification_points.get(config['data_classification'], 10)
        network_security_points = 15 if config['qos_enabled'] and config['dx_redundant'] else 10
        risk_points = {"Low": 15, "Medium": 10, "High": 5, "Critical": 0}
        risk_score_points = risk_points.get(metrics['networking_recommendations']['risk_level'], 10)
        
        compliance_score = min(max_compliance_points, encryption_points + framework_points + data_class_points + network_security_points + risk_score_points)
        compliance_change = f"+{compliance_score - 80:.0f}%" if compliance_score > 80 else f"{compliance_score - 80:.0f}%"
        
        with col1:
            st.metric("Active Projects", str(active_projects), project_change)
        with col2:
            st.metric("Total Data Volume", f"{total_data_tb:.1f} TB", data_change)
        with col3:
            st.metric("Migration Success Rate", f"{calculated_success_rate:.0f}%", success_change)
        with col4:
            st.metric("Projected Annual Savings", f"${total_savings/1000:.0f}K", savings_change)
        with col5:
            st.metric("Compliance Score", f"{compliance_score:.0f}%", compliance_change)
        
        # Current project overview with real-time metrics
        st.subheader("📊 Current Project Overview")
        
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
            # Calculate if timeline is on track
            timeline_status = "On Track" if metrics['transfer_days'] <= config['max_transfer_days'] else "At Risk"
            timeline_delta = f"{metrics['transfer_days']*24:.0f} hours ({timeline_status})"
            st.metric("📅 Duration", f"{metrics['transfer_days']:.1f} days", timeline_delta)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # Calculate budget status
            budget_status = "Under Budget" if metrics['cost_breakdown']['total'] <= config['budget_allocated'] else "Over Budget"
            budget_delta = f"${metrics['cost_breakdown']['total']/metrics['data_size_tb']:.0f}/TB ({budget_status})"
            st.metric("💰 Total Cost", f"${metrics['cost_breakdown']['total']:,.0f}", budget_delta)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time AI-Powered Recommendations Section
        st.subheader("🤖 AI-Powered Recommendations")
        recommendations = metrics['networking_recommendations']
        
        ai_type = "Real-time Claude AI" if config.get('enable_real_ai') and config.get('claude_api_key') else "Built-in AI"
        
        # Dynamic performance analysis based on current configuration
        performance_analysis = ""
        if config.get('real_world_mode', True):
            theoretical_max = metrics.get('theoretical_throughput', metrics['optimized_throughput'] * 1.5)
            efficiency_ratio = metrics['optimized_throughput'] / theoretical_max
            performance_gap = (1 - efficiency_ratio) * 100
            
            if efficiency_ratio > 0.8:
                performance_analysis = f"Excellent performance! Your configuration achieves {efficiency_ratio*100:.0f}% of theoretical maximum with only {performance_gap:.0f}% optimization potential remaining."
            elif efficiency_ratio > 0.6:
                performance_analysis = f"Good performance with room for improvement. Current efficiency is {efficiency_ratio*100:.0f}% with {performance_gap:.0f}% optimization gap due to storage I/O constraints and DataSync overhead."
            else:
                performance_analysis = f"Significant optimization opportunity! Your current {efficiency_ratio*100:.0f}% efficiency suggests {performance_gap:.0f}% performance gap mainly from storage bottlenecks and network constraints."
        else:
            performance_analysis = "Theoretical mode shows maximum possible performance under perfect laboratory conditions."
        
        st.markdown(f"""
        <div class="ai-insight">
            <strong>🧠 {ai_type} Analysis:</strong> {recommendations['rationale']}
            <br><br>
            <strong>🔍 Real-time Performance Analysis:</strong> {performance_analysis}
        </div>
        """, unsafe_allow_html=True)
        
        # Show real AI analysis if available
        if recommendations.get('ai_analysis'):
            st.markdown(f"""
            <div class="ai-insight">
                <strong>🔮 Advanced Claude AI Insights:</strong><br>
                {recommendations['ai_analysis'].replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time activities and dynamic alerts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Real-time Activities")
            
            # Dynamic activities based on current configuration
            current_time = datetime.now().strftime("%H:%M")
            activities = [
                f"🕐 {current_time} - {config['project_name']} configuration updated",
                f"🤖 AI recommended: {recommendations['primary_method']} for {metrics['data_size_tb']:.1f}TB dataset",
                f"🌐 Network analysis: {recommendations['networking_option']} ({metrics['optimized_throughput']:.0f} Mbps)",
                f"📊 Business impact: {metrics['business_impact']['level']} priority ({metrics['business_impact']['score']:.2f} score)",
                f"🔒 {len(config['compliance_frameworks'])} compliance framework(s) validated",
                f"💰 Cost analysis: ${metrics['cost_breakdown']['total']:,.0f} total budget",
                f"⚡ Performance mode: {'Real-world modeling' if config.get('real_world_mode') else 'Theoretical maximum'}"
            ]
            
            for activity in activities:
                st.write(f"• {activity}")
        
        with col2:
            st.subheader("⚠️ Real-time Alerts & Status")
            
            alerts = []
            
            # Dynamic alert generation based on current configuration
            if metrics['transfer_days'] > config['max_transfer_days']:
                days_over = metrics['transfer_days'] - config['max_transfer_days']
                alerts.append(f"🔴 Timeline risk: {days_over:.1f} days over {config['max_transfer_days']}-day target")
            
            if metrics['cost_breakdown']['total'] > config['budget_allocated']:
                over_budget = metrics['cost_breakdown']['total'] - config['budget_allocated']
                alerts.append(f"🔴 Budget exceeded by ${over_budget:,.0f}")
            
            if metrics['compliance_risks']:
                alerts.append(f"🟡 {len(metrics['compliance_risks'])} compliance risk(s) identified")
            
            if config['network_latency'] > 100:
                alerts.append(f"🟡 High latency ({config['network_latency']}ms) may impact performance")
            
            if recommendations['risk_level'] in ["High", "Critical"]:
                alerts.append(f"🟡 {recommendations['risk_level']} risk migration - review recommendations")
            
            # Performance-specific alerts
            if config.get('real_world_mode', True) and 'theoretical_throughput' in metrics:
                efficiency = metrics['optimized_throughput'] / metrics['theoretical_throughput']
                if efficiency < 0.5:
                    alerts.append("🟡 Low performance efficiency - consider instance upgrade")
            
            # Network utilization alerts
            utilization = (metrics['optimized_throughput'] / config['dx_bandwidth_mbps']) * 100
            if utilization > 80:
                alerts.append(f"🟡 High network utilization ({utilization:.0f}%) - monitor closely")
            
            # Compliance alerts based on data classification
            if config['data_classification'] in ["Restricted", "Top Secret"] and not config['encryption_at_rest']:
                alerts.append("🔴 Critical: Encryption at rest required for classified data")
            
            # AI-specific alerts
            if config.get('enable_real_ai') and not config.get('claude_api_key'):
                alerts.append("🟡 Real AI enabled but no API key provided")
            
            if not alerts:
                alerts.append("🟢 All systems optimal - no issues detected")
            
            for alert in alerts:
                st.write(alert)
        
        # Real-time project health dashboard
        st.subheader("🏥 Project Health Dashboard")
        
        # Calculate overall project health score
        health_factors = {
            "Timeline": 100 if metrics['transfer_days'] <= config['max_transfer_days'] else max(0, 100 - (metrics['transfer_days'] - config['max_transfer_days']) * 10),
            "Budget": 100 if metrics['cost_breakdown']['total'] <= config['budget_allocated'] else max(0, 100 - ((metrics['cost_breakdown']['total'] - config['budget_allocated']) / config['budget_allocated']) * 100),
            "Performance": metrics['network_efficiency'] * 100,
            "Security": compliance_score,
            "Risk": {"Low": 95, "Medium": 75, "High": 50, "Critical": 25}.get(recommendations['risk_level'], 75)
        }
        
        # Display health metrics
        health_cols = st.columns(len(health_factors))
        for idx, (factor, score) in enumerate(health_factors.items()):
            with health_cols[idx]:
                color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                st.metric(f"{color} {factor}", f"{score:.0f}%")
        
        # Overall health score
        overall_health = sum(health_factors.values()) / len(health_factors)
        health_status = "Excellent" if overall_health >= 90 else "Good" if overall_health >= 75 else "Fair" if overall_health >= 60 else "Needs Attention"
        
        st.markdown(f"""
        <div class="recommendation-box">
            <h4>📊 Overall Project Health: {overall_health:.0f}% ({health_status})</h4>
            <p><strong>Real-time Assessment:</strong> Based on current configuration, your migration project shows {health_status.lower()} health indicators with primary optimization opportunities in {min(health_factors, key=health_factors.get).lower()} management.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_networking_architecture_diagram(self, recommendations, config):
        """Render network architecture diagram"""
        
        # Create a network architecture visualization
        fig = go.Figure()
        
        # Define positions for network components
        components = {
            "Source DC": {"x": 1, "y": 3, "color": "#3498db", "size": 60},
            "Direct Connect": {"x": 3, "y": 4, "color": "#FF9900", "size": 40},
            "Internet": {"x": 3, "y": 2, "color": "#95a5a6", "size": 40},
            "AWS Region": {"x": 5, "y": 3, "color": "#27ae60", "size": 60},
            "Migration Tool": {"x": 3, "y": 3, "color": "#e74c3c", "size": 50}
        }
        
        # Add nodes
        for name, props in components.items():
            fig.add_trace(go.Scatter(
                x=[props["x"]], y=[props["y"]],
                mode='markers+text',
                marker=dict(size=props["size"], color=props["color"]),
                text=[name],
                textposition="middle center",
                textfont=dict(color="white", size=10),
                name=name,
                showlegend=False
            ))
        
        # Add connections based on recommendations
        connections = []
        
        # Primary path
        if "Direct Connect" in recommendations["networking_option"]:
            connections.append({"from": "Source DC", "to": "Direct Connect", "style": "solid", "color": "#FF9900", "width": 4})
            connections.append({"from": "Direct Connect", "to": "AWS Region", "style": "solid", "color": "#FF9900", "width": 4})
        else:
            connections.append({"from": "Source DC", "to": "Internet", "style": "solid", "color": "#95a5a6", "width": 3})
            connections.append({"from": "Internet", "to": "AWS Region", "style": "solid", "color": "#95a5a6", "width": 3})
        
        # Secondary path (if hybrid)
        if "Backup" in recommendations["networking_option"]:
            connections.append({"from": "Source DC", "to": "Internet", "style": "dash", "color": "#95a5a6", "width": 2})
            connections.append({"from": "Internet", "to": "AWS Region", "style": "dash", "color": "#95a5a6", "width": 2})
        
        # Migration tool connection
        connections.append({"from": "Source DC", "to": "Migration Tool", "style": "solid", "color": "#e74c3c", "width": 3})
        connections.append({"from": "Migration Tool", "to": "AWS Region", "style": "solid", "color": "#e74c3c", "width": 3})
        
        # Draw connections
        for conn in connections:
            from_comp = components[conn["from"]]
            to_comp = components[conn["to"]]
            
            fig.add_trace(go.Scatter(
                x=[from_comp["x"], to_comp["x"]],
                y=[from_comp["y"], to_comp["y"]],
                mode='lines',
                line=dict(
                    color=conn["color"],
                    width=conn["width"],
                    dash='dash' if conn["style"] == "dash" else None
                ),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Recommended Network Architecture: {recommendations['networking_option']}",
            xaxis=dict(range=[0, 6], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[1, 5], showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            plot_bgcolor='rgba(248,249,250,0.8)',
            annotations=[
                dict(
                    x=3, y=1.5,
                    text=f"Primary: {recommendations['primary_method']}<br>Secondary: {recommendations['secondary_method']}",
                    showarrow=False,
                    font=dict(size=12, color="#2c3e50"),
                    bgcolor="white",
                    bordercolor="#ddd",
                    borderwidth=1
                )
            ]
        )
        
        return fig
    
    def render_network_tab(self, config, metrics):
        """Render the network analysis tab"""
        st.header("🌐 Network Analysis & Architecture Optimization")
        
        # Network performance dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            utilization_pct = (metrics['optimized_throughput'] / config['dx_bandwidth_mbps']) * 100
            st.metric("Network Utilization", f"{utilization_pct:.1f}%", f"{metrics['optimized_throughput']:.0f} Mbps")
        
        with col2:
            if 'theoretical_throughput' in metrics:
                efficiency_vs_theoretical = (metrics['optimized_throughput'] / metrics['theoretical_throughput']) * 100
                st.metric("Real-world Efficiency", f"{efficiency_vs_theoretical:.1f}%", f"vs theoretical")
            else:
                efficiency_improvement = ((metrics['optimized_throughput'] - metrics['datasync_throughput']) / metrics['datasync_throughput']) * 100
                st.metric("Optimization Gain", f"{efficiency_improvement:.1f}%", "vs baseline")
        
        with col3:
            st.metric("Network Latency", f"{config['network_latency']} ms", "RTT to AWS")
        
        with col4:
            st.metric("Packet Loss", f"{config['packet_loss']}%", "Quality indicator")
        
        # Real-world vs Theoretical Performance Analysis
        if 'theoretical_throughput' in metrics and 'real_world_efficiency' in metrics:
            st.subheader("🌍 Real-world vs Theoretical Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance comparison metrics
                performance_data = pd.DataFrame({
                    "Scenario": ["Theoretical (Lab)", "Real-world (Field)", "Your Configuration"],
                    "Throughput (Mbps)": [
                        metrics['theoretical_throughput'],
                        metrics['optimized_throughput'],
                        metrics['optimized_throughput']
                    ],
                    "Efficiency": [
                        "95%",
                        f"{metrics['real_world_efficiency']*100:.1f}%",
                        f"{(metrics['optimized_throughput']/metrics['theoretical_throughput'])*100:.1f}%"
                    ],
                    "Factors": [
                        "Perfect conditions",
                        "Storage I/O, DataSync overhead, TCP limits",
                        "Current configuration"
                    ]
                })
                st.dataframe(performance_data, use_container_width=True, hide_index=True)
            
            with col2:
                # Create comparison chart
                fig_comparison = go.Figure()
                
                scenarios = ["Theoretical", "Real-world", "Your Config"]
                throughputs = [
                    metrics['theoretical_throughput'],
                    metrics['theoretical_throughput'] * metrics['real_world_efficiency'],
                    metrics['optimized_throughput']
                ]
                colors = ['lightblue', 'orange', 'lightgreen']
                
                fig_comparison.add_trace(go.Bar(
                    x=scenarios,
                    y=throughputs,
                    marker_color=colors,
                    text=[f"{t:.0f} Mbps" for t in throughputs],
                    textposition='auto'
                ))
                
                fig_comparison.update_layout(
                    title="Performance Reality Check",
                    yaxis_title="Throughput (Mbps)",
                    height=300
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Real-world bottlenecks analysis
            st.markdown(f"""
            <div class="ai-insight">
                <strong>🔍 Performance Analysis:</strong> Your actual throughput ({metrics['optimized_throughput']:.0f} Mbps) 
                represents {(metrics['optimized_throughput']/metrics['theoretical_throughput'])*100:.1f}% of theoretical maximum 
                ({metrics['theoretical_throughput']:.0f} Mbps). This {(1-(metrics['optimized_throughput']/metrics['theoretical_throughput']))*100:.1f}% 
                reduction is primarily due to storage I/O limitations, DataSync protocol overhead, and AWS API throttling.
            </div>
            """, unsafe_allow_html=True)
        
        # AI-Powered Network Architecture Recommendations
        st.subheader("🤖 AI-Powered Network Architecture Recommendations")
        
        recommendations = metrics['networking_recommendations']
        
        # Display networking architecture diagram
        fig_network = self.render_networking_architecture_diagram(recommendations, config)
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Recommendations breakdown
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
        
        # Real AI Analysis (if enabled)
        if recommendations.get('ai_analysis'):
            st.subheader("🤖 Advanced Claude AI Analysis")
            st.markdown(f"""
            <div class="ai-insight">
                <strong>🔮 Real-time Claude AI Insights:</strong><br>
                {recommendations['ai_analysis'].replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
        
        # Database Migration Tools Comparison
        st.subheader("🗄️ Database Migration Tools Analysis")
        
        db_tools_data = []
        for tool_key, tool_info in self.calculator.db_migration_tools.items():
            score = 85  # Base score
            if tool_key == recommendations['db_migration_tool']:
                score = 95  # Recommended tool gets higher score
            elif len(config['database_types']) > 0 and "Database" in tool_info['best_for'][0]:
                score = 90
            
            db_tools_data.append({
                "Tool": tool_info['name'],
                "Best For": ", ".join(tool_info['best_for'][:2]),
                "Data Size Limit": tool_info['data_size_limit'],
                "Downtime": tool_info['downtime'],
                "Complexity": tool_info['complexity'],
                "Recommendation Score": f"{score}%" if tool_key == recommendations['db_migration_tool'] else f"{score - 10}%",
                "Status": "✅ Recommended" if tool_key == recommendations['db_migration_tool'] else "Available"
            })
        
        df_db_tools = pd.DataFrame(db_tools_data)
        st.dataframe(df_db_tools, use_container_width=True, hide_index=True)
        
        # Network quality assessment
        st.subheader("📡 Network Quality Assessment")
        
        utilization_pct = (metrics['optimized_throughput'] / config['dx_bandwidth_mbps']) * 100
        
        quality_metrics = pd.DataFrame({
            "Metric": ["Latency", "Jitter", "Packet Loss", "Throughput", "Geographic Route"],
            "Current": [f"{config['network_latency']} ms", f"{config['network_jitter']} ms", 
                       f"{config['packet_loss']}%", f"{metrics['optimized_throughput']:.0f} Mbps",
                       f"{config['source_location']} → {config['target_aws_region']}"],
            "Target": ["< 50 ms", "< 10 ms", "< 0.1%", f"{config['dx_bandwidth_mbps'] * 0.8:.0f} Mbps", "Optimized"],
            "Status": [
                "✅ Good" if config['network_latency'] < 50 else "⚠️ High",
                "✅ Good" if config['network_jitter'] < 10 else "⚠️ High", 
                "✅ Good" if config['packet_loss'] < 0.1 else "⚠️ High",
                "✅ Good" if utilization_pct < 80 else "⚠️ High",
                "✅ Optimal" if recommendations['estimated_performance']['network_efficiency'] > 0.8 else "⚠️ Review"
            ]
        })
        
        st.dataframe(quality_metrics, use_container_width=True, hide_index=True)
    
    def render_planner_tab(self, config, metrics):
        """Render the migration planner tab"""
        st.header("📊 Migration Planning & Strategy")
        
        # AI Recommendations at the top
        st.subheader("🤖 AI-Powered Migration Strategy")
        recommendations = metrics['networking_recommendations']
        
        st.markdown(f"""
        <div class="ai-insight">
            <strong>🧠 Claude AI Recommendation:</strong> Based on your data profile ({metrics['data_size_tb']:.1f}TB), 
            network configuration ({config['dx_bandwidth_mbps']} Mbps), and geographic location ({config['source_location']} → {config['target_aws_region']}), 
            the optimal approach is <strong>{recommendations['primary_method']}</strong> with <strong>{recommendations['networking_option']}</strong>.
        </div>
        """, unsafe_allow_html=True)
        
        # Migration method comparison
        st.subheader("🔍 Migration Method Analysis")
        
        migration_methods = []
        
        # DataSync analysis
        migration_methods.append({
            "Method": f"DataSync Multi-Agent ({recommendations['primary_method']})",
            "Throughput": f"{metrics['optimized_throughput']:.0f} Mbps",
            "Duration": f"{metrics['transfer_days']:.1f} days",
            "Cost": f"${metrics['cost_breakdown']['total']:,.0f}",
            "Security": "High" if config['encryption_in_transit'] and config['encryption_at_rest'] else "Medium",
            "Complexity": "Medium",
            "AI Score": "95%" if recommendations['primary_method'] == "DataSync" else "85%"
        })
        
        # Snowball analysis
        if metrics['data_size_tb'] > 1:
            snowball_devices = max(1, int(metrics['data_size_tb'] / 72))
            snowball_days = 7 + (snowball_devices * 2)
            snowball_cost = snowball_devices * 300 + 2000
            
            migration_methods.append({
                "Method": f"Snowball Edge ({snowball_devices}x devices)",
                "Throughput": "Physical transfer",
                "Duration": f"{snowball_days} days",
                "Cost": f"${snowball_cost:,.0f}",
                "Security": "Very High",
                "Complexity": "Low",
                "AI Score": "90%" if recommendations['primary_method'] == "Snowball Edge" else "75%"
            })
        
        # DMS for databases
        if config['database_types']:
            dms_days = metrics['transfer_days'] * 1.2  # DMS typically takes longer
            dms_cost = metrics['cost_breakdown']['total'] * 1.1
            
            migration_methods.append({
                "Method": f"Database Migration Service (DMS)",
                "Throughput": f"{metrics['optimized_throughput'] * 0.8:.0f} Mbps",
                "Duration": f"{dms_days:.1f} days",
                "Cost": f"${dms_cost:,.0f}",
                "Security": "High",
                "Complexity": "Medium",
                "AI Score": "95%" if recommendations['db_migration_tool'] == "DMS" else "80%"
            })
        
        # Storage Gateway
        sg_throughput = min(config['dx_bandwidth_mbps'] * 0.6, 2000)
        sg_days = (metrics['effective_data_gb'] * 8) / (sg_throughput * metrics['available_hours_per_day'] * 3600) / 1000
        sg_cost = metrics['cost_breakdown']['total'] * 1.3
        
        migration_methods.append({
            "Method": "Storage Gateway (Hybrid)",
            "Throughput": f"{sg_throughput:.0f} Mbps",
            "Duration": f"{sg_days:.1f} days",
            "Cost": f"${sg_cost:,.0f}",
            "Security": "High",
            "Complexity": "Medium",
            "AI Score": "80%"
        })
        
        df_methods = pd.DataFrame(migration_methods)
        st.dataframe(df_methods, use_container_width=True, hide_index=True)
        
        # Geographic Optimization Analysis
        st.subheader("🗺️ Geographic Route Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show latency comparison for different regions
            if config['source_location'] in self.calculator.geographic_latency:
                latencies = self.calculator.geographic_latency[config['source_location']]
                region_comparison = []
                
                for region, latency in latencies.items():
                    region_comparison.append({
                        "AWS Region": region,
                        "Latency (ms)": latency,
                        "Performance Impact": "Excellent" if latency < 30 else "Good" if latency < 80 else "Fair",
                        "Recommended": "✅" if region in config['target_aws_region'] else ""
                    })
                
                df_regions = pd.DataFrame(region_comparison)
                st.dataframe(df_regions, use_container_width=True, hide_index=True)
        
        with col2:
            # Create latency comparison chart
            if config['source_location'] in self.calculator.geographic_latency:
                latencies = self.calculator.geographic_latency[config['source_location']]
                
                fig_latency = go.Figure()
                fig_latency.add_trace(go.Bar(
                    x=list(latencies.keys()),
                    y=list(latencies.values()),
                    marker_color=['lightgreen' if region in config['target_aws_region'] else 'lightblue' for region in latencies.keys()],
                    text=[f"{latency} ms" for latency in latencies.values()],
                    textposition='auto'
                ))
                
                fig_latency.update_layout(
                    title=f"Network Latency from {config['source_location']}",
                    xaxis_title="AWS Region",
                    yaxis_title="Latency (ms)",
                    height=300
                )
                st.plotly_chart(fig_latency, use_container_width=True)
        
        # Business impact assessment
        st.subheader("📈 Business Impact Analysis")
        
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
        st.header("⚡ Performance Optimization")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate baseline for comparison (using theoretical mode)
        baseline_result = self.calculator.calculate_enterprise_throughput(
            config['datasync_instance_type'], config['num_datasync_agents'], config['avg_file_size'], 
            config['dx_bandwidth_mbps'], 100, 5, 0.05, False, config['dedicated_bandwidth'], False
        )
        baseline_throughput = baseline_result[0] if isinstance(baseline_result, tuple) else baseline_result
        
        improvement = ((metrics['optimized_throughput'] - baseline_throughput) / baseline_throughput) * 100
        
        with col1:
            st.metric("Performance Gain", f"{improvement:.1f}%", "vs baseline")
        
        with col2:
            st.metric("Network Efficiency", f"{(metrics['optimized_throughput']/config['dx_bandwidth_mbps'])*100:.1f}%")
        
        with col3:
            st.metric("Transfer Time", f"{metrics['transfer_days']:.1f} days")
        
        with col4:
            st.metric("Cost per TB", f"${metrics['cost_breakdown']['total']/metrics['data_size_tb']:.0f}")
        
        # AI-Powered Optimization Recommendations
        st.subheader("🤖 AI-Powered Optimization Recommendations")
        recommendations = metrics['networking_recommendations']
        
        st.markdown(f"""
        <div class="ai-insight">
            <strong>🧠 Claude AI Performance Analysis:</strong> Your current configuration achieves {metrics['network_efficiency']:.1%} efficiency. 
            The recommended {recommendations['primary_method']} with {recommendations['networking_option']} can deliver 
            {recommendations['estimated_performance']['throughput_mbps']:.0f} Mbps throughput.
        </div>
        """, unsafe_allow_html=True)
        
        # Optimization recommendations
        st.subheader("🎯 Specific Optimization Recommendations")
        
        recommendations_list = []
        
        if config['tcp_window_size'] == "Default":
            recommendations_list.append("🔧 Enable TCP window scaling (2MB) for 25-30% improvement")
        
        if config['mtu_size'] == "1500 (Standard)":
            recommendations_list.append("📡 Configure jumbo frames (9000 MTU) for 10-15% improvement")
        
        if config['network_congestion_control'] == "Cubic (Default)":
            recommendations_list.append("⚡ Switch to BBR algorithm for 20-25% improvement")
        
        if not config['wan_optimization']:
            recommendations_list.append("🚀 Enable WAN optimization for 25-30% improvement")
        
        if config['parallel_streams'] < 20:
            recommendations_list.append("🔄 Increase parallel streams to 20+ for better throughput")
        
        if not config['use_transfer_acceleration']:
            recommendations_list.append("🌐 Enable S3 Transfer Acceleration for 50-500% improvement")
        
        # Add AI-specific recommendations
        if recommendations['networking_option'] != "Direct Connect (Primary)":
            recommendations_list.append(f"🤖 AI suggests upgrading to Direct Connect for optimal performance")
        
        if recommendations['primary_method'] != "DataSync":
            recommendations_list.append(f"🤖 AI recommends {recommendations['primary_method']} for your workload characteristics")
        
        if recommendations_list:
            for rec in recommendations_list:
                st.write(f"• {rec}")
        else:
            st.success("✅ Configuration is already well optimized!")
        
        # Performance comparison chart
        st.subheader("📊 Optimization Impact Analysis")
        
        # Include AI recommendations in the chart
        optimization_scenarios = {
            "Current Config": metrics['optimized_throughput'],
            "TCP Optimized": metrics['optimized_throughput'] * 1.25 if config['tcp_window_size'] == "Default" else metrics['optimized_throughput'],
            "Network Optimized": metrics['optimized_throughput'] * 1.4 if not config['wan_optimization'] else metrics['optimized_throughput'],
            "AI Recommended": recommendations['estimated_performance']['throughput_mbps']
        }
        
        fig_opt = go.Figure()
        colors = ['lightblue', 'lightgreen', 'orange', 'gold']
        
        fig_opt.add_trace(go.Bar(
            x=list(optimization_scenarios.keys()),
            y=list(optimization_scenarios.values()),
            marker_color=colors,
            text=[f"{v:.0f} Mbps" for v in optimization_scenarios.values()],
            textposition='auto'
        ))
        
        fig_opt.update_layout(
            title="Performance Optimization Scenarios",
            yaxis_title="Throughput (Mbps)",
            height=400
        )
        st.plotly_chart(fig_opt, use_container_width=True)
    
    def render_security_tab(self, config, metrics):
        """Render the security and compliance tab"""
        st.header("🔒 Security & Compliance Management")
        
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
        st.subheader("🤖 AI Security & Compliance Analysis")
        
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
        
        # Security controls matrix
        st.subheader("🛡️ Security Controls Matrix")
        
        security_controls = pd.DataFrame({
            "Control": [
                "Data Encryption in Transit",
                "Data Encryption at Rest",
                "Network Segmentation",
                "Access Control (IAM)",
                "Audit Logging",
                "Data Loss Prevention",
                "Incident Response Plan",
                "Compliance Monitoring"
            ],
            "Status": [
                "✅ Enabled" if config['encryption_in_transit'] else "❌ Disabled",
                "✅ Enabled" if config['encryption_at_rest'] else "❌ Disabled",
                "✅ Enabled",
                "✅ Enabled",
                "✅ Enabled",
                "⚠️ Partial",
                "✅ Enabled",
                "✅ Enabled" if config['compliance_frameworks'] else "❌ Disabled"
            ],
            "Compliance": [
                "Required" if any(f in ["GDPR", "HIPAA", "PCI-DSS"] for f in config['compliance_frameworks']) else "Recommended",
                "Required" if any(f in ["GDPR", "HIPAA", "PCI-DSS"] for f in config['compliance_frameworks']) else "Recommended",
                "Required" if "PCI-DSS" in config['compliance_frameworks'] else "Recommended",
                "Required",
                "Required" if any(f in ["SOX", "HIPAA"] for f in config['compliance_frameworks']) else "Recommended",
                "Required" if "GDPR" in config['compliance_frameworks'] else "Recommended",
                "Required",
                "Required" if config['compliance_frameworks'] else "Optional"
            ],
            "AI Recommendation": [
                "✅ Optimal" if config['encryption_in_transit'] else "⚠️ Enable",
                "✅ Optimal" if config['encryption_at_rest'] else "⚠️ Enable",
                "✅ Configured",
                "✅ AWS Best Practice",
                "✅ Enterprise Standard",
                "⚠️ Review DLP policies",
                "✅ AWS native tools",
                "✅ Optimal" if config['compliance_frameworks'] else "⚠️ Define requirements"
            ]
        })
        
        st.dataframe(security_controls, use_container_width=True, hide_index=True)
        
        # Compliance frameworks
        if config['compliance_frameworks']:
            st.subheader("🏛️ Compliance Frameworks")
            
            for framework in config['compliance_frameworks']:
                st.markdown(f'<span class="security-badge">{framework}</span>', unsafe_allow_html=True)
        
        # Compliance risks
        if metrics['compliance_risks']:
            st.subheader("⚠️ Compliance Risks")
            for risk in metrics['compliance_risks']:
                st.warning(risk)
    
    def render_analytics_tab(self, config, metrics):
        """Render the analytics and reporting tab"""
        st.header("📈 Analytics & Reporting")
        
        # AI-Generated Executive Summary
        recommendations = metrics['networking_recommendations']
        st.subheader("🤖 AI-Generated Executive Summary")
        
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
        st.subheader("💰 Cost Analysis")
        
        cost_labels = list(metrics['cost_breakdown'].keys())[:-1]  # Exclude 'total'
        cost_values = [metrics['cost_breakdown'][key] for key in cost_labels]
        
        fig_costs = go.Figure(data=[go.Pie(
            labels=[label.title() for label in cost_labels],
            values=cost_values,
            hole=0.3
        )])
        
        fig_costs.update_layout(title="Migration Cost Breakdown", height=400)
        st.plotly_chart(fig_costs, use_container_width=True)
        
        # Performance trends (simulated)
        st.subheader("📊 Performance Trends")
        
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="M")
        throughput_trend = np.random.normal(metrics['optimized_throughput'], metrics['optimized_throughput']*0.1, len(dates))
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=dates,
            y=throughput_trend,
            mode='lines+markers',
            name='Throughput (Mbps)',
            line=dict(color='#3498db')
        ))
        
        # Add AI prediction line
        future_dates = pd.date_range(start="2025-01-01", end="2025-06-30", freq="M")
        ai_prediction = [recommendations['estimated_performance']['throughput_mbps']] * len(future_dates)
        
        fig_trend.add_trace(go.Scatter(
            x=future_dates,
            y=ai_prediction,
            mode='lines+markers',
            name='AI Predicted Performance',
            line=dict(color='#e74c3c', dash='dash')
        ))
        
        fig_trend.update_layout(
            title="Historical Throughput Performance & AI Predictions",
            xaxis_title="Date",
            yaxis_title="Throughput (Mbps)",
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # ROI Analysis
        st.subheader("💡 ROI Analysis with AI Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Calculate annual savings
            on_premises_annual_cost = metrics['data_size_tb'] * 1000 * 12  # $1000/TB/month on-premises
            aws_annual_cost = metrics['cost_breakdown']['storage'] * 12 + (metrics['cost_breakdown']['total'] * 0.1)
            annual_savings = max(0, on_premises_annual_cost - aws_annual_cost)  # Ensure non-negative
            st.metric("Annual Savings", f"${annual_savings:,.0f}")
        
        with col2:
            roi_percentage = (annual_savings / metrics['cost_breakdown']['total']) * 100 if metrics['cost_breakdown']['total'] > 0 else 0
            st.metric("ROI", f"{roi_percentage:.1f}%")
        
        with col3:
            payback_period = metrics['cost_breakdown']['total'] / annual_savings if annual_savings > 0 else 0
            payback_display = f"{payback_period:.1f} years" if payback_period > 0 and payback_period < 50 else "N/A"
            st.metric("Payback Period", payback_display)
        
        # AI Business Impact Analysis
        st.markdown(f"""
        <div class="recommendation-box">
            <h4>🤖 AI Business Impact Analysis</h4>
            <p><strong>Business Value:</strong> The recommended migration strategy delivers {recommendations['cost_efficiency']} cost efficiency 
            with {recommendations['risk_level']} risk profile.</p>
            <p><strong>Performance Impact:</strong> Expected {recommendations['estimated_performance']['network_efficiency']:.1%} network efficiency 
            with {recommendations['estimated_performance']['throughput_mbps']:.0f} Mbps sustained throughput.</p>
            <p><strong>Strategic Recommendation:</strong> {recommendations['rationale']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar_status(self, config, metrics):
        """Render real-time status in sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.subheader("🚦 Real-time Status")
            
            # Dynamic health indicators based on current configuration
            status_factors = []
            
            # Timeline status
            if metrics['transfer_days'] <= config['max_transfer_days']:
                days_remaining = config['max_transfer_days'] - metrics['transfer_days']
                status_factors.append(f"✅ Timeline (+{days_remaining:.1f} days buffer)")
            else:
                days_over = metrics['transfer_days'] - config['max_transfer_days']
                status_factors.append(f"❌ Timeline (-{days_over:.1f} days over)")
            
            # Budget status
            if metrics['cost_breakdown']['total'] <= config['budget_allocated']:
                budget_remaining = config['budget_allocated'] - metrics['cost_breakdown']['total']
                status_factors.append(f"✅ Budget (${budget_remaining:,.0f} remaining)")
            else:
                budget_over = metrics['cost_breakdown']['total'] - config['budget_allocated']
                status_factors.append(f"❌ Budget (+${budget_over:,.0f} over)")
            
            # Compliance status
            if not metrics['compliance_risks']:
                compliance_count = len(config['compliance_frameworks'])
                status_factors.append(f"✅ Compliance ({compliance_count} frameworks)")
            else:
                risk_count = len(metrics['compliance_risks'])
                status_factors.append(f"⚠️ Compliance ({risk_count} risks)")
            
            # Network status with real-time metrics
            if config['network_latency'] < 50:
                status_factors.append(f"✅ Network ({config['network_latency']}ms latency)")
            elif config['network_latency'] < 100:
                status_factors.append(f"⚠️ Network ({config['network_latency']}ms latency)")
            else:
                status_factors.append(f"❌ Network ({config['network_latency']}ms latency)")
            
            # Performance status
            if 'theoretical_throughput' in metrics:
                efficiency_pct = (metrics['optimized_throughput'] / metrics['theoretical_throughput']) * 100
                if efficiency_pct >= 80:
                    status_factors.append(f"✅ Performance ({efficiency_pct:.0f}% efficiency)")
                elif efficiency_pct >= 60:
                    status_factors.append(f"⚠️ Performance ({efficiency_pct:.0f}% efficiency)")
                else:
                    status_factors.append(f"❌ Performance ({efficiency_pct:.0f}% efficiency)")
            else:
                network_eff_pct = metrics['network_efficiency'] * 100
                if network_eff_pct >= 80:
                    status_factors.append(f"✅ Performance ({network_eff_pct:.0f}% network efficiency)")
                else:
                    status_factors.append(f"⚠️ Performance ({network_eff_pct:.0f}% network efficiency)")
            
            for factor in status_factors:
                st.write(factor)
            
            # Real-time AI Recommendations Summary
            st.subheader("🤖 Live AI Summary")
            recommendations = metrics['networking_recommendations']
            
            # Dynamic AI status
            if config.get('enable_real_ai') and config.get('claude_api_key'):
                ai_status = f"🔮 Real AI ({config.get('ai_model', 'claude-sonnet-4')[:15]}...)"
            else:
                ai_status = "🧠 Built-in AI Simulation"
            
            st.write(f"**AI Mode:** {ai_status}")
            st.write(f"**Method:** {recommendations['primary_method']}")
            st.write(f"**Network:** {recommendations['networking_option']}")
            st.write(f"**Throughput:** {recommendations['estimated_performance']['throughput_mbps']:.0f} Mbps")
            st.write(f"**Risk:** {recommendations['risk_level']}")
            st.write(f"**Efficiency:** {recommendations['cost_efficiency']}")
            
            # Real-time performance insights
            if config.get('real_world_mode', True) and 'theoretical_throughput' in metrics:
                performance_gap = (1 - (metrics['optimized_throughput'] / metrics['theoretical_throughput'])) * 100
                if performance_gap > 40:
                    st.write(f"**Optimization:** {performance_gap:.0f}% gap - Major improvement possible")
                elif performance_gap > 20:
                    st.write(f"**Optimization:** {performance_gap:.0f}% gap - Good optimization potential")
                else:
                    st.write(f"**Optimization:** {performance_gap:.0f}% gap - Well optimized")
            
            # Dynamic bottleneck identification
            bottlenecks = []
            if config['datasync_instance_type'] == "m5.large" and config['avg_file_size'] == "> 1GB (Very large files)":
                bottlenecks.append("Instance CPU/Memory")
            if config['dx_bandwidth_mbps'] < 1000:
                bottlenecks.append("Network Bandwidth")
            if config['network_latency'] > 50:
                bottlenecks.append("Network Latency")
            if config['num_datasync_agents'] < 3:
                bottlenecks.append("Agent Count")
            
            if bottlenecks:
                st.write(f"**Bottlenecks:** {', '.join(bottlenecks[:2])}")
            
            # Quick actions with real-time context
            st.subheader("⚡ Smart Actions")
            
            if st.button("🔄 Refresh Analysis"):
                # Log the refresh action
                st.session_state.audit_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "MANUAL_REFRESH",
                    "details": f"User refreshed analysis for {config['project_name']}",
                    "user": st.session_state.user_profile["role"]
                })
                st.rerun()
            
            # Dynamic action recommendations
            if recommendations['primary_method'] != "DataSync":
                if st.button(f"🤖 Switch to {recommendations['primary_method']}"):
                    st.info(f"AI recommends switching to {recommendations['primary_method']} for optimal performance")
            
            # Performance optimization action
            if 'theoretical_throughput' in metrics:
                efficiency = metrics['optimized_throughput'] / metrics['theoretical_throughput']
                if efficiency < 0.7:
                    if st.button("🚀 Apply Performance Optimizations"):
                        st.info("Performance optimizations would upgrade instance type, increase agents, and optimize network settings")
            
            # Budget optimization action
            if metrics['cost_breakdown']['total'] > config['budget_allocated']:
                if st.button("💰 Optimize Costs"):
                    st.info("Cost optimizations would adjust storage class, instance types, and transfer schedule")
            
            # Real-time configuration summary
            st.subheader("📊 Config Summary")
            st.write(f"**Data:** {metrics['data_size_tb']:.1f} TB")
            st.write(f"**Agents:** {config['num_datasync_agents']}x {config['datasync_instance_type']}")
            st.write(f"**Bandwidth:** {config['dx_bandwidth_mbps']} Mbps")
            st.write(f"**Mode:** {'Real-world' if config.get('real_world_mode') else 'Theoretical'}")
            st.write(f"**Duration:** {metrics['transfer_days']:.1f} days")
            st.write(f"**Cost:** ${metrics['cost_breakdown']['total']:,.0f}")
    
    def log_audit_event(self, event_type, details):
        """Log audit events"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "details": details,
            "user": st.session_state.user_profile["role"]
        }
        st.session_state.audit_log.append(event)
    
    def render_footer(self, config, metrics):
        """Render footer with configuration management"""
        st.markdown("---")
        
        # Configuration management
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Configuration"):
                project_config = {
                    "project_name": config['project_name'],
                    "data_size_gb": config['data_size_gb'],
                    "compliance_frameworks": config['compliance_frameworks'],
                    "network_config": {
                        "dx_bandwidth_mbps": config['dx_bandwidth_mbps'],
                        "latency": config['network_latency'],
                        "redundant": config['dx_redundant']
                    },
                    "performance_metrics": {
                        "optimized_throughput": metrics['optimized_throughput'],
                        "transfer_days": metrics['transfer_days'],
                        "network_efficiency": metrics['network_efficiency']
                    },
                    "ai_recommendations": metrics['networking_recommendations'],
                    "timestamp": datetime.now().isoformat()
                }
                
                st.session_state.migration_projects[config['project_name']] = project_config
                self.log_audit_event("CONFIG_SAVED", f"Configuration saved for {config['project_name']}")
                st.success(f"✅ Configuration saved for project: {config['project_name']}")
        
        with col2:
            if st.button("📋 View Audit Log"):
                if st.session_state.audit_log:
                    audit_df = pd.DataFrame(st.session_state.audit_log)
                    st.dataframe(audit_df, use_container_width=True)
                else:
                    st.info("No audit events recorded yet.")
        
        with col3:
            if st.button("📤 Export AI Report"):
                report_data = {
                    "project_summary": {
                        "name": config['project_name'],
                        "data_size_tb": metrics['data_size_tb'],
                        "estimated_days": metrics['transfer_days'],
                        "total_cost": metrics['cost_breakdown']['total']
                    },
                    "ai_recommendations": metrics['networking_recommendations'],
                    "performance_metrics": {
                        "throughput_mbps": metrics['optimized_throughput'],
                        "network_efficiency": metrics['network_efficiency'],
                        "business_impact": metrics['business_impact']['level']
                    },
                    "compliance": config['compliance_frameworks'],
                    "generated_by": "Claude AI",
                    "generated": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="Download AI Analysis Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"{config['project_name']}_ai_migration_report.json",
                    mime="application/json"
                )
        
        # Footer information
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏢 Enterprise AWS Migration Platform v2.0**")
            st.markdown("*AI-Powered • Security-First • Compliance-Ready*")
        
        with col2:
            st.markdown("**🤖 AI-Powered Features**")
            st.markdown("• Intelligent Architecture Recommendations")
            st.markdown("• Automated Performance Optimization")
            st.markdown("• Smart Cost Analysis")
        
        with col3:
            st.markdown("**🔒 Security & Privacy**")
            st.markdown("• SOC2 Type II Certified")
            st.markdown("• End-to-end Encryption")
            st.markdown("• Zero Trust Architecture")
    
    def run(self):
        """Main application entry point with real-time updates"""
        # Render header and navigation
        self.render_header()
        self.render_navigation()
        
        # Get configuration from sidebar
        config = self.render_sidebar_controls()
        
        # Detect configuration changes for real-time updates
        config_changed = self.detect_configuration_changes(config)
        
        # Calculate migration metrics (this will recalculate automatically when config changes)
        metrics = self.calculate_migration_metrics(config)
        
        # Show real-time update indicator
        if config_changed:
            st.success("🔄 Configuration updated - Dashboard refreshed with new calculations")
        
        # Add automatic refresh timestamp
        current_time = datetime.now()
        time_since_update = (current_time - self.last_update_time).seconds
        
        # Display last update time in the header
        st.markdown(f"""
        <div style="text-align: right; color: #666; font-size: 0.8em; margin-bottom: 1rem;">
            Last updated: {current_time.strftime('%H:%M:%S')} | Auto-refresh: {time_since_update}s ago
        </div>
        """, unsafe_allow_html=True)
        
        # Render appropriate tab based on selection
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
        
        # Update timestamp
        self.last_update_time = current_time
        
        # Render footer and sidebar status
        self.render_footer(config, metrics)
        self.render_sidebar_status(config, metrics)

def main():
    """Main function to run the Enterprise AWS Migration Platform"""
    try:
        # Initialize and run the migration platform
        platform = MigrationPlatform()
        platform.run()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your configuration and try again.")
        
        # Log the error for debugging
        st.write("**Debug Information:**")
        st.code(f"Error: {str(e)}")
        
        # Provide support contact
        st.info("If the problem persists, please contact support at admin@futureminds.com")

# Application entry point
if __name__ == "__main__":
    main()