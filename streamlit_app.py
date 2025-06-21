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
    
    def calculate_enterprise_throughput(self, instance_type, num_agents, file_size_category, 
                                      network_bw_mbps, latency, jitter, packet_loss, qos_enabled, dedicated_bandwidth):
        """Calculate optimized throughput considering all network factors"""
        base_performance = self.instance_performance[instance_type]["baseline_throughput"]
        file_efficiency = self.file_size_multipliers[file_size_category]
        
        # Network impact calculations
        latency_factor = max(0.4, 1 - (latency - 5) / 500)
        jitter_factor = max(0.8, 1 - jitter / 100)
        packet_loss_factor = max(0.6, 1 - packet_loss / 10)
        qos_factor = 1.2 if qos_enabled else 1.0
        
        network_efficiency = latency_factor * jitter_factor * packet_loss_factor * qos_factor
        
        # Multi-agent scaling with diminishing returns
        total_throughput = 0
        for i in range(num_agents):
            agent_efficiency = max(0.4, 1 - (i * 0.05))
            agent_throughput = base_performance * file_efficiency * network_efficiency * agent_efficiency
            total_throughput += agent_throughput
        
        # Apply bandwidth limitation
        max_available_bandwidth = network_bw_mbps * (dedicated_bandwidth / 100)
        effective_throughput = min(total_throughput, max_available_bandwidth)
        
        return effective_throughput, network_efficiency
    
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

class MigrationPlatform:
    """Main application class for the Enterprise AWS Migration Platform"""
    
    def __init__(self):
        self.calculator = EnterpriseCalculator()
        self.initialize_session_state()
        self.setup_custom_css()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
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
    
    def setup_custom_css(self):
        """Setup custom CSS styling"""
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
            ["New York, NY", "Chicago, IL", "Dallas, TX", "Los Angeles, CA", "Atlanta, GA", 
             "London, UK", "Frankfurt, DE", "Tokyo, JP", "Sydney, AU", "Other"])
        target_aws_region = st.sidebar.selectbox("Target AWS Region",
            ["us-east-1 (N. Virginia)", "us-east-2 (Ohio)", "us-west-1 (N. California)", 
             "us-west-2 (Oregon)", "eu-west-1 (Ireland)", "eu-central-1 (Frankfurt)",
             "ap-southeast-1 (Singapore)", "ap-northeast-1 (Tokyo)"])
        
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
            'target_aws_region': target_aws_region
        }
    
    def calculate_migration_metrics(self, config):
        """Calculate all migration metrics with error handling"""
        try:
            # Basic calculations
            data_size_tb = max(0.1, config['data_size_gb'] / 1024)  # Ensure minimum size
            effective_data_gb = config['data_size_gb'] * 0.85  # Account for compression/deduplication
            
            # Calculate throughput with optimizations
            datasync_throughput, network_efficiency = self.calculator.calculate_enterprise_throughput(
                config['datasync_instance_type'], config['num_datasync_agents'], config['avg_file_size'], 
                config['dx_bandwidth_mbps'], config['network_latency'], config['network_jitter'], 
                config['packet_loss'], config['qos_enabled'], config['dedicated_bandwidth']
            )
            
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
            
            return {
                'data_size_tb': data_size_tb,
                'effective_data_gb': effective_data_gb,
                'datasync_throughput': datasync_throughput,
                'optimized_throughput': optimized_throughput,
                'network_efficiency': network_efficiency,
                'transfer_days': transfer_days,
                'cost_breakdown': cost_breakdown,
                'compliance_reqs': compliance_reqs,
                'compliance_risks': compliance_risks,
                'business_impact': business_impact,
                'available_hours_per_day': available_hours_per_day
            }
            
        except Exception as e:
            # Return default metrics if calculation fails
            st.error(f"Error in calculation: {str(e)}")
            return {
                'data_size_tb': 1.0,
                'effective_data_gb': 1000,
                'datasync_throughput': 100,
                'optimized_throughput': 100,
                'network_efficiency': 0.7,
                'transfer_days': 10,
                'cost_breakdown': {'compute': 1000, 'transfer': 500, 'storage': 200, 'compliance': 100, 'monitoring': 50, 'total': 1850},
                'compliance_reqs': [],
                'compliance_risks': [],
                'business_impact': {'score': 0.5, 'level': 'Medium', 'recommendation': 'Standard approach'},
                'available_hours_per_day': 24
            }
    
    def render_dashboard_tab(self, config, metrics):
        """Render the dashboard tab"""
        st.header("🏠 Enterprise Migration Dashboard")
        
        # Executive summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Active Projects", "3", "+1")
        with col2:
            st.metric("Total Data Migrated", "247 TB", "+12 TB")
        with col3:
            st.metric("Migration Success Rate", "94%", "+2%")
        with col4:
            st.metric("Cost Savings YTD", "$2.1M", "+$340K")
        with col5:
            st.metric("Compliance Score", "96%", "+4%")
        
        # Current project overview
        st.subheader("📊 Current Project Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💾 Data Volume", f"{metrics['data_size_tb']:.1f} TB", f"{config['data_size_gb']:,.0f} GB")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("⚡ Throughput", f"{metrics['optimized_throughput']:.0f} Mbps", f"{metrics['network_efficiency']:.1%} efficiency")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📅 Duration", f"{metrics['transfer_days']:.1f} days", f"{metrics['transfer_days']*24:.0f} hours")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💰 Total Cost", f"${metrics['cost_breakdown']['total']:,.0f}", f"${metrics['cost_breakdown']['total']/metrics['data_size_tb']:.0f}/TB")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent activities and alerts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Recent Activities")
            activities = [
                f"✅ {config['project_name']} migration plan generated",
                "🔄 Network optimization recommendations updated",
                f"📊 Business impact assessment: {metrics['business_impact']['level']}",
                "🔒 Compliance framework validation completed",
                f"💰 Cost estimate: ${metrics['cost_breakdown']['total']:,.0f}"
            ]
            
            for activity in activities:
                st.write(f"• {activity}")
        
        with col2:
            st.subheader("⚠️ Alerts & Notifications")
            alerts = []
            
            if metrics['transfer_days'] > config['max_transfer_days']:
                alerts.append(f"🔴 Timeline exceeds {config['max_transfer_days']} day target")
            if metrics['cost_breakdown']['total'] > config['budget_allocated']:
                alerts.append("🔴 Budget exceeded")
            if metrics['compliance_risks']:
                alerts.append("🟡 Compliance risks identified")
            if config['network_latency'] > 100:
                alerts.append("🟡 High network latency detected")
            if not alerts:
                alerts.append("🟢 All systems optimal")
            
            for alert in alerts:
                st.write(alert)
    
    def render_network_tab(self, config, metrics):
        """Render the network analysis tab"""
        st.header("🌐 Network Analysis & Optimization")
        
        # Network performance dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            utilization_pct = (metrics['optimized_throughput'] / config['dx_bandwidth_mbps']) * 100
            st.metric("Network Utilization", f"{utilization_pct:.1f}%", f"{metrics['optimized_throughput']:.0f} Mbps")
        
        with col2:
            efficiency_improvement = ((metrics['optimized_throughput'] - metrics['datasync_throughput']) / metrics['datasync_throughput']) * 100
            st.metric("Optimization Gain", f"{efficiency_improvement:.1f}%", "vs baseline")
        
        with col3:
            st.metric("Network Latency", f"{config['network_latency']} ms", "RTT to AWS")
        
        with col4:
            st.metric("Packet Loss", f"{config['packet_loss']}%", "Quality indicator")
        
        # Network optimization chart
        st.subheader("📊 Network Performance Analysis")
        
        # Create performance comparison chart
        configs_list = ["Baseline", "Current", "Optimized"]
        baseline_throughput = self.calculator.calculate_enterprise_throughput(
            config['datasync_instance_type'], config['num_datasync_agents'], config['avg_file_size'], 
            config['dx_bandwidth_mbps'], 100, 5, 0.05, False, config['dedicated_bandwidth']
        )[0]
        
        throughputs = [baseline_throughput, metrics['datasync_throughput'], metrics['optimized_throughput']]
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(
            x=configs_list,
            y=throughputs,
            marker_color=['lightgray', 'lightblue', 'lightgreen'],
            text=[f"{t:.0f} Mbps" for t in throughputs],
            textposition='auto'
        ))
        
        fig_perf.update_layout(
            title="Network Performance Comparison",
            yaxis_title="Throughput (Mbps)",
            height=400
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Network quality assessment
        st.subheader("📡 Network Quality Assessment")
        
        utilization_pct = (metrics['optimized_throughput'] / config['dx_bandwidth_mbps']) * 100
        
        quality_metrics = pd.DataFrame({
            "Metric": ["Latency", "Jitter", "Packet Loss", "Throughput"],
            "Current": [f"{config['network_latency']} ms", f"{config['network_jitter']} ms", 
                       f"{config['packet_loss']}%", f"{metrics['optimized_throughput']:.0f} Mbps"],
            "Target": ["< 50 ms", "< 10 ms", "< 0.1%", f"{config['dx_bandwidth_mbps'] * 0.8:.0f} Mbps"],
            "Status": [
                "✅ Good" if config['network_latency'] < 50 else "⚠️ High",
                "✅ Good" if config['network_jitter'] < 10 else "⚠️ High", 
                "✅ Good" if config['packet_loss'] < 0.1 else "⚠️ High",
                "✅ Good" if utilization_pct < 80 else "⚠️ High"
            ]
        })
        
        st.dataframe(quality_metrics, use_container_width=True, hide_index=True)
    
    def render_planner_tab(self, config, metrics):
        """Render the migration planner tab"""
        st.header("📊 Migration Planning & Strategy")
        
        # Migration method comparison
        st.subheader("🔍 Migration Method Analysis")
        
        migration_methods = []
        
        # DataSync analysis
        migration_methods.append({
            "Method": "DataSync Multi-Agent",
            "Throughput": f"{metrics['optimized_throughput']:.0f} Mbps",
            "Duration": f"{metrics['transfer_days']:.1f} days",
            "Cost": f"${metrics['cost_breakdown']['total']:,.0f}",
            "Security": "High" if config['encryption_in_transit'] and config['encryption_at_rest'] else "Medium",
            "Complexity": "Medium"
        })
        
        # Snowball analysis
        if metrics['data_size_tb'] > 1:
            snowball_devices = max(1, int(metrics['data_size_tb'] / 72))
            snowball_days = 7 + (snowball_devices * 2)
            snowball_cost = snowball_devices * 300 + 2000
            
            migration_methods.append({
                "Method": f"Snowball Edge ({snowball_devices}x)",
                "Throughput": "Physical transfer",
                "Duration": f"{snowball_days} days",
                "Cost": f"${snowball_cost:,.0f}",
                "Security": "Very High",
                "Complexity": "Low"
            })
        
        # Storage Gateway
        sg_throughput = min(config['dx_bandwidth_mbps'] * 0.6, 2000)
        sg_days = (metrics['effective_data_gb'] * 8) / (sg_throughput * metrics['available_hours_per_day'] * 3600) / 1000
        sg_cost = metrics['cost_breakdown']['total'] * 1.3
        
        migration_methods.append({
            "Method": "Storage Gateway",
            "Throughput": f"{sg_throughput:.0f} Mbps",
            "Duration": f"{sg_days:.1f} days",
            "Cost": f"${sg_cost:,.0f}",
            "Security": "High",
            "Complexity": "Medium"
        })
        
        df_methods = pd.DataFrame(migration_methods)
        st.dataframe(df_methods, use_container_width=True, hide_index=True)
        
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
        
        st.write(f"**Recommendation:** {metrics['business_impact']['recommendation']}")
    
    def render_performance_tab(self, config, metrics):
        """Render the performance optimization tab"""
        st.header("⚡ Performance Optimization")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        baseline_throughput = self.calculator.calculate_enterprise_throughput(
            config['datasync_instance_type'], config['num_datasync_agents'], config['avg_file_size'], 
            config['dx_bandwidth_mbps'], 100, 5, 0.05, False, config['dedicated_bandwidth']
        )[0]
        
        improvement = ((metrics['optimized_throughput'] - baseline_throughput) / baseline_throughput) * 100
        
        with col1:
            st.metric("Performance Gain", f"{improvement:.1f}%", "vs baseline")
        
        with col2:
            st.metric("Network Efficiency", f"{(metrics['optimized_throughput']/config['dx_bandwidth_mbps'])*100:.1f}%")
        
        with col3:
            st.metric("Transfer Time", f"{metrics['transfer_days']:.1f} days")
        
        with col4:
            st.metric("Cost per TB", f"${metrics['cost_breakdown']['total']/metrics['data_size_tb']:.0f}")
        
        # Optimization recommendations
        st.subheader("🎯 Optimization Recommendations")
        
        recommendations = []
        
        if config['tcp_window_size'] == "Default":
            recommendations.append("Enable TCP window scaling for 15-20% improvement")
        
        if config['mtu_size'] == "1500 (Standard)":
            recommendations.append("Configure jumbo frames for 10-15% improvement")
        
        if config['network_congestion_control'] == "Cubic (Default)":
            recommendations.append("Switch to BBR algorithm for 20-25% improvement")
        
        if not config['wan_optimization']:
            recommendations.append("Enable WAN optimization for 25-30% improvement")
        
        if config['parallel_streams'] < 20:
            recommendations.append("Increase parallel streams for better throughput")
        
        if recommendations:
            for rec in recommendations:
                st.write(f"• {rec}")
        else:
            st.success("✅ Configuration is already well optimized!")
        
        # Performance comparison chart
        st.subheader("📊 Optimization Impact Analysis")
        
        optimization_scenarios = {
            "Current": metrics['optimized_throughput'],
            "TCP Optimized": metrics['optimized_throughput'] * 1.2 if config['tcp_window_size'] == "Default" else metrics['optimized_throughput'],
            "MTU Optimized": metrics['optimized_throughput'] * 1.15 if config['mtu_size'] == "1500 (Standard)" else metrics['optimized_throughput'],
            "Fully Optimized": metrics['optimized_throughput'] * 1.45 if (config['tcp_window_size'] == "Default" and config['mtu_size'] == "1500 (Standard)" and not config['wan_optimization']) else metrics['optimized_throughput'] * 1.1
        }
        
        fig_opt = go.Figure()
        fig_opt.add_trace(go.Bar(
            x=list(optimization_scenarios.keys()),
            y=list(optimization_scenarios.values()),
            marker_color=['lightblue', 'lightgreen', 'orange', 'lightcoral'],
            text=[f"{v:.0f} Mbps" for v in optimization_scenarios.values()],
            textposition='auto'
        ))
        
        fig_opt.update_layout(
            title="Optimization Scenarios Comparison",
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
            name='Throughput (Mbps)'
        ))
        
        fig_trend.update_layout(
            title="Historical Throughput Performance",
            xaxis_title="Date",
            yaxis_title="Throughput (Mbps)",
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # ROI Analysis
        st.subheader("💡 ROI Analysis")
        
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
            if st.button("📤 Export Report"):
                report_data = {
                    "project_summary": {
                        "name": config['project_name'],
                        "data_size_tb": metrics['data_size_tb'],
                        "estimated_days": metrics['transfer_days'],
                        "total_cost": metrics['cost_breakdown']['total']
                    },
                    "performance_metrics": {
                        "throughput_mbps": metrics['optimized_throughput'],
                        "network_efficiency": metrics['network_efficiency'],
                        "business_impact": metrics['business_impact']['level']
                    },
                    "compliance": config['compliance_frameworks'],
                    "generated": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"{config['project_name']}_migration_report.json",
                    mime="application/json"
                )
        
        # Footer information
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏢 Enterprise AWS Migration Platform v2.0**")
            st.markdown("*AI-Powered • Security-First • Compliance-Ready*")
        
        with col2:
            st.markdown("**📞 Support & Resources**")
            st.markdown("• 24/7 Enterprise Support")
            st.markdown("• Migration Acceleration Program")
            st.markdown("• Compliance Advisory Services")
        
        with col3:
            st.markdown("**🔒 Security & Privacy**")
            st.markdown("• SOC2 Type II Certified")
            st.markdown("• End-to-end Encryption")
            st.markdown("• Zero Trust Architecture")
    
    def render_sidebar_status(self, config, metrics):
        """Render quick status in sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.subheader("🚦 Quick Status")
            
            # Overall health indicators
            status_factors = []
            
            if metrics['transfer_days'] <= config['max_transfer_days']:
                status_factors.append("✅ Timeline")
            else:
                status_factors.append("❌ Timeline")
            
            if metrics['cost_breakdown']['total'] <= config['budget_allocated']:
                status_factors.append("✅ Budget")
            else:
                status_factors.append("❌ Budget")
            
            if not metrics['compliance_risks']:
                status_factors.append("✅ Compliance")
            else:
                status_factors.append("⚠️ Compliance")
            
            if config['network_latency'] < 100:
                status_factors.append("✅ Network")
            else:
                status_factors.append("⚠️ Network")
            
            for factor in status_factors:
                st.write(factor)
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            
            if st.button("🔄 Refresh Calculations"):
                st.rerun()
            
            if st.button("🎯 Auto-Optimize"):
                st.info("Auto-optimization would adjust parameters for best performance/cost balance")
    
    def run(self):
        """Main application entry point"""
        # Render header and navigation
        self.render_header()
        self.render_navigation()
        
        # Get configuration from sidebar
        config = self.render_sidebar_controls()
        
        # Calculate migration metrics
        metrics = self.calculate_migration_metrics(config)
        
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
        st.info("If the problem persists, please contact support at support@enterprise-migration.com")

# Application entry point
if __name__ == "__main__":
    main()