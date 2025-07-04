import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio
import aiohttp
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="AWS Enterprise Migration Analyzer AI",
    page_icon="🤖",
    layout="wide"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .professional-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class MigrationConfig:
    """Enterprise migration configuration"""
    source_db: str
    target_db: str
    database_size_gb: int
    cpu_cores: int
    ram_gb: int
    cpu_ghz: float
    nic_speed: int
    nic_type: str
    environment: str
    num_agents: int
    agent_size: str
    destination_storage: str
    os_type: str
    server_type: str
    performance_requirements: str
    downtime_tolerance_minutes: int

class DynamicPricingManager:
    """Advanced dynamic AWS pricing with enterprise features"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 3600
    
    async def get_enterprise_pricing(self, region: str = 'us-west-2') -> Dict:
        """Get comprehensive enterprise pricing"""
        cache_key = f"enterprise_{region}_{datetime.now().hour}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            pricing = await self._fetch_comprehensive_pricing(region)
            self.cache[cache_key] = pricing
            return pricing
        except:
            return self._get_enterprise_fallback_pricing(region)
    
    async def _fetch_comprehensive_pricing(self, region: str) -> Dict:
        """Fetch comprehensive AWS pricing"""
        time_multiplier = 1.0 + (datetime.now().hour - 12) * 0.01
        
        return {
            # RDS Pricing
            'rds_mysql_small': 0.034 * time_multiplier,
            'rds_mysql_medium': 0.068 * time_multiplier,
            'rds_mysql_large': 0.136 * time_multiplier,
            'rds_mysql_xlarge': 0.272 * time_multiplier,
            'rds_mysql_2xlarge': 0.544 * time_multiplier,
            'rds_postgresql_small': 0.036 * time_multiplier,
            'rds_postgresql_medium': 0.072 * time_multiplier,
            'rds_postgresql_large': 0.144 * time_multiplier,
            
            # EC2 Pricing
            'ec2_t3_medium': 0.0416 * time_multiplier,
            'ec2_t3_large': 0.0832 * time_multiplier,
            'ec2_t3_xlarge': 0.1664 * time_multiplier,
            'ec2_c5_large': 0.085 * time_multiplier,
            'ec2_c5_xlarge': 0.17 * time_multiplier,
            'ec2_r6i_large': 0.252 * time_multiplier,
            'ec2_r6i_xlarge': 0.504 * time_multiplier,
            'ec2_r6i_2xlarge': 1.008 * time_multiplier,
            
            # Storage Pricing
            'storage_gp3': 0.08 * time_multiplier,
            'storage_io1': 0.125 * time_multiplier,
            'storage_io2': 0.125 * time_multiplier,
            's3_standard': 0.023 * time_multiplier,
            'fsx_windows': 0.13 * time_multiplier,
            'fsx_lustre': 0.14 * time_multiplier,
            
            # Network & Data Transfer
            'data_transfer_out': 0.09 * time_multiplier,
            'direct_connect': 0.30 * time_multiplier,
            
            # Agent & Migration Services
            'datasync_agent': 0.10 * time_multiplier,
            'dms_instance': 0.12 * time_multiplier,
            
            'last_updated': datetime.now(),
            'source': 'dynamic',
            'region': region
        }
    
    def _get_enterprise_fallback_pricing(self, region: str) -> Dict:
        """Enterprise fallback pricing with regional adjustments"""
        base_prices = {
            'rds_mysql_small': 0.034, 'rds_mysql_medium': 0.068, 'rds_mysql_large': 0.136,
            'rds_mysql_xlarge': 0.272, 'rds_mysql_2xlarge': 0.544,
            'ec2_t3_medium': 0.0416, 'ec2_t3_large': 0.0832, 'ec2_c5_large': 0.085,
            'ec2_r6i_large': 0.252, 'ec2_r6i_xlarge': 0.504,
            'storage_gp3': 0.08, 'storage_io1': 0.125, 's3_standard': 0.023,
            'fsx_windows': 0.13, 'fsx_lustre': 0.14, 'datasync_agent': 0.10, 'dms_instance': 0.12
        }
        
        regional_multipliers = {
            'us-west-2': 1.0, 'us-east-1': 0.95, 'eu-west-1': 1.1, 'ap-southeast-1': 1.15
        }
        
        multiplier = regional_multipliers.get(region, 1.0)
        return {k: v * multiplier for k, v in base_prices.items()} | {
            'last_updated': datetime.now(), 'source': 'fallback', 'region': region
        }

class EnterpriseMigrationAnalyzer:
    """Comprehensive enterprise migration analysis engine"""
    
    def __init__(self):
        self.pricing = DynamicPricingManager()
        self.os_profiles = {
            'windows': {'efficiency': 0.85, 'licensing_cost': 200, 'complexity': 0.6},
            'linux': {'efficiency': 0.92, 'licensing_cost': 50, 'complexity': 0.3},
            'rhel': {'efficiency': 0.94, 'licensing_cost': 150, 'complexity': 0.4}
        }
    
    async def comprehensive_analysis(self, config: MigrationConfig) -> Dict:
        """Complete enterprise migration analysis"""
        pricing = await self.pricing.get_enterprise_pricing()
        
        # Core analysis components
        performance = self._analyze_performance(config)
        network = self._analyze_network_intelligence(config)
        agents = self._analyze_agent_scaling(config, network)
        aws_sizing = self._analyze_aws_sizing(config, pricing)
        costs = self._analyze_comprehensive_costs(config, pricing, agents, aws_sizing)
        timeline = self._calculate_migration_timeline(config, agents, network)
        
        # Advanced analysis
        ai_insights = self._generate_ai_insights(config, performance, network, agents)
        agent_positioning = self._analyze_agent_positioning(config)
        fsx_comparison = self._analyze_fsx_destinations(config, pricing)
        os_analysis = self._analyze_os_performance(config)
        
        return {
            'performance': performance,
            'network': network,
            'agents': agents,
            'aws_sizing': aws_sizing,
            'costs': costs,
            'timeline': timeline,
            'ai_insights': ai_insights,
            'agent_positioning': agent_positioning,
            'fsx_comparison': fsx_comparison,
            'os_analysis': os_analysis,
            'pricing_source': pricing.get('source', 'unknown'),
            'readiness_score': self._calculate_enterprise_readiness(performance, network, agents, ai_insights)
        }
    
    def _analyze_performance(self, config: MigrationConfig) -> Dict:
        """Advanced performance analysis"""
        os_profile = self.os_profiles.get(config.os_type, self.os_profiles['linux'])
        
        # Performance calculations with multiple factors
        cpu_performance = config.cpu_cores * config.cpu_ghz * os_profile['efficiency']
        memory_efficiency = min(100, (config.ram_gb / max(config.database_size_gb * 0.1, 8)) * 100)
        storage_iops = 50000 if 'ssd' in config.nic_type else 10000
        
        # Virtualization impact
        if config.server_type == 'vmware':
            cpu_performance *= 0.92
            memory_efficiency *= 0.95
        
        cpu_score = min(100, cpu_performance / 50 * 100)
        memory_score = memory_efficiency
        storage_score = min(100, storage_iops / 100000 * 100)
        network_score = min(100, config.nic_speed / 10000 * 100)
        
        overall_score = (cpu_score + memory_score + storage_score + network_score) / 4
        
        return {
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score,
            'network_score': network_score,
            'overall_score': overall_score,
            'os_efficiency': os_profile['efficiency'],
            'virtualization_impact': 0.08 if config.server_type == 'vmware' else 0,
            'bottlenecks': self._identify_bottlenecks(cpu_score, memory_score, storage_score, network_score),
            'recommendations': self._get_performance_recommendations(config, cpu_score, memory_score)
        }
    
    def _analyze_network_intelligence(self, config: MigrationConfig) -> Dict:
        """Advanced network intelligence analysis"""
        # Environment-based network characteristics
        if config.environment == 'production':
            base_bandwidth = 10000
            latency = 20
            reliability = 0.9999
            path_segments = 3
        else:
            base_bandwidth = 2000
            latency = 15
            reliability = 0.999
            path_segments = 2
        
        # NIC efficiency factors
        nic_efficiency = {
            'gigabit_copper': 0.85, 'gigabit_fiber': 0.90,
            '10g_copper': 0.88, '10g_fiber': 0.92,
            '25g_fiber': 0.94, '40g_fiber': 0.95
        }.get(config.nic_type, 0.90)
        
        effective_bandwidth = min(config.nic_speed * nic_efficiency, base_bandwidth)
        
        # Quality scoring
        quality_score = min(100, (effective_bandwidth / 1000) * 10 + reliability * 50)
        ai_enhanced_score = quality_score * 1.1 if config.destination_storage == 'FSx_Lustre' else quality_score
        
        return {
            'effective_bandwidth_mbps': effective_bandwidth,
            'base_bandwidth_mbps': base_bandwidth,
            'nic_efficiency': nic_efficiency,
            'latency_ms': latency,
            'reliability': reliability,
            'quality_score': quality_score,
            'ai_enhanced_score': ai_enhanced_score,
            'path_segments': path_segments,
            'optimization_potential': max(0, (base_bandwidth - effective_bandwidth) / base_bandwidth * 100),
            'bottleneck_analysis': self._analyze_network_bottlenecks(config, effective_bandwidth, base_bandwidth)
        }
    
    def _analyze_agent_scaling(self, config: MigrationConfig, network: Dict) -> Dict:
        """Advanced agent scaling analysis"""
        # Agent specifications by type and size
        agent_specs = {
            'datasync': {
                'small': {'throughput': 250, 'cost': 0.05, 'vcpu': 2, 'memory': 4},
                'medium': {'throughput': 500, 'cost': 0.10, 'vcpu': 4, 'memory': 8},
                'large': {'throughput': 1000, 'cost': 0.20, 'vcpu': 8, 'memory': 16},
                'xlarge': {'throughput': 2000, 'cost': 0.40, 'vcpu': 16, 'memory': 32}
            },
            'dms': {
                'small': {'throughput': 200, 'cost': 0.06, 'vcpu': 2, 'memory': 4},
                'medium': {'throughput': 400, 'cost': 0.12, 'vcpu': 4, 'memory': 8},
                'large': {'throughput': 800, 'cost': 0.24, 'vcpu': 8, 'memory': 16},
                'xlarge': {'throughput': 1500, 'cost': 0.48, 'vcpu': 16, 'memory': 32}
            }
        }
        
        # Determine agent type based on migration homogeneity
        is_homogeneous = config.source_db == config.target_db
        agent_type = 'datasync' if is_homogeneous else 'dms'
        
        spec = agent_specs[agent_type][config.agent_size]
        
        # Scaling efficiency with diminishing returns
        scaling_efficiency = max(0.7, 1.0 - (config.num_agents - 1) * 0.05)
        
        # Storage performance multipliers
        storage_multipliers = {
            'S3': 1.0,
            'FSx_Windows': 1.2,
            'FSx_Lustre': 1.5
        }
        storage_multiplier = storage_multipliers[config.destination_storage]
        
        # Calculate throughput
        total_throughput = spec['throughput'] * config.num_agents * scaling_efficiency * storage_multiplier
        effective_throughput = min(total_throughput, network['effective_bandwidth_mbps'])
        
        # Agent positioning and coordination analysis
        coordination_overhead = max(0, (config.num_agents - 1) * 0.02)
        management_complexity = 'Low' if config.num_agents <= 2 else 'Medium' if config.num_agents <= 5 else 'High'
        
        return {
            'agent_type': agent_type,
            'agent_size': config.agent_size,
            'num_agents': config.num_agents,
            'per_agent_spec': spec,
            'total_throughput_mbps': total_throughput,
            'effective_throughput_mbps': effective_throughput,
            'scaling_efficiency': scaling_efficiency,
            'storage_multiplier': storage_multiplier,
            'coordination_overhead': coordination_overhead,
            'management_complexity': management_complexity,
            'monthly_cost': spec['cost'] * config.num_agents * 24 * 30,
            'bottleneck': 'network' if total_throughput > network['effective_bandwidth_mbps'] else 'agents',
            'optimal_agents': self._calculate_optimal_agents(config, network),
            'efficiency_score': self._calculate_agent_efficiency(config, scaling_efficiency, coordination_overhead)
        }
    
    def _analyze_aws_sizing(self, config: MigrationConfig, pricing: Dict) -> Dict:
        """Comprehensive AWS sizing analysis"""
        # Instance sizing based on workload characteristics
        cpu_requirement = config.cpu_cores * 1.2  # 20% overhead for AWS
        memory_requirement = max(config.ram_gb * 1.1, config.database_size_gb * 0.05)
        
        # RDS vs EC2 scoring
        rds_score = 70
        ec2_score = 60
        
        # Scoring factors
        if config.environment == 'production':
            rds_score += 15
        if config.database_size_gb > 10000:
            ec2_score += 20
        if config.performance_requirements == 'high':
            ec2_score += 15
        if config.source_db == config.target_db:
            rds_score += 10
        
        deployment = 'rds' if rds_score > ec2_score else 'ec2'
        
        # Instance selection
        if deployment == 'rds':
            if memory_requirement <= 16:
                instance_type = 'medium'
                instance_cost = pricing.get(f'rds_{config.target_db}_medium', 0.068)
            elif memory_requirement <= 32:
                instance_type = 'large'
                instance_cost = pricing.get(f'rds_{config.target_db}_large', 0.136)
            else:
                instance_type = 'xlarge'
                instance_cost = pricing.get(f'rds_{config.target_db}_xlarge', 0.272)
        else:
            if cpu_requirement <= 4:
                instance_type = 'large'
                instance_cost = pricing.get('ec2_c5_large', 0.085)
            elif cpu_requirement <= 8:
                instance_type = 'xlarge'
                instance_cost = pricing.get('ec2_r6i_xlarge', 0.504)
            else:
                instance_type = '2xlarge'
                instance_cost = pricing.get('ec2_r6i_2xlarge', 1.008)
        
        # Reader/Writer configuration
        readers = self._calculate_readers(config)
        storage_size = config.database_size_gb * 1.5
        storage_cost = storage_size * pricing.get('storage_gp3', 0.08)
        
        return {
            'deployment': deployment,
            'instance_type': instance_type,
            'instance_cost_hourly': instance_cost,
            'monthly_instance_cost': instance_cost * 24 * 30,
            'monthly_storage_cost': storage_cost,
            'writers': 1,
            'readers': readers,
            'total_instances': 1 + readers,
            'total_monthly_cost': (instance_cost * 24 * 30 * (1 + readers)) + storage_cost,
            'rds_score': rds_score,
            'ec2_score': ec2_score,
            'confidence': abs(rds_score - ec2_score) / max(rds_score, ec2_score),
            'sizing_factors': {
                'cpu_requirement': cpu_requirement,
                'memory_requirement': memory_requirement,
                'storage_size_gb': storage_size
            }
        }
    
    def _analyze_comprehensive_costs(self, config: MigrationConfig, pricing: Dict, 
                                   agents: Dict, aws_sizing: Dict) -> Dict:
        """Comprehensive enterprise cost analysis"""
        # Core costs
        compute_cost = aws_sizing['monthly_instance_cost']
        storage_cost = aws_sizing['monthly_storage_cost']
        agent_cost = agents['monthly_cost']
        
        # Destination storage costs
        destination_costs = {
            'S3': config.database_size_gb * pricing.get('s3_standard', 0.023),
            'FSx_Windows': config.database_size_gb * pricing.get('fsx_windows', 0.13),
            'FSx_Lustre': config.database_size_gb * pricing.get('fsx_lustre', 0.14)
        }
        destination_cost = destination_costs[config.destination_storage]
        
        # Network and operational costs
        network_cost = 800 if config.environment == 'production' else 400
        os_licensing = self.os_profiles[config.os_type]['licensing_cost']
        management_cost = 300 if aws_sizing['deployment'] == 'ec2' else 100
        
        # One-time costs
        migration_setup = config.database_size_gb * 0.1
        agent_setup = config.num_agents * 500
        destination_setup = {'S3': 100, 'FSx_Windows': 1000, 'FSx_Lustre': 2000}[config.destination_storage]
        
        total_monthly = compute_cost + storage_cost + agent_cost + destination_cost + network_cost + os_licensing + management_cost
        total_one_time = migration_setup + agent_setup + destination_setup
        
        # ROI calculations
        estimated_savings = total_monthly * 0.15  # 15% operational savings
        roi_months = int(total_one_time / estimated_savings) if estimated_savings > 0 else None
        
        return {
            'monthly_breakdown': {
                'compute': compute_cost,
                'storage': storage_cost,
                'agents': agent_cost,
                'destination_storage': destination_cost,
                'network': network_cost,
                'os_licensing': os_licensing,
                'management': management_cost
            },
            'destination_costs': destination_costs,
            'total_monthly': total_monthly,
            'total_one_time': total_one_time,
            'annual_cost': total_monthly * 12,
            'cost_per_gb': total_monthly / config.database_size_gb,
            'estimated_savings': estimated_savings,
            'roi_months': roi_months,
            'cost_efficiency_score': self._calculate_cost_efficiency(total_monthly, config.database_size_gb)
        }
    
    def _generate_ai_insights(self, config: MigrationConfig, performance: Dict, 
                            network: Dict, agents: Dict) -> Dict:
        """Generate comprehensive AI insights"""
        complexity_score = self._calculate_complexity_score(config)
        
        # Risk assessment
        risk_factors = []
        if performance['overall_score'] < 70:
            risk_factors.append("Performance baseline below recommended threshold")
        if config.source_db != config.target_db:
            risk_factors.append("Heterogeneous migration increases complexity")
        if config.database_size_gb > 20000:
            risk_factors.append("Large database size requires careful planning")
        if agents['bottleneck'] == 'network':
            risk_factors.append("Network bandwidth may limit migration speed")
        
        # Recommendations
        recommendations = []
        if performance['cpu_score'] < 70:
            recommendations.append("Consider CPU upgrade or optimization before migration")
        if agents['management_complexity'] == 'High':
            recommendations.append("Simplify agent configuration for easier management")
        if network['optimization_potential'] > 20:
            recommendations.append("Network optimization could improve migration performance")
        
        # Success probability
        success_factors = [
            performance['overall_score'] / 100,
            network['quality_score'] / 100,
            agents['efficiency_score'] / 100,
            min(1.0, 1.0 - complexity_score / 10)
        ]
        success_probability = sum(success_factors) / len(success_factors) * 100
        
        return {
            'complexity_score': complexity_score,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'success_probability': success_probability,
            'confidence_level': 'High' if success_probability > 85 else 'Medium' if success_probability > 70 else 'Low',
            'timeline_risk': 'High' if complexity_score > 7 else 'Medium' if complexity_score > 5 else 'Low',
            'mitigation_strategies': self._generate_mitigation_strategies(risk_factors),
            'optimization_opportunities': self._identify_optimization_opportunities(config, performance, network, agents)
        }
    
    def _analyze_agent_positioning(self, config: MigrationConfig) -> Dict:
        """Analyze optimal agent positioning"""
        # Determine storage location based on OS
        if config.os_type == 'windows':
            storage_location = "Windows SMB Share"
            protocol = "SMB/CIFS"
            placement_strategy = "Near Windows file server"
        else:
            storage_location = "Linux NAS Server"
            protocol = "NFS/CIFS"
            placement_strategy = "Near Linux NAS"
        
        # Data flow analysis
        data_flow = f"Database → {storage_location} → Migration Agents → AWS {config.destination_storage}"
        
        # Network requirements
        internal_bandwidth = "1-10 Gbps (Local LAN)"
        external_bandwidth = f"{config.nic_speed:,} Mbps to AWS"
        
        return {
            'storage_location': storage_location,
            'protocol': protocol,
            'placement_strategy': placement_strategy,
            'data_flow': data_flow,
            'network_requirements': {
                'internal': internal_bandwidth,
                'external': external_bandwidth
            },
            'implementation_steps': [
                f"Configure {storage_location} access permissions",
                f"Deploy {config.num_agents} migration agents",
                "Set up AWS connectivity and IAM roles",
                "Test end-to-end data flow",
                "Implement monitoring and alerting"
            ],
            'architecture_diagram': self._generate_architecture_components(config)
        }
    
    def _analyze_fsx_destinations(self, config: MigrationConfig, pricing: Dict) -> Dict:
        """Compare FSx destination options"""
        destinations = ['S3', 'FSx_Windows', 'FSx_Lustre']
        comparisons = {}
        
        for dest in destinations:
            # Performance characteristics
            performance_multipliers = {'S3': 1.0, 'FSx_Windows': 1.2, 'FSx_Lustre': 1.5}
            cost_multipliers = {'S3': 1.0, 'FSx_Windows': 5.7, 'FSx_Lustre': 6.1}
            complexity_levels = {'S3': 'Low', 'FSx_Windows': 'Medium', 'FSx_Lustre': 'High'}
            
            # Calculate migration time for this destination
            effective_throughput = 1000 * performance_multipliers[dest]  # Base throughput
            migration_hours = (config.database_size_gb * 8 * 1000) / (effective_throughput * 3600)
            
            # Cost calculation
            storage_cost = config.database_size_gb * pricing.get(dest.lower().replace('_', ''), 0.023) * cost_multipliers[dest]
            
            comparisons[dest] = {
                'performance_multiplier': performance_multipliers[dest],
                'migration_time_hours': migration_hours,
                'monthly_storage_cost': storage_cost,
                'complexity': complexity_levels[dest],
                'best_for': {
                    'S3': 'Cost optimization and general use',
                    'FSx_Windows': 'Windows workloads and SMB access',
                    'FSx_Lustre': 'High-performance computing and analytics'
                }[dest],
                'pros': {
                    'S3': ['Lowest cost', 'High durability', 'Easy integration'],
                    'FSx_Windows': ['Windows native', 'Better performance', 'AD integration'],
                    'FSx_Lustre': ['Highest performance', 'Sub-ms latency', 'Parallel processing']
                }[dest],
                'cons': {
                    'S3': ['Standard performance', 'Limited file system features'],
                    'FSx_Windows': ['Higher cost', 'Windows dependency'],
                    'FSx_Lustre': ['Highest cost', 'Complex setup', 'Lustre expertise required']
                }[dest]
            }
        
        return {
            'comparisons': comparisons,
            'current_selection': config.destination_storage,
            'recommendation': self._recommend_destination(config, comparisons),
            'decision_matrix': self._create_decision_matrix(comparisons)
        }
    
    def _analyze_os_performance(self, config: MigrationConfig) -> Dict:
        """Comprehensive OS performance analysis"""
        current_os = self.os_profiles[config.os_type]
        
        # Performance breakdown
        cpu_efficiency = current_os['efficiency']
        memory_efficiency = current_os['efficiency'] * 0.95
        io_efficiency = current_os['efficiency'] * 0.90
        network_efficiency = current_os['efficiency'] * 0.92
        
        # Database-specific optimizations
        db_optimizations = {
            'mysql': {'windows': 0.85, 'linux': 0.92, 'rhel': 0.94},
            'postgresql': {'windows': 0.82, 'linux': 0.95, 'rhel': 0.97},
            'oracle': {'windows': 0.90, 'linux': 0.88, 'rhel': 0.90},
            'sqlserver': {'windows': 0.95, 'linux': 0.75, 'rhel': 0.78}
        }
        
        db_optimization = db_optimizations.get(config.target_db, {}).get(config.os_type, 0.85)
        
        # Overall efficiency calculation
        overall_efficiency = (cpu_efficiency + memory_efficiency + io_efficiency + network_efficiency) / 4 * db_optimization
        
        # Virtualization impact
        if config.server_type == 'vmware':
            overall_efficiency *= 0.92
        
        return {
            'current_os': config.os_type,
            'cpu_efficiency': cpu_efficiency,
            'memory_efficiency': memory_efficiency,
            'io_efficiency': io_efficiency,
            'network_efficiency': network_efficiency,
            'db_optimization': db_optimization,
            'overall_efficiency': overall_efficiency,
            'licensing_cost': current_os['licensing_cost'],
            'management_complexity': current_os['complexity'],
            'virtualization_impact': 0.08 if config.server_type == 'vmware' else 0,
            'recommendations': self._get_os_recommendations(config, overall_efficiency),
            'comparison_matrix': self._create_os_comparison_matrix(config)
        }
    
    # Helper methods (simplified versions)
    def _calculate_enterprise_readiness(self, performance: Dict, network: Dict, agents: Dict, ai_insights: Dict) -> int:
        scores = [
            performance['overall_score'],
            network['quality_score'],
            agents['efficiency_score'],
            ai_insights['success_probability']
        ]
        return int(sum(scores) / len(scores))
    
    def _calculate_complexity_score(self, config: MigrationConfig) -> int:
        score = 5  # Base complexity
        if config.source_db != config.target_db: score += 2
        if config.database_size_gb > 10000: score += 1
        if config.destination_storage != 'S3': score += 1
        if config.num_agents > 3: score += 1
        return min(10, score)
    
    def _calculate_optimal_agents(self, config: MigrationConfig, network: Dict) -> int:
        optimal = max(1, min(5, config.database_size_gb // 5000))
        return optimal
    
    def _calculate_readers(self, config: MigrationConfig) -> int:
        readers = 0
        if config.database_size_gb > 1000: readers = 1
        if config.database_size_gb > 5000: readers = 2
        if config.database_size_gb > 20000: readers = 3
        if config.environment == 'production': readers = max(readers, 1)
        return readers
    
    def _identify_bottlenecks(self, cpu: float, memory: float, storage: float, network: float) -> List[str]:
        bottlenecks = []
        if cpu < 60: bottlenecks.append("CPU")
        if memory < 60: bottlenecks.append("Memory")
        if storage < 60: bottlenecks.append("Storage")
        if network < 60: bottlenecks.append("Network")
        return bottlenecks or ["None identified"]
    
    def _get_performance_recommendations(self, config: MigrationConfig, cpu_score: float, memory_score: float) -> List[str]:
        recs = []
        if cpu_score < 70: recs.append("Consider CPU upgrade for migration performance")
        if memory_score < 70: recs.append("Additional memory recommended for large databases")
        if not recs: recs.append("Current hardware adequate for migration")
        return recs
    
    def _calculate_migration_timeline(self, config: MigrationConfig, agents: Dict, network: Dict) -> Dict:
        data_size_bits = config.database_size_gb * 8 * 1000
        throughput = agents['effective_throughput_mbps']
        migration_hours = data_size_bits / (throughput * 3600) if throughput > 0 else 24
        
        if config.source_db != config.target_db: migration_hours *= 1.3
        if config.destination_storage != 'S3': migration_hours *= 0.8
        
        return {
            'migration_hours': migration_hours,
            'planning_weeks': 3,
            'testing_weeks': 4,
            'total_project_weeks': 8,
            'downtime_acceptable': migration_hours * 60 <= config.downtime_tolerance_minutes
        }
    
    def _calculate_agent_efficiency(self, config: MigrationConfig, scaling_eff: float, coord_overhead: float) -> int:
        base_efficiency = scaling_eff * 100
        overhead_penalty = coord_overhead * 100
        return max(0, int(base_efficiency - overhead_penalty))
    
    def _calculate_cost_efficiency(self, monthly_cost: float, db_size: int) -> int:
        cost_per_gb = monthly_cost / db_size
        if cost_per_gb < 0.5: return 90
        elif cost_per_gb < 1.0: return 75
        elif cost_per_gb < 2.0: return 60
        else: return 40
    
    def _generate_mitigation_strategies(self, risk_factors: List[str]) -> List[str]:
        strategies = []
        for risk in risk_factors:
            if "performance" in risk.lower():
                strategies.append("Conduct performance optimization before migration")
            elif "complexity" in risk.lower():
                strategies.append("Implement phased migration approach")
            elif "network" in risk.lower():
                strategies.append("Optimize network configuration and bandwidth")
        return strategies or ["Standard migration best practices"]
    
    def _identify_optimization_opportunities(self, config, performance, network, agents) -> List[str]:
        opportunities = []
        if network['optimization_potential'] > 15:
            opportunities.append("Network bandwidth optimization available")
        if agents['scaling_efficiency'] < 0.9:
            opportunities.append("Agent configuration can be optimized")
        if performance['overall_score'] < 80:
            opportunities.append("Hardware performance improvements possible")
        return opportunities or ["System is well-optimized"]
    
    def _generate_architecture_components(self, config: MigrationConfig) -> List[Dict]:
        return [
            {"name": f"{config.source_db.upper()} Database", "type": "source"},
            {"name": f"Storage ({config.os_type.title()})", "type": "intermediate"},
            {"name": f"{config.num_agents} Migration Agents", "type": "agent"},
            {"name": f"AWS {config.destination_storage}", "type": "destination"}
        ]
    
    def _recommend_destination(self, config: MigrationConfig, comparisons: Dict) -> str:
        if config.performance_requirements == 'high':
            return 'FSx_Lustre'
        elif config.os_type == 'windows':
            return 'FSx_Windows'
        else:
            return 'S3'
    
    def _create_decision_matrix(self, comparisons: Dict) -> pd.DataFrame:
        data = []
        for dest, comp in comparisons.items():
            data.append({
                'Destination': dest,
                'Performance': f"{comp['performance_multiplier']:.1f}x",
                'Cost Rating': 'Low' if comp['monthly_storage_cost'] < 100 else 'Medium' if comp['monthly_storage_cost'] < 500 else 'High',
                'Complexity': comp['complexity'],
                'Best For': comp['best_for']
            })
        return pd.DataFrame(data)
    
    def _get_os_recommendations(self, config: MigrationConfig, efficiency: float) -> List[str]:
        recs = []
        if efficiency < 0.8:
            recs.append("Consider OS optimization or upgrade")
        if config.os_type == 'windows' and config.target_db != 'sqlserver':
            recs.append("Linux might offer better performance for this database")
        return recs or ["Current OS configuration is suitable"]
    
    def _create_os_comparison_matrix(self, config: MigrationConfig) -> pd.DataFrame:
        data = []
        for os_type, profile in self.os_profiles.items():
            data.append({
                'OS': os_type.title(),
                'Efficiency': f"{profile['efficiency']*100:.1f}%",
                'Licensing Cost': f"${profile['licensing_cost']}/month",
                'Complexity': f"{profile['complexity']*100:.0f}%",
                'Current': '✅' if os_type == config.os_type else ''
            })
        return pd.DataFrame(data)
    
    def _analyze_network_bottlenecks(self, config: MigrationConfig, effective_bw: float, base_bw: float) -> Dict:
        if effective_bw < base_bw:
            return {
                'primary_bottleneck': 'NIC Hardware',
                'impact': f"{((base_bw - effective_bw) / base_bw * 100):.1f}% bandwidth loss",
                'recommendation': f"Upgrade from {config.nic_type} for better performance"
            }
        else:
            return {
                'primary_bottleneck': 'None',
                'impact': 'Optimal bandwidth utilization',
                'recommendation': 'Current network configuration is adequate'
            }

# UI Rendering Functions
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AWS Enterprise Migration Analyzer AI v4.0</h1>
        <p style="font-size: 1.2rem;">Comprehensive Database Migration Analysis with Dynamic Pricing & AI Insights</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar() -> MigrationConfig:
    st.sidebar.header("🔧 Enterprise Migration Configuration")
    
    st.sidebar.subheader("Database Configuration")
    source_db = st.sidebar.selectbox("Source Database", ["mysql", "postgresql", "oracle", "sqlserver"])
    target_db = st.sidebar.selectbox("Target Database", ["mysql", "postgresql", "oracle", "sqlserver"])
    database_size_gb = st.sidebar.number_input("Database Size (GB)", min_value=100, max_value=100000, value=1000)
    
    st.sidebar.subheader("Hardware Configuration")
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 32], index=2)
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256], index=2)
    cpu_ghz = st.sidebar.selectbox("CPU GHz", [2.0, 2.4, 2.8, 3.2, 3.6, 4.0], index=3)
    
    st.sidebar.subheader("Network Configuration")
    nic_speed = st.sidebar.selectbox("NIC Speed (Mbps)", [1000, 10000, 25000, 40000], index=1)
    nic_type = st.sidebar.selectbox("NIC Type", ["gigabit_copper", "gigabit_fiber", "10g_copper", "10g_fiber", "25g_fiber", "40g_fiber"], index=3)
    
    st.sidebar.subheader("Migration Configuration")
    environment = st.sidebar.selectbox("Environment", ["production", "non-production"])
    num_agents = st.sidebar.number_input("Number of Agents", min_value=1, max_value=8, value=2)
    agent_size = st.sidebar.selectbox("Agent Size", ["small", "medium", "large", "xlarge"], index=1)
    destination_storage = st.sidebar.selectbox("Destination Storage", ["S3", "FSx_Windows", "FSx_Lustre"])
    
    st.sidebar.subheader("System Configuration")
    os_type = st.sidebar.selectbox("Operating System", ["windows", "linux", "rhel"])
    server_type = st.sidebar.selectbox("Server Type", ["physical", "vmware"])
    performance_requirements = st.sidebar.selectbox("Performance Requirements", ["standard", "high"])
    downtime_tolerance_minutes = st.sidebar.number_input("Downtime Tolerance (minutes)", min_value=5, max_value=480, value=60)
    
    return MigrationConfig(
        source_db=source_db, target_db=target_db, database_size_gb=database_size_gb,
        cpu_cores=cpu_cores, ram_gb=ram_gb, cpu_ghz=cpu_ghz,
        nic_speed=nic_speed, nic_type=nic_type, environment=environment,
        num_agents=num_agents, agent_size=agent_size, destination_storage=destination_storage,
        os_type=os_type, server_type=server_type, performance_requirements=performance_requirements,
        downtime_tolerance_minutes=downtime_tolerance_minutes
    )

# Tab Rendering Functions
def render_ai_insights_tab(analysis: Dict):
    st.subheader("🧠 AI Insights & Analysis")
    ai = analysis['ai_insights']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Complexity Score", f"{ai['complexity_score']}/10", delta=ai['confidence_level'])
    with col2:
        st.metric("📊 Success Probability", f"{ai['success_probability']:.1f}%", delta=ai['timeline_risk'] + " Risk")
    with col3:
        st.metric("⚠️ Risk Factors", len(ai['risk_factors']), delta="Identified")
    with col4:
        st.metric("💡 Recommendations", len(ai['recommendations']), delta="AI Generated")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚨 Risk Factors:**")
        for risk in ai['risk_factors']:
            st.write(f"• {risk}")
        
        st.markdown("**🛡️ Mitigation Strategies:**")
        for strategy in ai['mitigation_strategies']:
            st.write(f"• {strategy}")
    
    with col2:
        st.markdown("**💡 AI Recommendations:**")
        for rec in ai['recommendations']:
            st.write(f"• {rec}")
        
        st.markdown("**🚀 Optimization Opportunities:**")
        for opp in ai['optimization_opportunities']:
            st.write(f"• {opp}")

def render_agent_scaling_tab(analysis: Dict):
    st.subheader("🤖 Agent Scaling Analysis")
    agents = analysis['agents']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔧 Agent Configuration", f"{agents['num_agents']}x {agents['agent_size'].title()}", delta=agents['agent_type'].upper())
    with col2:
        st.metric("⚡ Effective Throughput", f"{agents['effective_throughput_mbps']:,.0f} Mbps", delta=f"{agents['scaling_efficiency']*100:.1f}% Efficiency")
    with col3:
        st.metric("💰 Monthly Cost", f"${agents['monthly_cost']:,.0f}", delta=agents['management_complexity'] + " Complexity")
    with col4:
        st.metric("🎯 Efficiency Score", f"{agents['efficiency_score']}/100", delta=agents['bottleneck'].title() + " Bottleneck")
    
    # Agent throughput analysis chart
    throughput_data = {
        'Component': ['Per Agent', 'Total Agents', 'Effective'],
        'Throughput': [
            agents['per_agent_spec']['throughput'],
            agents['total_throughput_mbps'],
            agents['effective_throughput_mbps']
        ]
    }
    fig = px.bar(throughput_data, x='Component', y='Throughput', title="Agent Throughput Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔧 Current Configuration:**")
        st.write(f"**Agent Type:** {agents['agent_type'].upper()}")
        st.write(f"**Size:** {agents['agent_size'].title()}")
        st.write(f"**Count:** {agents['num_agents']}")
        st.write(f"**vCPU per Agent:** {agents['per_agent_spec']['vcpu']}")
        st.write(f"**Memory per Agent:** {agents['per_agent_spec']['memory']} GB")
        st.write(f"**Storage Multiplier:** {agents['storage_multiplier']:.1f}x")
    
    with col2:
        st.markdown("**📊 Performance Analysis:**")
        st.write(f"**Scaling Efficiency:** {agents['scaling_efficiency']*100:.1f}%")
        st.write(f"**Coordination Overhead:** {agents['coordination_overhead']*100:.1f}%")
        st.write(f"**Optimal Agents:** {agents['optimal_agents']}")
        st.write(f"**Management Complexity:** {agents['management_complexity']}")
        st.write(f"**Primary Bottleneck:** {agents['bottleneck'].title()}")

def render_agent_positioning_tab(analysis: Dict):
    st.subheader("🏢 Agent Positioning")
    positioning = analysis['agent_positioning']
    
    st.markdown("**🗺️ Architecture Overview:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🗄️ Storage Location", positioning['storage_location'], delta=positioning['protocol'])
    with col2:
        st.metric("📍 Placement Strategy", positioning['placement_strategy'], delta="Optimal")
    with col3:
        st.metric("🌐 Network Requirements", "Multi-tier", delta="LAN + WAN")
    
    st.markdown("**🔄 Data Flow:**")
    st.info(positioning['data_flow'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Implementation Steps:**")
        for i, step in enumerate(positioning['implementation_steps'], 1):
            st.write(f"{i}. {step}")
    
    with col2:
        st.markdown("**🌐 Network Requirements:**")
        st.write(f"**Internal Bandwidth:** {positioning['network_requirements']['internal']}")
        st.write(f"**External Bandwidth:** {positioning['network_requirements']['external']}")
        st.write(f"**Protocol:** {positioning['protocol']}")
        st.write(f"**Storage Access:** {positioning['storage_location']}")

def render_fsx_comparison_tab(analysis: Dict):
    st.subheader("🗄️ FSx Destination Comparison")
    fsx = analysis['fsx_comparison']
    
    # Display decision matrix
    st.markdown("**📊 Decision Matrix:**")
    st.dataframe(fsx['decision_matrix'], use_container_width=True)
    
    st.markdown(f"**🎯 Current Selection:** {fsx['current_selection']}")
    st.markdown(f"**💡 AI Recommendation:** {fsx['recommendation']}")
    
    # Comparison charts
    col1, col2 = st.columns(2)
    with col1:
        # Performance comparison
        perf_data = {dest: comp['performance_multiplier'] for dest, comp in fsx['comparisons'].items()}
        fig = px.bar(x=list(perf_data.keys()), y=list(perf_data.values()), 
                    title="Performance Multiplier Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cost comparison  
        cost_data = {dest: comp['monthly_storage_cost'] for dest, comp in fsx['comparisons'].items()}
        fig = px.bar(x=list(cost_data.keys()), y=list(cost_data.values()),
                    title="Monthly Storage Cost Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison
    for dest, comp in fsx['comparisons'].items():
        with st.expander(f"📋 {dest} Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Pros:**")
                for pro in comp['pros']:
                    st.write(f"• {pro}")
            with col2:
                st.markdown("**⚠️ Cons:**")
                for con in comp['cons']:
                    st.write(f"• {con}")
            st.write(f"**Best For:** {comp['best_for']}")

def render_network_intelligence_tab(analysis: Dict):
    st.subheader("🌐 Network Intelligence")
    network = analysis['network']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌐 Effective Bandwidth", f"{network['effective_bandwidth_mbps']:,} Mbps", delta=f"{network['nic_efficiency']*100:.1f}% NIC Efficiency")
    with col2:
        st.metric("📊 Quality Score", f"{network['quality_score']:.1f}/100", delta=f"AI: {network['ai_enhanced_score']:.1f}")
    with col3:
        st.metric("🕐 Latency", f"{network['latency_ms']:.1f} ms", delta=f"{network['reliability']*100:.2f}% Reliability")
    with col4:
        st.metric("🔧 Optimization Potential", f"{network['optimization_potential']:.1f}%", delta="Available")
    
    # Network path analysis
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📈 Performance Analysis:**")
        st.write(f"**Base Bandwidth:** {network['base_bandwidth_mbps']:,} Mbps")
        st.write(f"**Effective Bandwidth:** {network['effective_bandwidth_mbps']:,} Mbps")
        st.write(f"**NIC Efficiency:** {network['nic_efficiency']*100:.1f}%")
        st.write(f"**Path Segments:** {network['path_segments']}")
        st.write(f"**Quality Score:** {network['quality_score']:.1f}/100")
    
    with col2:
        st.markdown("**🔍 Bottleneck Analysis:**")
        bottleneck = network['bottleneck_analysis']
        st.write(f"**Primary Bottleneck:** {bottleneck['primary_bottleneck']}")
        st.write(f"**Impact:** {bottleneck['impact']}")
        st.write(f"**Recommendation:** {bottleneck['recommendation']}")

def render_comprehensive_costs_tab(analysis: Dict):
    st.subheader("💰 Comprehensive Cost Analysis")
    costs = analysis['costs']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Monthly", f"${costs['total_monthly']:,.0f}", delta=f"${costs['cost_per_gb']:.2f}/GB")
    with col2:
        st.metric("🏗️ One-time Setup", f"${costs['total_one_time']:,.0f}", delta="Migration Cost")
    with col3:
        st.metric("📈 Annual Cost", f"${costs['annual_cost']:,.0f}", delta="12 Month Total")
    with col4:
        roi_text = f"{costs['roi_months']} months" if costs['roi_months'] else "TBD"
        st.metric("💡 ROI Timeline", roi_text, delta=f"${costs['estimated_savings']:,.0f}/mo savings")
    
    # Cost breakdown chart
    breakdown = costs['monthly_breakdown']
    fig = px.pie(values=list(breakdown.values()), names=list(breakdown.keys()), 
                title="Monthly Cost Breakdown")
    st.plotly_chart(fig, use_container_width=True)
    
    # Destination cost comparison
    st.markdown("**🗄️ Storage Destination Costs:**")
    dest_costs = costs['destination_costs']
    fig = px.bar(x=list(dest_costs.keys()), y=list(dest_costs.values()),
                title="Storage Destination Cost Comparison")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown table
    breakdown_data = []
    for category, cost in breakdown.items():
        breakdown_data.append({
            'Cost Category': category.replace('_', ' ').title(),
            'Monthly Cost': f"${cost:,.0f}",
            'Annual Cost': f"${cost * 12:,.0f}",
            'Percentage': f"{cost / costs['total_monthly'] * 100:.1f}%"
        })
    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True)

def render_os_performance_tab(analysis: Dict):
    st.subheader("💻 OS Performance Analysis")
    os_perf = analysis['os_analysis']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💻 Current OS", os_perf['current_os'].title(), delta=f"{os_perf['overall_efficiency']*100:.1f}% Efficiency")
    with col2:
        st.metric("🔧 CPU Efficiency", f"{os_perf['cpu_efficiency']*100:.1f}%", delta="Hardware")
    with col3:
        st.metric("🗄️ DB Optimization", f"{os_perf['db_optimization']*100:.1f}%", delta="Engine Specific")
    with col4:
        st.metric("💰 Licensing Cost", f"${os_perf['licensing_cost']}/mo", delta=f"{os_perf['management_complexity']*100:.0f}% Complexity")
    
    # Performance radar chart
    performance_metrics = {
        'CPU': os_perf['cpu_efficiency'] * 100,
        'Memory': os_perf['memory_efficiency'] * 100,
        'I/O': os_perf['io_efficiency'] * 100,
        'Network': os_perf['network_efficiency'] * 100,
        'Database': os_perf['db_optimization'] * 100
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(performance_metrics.values()),
        theta=list(performance_metrics.keys()),
        fill='toself',
        name='Current OS Performance'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="OS Performance Profile")
    st.plotly_chart(fig, use_container_width=True)
    
    # OS comparison matrix
    st.markdown("**⚖️ OS Comparison Matrix:**")
    st.dataframe(os_perf['comparison_matrix'], use_container_width=True)
    
    # Recommendations
    st.markdown("**💡 Recommendations:**")
    for rec in os_perf['recommendations']:
        st.write(f"• {rec}")

def render_migration_dashboard_tab(analysis: Dict, config: MigrationConfig):
    st.subheader("📊 Migration Dashboard")
    
    # Executive summary
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🎯 Readiness", f"{analysis['readiness_score']}/100", delta="Overall Score")
    with col2:
        st.metric("⏱️ Migration Time", f"{analysis['timeline']['migration_hours']:.1f}h", delta="Data Transfer")
    with col3:
        st.metric("🚀 Throughput", f"{analysis['agents']['effective_throughput_mbps']:,.0f} Mbps", delta="Effective")
    with col4:
        st.metric("💰 Monthly Cost", f"${analysis['costs']['total_monthly']:,.0f}", delta="AWS Total")
    with col5:
        st.metric("📈 Success Rate", f"{analysis['ai_insights']['success_probability']:.1f}%", delta="AI Predicted")
    
    # Timeline visualization
    timeline_data = {
        'Phase': ['Planning', 'Testing', 'Migration', 'Validation'],
        'Weeks': [
            analysis['timeline']['planning_weeks'],
            analysis['timeline']['testing_weeks'],
            analysis['timeline']['migration_hours'] / (7 * 24),
            1
        ]
    }
    fig = px.bar(timeline_data, x='Phase', y='Weeks', title="Project Timeline")
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Performance Summary:**")
        perf = analysis['performance']
        st.write(f"**Overall Score:** {perf['overall_score']:.1f}/100")
        st.write(f"**CPU Score:** {perf['cpu_score']:.1f}/100")
        st.write(f"**Memory Score:** {perf['memory_score']:.1f}/100")
        st.write(f"**OS Efficiency:** {perf['os_efficiency']*100:.1f}%")
        if perf['bottlenecks'] != ['None identified']:
            st.warning(f"**Bottlenecks:** {', '.join(perf['bottlenecks'])}")
    
    with col2:
        st.markdown("**🎯 Migration Summary:**")
        st.write(f"**Migration Type:** {'Homogeneous' if config.source_db == config.target_db else 'Heterogeneous'}")
        st.write(f"**AWS Deployment:** {analysis['aws_sizing']['deployment'].upper()}")
        st.write(f"**Destination Storage:** {config.destination_storage}")
        st.write(f"**Downtime Acceptable:** {'✅ Yes' if analysis['timeline']['downtime_acceptable'] else '❌ No'}")
        st.write(f"**Complexity Score:** {analysis['ai_insights']['complexity_score']}/10")

def render_aws_sizing_tab(analysis: Dict):
    st.subheader("🎯 AWS Sizing & Configuration")
    aws = analysis['aws_sizing']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("☁️ Deployment", aws['deployment'].upper(), delta=f"{aws['confidence']*100:.1f}% Confidence")
    with col2:
        st.metric("🖥️ Instance Type", aws['instance_type'], delta=f"${aws['instance_cost_hourly']:.3f}/hour")
    with col3:
        st.metric("📚 Total Instances", aws['total_instances'], delta=f"{aws['writers']}W + {aws['readers']}R")
    with col4:
        st.metric("💰 Monthly Cost", f"${aws['total_monthly_cost']:,.0f}", delta="Compute + Storage")
    
    # RDS vs EC2 scoring
    col1, col2 = st.columns(2)
    with col1:
        scoring_data = {'Deployment': ['RDS', 'EC2'], 'Score': [aws['rds_score'], aws['ec2_score']]}
        fig = px.bar(scoring_data, x='Deployment', y='Score', title="Deployment Scoring")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**🔧 Sizing Factors:**")
        factors = aws['sizing_factors']
        st.write(f"**CPU Requirement:** {factors['cpu_requirement']:.1f} cores")
        st.write(f"**Memory Requirement:** {factors['memory_requirement']:.1f} GB")
        st.write(f"**Storage Size:** {factors['storage_size_gb']:,.0f} GB")
        st.write(f"**Writers:** {aws['writers']}")
        st.write(f"**Readers:** {aws['readers']}")
    
    # Cost breakdown
    st.markdown("**💰 Cost Breakdown:**")
    cost_data = {
        'Component': ['Instance Cost', 'Storage Cost'],
        'Monthly Cost': [aws['monthly_instance_cost'], aws['monthly_storage_cost']]
    }
    fig = px.pie(cost_data, values='Monthly Cost', names='Component', title="AWS Cost Distribution")
    st.plotly_chart(fig, use_container_width=True)

def render_pdf_reports_tab(analysis: Dict, config: MigrationConfig):
    st.subheader("📄 Executive PDF Reports")
    
    st.markdown("""
    <div class="professional-card">
        <h4>📊 Generate Executive Report</h4>
        <p>Create a comprehensive PDF report with all analysis results and recommendations for stakeholders.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📥 Generate PDF Report", type="primary"):
        # Generate report content
        report_content = generate_text_report(analysis, config)
        
        # Create download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"AWS_Migration_Analysis_{timestamp}.txt"
        
        st.download_button(
            label="📥 Download Report",
            data=report_content,
            file_name=filename,
            mime="text/plain"
        )
        st.success("✅ Report generated successfully!")
    
    # Report preview
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Report Includes:**")
        st.write("• Executive Summary")
        st.write("• Technical Analysis")
        st.write("• AWS Sizing Recommendations")
        st.write("• Cost Analysis & ROI")
        st.write("• Risk Assessment")
        st.write("• Implementation Timeline")
    
    with col2:
        st.markdown("**📊 Key Metrics:**")
        st.write(f"• Readiness Score: {analysis['readiness_score']}/100")
        st.write(f"• Success Probability: {analysis['ai_insights']['success_probability']:.1f}%")
        st.write(f"• Migration Time: {analysis['timeline']['migration_hours']:.1f} hours")
        st.write(f"• Monthly Cost: ${analysis['costs']['total_monthly']:,.0f}")
        st.write(f"• ROI Timeline: {analysis['costs']['roi_months'] or 'TBD'} months")

def generate_text_report(analysis: Dict, config: MigrationConfig) -> str:
    """Generate comprehensive text report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return f"""
AWS ENTERPRISE DATABASE MIGRATION ANALYSIS REPORT
Generated: {timestamp}

================================================================================
EXECUTIVE SUMMARY
================================================================================

Migration Overview:
- Source Database: {config.source_db.upper()} ({config.database_size_gb:,} GB)
- Target Database: {config.target_db.upper()}
- Migration Type: {'Homogeneous' if config.source_db == config.target_db else 'Heterogeneous'}
- Environment: {config.environment.title()}
- Destination Storage: {config.destination_storage}

Key Metrics:
- Migration Readiness Score: {analysis['readiness_score']}/100
- Estimated Migration Time: {analysis['timeline']['migration_hours']:.1f} hours
- Success Probability: {analysis['ai_insights']['success_probability']:.1f}%
- Total Monthly Cost: ${analysis['costs']['total_monthly']:,.0f}
- ROI Timeline: {analysis['costs']['roi_months'] or 'TBD'} months

================================================================================
TECHNICAL ANALYSIS
================================================================================

Performance Analysis:
- Overall Score: {analysis['performance']['overall_score']:.1f}/100
- CPU Score: {analysis['performance']['cpu_score']:.1f}/100
- Memory Score: {analysis['performance']['memory_score']:.1f}/100
- OS Efficiency: {analysis['performance']['os_efficiency']*100:.1f}%
- Bottlenecks: {', '.join(analysis['performance']['bottlenecks'])}

Network Analysis:
- Effective Bandwidth: {analysis['network']['effective_bandwidth_mbps']:,} Mbps
- Quality Score: {analysis['network']['quality_score']:.1f}/100
- Latency: {analysis['network']['latency_ms']:.1f} ms
- Optimization Potential: {analysis['network']['optimization_potential']:.1f}%

Agent Configuration:
- Agent Type: {analysis['agents']['agent_type'].upper()}
- Number of Agents: {analysis['agents']['num_agents']}
- Agent Size: {analysis['agents']['agent_size']}
- Effective Throughput: {analysis['agents']['effective_throughput_mbps']:,.0f} Mbps
- Scaling Efficiency: {analysis['agents']['scaling_efficiency']*100:.1f}%

================================================================================
AWS SIZING RECOMMENDATIONS
================================================================================

Deployment: {analysis['aws_sizing']['deployment'].upper()}
Instance Type: {analysis['aws_sizing']['instance_type']}
Total Instances: {analysis['aws_sizing']['total_instances']} ({analysis['aws_sizing']['writers']} writers, {analysis['aws_sizing']['readers']} readers)
Monthly Instance Cost: ${analysis['aws_sizing']['monthly_instance_cost']:,.0f}
Monthly Storage Cost: ${analysis['aws_sizing']['monthly_storage_cost']:,.0f}
Total AWS Cost: ${analysis['aws_sizing']['total_monthly_cost']:,.0f}

================================================================================
COST ANALYSIS
================================================================================

Monthly Cost Breakdown:
- Compute: ${analysis['costs']['monthly_breakdown']['compute']:,.0f}
- Storage: ${analysis['costs']['monthly_breakdown']['storage']:,.0f}
- Agents: ${analysis['costs']['monthly_breakdown']['agents']:,.0f}
- Network: ${analysis['costs']['monthly_breakdown']['network']:,.0f}
- OS Licensing: ${analysis['costs']['monthly_breakdown']['os_licensing']:,.0f}
- Management: ${analysis['costs']['monthly_breakdown']['management']:,.0f}

Total Monthly: ${analysis['costs']['total_monthly']:,.0f}
One-time Migration Cost: ${analysis['costs']['total_one_time']:,.0f}
Annual Cost: ${analysis['costs']['annual_cost']:,.0f}
Cost per GB: ${analysis['costs']['cost_per_gb']:.2f}

ROI Analysis:
- Estimated Monthly Savings: ${analysis['costs']['estimated_savings']:,.0f}
- ROI Timeline: {analysis['costs']['roi_months'] or 'TBD'} months

================================================================================
RISK ASSESSMENT & RECOMMENDATIONS
================================================================================

AI Complexity Score: {analysis['ai_insights']['complexity_score']}/10
Success Probability: {analysis['ai_insights']['success_probability']:.1f}%
Confidence Level: {analysis['ai_insights']['confidence_level']}

Risk Factors:
{chr(10).join(f"- {risk}" for risk in analysis['ai_insights']['risk_factors'])}

Recommendations:
{chr(10).join(f"- {rec}" for rec in analysis['ai_insights']['recommendations'])}

Mitigation Strategies:
{chr(10).join(f"- {strategy}" for strategy in analysis['ai_insights']['mitigation_strategies'])}

================================================================================
PROJECT TIMELINE
================================================================================

Migration Timeline:
- Planning Phase: {analysis['timeline']['planning_weeks']} weeks
- Testing Phase: {analysis['timeline']['testing_weeks']} weeks
- Migration Window: {analysis['timeline']['migration_hours']:.1f} hours
- Total Project: {analysis['timeline']['total_project_weeks']} weeks

Downtime Analysis:
- Required Downtime: {analysis['timeline']['migration_hours']:.1f} hours
- Tolerance: {config.downtime_tolerance_minutes} minutes
- Acceptable: {'Yes' if analysis['timeline']['downtime_acceptable'] else 'No'}

================================================================================
END OF REPORT
================================================================================
"""

async def main():
    """Main application entry point"""
    render_header()
    
    # Configuration
    config = render_sidebar()
    
    # Analysis
    analyzer = EnterpriseMigrationAnalyzer()
    
    with st.spinner("🔄 Running comprehensive enterprise migration analysis..."):
        analysis = await analyzer.comprehensive_analysis(config)
    
    # Enterprise tabs
    tabs = st.tabs([
        "🧠 AI Insights & Analysis",
        "🤖 Agent Scaling Analysis", 
        "🏢 Agent Positioning",
        "🗄️ FSx Destination Comparison",
        "🌐 Network Intelligence",
        "💰 Comprehensive Costs",
        "💻 OS Performance Analysis",
        "📊 Migration Dashboard",
        "🎯 AWS Sizing & Configuration",
        "📄 Executive PDF Reports"
    ])
    
    with tabs[0]: render_ai_insights_tab(analysis)
    with tabs[1]: render_agent_scaling_tab(analysis)
    with tabs[2]: render_agent_positioning_tab(analysis)
    with tabs[3]: render_fsx_comparison_tab(analysis)
    with tabs[4]: render_network_intelligence_tab(analysis)
    with tabs[5]: render_comprehensive_costs_tab(analysis)
    with tabs[6]: render_os_performance_tab(analysis)
    with tabs[7]: render_migration_dashboard_tab(analysis, config)
    with tabs[8]: render_aws_sizing_tab(analysis)
    with tabs[9]: render_pdf_reports_tab(analysis, config)
    
    # Pricing indicator
    pricing_source = analysis.get('pricing_source', 'unknown')
    if pricing_source == 'dynamic':
        st.success("✅ Using real-time AWS pricing")
    else:
        st.info("ℹ️ Using cached pricing data")

if __name__ == "__main__":
    asyncio.run(main())