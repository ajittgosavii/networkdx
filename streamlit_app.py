import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import math
import random
from typing import Dict, List, Tuple, Optional
import json
import requests
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import anthropic
import asyncio
import logging
from dataclasses import dataclass
 
import base64
from io import BytesIO
# Optional imports for PDF generation
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mpatches = None
    PdfPages = None
 



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AWS Enterprise Database Migration Analyzer AI v3.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add this function right after the imports and before any classes
def get_nic_efficiency(nic_type):
    """Get NIC efficiency based on type"""
    efficiencies = {
        'gigabit_copper': 0.85,
        'gigabit_fiber': 0.90,
        '10g_copper': 0.88, 
        '10g_fiber': 0.92,
        '25g_fiber': 0.94,
        '40g_fiber': 0.95
    }
    return efficiencies.get(nic_type, 0.90)

def render_bandwidth_waterfall_analysis(analysis, config):
    """Show complete bandwidth degradation from user hardware to final throughput"""
    st.markdown("**🌊 Bandwidth Waterfall Analysis: From Your Hardware to Migration Throughput**")
    
    # Get ACTUAL values from user inputs and analysis
    network_perf = analysis.get('network_performance', {})
    os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
    agent_analysis = analysis.get('agent_analysis', {})
    
    # CORRECT FLOW: Start with user's actual hardware
    user_nic_speed = config.get('nic_speed', 1000)  # User's actual NIC selection
    nic_type = config.get('nic_type', 'gigabit_fiber')
    environment = config.get('environment', 'non-production')
    
    # Network path limitations from environment analysis
    network_path_limit = network_perf.get('effective_bandwidth_mbps', 1000)
    
    # Step 1: User's Raw NIC Capacity
    raw_user_capacity = user_nic_speed
    
    # Step 2: NIC Hardware Efficiency
    nic_efficiency = get_nic_efficiency(nic_type)
    after_nic = raw_user_capacity * nic_efficiency
    
    # Step 3: OS Network Stack
    os_network_efficiency = os_impact.get('network_efficiency', 0.90)
    after_os = after_nic * os_network_efficiency
    
    # Step 4: Virtualization Impact (if VMware)
    server_type = config.get('server_type', 'physical')
    if server_type == 'vmware':
        virtualization_efficiency = 0.92
        after_virtualization = after_os * virtualization_efficiency
    else:
        virtualization_efficiency = 1.0
        after_virtualization = after_os
    
    # Step 5: Protocol Overhead
    if 'production' in environment:
        protocol_efficiency = 0.82
    else:
        protocol_efficiency = 0.85
    after_protocol = after_virtualization * protocol_efficiency
    
    # Step 6: NETWORK PATH LIMITATION (The Key Bottleneck Check)
    # This is where we hit the actual network infrastructure limit
    after_network_limit = min(after_protocol, network_path_limit)
    network_is_bottleneck = after_protocol > network_path_limit
    
    # Step 7: Migration Agent Processing
    total_agent_capacity = agent_analysis.get('total_max_throughput_mbps', after_network_limit * 0.75)
    actual_throughput = agent_analysis.get('total_effective_throughput', 0)
    
    if actual_throughput > 0:
        final_throughput = actual_throughput
    else:
        final_throughput = min(total_agent_capacity, after_network_limit)
    
    # Build the waterfall stages
    stages = ['Your NIC\nCapacity']
    throughputs = [raw_user_capacity]
    efficiencies = [100]
    descriptions = [f"{user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC"]
    
    # NIC Processing
    stages.append('After NIC\nProcessing')
    throughputs.append(after_nic)
    efficiencies.append(nic_efficiency * 100)
    descriptions.append(f"{nic_type.replace('_', ' ').title()} hardware efficiency")
    
    # OS Processing
    os_name = config.get('operating_system', 'unknown').replace('_', ' ').title()
    stages.append(f'After OS\nNetwork Stack')
    throughputs.append(after_os)
    efficiencies.append(os_network_efficiency * 100)
    descriptions.append(f"{os_name} network processing")
    
    # Virtualization (if applicable)
    if server_type == 'vmware':
        stages.append('After VMware\nVirtualization')
        throughputs.append(after_virtualization)
        efficiencies.append(virtualization_efficiency * 100)
        descriptions.append('VMware hypervisor overhead')
    
    # Protocol Overhead
    stages.append('After Protocol\nOverhead')
    throughputs.append(after_protocol)
    efficiencies.append(protocol_efficiency * 100)
    descriptions.append(f"{environment.title()} security protocols")
    
    # Network Path Limitation (CRITICAL BOTTLENECK POINT)
    stages.append('After Network\nPath Limit')
    throughputs.append(after_network_limit)
    if network_is_bottleneck:
        # Network became the bottleneck
        efficiencies.append((network_path_limit / after_protocol) * 100)
        if environment == 'production':
            descriptions.append(f"Production path: {network_path_limit:,.0f} Mbps available")
        else:
            descriptions.append(f"Non-prod DX limit: {network_path_limit:,.0f} Mbps")
    else:
        # User's hardware was already the bottleneck
        efficiencies.append(100)
        descriptions.append(f"Network supports {network_path_limit:,.0f} Mbps (no additional limit)")
    
    # Final Migration Throughput
    tool_name = agent_analysis.get('primary_tool', 'DMS').upper()
    num_agents = config.get('number_of_agents', 1)
    stages.append(f'Final Migration\nThroughput')
    throughputs.append(final_throughput)
    
    if after_network_limit > 0:
        agent_efficiency = (final_throughput / after_network_limit) * 100
    else:
        agent_efficiency = 75
    efficiencies.append(agent_efficiency)
    descriptions.append(f"{num_agents}x {tool_name} agents processing")
    
    # Create the visualization
    waterfall_data = {
        'Stage': stages,
        'Throughput (Mbps)': throughputs,
        'Efficiency': efficiencies,
        'Description': descriptions
    }
    
    fig = px.bar(
        waterfall_data,
        x='Stage',
        y='Throughput (Mbps)',
        title=f"Your Migration Bandwidth: {user_nic_speed:,.0f} Mbps Hardware → {final_throughput:.0f} Mbps Migration Speed",
        color='Efficiency',
        color_continuous_scale='RdYlGn',
        text='Throughput (Mbps)',
        hover_data=['Description']
    )
    
    fig.update_traces(texttemplate='%{text:.0f} Mbps', textposition='outside')
    fig.update_layout(height=500)
    
    # Add loss annotations
    for i in range(1, len(throughputs)):
        loss = throughputs[i-1] - throughputs[i]
        if loss > 0:
            loss_pct = (loss / throughputs[i-1]) * 100
            fig.add_annotation(
                x=i, y=throughputs[i] + (max(throughputs) * 0.05),
                text=f"-{loss:.0f} Mbps<br>(-{loss_pct:.1f}%)",
                showarrow=True,
                arrowcolor="red",
                arrowhead=2,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="red",
                font=dict(size=10)
            )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Analysis Summary
    total_loss = user_nic_speed - final_throughput
    total_loss_pct = (total_loss / user_nic_speed) * 100
    
    # Identify the primary bottleneck
    if network_is_bottleneck:
        primary_bottleneck = f"Network path ({network_path_limit:,.0f} Mbps limit)"
        bottleneck_type = "Infrastructure"
    elif final_throughput < after_network_limit * 0.9:
        primary_bottleneck = f"Migration agents ({num_agents}x {tool_name})"
        bottleneck_type = "Agent Capacity"
    elif after_nic < user_nic_speed * 0.95:
        primary_bottleneck = f"NIC efficiency ({nic_type.replace('_', ' ')})"
        bottleneck_type = "Hardware"
    else:
        primary_bottleneck = "Protocol and OS overhead"
        bottleneck_type = "Software"
    
    # Show different messages based on bottleneck type
    if bottleneck_type == "Infrastructure":
        message_type = "warning"
        if environment == 'production':
            bottleneck_explanation = f"Your {user_nic_speed:,.0f} Mbps NIC can handle more, but production network infrastructure provides {network_path_limit:,.0f} Mbps"
        else:
            bottleneck_explanation = f"Your {user_nic_speed:,.0f} Mbps NIC can handle more, but non-production Direct Connect is limited to {network_path_limit:,.0f} Mbps"
    elif bottleneck_type == "Hardware":
        message_type = "info"
        bottleneck_explanation = f"Your {user_nic_speed:,.0f} Mbps NIC is the limiting factor (network supports {network_path_limit:,.0f} Mbps)"
    else:
        message_type = "error"
        bottleneck_explanation = f"Agent configuration or processing overhead is limiting throughput below network capacity"
    
    if message_type == "warning":
        st.warning(f"""
        ⚠️ **Network Infrastructure Bottleneck Detected:**
        • **Your Hardware:** {user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC
        • **Network Limitation:** {network_path_limit:,.0f} Mbps ({environment} environment)
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Explanation:** {bottleneck_explanation}
        • **Recommendation:** Plan migration times using {final_throughput:.0f} Mbps actual speed
        """)
    elif message_type == "info":
        st.info(f"""
        💡 **Hardware Bottleneck (Expected):**
        • **Your Hardware:** {user_nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} NIC  
        • **Network Capacity:** {network_path_limit:,.0f} Mbps available
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Explanation:** {bottleneck_explanation}
        • **Recommendation:** Consider NIC upgrade if faster migration needed
        """)
    else:
        st.error(f"""
        🔍 **Agent/Processing Bottleneck:**
        • **Available Bandwidth:** {min(user_nic_speed, network_path_limit):,.0f} Mbps
        • **Final Migration Speed:** {final_throughput:.0f} Mbps
        • **Efficiency Loss:** {total_loss_pct:.1f}%
        • **Primary Issue:** {primary_bottleneck}
        • **Recommendation:** Optimize agent configuration or increase agent count
        """)
    
    return final_throughput

def render_performance_impact_table(analysis, config):
    """Detailed table showing each performance impact layer - FULLY DYNAMIC"""
    
    st.markdown("**📊 Detailed Performance Impact Analysis (Based on Your Configuration):**")
    
    # Get ACTUAL values from user configuration
    network_perf = analysis.get('network_performance', {})
    os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
    agent_analysis = analysis.get('agent_analysis', {})
    
    # DYNAMIC: Start with user's actual NIC speed
    base_bandwidth = config.get('nic_speed', 1000)
    
    impact_data = []
    running_throughput = base_bandwidth
    
    # Layer 1: Raw Network (user's actual NIC)
    nic_type = config.get('nic_type', 'gigabit_fiber')
    impact_data.append({
        'Layer': '🌐 Raw Network Capacity',
        'Component': f"{base_bandwidth:,.0f} Mbps {nic_type.replace('_', ' ')} connection",
        'Throughput (Mbps)': f"{running_throughput:.0f}",
        'Efficiency (%)': "100.0%",
        'Loss (Mbps)': "0",
        'Impact Type': 'User Configuration'
    })
    
    # Layer 2: NIC Hardware (user's actual NIC type)
    nic_efficiency = get_nic_efficiency(nic_type)
    nic_loss = running_throughput * (1 - nic_efficiency)
    running_throughput *= nic_efficiency
    
    impact_data.append({
        'Layer': '🔌 NIC Hardware',
        'Component': f"{nic_type.replace('_', ' ').title()} hardware limitations",
        'Throughput (Mbps)': f"{running_throughput:.0f}",
        'Efficiency (%)': f"{nic_efficiency * 100:.1f}%",
        'Loss (Mbps)': f"{nic_loss:.0f}",
        'Impact Type': 'Hardware Limitation'
    })
    
    # Layer 3: Operating System (user's actual OS selection)
    os_efficiency = os_impact.get('network_efficiency', 0.90)
    os_loss = running_throughput * (1 - os_efficiency)
    running_throughput *= os_efficiency
    
    os_name = config.get('operating_system', 'Unknown').replace('_', ' ').title()
    impact_data.append({
        'Layer': '💻 OS Network Stack',
        'Component': f"{os_name} network stack overhead",
        'Throughput (Mbps)': f"{running_throughput:.0f}",
        'Efficiency (%)': f"{os_efficiency * 100:.1f}%",
        'Loss (Mbps)': f"{os_loss:.0f}",
        'Impact Type': 'Software Overhead'
    })
    
    # Layer 4: Virtualization (only if user selected VMware)
    server_type = config.get('server_type', 'physical')
    if server_type == 'vmware':
        virt_efficiency = 0.92
        virt_loss = running_throughput * (1 - virt_efficiency)
        running_throughput *= virt_efficiency
        
        impact_data.append({
            'Layer': '☁️ Virtualization',
            'Component': 'VMware hypervisor overhead (user selected VMware)',
            'Throughput (Mbps)': f"{running_throughput:.0f}",
            'Efficiency (%)': f"{virt_efficiency * 100:.1f}%",
            'Loss (Mbps)': f"{virt_loss:.0f}",
            'Impact Type': 'Platform Overhead'
        })
    
    # Layer 5: Protocol Overhead (based on environment)
    env_type = config.get('environment', 'non-production')
    if env_type == 'production':
        protocol_efficiency = 0.82  # More security in production
        protocol_desc = "Production environment security protocols"
    else:
        protocol_efficiency = 0.85  # Standard overhead
        protocol_desc = "Non-production environment protocols"
    
    protocol_loss = running_throughput * (1 - protocol_efficiency)
    running_throughput *= protocol_efficiency
    
    impact_data.append({
        'Layer': '🔗 Protocol Overhead',
        'Component': protocol_desc,
        'Throughput (Mbps)': f"{running_throughput:.0f}",
        'Efficiency (%)': f"{protocol_efficiency * 100:.1f}%",
        'Loss (Mbps)': f"{protocol_loss:.0f}",
        'Impact Type': 'Protocol Processing'
    })
    
    # Layer 6: Migration Appliance (user's actual agent configuration)
    num_agents = config.get('number_of_agents', 1)
    tool_name = agent_analysis.get('primary_tool', 'DMS').upper()
    agent_size = agent_analysis.get('agent_size', 'medium')
    
    # Get actual appliance efficiency from agent analysis
    actual_throughput = agent_analysis.get('total_effective_throughput', 0)
    if actual_throughput > 0 and running_throughput > 0:
        appliance_efficiency = min(1.0, actual_throughput / running_throughput)
    else:
        appliance_efficiency = 0.75  # Default
    
    appliance_loss = running_throughput * (1 - appliance_efficiency)
    final_running_throughput = actual_throughput if actual_throughput > 0 else running_throughput * appliance_efficiency
    
    impact_data.append({
        'Layer': '🤖 Migration Appliance',
        'Component': f"{num_agents}x {tool_name} {agent_size} agents (user config)",
        'Throughput (Mbps)': f"{final_running_throughput:.0f}",
        'Efficiency (%)': f"{appliance_efficiency * 100:.1f}%",
        'Loss (Mbps)': f"{appliance_loss:.0f}",
        'Impact Type': 'Migration Tool Processing'
    })
    
    # Create DataFrame and display
    df_impact = pd.DataFrame(impact_data)
    st.dataframe(df_impact, use_container_width=True)
    
    # DYNAMIC Summary based on user's actual configuration
    total_loss = base_bandwidth - final_running_throughput
    total_loss_pct = (total_loss / base_bandwidth) * 100
    
    # Find the layer with the biggest impact
    max_loss = 0
    biggest_impact_layer = ""
    for row in impact_data[1:]:  # Skip baseline
        loss = float(row['Loss (Mbps)'])
        if loss > max_loss:
            max_loss = loss
            biggest_impact_layer = row['Layer']
    
    st.info(f"""
    💡 **Your Configuration Analysis:**
    • **Your Setup:** {os_name} on {server_type.title()} with {nic_type.replace('_', ' ')} NIC
    • **Migration Tools:** {num_agents}x {tool_name} {agent_size} agents to {config.get('destination_storage_type', 'S3')}
    • **Biggest Impact:** {biggest_impact_layer} (-{max_loss:.0f} Mbps)
    • **Final Result:** {base_bandwidth:,.0f} Mbps → {final_running_throughput:.0f} Mbps ({100-total_loss_pct:.1f}% efficiency)
    """)
# Professional CSS styling with corporate colors
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(30,58,138,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
        font-weight: 600;
    }
    
    .professional-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
        border: 1px solid #e5e7eb;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #3b82f6;
        border: 1px solid #e5e7eb;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #fefdf8 0%, #fefce8 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #f59e0b;
        border: 1px solid #e5e7eb;
    }
    
    .pricing-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #0ea5e9;
        border: 1px solid #e5e7eb;
    }
    
    .network-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #22c55e;
        border: 1px solid #e5e7eb;
    }
    
    .storage-card {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #a855f7;
        border: 1px solid #e5e7eb;
    }
    
    .performance-card {
        background: linear-gradient(135deg, #fef7f0 0%, #fed7aa 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #f97316;
        border: 1px solid #e5e7eb;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 6px;
        color: #374151;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 3px solid #64748b;
        border: 1px solid #e5e7eb;
    }
    
    .api-status-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        color: #374151;
        font-size: 0.9rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-card h4, .metric-card h5 {
        margin: 0 0 0.8rem 0;
        color: #1f2937;
        font-weight: 600;
    }
    
    .metric-card p {
        margin: 0.4rem 0;
        color: #6b7280;
        line-height: 1.4;
    }
    
    .network-diagram-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    .status-online { background-color: #22c55e; }
    .status-offline { background-color: #ef4444; }
    .status-warning { background-color: #f59e0b; }
    
    .enterprise-footer {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        padding: 2rem;
        border-radius: 6px;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .analysis-section {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .analysis-section h4, .analysis-section h5 {
        margin: 0 0 1rem 0;
        color: #1f2937;
        font-weight: 600;
    }
    
    .analysis-section p {
        margin: 0.5rem 0;
        color: #6b7280;
        line-height: 1.5;
    }
    
    .analysis-section ul {
        margin: 0.8rem 0;
        padding-left: 1.5rem;
    }
    
    .analysis-section li {
        margin: 0.4rem 0;
        color: #6b7280;
        line-height: 1.4;
    }
    
    /* Compact metric cards */
    .compact-metric {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #3b82f6;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    .compact-metric h5 {
        margin: 0 0 0.5rem 0;
        color: #1f2937;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .compact-metric p {
        margin: 0.2rem 0;
        color: #6b7280;
        font-size: 0.85rem;
        line-height: 1.3;
    }
    
    /* PDF download section */
    .pdf-section {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 2rem;
        border-radius: 6px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
        text-align: center;
    }
    
    /* Professional tables */
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Form controls */
    .stSelectbox > div > div {
        border-radius: 6px;
    }
    
    .stNumberInput > div > div {
        border-radius: 6px;
    }
    
    /* Professional buttons */
    .stButton > button {
        border-radius: 6px;
        border: none;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
    }
    
    /* Professional tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .stSidebar .stSelectbox label,
    .stSidebar .stNumberInput label {
        font-weight: 500;
        color: #1f2937;
        font-size: 0.9rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .professional-card, .insight-card, .recommendation-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class APIStatus:
    anthropic_connected: bool = False
    aws_pricing_connected: bool = False
    aws_compute_optimizer_connected: bool = False
    last_update: datetime = None
    error_message: str = None

class AnthropicAIManager:
    """Enhanced Anthropic AI manager with improved error handling and connection"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.error_message = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize connection to Anthropic API"""
        try:
            # Try to get API key from Streamlit secrets
            if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
                api_key = st.secrets['ANTHROPIC_API_KEY']
            else:
                # Try environment variable
                import os
                api_key = os.getenv('ANTHROPIC_API_KEY')
            
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.connected = True
                self.error_message = None
            else:
                self.connected = False
                self.error_message = "API key not found in secrets or environment"
                
        except Exception as e:
            self.connected = False
            self.error_message = str(e)
    
    
    async def analyze_migration_workload(self, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced AI-powered workload analysis with detailed insights"""
        if not self.connected:
            return self._fallback_workload_analysis(config, performance_data)
        
        try:
            # Enhanced prompt for more detailed analysis
            prompt = f"""
            As a senior AWS migration consultant with deep expertise in database migrations, provide a comprehensive analysis of this migration scenario:

            CURRENT INFRASTRUCTURE:
            - Source Database: {config['source_database_engine']} ({config['database_size_gb']} GB)
            - Target Database: {config['database_engine']}
            - Operating System: {config['operating_system']}
            - Platform: {config['server_type']}
            - Hardware: {config['cpu_cores']} cores @ {config['cpu_ghz']} GHz, {config['ram_gb']} GB RAM
            - Network: {config['nic_type']} ({config['nic_speed']} Mbps)
            - Environment: {config['environment']}
            - Performance Requirement: {config['performance_requirements']}
            - Downtime Tolerance: {config['downtime_tolerance_minutes']} minutes
            - Migration Agents: {config.get('number_of_agents', 1)} agents configured
            - Destination Storage: {config.get('destination_storage_type', 'S3')}

            CURRENT PERFORMANCE METRICS:
            - Database TPS: {performance_data.get('database_performance', {}).get('effective_tps', 'Unknown')}
            - Storage IOPS: {performance_data.get('storage_performance', {}).get('effective_iops', 'Unknown')}
            - Network Bandwidth: {performance_data.get('network_performance', {}).get('effective_bandwidth_mbps', 'Unknown')} Mbps
            - OS Efficiency: {performance_data.get('os_impact', {}).get('total_efficiency', 0) * 100:.1f}%
            - Overall Performance Score: {performance_data.get('performance_score', 0):.1f}/100

            Please provide a detailed assessment including:

            1. MIGRATION COMPLEXITY (1-10 scale with detailed justification)
            2. RISK ASSESSMENT with specific risk percentages and mitigation strategies
            3. PERFORMANCE OPTIMIZATION recommendations with expected improvement percentages
            4. DETAILED TIMELINE with phase-by-phase breakdown
            5. RESOURCE ALLOCATION with specific AWS instance recommendations
            6. COST OPTIMIZATION strategies with potential savings
            7. BEST PRACTICES specific to this configuration with implementation steps
            8. TESTING STRATEGY with checkpoints and validation criteria
            9. ROLLBACK PROCEDURES and contingency planning
            10. POST-MIGRATION monitoring and optimization recommendations
            11. AGENT SCALING IMPACT analysis based on {config.get('number_of_agents', 1)} agents
            12. DESTINATION STORAGE IMPACT for {config.get('destination_storage_type', 'S3')} including performance and cost implications

            Provide quantitative analysis wherever possible, including specific metrics, percentages, and measurable outcomes.
            Format the response as detailed sections with clear recommendations and actionable insights.
            """
            
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse AI response
            ai_response = message.content[0].text
            
            # Enhanced parsing for detailed analysis
            ai_analysis = self._parse_detailed_ai_response(ai_response, config, performance_data)
            
            return {
                'ai_complexity_score': ai_analysis.get('complexity_score', 6),
                'risk_factors': ai_analysis.get('risk_factors', []),
                'risk_percentages': ai_analysis.get('risk_percentages', {}),
                'mitigation_strategies': ai_analysis.get('mitigation_strategies', []),
                'performance_recommendations': ai_analysis.get('performance_recommendations', []),
                'performance_improvements': ai_analysis.get('performance_improvements', {}),
                'timeline_suggestions': ai_analysis.get('timeline_suggestions', []),
                'resource_allocation': ai_analysis.get('resource_allocation', {}),
                'cost_optimization': ai_analysis.get('cost_optimization', []),
                'best_practices': ai_analysis.get('best_practices', []),
                'testing_strategy': ai_analysis.get('testing_strategy', []),
                'rollback_procedures': ai_analysis.get('rollback_procedures', []),
                'post_migration_monitoring': ai_analysis.get('post_migration_monitoring', []),
                'confidence_level': ai_analysis.get('confidence_level', 'medium'),
                'detailed_assessment': ai_analysis.get('detailed_assessment', {}),
                'agent_scaling_impact': ai_analysis.get('agent_scaling_impact', {}),
                'destination_storage_impact': ai_analysis.get('destination_storage_impact', {}),
                'raw_ai_response': ai_response
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            st.error(f"AI Analysis Error: {str(e)}")
            return self._fallback_workload_analysis(config, performance_data)
    
    def _calculate_optimal_agents(self, config: Dict) -> int:
        """Calculate optimal number of agents based on database size and requirements"""
        database_size = config['database_size_gb']
        
        if database_size < 1000:
            return 1
        elif database_size < 5000:
            return 2
        elif database_size < 20000:
            return 3
        elif database_size < 50000:
            return 4
        else:
            return 5
    
    def _calculate_storage_performance_impact(self, storage_type: str) -> Dict:
        """Calculate performance impact for different storage destinations"""
        storage_profiles = {
            'S3': {
                'throughput_multiplier': 1.0,
                'latency_impact': 1.0,
                'iops_capability': 'Standard',
                'performance_rating': 'Good'
            },
            'FSx_Windows': {
                'throughput_multiplier': 1.3,
                'latency_impact': 0.7,
                'iops_capability': 'High',
                'performance_rating': 'Very Good'
            },
            'FSx_Lustre': {
                'throughput_multiplier': 2.0,
                'latency_impact': 0.4,
                'iops_capability': 'Very High',
                'performance_rating': 'Excellent'
            }
        }
        return storage_profiles.get(storage_type, storage_profiles['S3'])
    
    def _calculate_storage_cost_impact(self, storage_type: str) -> Dict:
        """Calculate cost impact for different storage destinations"""
        cost_profiles = {
            'S3': {
                'base_cost_multiplier': 1.0,
                'operational_cost': 'Low',
                'setup_cost': 'Minimal',
                'long_term_value': 'Excellent'
            },
            'FSx_Windows': {
                'base_cost_multiplier': 2.5,
                'operational_cost': 'Medium',
                'setup_cost': 'Medium',
                'long_term_value': 'Good'
            },
            'FSx_Lustre': {
                'base_cost_multiplier': 4.0,
                'operational_cost': 'High',
                'setup_cost': 'High',
                'long_term_value': 'Good for HPC'
            }
        }
        return cost_profiles.get(storage_type, cost_profiles['S3'])
    
    def _get_storage_complexity_factor(self, storage_type: str) -> float:
        """Get complexity factor for storage type"""
        complexity_factors = {
            'S3': 1.0,
            'FSx_Windows': 1.8,
            'FSx_Lustre': 2.2
        }
        return complexity_factors.get(storage_type, 1.0)
    
    def _get_storage_recommendations(self, storage_type: str, config: Dict) -> List[str]:
        """Get recommendations for storage type"""
        recommendations = {
            'S3': [
                "Cost-effective for most workloads",
                "Excellent durability and availability",
                "Simple integration with migration tools",
                "Automatic scaling and management"
            ],
            'FSx_Windows': [
                "Ideal for Windows-based applications",
                "Native Windows file system features",
                "Active Directory integration",
                "Better performance for file-based workloads"
            ],
            'FSx_Lustre': [
                "Best for high-performance computing",
                "Extremely high throughput and IOPS",
                "Optimized for parallel processing",
                "Ideal for analytics and ML workloads"
            ]
        }
        return recommendations.get(storage_type, recommendations['S3'])
    
    def _generate_mitigation_strategies(self, risk_factors: List[str], config: Dict) -> List[str]:
        """Generate specific mitigation strategies"""
        strategies = []
        
        if any('schema' in risk.lower() for risk in risk_factors):
            strategies.append("Conduct comprehensive schema conversion testing with AWS SCT")
        
        if any('database size' in risk.lower() for risk in risk_factors):
            strategies.append("Implement parallel data transfer using multiple DMS tasks")
        
        if any('downtime' in risk.lower() for risk in risk_factors):
            strategies.append("Implement read replica for near-zero downtime migration")
        
        if any('performance' in risk.lower() for risk in risk_factors):
            strategies.append("Conduct pre-migration performance tuning")
        
        return strategies
    
    def _generate_cost_optimization(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate cost optimization strategies"""
        optimizations = []
        
        if config['database_size_gb'] < 1000:
            optimizations.append("Consider Reserved Instances for 20-30% cost savings")
        
        if config['environment'] == 'non-production':
            optimizations.append("Use Spot Instances for development/testing")
        
        optimizations.append("Implement automated scaling policies")
        
        return optimizations
    
    def _generate_best_practices(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate specific best practices"""
        practices = [
            "Implement comprehensive backup strategy",
            "Use AWS Migration Hub for tracking",
            "Establish detailed communication plan"
        ]
        
        if complexity_score > 7:
            practices.append("Engage AWS Professional Services")
        
        return practices
    
    def _generate_testing_strategy(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate comprehensive testing strategy"""
        strategy = [
            "Unit Testing: Validate migration components",
            "Integration Testing: End-to-end workflow",
            "Performance Testing: AWS environment validation"
        ]
        
        if config['source_database_engine'] != config['database_engine']:
            strategy.append("Schema Conversion Testing")
        
        return strategy
    
    def _generate_rollback_procedures(self, config: Dict) -> List[str]:
        """Generate rollback procedures"""
        procedures = [
            "Maintain synchronized read replica",
            "Create point-in-time recovery snapshot",
            "Prepare DNS switching procedures",
            "Test rollback procedures in staging"
        ]
        
        return procedures
    
    def _generate_monitoring_recommendations(self, config: Dict) -> List[str]:
        """Generate post-migration monitoring recommendations"""
        monitoring = [
            "Implement CloudWatch detailed monitoring",
            "Set up automated alerts",
            "Monitor application response times",
            "Track database connection patterns"
        ]
        
        return monitoring
    
    def _identify_critical_success_factors(self, config: Dict, complexity_score: int) -> List[str]:
        """Identify critical success factors"""
        factors = [
            "Stakeholder alignment on timeline",
            "Comprehensive testing in staging",
            "Skilled migration team"
        ]
        
        if complexity_score > 7:
            factors.append("Dedicated AWS solutions architect")
        
        return factors
    
    def _fallback_workload_analysis(self, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced fallback analysis when AI is not available"""
        
        complexity_score = 5
        if config['source_database_engine'] != config['database_engine']:
            complexity_score += 2
        if config['database_size_gb'] > 5000:
            complexity_score += 1
        
        return {
            'ai_complexity_score': min(10, complexity_score),
            'risk_factors': [
                "Migration complexity varies with database differences",
                "Large database sizes increase duration"
            ],
            'mitigation_strategies': [
                "Conduct thorough testing",
                "Plan adequate migration windows"
            ],
            'performance_recommendations': [
                "Optimize database before migration",
                "Monitor performance throughout"
            ],
            'confidence_level': 'medium',
            'raw_ai_response': 'AI analysis not available - using fallback'
        }
    
    
    
    def _parse_detailed_ai_response(self, ai_response: str, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced parsing for detailed AI analysis"""
        
        # Calculate complexity score based on multiple factors
        complexity_factors = []
        base_complexity = 5
        
        # Database engine complexity
        if config['source_database_engine'] != config['database_engine']:
            complexity_factors.append(('Heterogeneous migration', 2))
            base_complexity += 2
        
        # Database size complexity
        if config['database_size_gb'] > 10000:
            complexity_factors.append(('Large database size', 1.5))
            base_complexity += 1.5
        elif config['database_size_gb'] > 5000:
            complexity_factors.append(('Medium database size', 0.5))
            base_complexity += 0.5
        
        # Performance requirements
        if config['performance_requirements'] == 'high':
            complexity_factors.append(('High performance requirements', 1))
            base_complexity += 1
        
        # Environment complexity
        if config['environment'] == 'production':
            complexity_factors.append(('Production environment', 0.5))
            base_complexity += 0.5
        
        # OS complexity
        if 'windows' in config['operating_system']:
            complexity_factors.append(('Windows licensing considerations', 0.5))
            base_complexity += 0.5
        
        # Downtime constraints
        if config['downtime_tolerance_minutes'] < 60:
            complexity_factors.append(('Strict downtime requirements', 1))
            base_complexity += 1
        
        # Agent scaling impact
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 3:
            complexity_factors.append(('Multi-agent coordination complexity', 0.5))
            base_complexity += 0.5
        elif num_agents == 1:
            complexity_factors.append(('Single agent bottleneck risk', 0.3))
            base_complexity += 0.3
        
        # Destination storage complexity
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Windows':
            complexity_factors.append(('FSx for Windows File System complexity', 0.8))
            base_complexity += 0.8
        elif destination_storage == 'FSx_Lustre':
            complexity_factors.append(('FSx for Lustre high-performance complexity', 1.0))
            base_complexity += 1.0
        
        complexity_score = min(10, max(1, base_complexity))
        
        # Generate detailed risk assessment
        risk_factors = []
        risk_percentages = {}
        
        if config['source_database_engine'] != config['database_engine']:
            risk_factors.append("Schema conversion complexity may cause compatibility issues")
            risk_percentages['schema_conversion_risk'] = 25
        
        if config['database_size_gb'] > 5000:
            risk_factors.append("Large database size increases migration time and failure probability")
            risk_percentages['large_database_risk'] = 15
        
        if config['downtime_tolerance_minutes'] < 120:
            risk_factors.append("Tight downtime window may require multiple attempts")
            risk_percentages['downtime_risk'] = 20
        
        # Agent-specific risks
        if num_agents == 1 and config['database_size_gb'] > 5000:
            risk_factors.append("Single agent may become throughput bottleneck")
            risk_percentages['agent_bottleneck_risk'] = 30
        elif num_agents > 5:
            risk_factors.append("Complex multi-agent coordination may cause synchronization issues")
            risk_percentages['coordination_risk'] = 15
        
        # Destination storage risks
        if destination_storage == 'FSx_Windows':
            risk_factors.append("FSx for Windows may require additional Active Directory integration")
            risk_percentages['fsx_windows_risk'] = 10
        elif destination_storage == 'FSx_Lustre':
            risk_factors.append("FSx for Lustre high-performance configuration complexity")
            risk_percentages['fsx_lustre_risk'] = 15
        
        perf_score = performance_data.get('performance_score', 0)
        if perf_score < 70:
            risk_factors.append("Current performance issues may impact migration success")
            risk_percentages['performance_risk'] = 30
        
        # Generate detailed recommendations
        performance_recommendations = []
        performance_improvements = {}
        
        if perf_score < 80:
            performance_recommendations.append("Optimize database queries and indexes before migration")
            performance_improvements['query_optimization'] = '15-25%'
        
        if config['ram_gb'] < 32:
            performance_recommendations.append("Consider memory upgrade for better migration throughput")
            performance_improvements['memory_upgrade'] = '20-30%'
        
        # Agent scaling recommendations
        if num_agents > 1:
            performance_recommendations.append(f"Optimize {num_agents} agents for parallel processing")
            performance_improvements['agent_scaling'] = f'{min(num_agents * 15, 75)}%'
        
        # Destination storage recommendations
        if destination_storage == 'FSx_Lustre':
            performance_recommendations.append("Leverage FSx for Lustre high-performance capabilities")
            performance_improvements['fsx_lustre_performance'] = '40-60%'
        elif destination_storage == 'FSx_Windows':
            performance_recommendations.append("Optimize FSx for Windows file sharing protocols")
            performance_improvements['fsx_windows_optimization'] = '20-35%'
        
        # Enhanced timeline with phases
        timeline_suggestions = [
            "Phase 1: Assessment and Planning (2-3 weeks)",
            "Phase 2: Environment Setup and Testing (2-4 weeks)", 
            "Phase 3: Data Validation and Performance Testing (1-2 weeks)",
            "Phase 4: Migration Execution (1-3 days)",
            "Phase 5: Post-Migration Validation and Optimization (1 week)"
        ]
        
        # Detailed resource allocation with agent scaling considerations
        resource_allocation = {
            'migration_team_size': 3 + (complexity_score // 3) + (num_agents // 3),
            'aws_specialists_needed': 1 if complexity_score < 6 else 2,
            'database_experts_required': 1 if config['source_database_engine'] == config['database_engine'] else 2,
            'testing_resources': '2-3 dedicated testers for ' + ('2 weeks' if complexity_score < 7 else '3-4 weeks'),
            'infrastructure_requirements': f"Staging environment with {config['cpu_cores']*2} cores and {config['ram_gb']*1.5} GB RAM",
            'agent_management_overhead': f"{num_agents} agents require {max(1, num_agents // 2)} dedicated monitoring specialists",
            'storage_specialists': 1 if destination_storage != 'S3' else 0
        }
        
        # Agent scaling impact analysis
        agent_scaling_impact = {
            'parallel_processing_benefit': min(num_agents * 20, 80),  # Diminishing returns
            'coordination_overhead': max(0, (num_agents - 1) * 5),
            'throughput_multiplier': min(num_agents * 0.8, 4.0),  # Not linear scaling
            'management_complexity': num_agents * 10,
            'optimal_agent_count': self._calculate_optimal_agents(config),
            'current_efficiency': min(100, (100 - (abs(num_agents - self._calculate_optimal_agents(config)) * 10)))
        }
        
        # Destination storage impact analysis
        destination_storage_impact = {
            'storage_type': destination_storage,
            'performance_impact': self._calculate_storage_performance_impact(destination_storage),
            'cost_impact': self._calculate_storage_cost_impact(destination_storage),
            'complexity_factor': self._get_storage_complexity_factor(destination_storage),
            'recommended_for': self._get_storage_recommendations(destination_storage, config)
        }
        
        return {
            'complexity_score': complexity_score,
            'complexity_factors': complexity_factors,
            'risk_factors': risk_factors,
            'risk_percentages': risk_percentages,
            'mitigation_strategies': self._generate_mitigation_strategies(risk_factors, config),
            'performance_recommendations': performance_recommendations,
            'performance_improvements': performance_improvements,
            'timeline_suggestions': timeline_suggestions,
            'resource_allocation': resource_allocation,
            'cost_optimization': self._generate_cost_optimization(config, complexity_score),
            'best_practices': self._generate_best_practices(config, complexity_score),
            'testing_strategy': self._generate_testing_strategy(config, complexity_score),
            'rollback_procedures': self._generate_rollback_procedures(config),
            'post_migration_monitoring': self._generate_monitoring_recommendations(config),
            'confidence_level': 'high' if complexity_score < 6 else 'medium' if complexity_score < 8 else 'requires_specialist_review',
            'agent_scaling_impact': agent_scaling_impact,
            'destination_storage_impact': destination_storage_impact,
            'detailed_assessment': {
                'overall_readiness': 'ready' if perf_score > 75 and complexity_score < 7 else 'needs_preparation' if perf_score > 60 else 'significant_preparation_required',
                'success_probability': max(60, 95 - (complexity_score * 5) - max(0, (70 - perf_score)) + (agent_scaling_impact['current_efficiency'] // 10)),
                'recommended_approach': 'direct_migration' if complexity_score < 6 and config['database_size_gb'] < 2000 else 'staged_migration',
                'critical_success_factors': self._identify_critical_success_factors(config, complexity_score)
            }
        }
    
    def _calculate_optimal_agents(self, config: Dict) -> int:
        """Calculate optimal number of agents based on database size and requirements"""
        database_size = config['database_size_gb']
        
        if database_size < 1000:
            return 1
        elif database_size < 5000:
            return 2
        elif database_size < 20000:
            return 3
        elif database_size < 50000:
            return 4
        else:
            return 5
    
    def _calculate_storage_performance_impact(self, storage_type: str) -> Dict:
        """Calculate performance impact for different storage destinations"""
        storage_profiles = {
            'S3': {
                'throughput_multiplier': 1.0,
                'latency_impact': 1.0,
                'iops_capability': 'Standard',
                'performance_rating': 'Good'
            },
            'FSx_Windows': {
                'throughput_multiplier': 1.3,
                'latency_impact': 0.7,
                'iops_capability': 'High',
                'performance_rating': 'Very Good'
            },
            'FSx_Lustre': {
                'throughput_multiplier': 2.0,
                'latency_impact': 0.4,
                'iops_capability': 'Very High',
                'performance_rating': 'Excellent'
            }
        }
        return storage_profiles.get(storage_type, storage_profiles['S3'])
    
    def _calculate_storage_cost_impact(self, storage_type: str) -> Dict:
        """Calculate cost impact for different storage destinations"""
        cost_profiles = {
            'S3': {
                'base_cost_multiplier': 1.0,
                'operational_cost': 'Low',
                'setup_cost': 'Minimal',
                'long_term_value': 'Excellent'
            },
            'FSx_Windows': {
                'base_cost_multiplier': 2.5,
                'operational_cost': 'Medium',
                'setup_cost': 'Medium',
                'long_term_value': 'Good'
            },
            'FSx_Lustre': {
                'base_cost_multiplier': 4.0,
                'operational_cost': 'High',
                'setup_cost': 'High',
                'long_term_value': 'Good for HPC'
            }
        }
        return cost_profiles.get(storage_type, cost_profiles['S3'])
    
    def _get_storage_complexity_factor(self, storage_type: str) -> float:
        """Get complexity factor for storage type"""
        complexity_factors = {
            'S3': 1.0,
            'FSx_Windows': 1.8,
            'FSx_Lustre': 2.2
        }
        return complexity_factors.get(storage_type, 1.0)
    
    def _get_storage_recommendations(self, storage_type: str, config: Dict) -> List[str]:
        """Get recommendations for storage type"""
        recommendations = {
            'S3': [
                "Cost-effective for most workloads",
                "Excellent durability and availability",
                "Simple integration with migration tools",
                "Automatic scaling and management"
            ],
            'FSx_Windows': [
                "Ideal for Windows-based applications",
                "Native Windows file system features",
                "Active Directory integration",
                "Better performance for file-based workloads"
            ],
            'FSx_Lustre': [
                "Best for high-performance computing",
                "Extremely high throughput and IOPS",
                "Optimized for parallel processing",
                "Ideal for analytics and ML workloads"
            ]
        }
        return recommendations.get(storage_type, recommendations['S3'])
    
    def _generate_mitigation_strategies(self, risk_factors: List[str], config: Dict) -> List[str]:
        """Generate specific mitigation strategies"""
        strategies = []
        
        if any('schema' in risk.lower() for risk in risk_factors):
            strategies.append("Conduct comprehensive schema conversion testing with AWS SCT")
            strategies.append("Create detailed schema mapping documentation")
            strategies.append("Implement phased schema migration with rollback checkpoints")
        
        if any('database size' in risk.lower() for risk in risk_factors):
            strategies.append("Implement parallel data transfer using multiple DMS tasks")
            strategies.append("Use AWS DataSync for initial bulk data transfer")
            strategies.append("Schedule migration during low-traffic periods")
        
        if any('downtime' in risk.lower() for risk in risk_factors):
            strategies.append("Implement read replica for near-zero downtime migration")
            strategies.append("Use AWS DMS ongoing replication for data synchronization")
            strategies.append("Prepare detailed rollback procedures with time estimates")
        
        if any('performance' in risk.lower() for risk in risk_factors):
            strategies.append("Conduct pre-migration performance tuning")
            strategies.append("Implement AWS CloudWatch monitoring throughout migration")
            strategies.append("Establish performance baselines and acceptance criteria")
        
        if any('agent' in risk.lower() for risk in risk_factors):
            strategies.append(f"Optimize {config.get('number_of_agents', 1)} agent configuration for workload")
            strategies.append("Implement agent health monitoring and automatic failover")
            strategies.append("Configure load balancing across multiple agents")
        
        if any('fsx' in risk.lower() for risk in risk_factors):
            strategies.append("Test FSx integration thoroughly in staging environment")
            strategies.append("Implement FSx performance monitoring and optimization")
            strategies.append("Plan for FSx-specific backup and recovery procedures")
        
        return strategies
    
    def _generate_cost_optimization(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate cost optimization strategies"""
        optimizations = []
        
        if config['database_size_gb'] < 1000:
            optimizations.append("Consider Reserved Instances for 20-30% cost savings")
        
        if config['environment'] == 'non-production':
            optimizations.append("Use Spot Instances for development/testing to reduce costs by 60-70%")
        
        if complexity_score < 6:
            optimizations.append("Leverage AWS Managed Services (RDS) to reduce operational overhead")
        
        optimizations.append("Implement automated scaling policies to optimize resource utilization")
        
        # Storage-specific optimizations
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'S3':
            optimizations.append("Use S3 Intelligent Tiering for backup storage cost optimization")
        elif destination_storage == 'FSx_Windows':
            optimizations.append("Right-size FSx for Windows based on actual usage patterns")
        elif destination_storage == 'FSx_Lustre':
            optimizations.append("Use FSx for Lustre scratch file systems for temporary high-performance needs")
        
        # Agent-specific optimizations
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 3:
            optimizations.append("Consider agent consolidation to reduce licensing and management costs")
        elif num_agents == 1 and config['database_size_gb'] > 5000:
            optimizations.append("Scale up to multiple agents for faster migration and reduced window costs")
        
        return optimizations
    
    def _generate_best_practices(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate specific best practices"""
        practices = []
        
        practices.append("Implement comprehensive backup strategy before migration initiation")
        practices.append("Use AWS Migration Hub for centralized migration tracking")
        practices.append("Establish detailed communication plan with stakeholders")
        
        if config['database_engine'] in ['mysql', 'postgresql']:
            practices.append("Leverage native database replication for minimal downtime")
        
        if complexity_score > 7:
            practices.append("Engage AWS Professional Services for complex migration scenarios")
        
        practices.append("Implement automated testing pipelines for validation")
        practices.append("Create detailed runbook with step-by-step procedures")
        
        # Agent-specific best practices
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 1:
            practices.append(f"Configure {num_agents} agents with proper load distribution")
            practices.append("Implement centralized agent monitoring and logging")
            practices.append("Test agent failover scenarios during non-production phases")
        
        # Storage-specific best practices
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Windows':
            practices.append("Configure FSx for Windows with appropriate backup policies")
            practices.append("Implement proper Active Directory integration for FSx")
        elif destination_storage == 'FSx_Lustre':
            practices.append("Optimize FSx for Lustre configuration for specific workload patterns")
            practices.append("Plan for FSx for Lustre data repository associations")
        
        return practices
    
    def _generate_testing_strategy(self, config: Dict, complexity_score: int) -> List[str]:
        """Generate comprehensive testing strategy"""
        strategy = []
        
        strategy.append("Unit Testing: Validate individual migration components")
        strategy.append("Integration Testing: Test end-to-end migration workflow")
        strategy.append("Performance Testing: Validate AWS environment performance")
        strategy.append("Data Integrity Testing: Verify data consistency and completeness")
        
        if config['source_database_engine'] != config['database_engine']:
            strategy.append("Schema Conversion Testing: Validate converted database objects")
            strategy.append("Application Compatibility Testing: Ensure application functionality")
        
        strategy.append("Disaster Recovery Testing: Validate backup and restore procedures")
        strategy.append("Security Testing: Verify access controls and encryption")
        
        # Agent-specific testing
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 1:
            strategy.append(f"Multi-Agent Testing: Validate {num_agents} agent coordination")
            strategy.append("Load Balancing Testing: Verify even distribution across agents")
        
        # Storage-specific testing
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage != 'S3':
            strategy.append(f"FSx Performance Testing: Validate {destination_storage} performance characteristics")
            strategy.append(f"FSx Integration Testing: Test application connectivity to {destination_storage}")
        
        return strategy
    
    def _generate_rollback_procedures(self, config: Dict) -> List[str]:
        """Generate rollback procedures"""
        procedures = []
        
        procedures.append("Maintain synchronized read replica during migration window")
        procedures.append("Create point-in-time recovery snapshot before cutover")
        procedures.append("Prepare DNS switching procedures for quick rollback")
        procedures.append("Document application configuration rollback steps")
        procedures.append("Establish go/no-go decision criteria with specific metrics")
        procedures.append("Test rollback procedures in staging environment")
        
        # Agent-specific rollback procedures
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 1:
            procedures.append(f"Coordinate {num_agents} agent shutdown procedures")
            procedures.append("Implement agent state synchronization for rollback")
        
        # Storage-specific rollback
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage != 'S3':
            procedures.append(f"Prepare {destination_storage} rollback and data recovery procedures")
            procedures.append(f"Test {destination_storage} backup and restore capabilities")
        
        return procedures
    
    def _generate_monitoring_recommendations(self, config: Dict) -> List[str]:
        """Generate post-migration monitoring recommendations"""
        monitoring = []
        
        monitoring.append("Implement CloudWatch detailed monitoring for all database metrics")
        monitoring.append("Set up automated alerts for performance degradation")
        monitoring.append("Monitor application response times and error rates")
        monitoring.append("Track database connection patterns and query performance")
        monitoring.append("Implement cost monitoring and optimization alerts")
        monitoring.append("Schedule regular performance reviews for first 30 days")
        
        # Agent-specific monitoring
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 1:
            monitoring.append(f"Monitor {num_agents} agent performance and health metrics")
            monitoring.append("Track agent load distribution and efficiency")
        
        # Storage-specific monitoring
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Windows':
            monitoring.append("Monitor FSx for Windows file system performance and utilization")
            monitoring.append("Track Windows file sharing protocol efficiency")
        elif destination_storage == 'FSx_Lustre':
            monitoring.append("Monitor FSx for Lustre high-performance metrics")
            monitoring.append("Track parallel processing efficiency and throughput")
        
        return monitoring
    
    def _identify_critical_success_factors(self, config: Dict, complexity_score: int) -> List[str]:
        """Identify critical success factors"""
        factors = []
        
        factors.append("Stakeholder alignment on migration timeline and expectations")
        factors.append("Comprehensive testing in staging environment matching production")
        factors.append("Skilled migration team with AWS and database expertise")
        
        if complexity_score > 7:
            factors.append("Dedicated AWS solutions architect involvement")
        
        if config['downtime_tolerance_minutes'] < 120:
            factors.append("Near-zero downtime migration strategy implementation")
        
        factors.append("Robust monitoring and alerting throughout migration process")
        factors.append("Clear rollback criteria and tested rollback procedures")
        
        # Agent-specific success factors
        num_agents = config.get('number_of_agents', 1)
        if num_agents > 2:
            factors.append(f"Proper coordination and management of {num_agents} migration agents")
            factors.append("Agent performance optimization and load balancing")
        
        # Storage-specific success factors
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage != 'S3':
            factors.append(f"Successful {destination_storage} integration and performance validation")
            factors.append(f"Proper {destination_storage} configuration and optimization")
        
        return factors
    
    def _fallback_workload_analysis(self, config: Dict, performance_data: Dict) -> Dict:
        """Enhanced fallback analysis when AI is not available"""
        
        # Calculate complexity based on configuration
        complexity_score = 5
        if config['source_database_engine'] != config['database_engine']:
            complexity_score += 2
        if config['database_size_gb'] > 5000:
            complexity_score += 1
        if config['performance_requirements'] == 'high':
            complexity_score += 1
        if config['downtime_tolerance_minutes'] < 60:
            complexity_score += 1
        
        # Storage complexity
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Windows':
            complexity_score += 0.8
        elif destination_storage == 'FSx_Lustre':
            complexity_score += 1.0
        
        # Agent scaling impact
        num_agents = config.get('number_of_agents', 1)
        optimal_agents = self._calculate_optimal_agents(config)
        agent_efficiency = min(100, (100 - (abs(num_agents - optimal_agents) * 10)))
        
        return {
            'ai_complexity_score': min(10, complexity_score),
            'risk_factors': [
                "Migration complexity varies with database engine differences",
                "Large database sizes increase migration duration",
                "Performance requirements may necessitate additional testing",
                f"Agent configuration ({num_agents} agents) may impact throughput",
                f"Destination storage ({destination_storage}) affects performance and complexity"
            ],
            'mitigation_strategies': [
                "Conduct thorough pre-migration testing",
                "Plan for adequate migration windows",
                "Implement comprehensive backup strategies",
                f"Optimize {num_agents} agent configuration for workload",
                f"Validate {destination_storage} integration thoroughly"
            ],
            'performance_recommendations': [
                "Optimize database before migration",
                "Consider read replicas for minimal downtime",
                "Monitor performance throughout migration",
                f"Fine-tune {num_agents} agent performance settings",
                f"Leverage {destination_storage} performance characteristics"
            ],
            'confidence_level': 'medium',
            'agent_scaling_impact': {
                'current_efficiency': agent_efficiency,
                'optimal_agent_count': optimal_agents,
                'throughput_multiplier': min(num_agents * 0.8, 4.0)
            },
            'destination_storage_impact': {
                'storage_type': destination_storage,
                'performance_impact': self._calculate_storage_performance_impact(destination_storage),
                'cost_impact': self._calculate_storage_cost_impact(destination_storage),
                'complexity_factor': self._get_storage_complexity_factor(destination_storage)
            },
            'resource_allocation': {
                'migration_team_size': 3 + (num_agents // 2),
                'aws_specialists_needed': 1,
                'database_experts_required': 1,
                'storage_specialists': 1 if destination_storage != 'S3' else 0
            },
            'detailed_assessment': {
                'success_probability': max(60, 85 - complexity_score * 3 + (agent_efficiency // 10))
            },
            'raw_ai_response': 'AI analysis not available - using fallback analysis'
        }

class AWSAPIManager:
    """Manage AWS API integration for real-time pricing and optimization"""
    
    def __init__(self):
        self.pricing_client = None
        self.connected = False
        self.error_message = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize connection to AWS APIs"""
        try:
            # Try to initialize AWS clients
            import os
            if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY'):
                self.pricing_client = boto3.client('pricing', region_name='us-east-1')
                self.connected = True
                self.error_message = None
            else:
                self.connected = False
                self.error_message = "AWS credentials not found"
                
        except Exception as e:
            self.connected = False
            self.error_message = str(e)
    
    async def get_real_time_pricing(self, region: str = 'us-west-2') -> Dict:
        """Fetch real-time AWS pricing data including FSx"""
        if not self.connected:
            return self._fallback_pricing_data(region)
        
        try:
            # Implementation would go here
            return self._fallback_pricing_data(region)
            
        except Exception as e:
            logger.error(f"Failed to fetch AWS pricing: {e}")
            return self._fallback_pricing_data(region)
    
    def _fallback_pricing_data(self, region: str) -> Dict:
        """Fallback pricing data when API is unavailable"""
        return {
            'region': region,
            'last_updated': datetime.now(),
            'data_source': 'fallback',
            'ec2_instances': {},
            'rds_instances': {},
            'storage': {},
            'fsx': {}
        }
    
    
    async def get_real_time_pricing(self, region: str = 'us-west-2') -> Dict:
        """Fetch real-time AWS pricing data including FSx"""
        if not self.connected:
            return self._fallback_pricing_data(region)
        
        try:
            # Get EC2 pricing
            ec2_pricing = await self._get_ec2_pricing(region)
            
            # Get RDS pricing
            rds_pricing = await self._get_rds_pricing(region)
            
            # Get storage pricing (S3, EBS)
            storage_pricing = await self._get_storage_pricing(region)
            
            # Get FSx pricing
            fsx_pricing = await self._get_fsx_pricing(region)
            
            return {
                'region': region,
                'last_updated': datetime.now(),
                'ec2_instances': ec2_pricing,
                'rds_instances': rds_pricing,
                'storage': storage_pricing,
                'fsx': fsx_pricing,
                'data_source': 'aws_api'
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch AWS pricing: {e}")
            return self._fallback_pricing_data(region)
    
    async def _get_fsx_pricing(self, region: str) -> Dict:
        """Get FSx pricing for Windows and Lustre"""
        try:
            fsx_pricing = {}
            
            # FSx for Windows File Server
            try:
                response = self.pricing_client.get_products(
                    ServiceCode='AmazonFSx',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'fileSystemType', 'Value': 'Windows'},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                        {'Type': 'TERM_MATCH', 'Field': 'storageType', 'Value': 'SSD'}
                    ],
                    MaxResults=1
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data.get('terms', {}).get('OnDemand', {})
                    if terms:
                        term_data = list(terms.values())[0]
                        price_dimensions = term_data.get('priceDimensions', {})
                        if price_dimensions:
                            price_info = list(price_dimensions.values())[0]
                            price_per_gb_month = float(price_info['pricePerUnit']['USD'])
                            
                            fsx_pricing['windows'] = {
                                'price_per_gb_month': price_per_gb_month,
                                'minimum_size_gb': 32,
                                'maximum_size_gb': 65536,
                                'throughput_capacity_mbps': [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                                'backup_retention': True,
                                'multi_az': True
                            }
                            
            except Exception as e:
                logger.warning(f"Failed to get FSx Windows pricing: {e}")
                fsx_pricing['windows'] = self._get_fallback_fsx_windows_pricing()
            
            # FSx for Lustre
            try:
                response = self.pricing_client.get_products(
                    ServiceCode='AmazonFSx',
                    Filters=[
                        {'Type': 'TERM_MATCH', 'Field': 'fileSystemType', 'Value': 'Lustre'},
                        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                        {'Type': 'TERM_MATCH', 'Field': 'storageType', 'Value': 'SSD'}
                    ],
                    MaxResults=1
                )
                
                if response['PriceList']:
                    price_data = json.loads(response['PriceList'][0])
                    terms = price_data.get('terms', {}).get('OnDemand', {})
                    if terms:
                        term_data = list(terms.values())[0]
                        price_dimensions = term_data.get('priceDimensions', {})
                        if price_dimensions:
                            price_info = list(price_dimensions.values())[0]
                            price_per_gb_month = float(price_info['pricePerUnit']['USD'])
                            
                            fsx_pricing['lustre'] = {
                                'price_per_gb_month': price_per_gb_month,
                                'minimum_size_gb': 1200,
                                'maximum_size_gb': 100800,
                                'throughput_per_tib': [50, 100, 200],
                                'deployment_type': ['SCRATCH_1', 'SCRATCH_2', 'PERSISTENT_1', 'PERSISTENT_2'],
                                'data_repository_association': True
                            }
                            
            except Exception as e:
                logger.warning(f"Failed to get FSx Lustre pricing: {e}")
                fsx_pricing['lustre'] = self._get_fallback_fsx_lustre_pricing()
            
            return fsx_pricing
            
        except Exception as e:
            logger.error(f"FSx pricing fetch failed: {e}")
            return self._fallback_fsx_pricing()
    
    async def _get_ec2_pricing(self, region: str) -> Dict:
        """Get EC2 instance pricing"""
        try:
            # AWS Pricing API calls for EC2 instances
            instance_types = ['t3.medium', 't3.large', 't3.xlarge', 'c5.large', 'c5.xlarge', 
                            'c5.2xlarge', 'c5.4xlarge', 'r6i.large', 'r6i.xlarge', 
                            'r6i.2xlarge', 'r6i.4xlarge', 'r6i.8xlarge']
            
            pricing_data = {}
            
            for instance_type in instance_types:
                try:
                    response = self.pricing_client.get_products(
                        ServiceCode='AmazonEC2',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                            {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                            {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                            {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'}
                        ],
                        MaxResults=1
                    )
                    
                    if response['PriceList']:
                        price_data = json.loads(response['PriceList'][0])
                        # Extract pricing information
                        terms = price_data.get('terms', {}).get('OnDemand', {})
                        if terms:
                            term_data = list(terms.values())[0]
                            price_dimensions = term_data.get('priceDimensions', {})
                            if price_dimensions:
                                price_info = list(price_dimensions.values())[0]
                                price_per_hour = float(price_info['pricePerUnit']['USD'])
                                
                                # Get instance specs
                                attributes = price_data.get('product', {}).get('attributes', {})
                                
                                pricing_data[instance_type] = {
                                    'vcpu': int(attributes.get('vcpu', 2)),
                                    'memory': self._extract_memory_gb(attributes.get('memory', '4 GiB')),
                                    'cost_per_hour': price_per_hour
                                }
                                
                except Exception as e:
                    logger.warning(f"Failed to get pricing for {instance_type}: {e}")
                    # Use fallback pricing
                    pricing_data[instance_type] = self._get_fallback_instance_pricing(instance_type)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"EC2 pricing fetch failed: {e}")
            return self._fallback_ec2_pricing()
    
    async def _get_rds_pricing(self, region: str) -> Dict:
        """Get RDS instance pricing"""
        try:
            instance_types = ['db.t3.medium', 'db.t3.large', 'db.r6g.large', 'db.r6g.xlarge', 
                            'db.r6g.2xlarge', 'db.r6g.4xlarge', 'db.r6g.8xlarge']
            
            pricing_data = {}
            
            for instance_type in instance_types:
                try:
                    response = self.pricing_client.get_products(
                        ServiceCode='AmazonRDS',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)},
                            {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': 'MySQL'},
                            {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Single-AZ'}
                        ],
                        MaxResults=1
                    )
                    
                    if response['PriceList']:
                        price_data = json.loads(response['PriceList'][0])
                        terms = price_data.get('terms', {}).get('OnDemand', {})
                        if terms:
                            term_data = list(terms.values())[0]
                            price_dimensions = term_data.get('priceDimensions', {})
                            if price_dimensions:
                                price_info = list(price_dimensions.values())[0]
                                price_per_hour = float(price_info['pricePerUnit']['USD'])
                                
                                attributes = price_data.get('product', {}).get('attributes', {})
                                
                                pricing_data[instance_type] = {
                                    'vcpu': int(attributes.get('vcpu', 2)),
                                    'memory': self._extract_memory_gb(attributes.get('memory', '4 GiB')),
                                    'cost_per_hour': price_per_hour
                                }
                                
                except Exception as e:
                    logger.warning(f"Failed to get RDS pricing for {instance_type}: {e}")
                    pricing_data[instance_type] = self._get_fallback_rds_pricing(instance_type)
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"RDS pricing fetch failed: {e}")
            return self._fallback_rds_pricing()
    
    async def _get_storage_pricing(self, region: str) -> Dict:
        """Get EBS and S3 storage pricing"""
        try:
            storage_types = ['gp3', 'io1', 'io2']
            pricing_data = {}
            
            for storage_type in storage_types:
                try:
                    volume_type_map = {
                        'gp3': 'General Purpose',
                        'io1': 'Provisioned IOPS',
                        'io2': 'Provisioned IOPS'
                    }
                    
                    response = self.pricing_client.get_products(
                        ServiceCode='AmazonEC2',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'productFamily', 'Value': 'Storage'},
                            {'Type': 'TERM_MATCH', 'Field': 'volumeType', 'Value': volume_type_map.get(storage_type, 'General Purpose')},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._region_to_location(region)}
                        ],
                        MaxResults=1
                    )
                    
                    if response['PriceList']:
                        price_data = json.loads(response['PriceList'][0])
                        terms = price_data.get('terms', {}).get('OnDemand', {})
                        if terms:
                            term_data = list(terms.values())[0]
                            price_dimensions = term_data.get('priceDimensions', {})
                            if price_dimensions:
                                price_info = list(price_dimensions.values())[0]
                                price_per_gb = float(price_info['pricePerUnit']['USD'])
                                
                                pricing_data[storage_type] = {
                                    'cost_per_gb_month': price_per_gb,
                                    'iops_included': 3000 if storage_type == 'gp3' else 0,
                                    'cost_per_iops_month': 0.065 if storage_type in ['io1', 'io2'] else 0
                                }
                                
                except Exception as e:
                    logger.warning(f"Failed to get storage pricing for {storage_type}: {e}")
                    pricing_data[storage_type] = self._get_fallback_storage_pricing(storage_type)
            
            # Add S3 pricing
            pricing_data['s3_standard'] = {
                'cost_per_gb_month': 0.023,
                'requests_per_1000': 0.0004,
                'data_transfer_out_per_gb': 0.09
            }
            
            return pricing_data
            
        except Exception as e:
            logger.error(f"Storage pricing fetch failed: {e}")
            return self._fallback_storage_pricing()
    
    def _region_to_location(self, region: str) -> str:
        """Convert AWS region to location name for pricing API"""
        region_map = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'Europe (Ireland)',
            'ap-southeast-1': 'Asia Pacific (Singapore)'
        }
        return region_map.get(region, 'US West (Oregon)')
    
    def _extract_memory_gb(self, memory_str: str) -> int:
        """Extract memory in GB from AWS memory string"""
        try:
            # Extract number from strings like "4 GiB", "8.0 GiB"
            import re
            match = re.search(r'([\d.]+)', memory_str)
            if match:
                return int(float(match.group(1)))
            return 4  # Default
        except:
            return 4
    
    def _fallback_pricing_data(self, region: str) -> Dict:
        """Fallback pricing data when API is unavailable"""
        return {
            'region': region,
            'last_updated': datetime.now(),
            'data_source': 'fallback',
            'ec2_instances': self._fallback_ec2_pricing(),
            'rds_instances': self._fallback_rds_pricing(),
            'storage': self._fallback_storage_pricing(),
            'fsx': self._fallback_fsx_pricing()
        }
    
    def _fallback_ec2_pricing(self) -> Dict:
        """Fallback EC2 pricing data"""
        return {
            't3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.0416},
            't3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.0832},
            't3.xlarge': {'vcpu': 4, 'memory': 16, 'cost_per_hour': 0.1664},
            'c5.large': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.085},
            'c5.xlarge': {'vcpu': 4, 'memory': 8, 'cost_per_hour': 0.17},
            'c5.2xlarge': {'vcpu': 8, 'memory': 16, 'cost_per_hour': 0.34},
            'c5.4xlarge': {'vcpu': 16, 'memory': 32, 'cost_per_hour': 0.68},
            'r6i.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.252},
            'r6i.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.504},
            'r6i.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.008},
            'r6i.4xlarge': {'vcpu': 16, 'memory': 128, 'cost_per_hour': 2.016},
            'r6i.8xlarge': {'vcpu': 32, 'memory': 256, 'cost_per_hour': 4.032}
        }
    
    def _fallback_rds_pricing(self) -> Dict:
        """Fallback RDS pricing data"""
        return {
            'db.t3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.068},
            'db.t3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.136},
            'db.r6g.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.48},
            'db.r6g.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.96},
            'db.r6g.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.92},
            'db.r6g.4xlarge': {'vcpu': 16, 'memory': 128, 'cost_per_hour': 3.84},
            'db.r6g.8xlarge': {'vcpu': 32, 'memory': 256, 'cost_per_hour': 7.68}
        }
    
    def _fallback_storage_pricing(self) -> Dict:
        """Fallback storage pricing data"""
        return {
            'gp3': {'cost_per_gb_month': 0.08, 'iops_included': 3000},
            'io1': {'cost_per_gb_month': 0.125, 'cost_per_iops_month': 0.065},
            'io2': {'cost_per_gb_month': 0.125, 'cost_per_iops_month': 0.065},
            's3_standard': {
                'cost_per_gb_month': 0.023,
                'requests_per_1000': 0.0004,
                'data_transfer_out_per_gb': 0.09
            }
        }
    
    def _fallback_fsx_pricing(self) -> Dict:
        """Fallback FSx pricing data"""
        return {
            'windows': self._get_fallback_fsx_windows_pricing(),
            'lustre': self._get_fallback_fsx_lustre_pricing()
        }
    
    def _get_fallback_fsx_windows_pricing(self) -> Dict:
        """Get fallback FSx Windows pricing"""
        return {
            'price_per_gb_month': 0.13,
            'minimum_size_gb': 32,
            'maximum_size_gb': 65536,
            'throughput_capacity_mbps': [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            'backup_retention': True,
            'multi_az': True
        }
    
    def _get_fallback_fsx_lustre_pricing(self) -> Dict:
        """Get fallback FSx Lustre pricing"""
        return {
            'price_per_gb_month': 0.14,
            'minimum_size_gb': 1200,
            'maximum_size_gb': 100800,
            'throughput_per_tib': [50, 100, 200],
            'deployment_type': ['SCRATCH_1', 'SCRATCH_2', 'PERSISTENT_1', 'PERSISTENT_2'],
            'data_repository_association': True
        }
    
    def _get_fallback_instance_pricing(self, instance_type: str) -> Dict:
        """Get fallback pricing for specific instance type"""
        fallback_data = self._fallback_ec2_pricing()
        return fallback_data.get(instance_type, {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.05})
    
    def _get_fallback_rds_pricing(self, instance_type: str) -> Dict:
        """Get fallback RDS pricing for specific instance type"""
        fallback_data = self._fallback_rds_pricing()
        return fallback_data.get(instance_type, {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.07})
    
    def _get_fallback_storage_pricing(self, storage_type: str) -> Dict:
        """Get fallback storage pricing for specific type"""
        fallback_data = self._fallback_storage_pricing()
        return fallback_data.get(storage_type, {'cost_per_gb_month': 0.08, 'iops_included': 0})

# Fixed OS Performance Manager
class OSPerformanceManager:
    """Enhanced OS performance manager with AI insights"""
    
    def __init__(self):
        self.operating_systems = {
            'windows_server_2019': {
                'name': 'Windows Server 2019',
                'cpu_efficiency': 0.88,
                'memory_efficiency': 0.85,
                'io_efficiency': 0.87,
                'network_efficiency': 0.90,
                'licensing_cost_factor': 2.0,
                'management_complexity': 0.6,
                'security_overhead': 0.05,
                'virtualization_overhead': 0.08,
                'database_optimizations': {
                    'mysql': 0.85,
                    'postgresql': 0.82,
                    'oracle': 0.90,
                    'sqlserver': 0.95,
                    'mongodb': 0.80
                },
                'ai_insights': {
                    'strengths': ['Good SQL Server integration', 'Familiar management tools', 'Enterprise features'],
                    'weaknesses': ['Higher licensing costs', 'More resource overhead'],
                    'migration_considerations': ['License optimization needed', 'Consider newer versions']
                }
            },
            'windows_server_2022': {
                'name': 'Windows Server 2022',
                'cpu_efficiency': 0.90,
                'memory_efficiency': 0.88,
                'io_efficiency': 0.89,
                'network_efficiency': 0.92,
                'licensing_cost_factor': 2.2,
                'management_complexity': 0.5,
                'security_overhead': 0.03,
                'virtualization_overhead': 0.06,
                'database_optimizations': {
                    'mysql': 0.87,
                    'postgresql': 0.85,
                    'oracle': 0.92,
                    'sqlserver': 0.98,
                    'mongodb': 0.83
                },
                'ai_insights': {
                    'strengths': ['Latest Windows features', 'Better security', 'Improved performance'],
                    'weaknesses': ['Higher licensing costs', 'Newer platform considerations'],
                    'migration_considerations': ['Modern platform benefits', 'License optimization']
                }
            },
            'rhel_8': {
                'name': 'Red Hat Enterprise Linux 8',
                'cpu_efficiency': 0.92,
                'memory_efficiency': 0.90,
                'io_efficiency': 0.91,
                'network_efficiency': 0.93,
                'licensing_cost_factor': 1.5,
                'management_complexity': 0.7,
                'security_overhead': 0.02,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.92,
                    'postgresql': 0.95,
                    'oracle': 0.88,
                    'sqlserver': 0.75,
                    'mongodb': 0.90
                },
                'ai_insights': {
                    'strengths': ['Excellent performance', 'Strong security', 'Enterprise support'],
                    'weaknesses': ['Commercial licensing', 'Learning curve'],
                    'migration_considerations': ['Great for database workloads', 'Consider support costs']
                }
            },
            'rhel_9': {
                'name': 'Red Hat Enterprise Linux 9',
                'cpu_efficiency': 0.94,
                'memory_efficiency': 0.92,
                'io_efficiency': 0.93,
                'network_efficiency': 0.95,
                'licensing_cost_factor': 1.6,
                'management_complexity': 0.6,
                'security_overhead': 0.01,
                'virtualization_overhead': 0.04,
                'database_optimizations': {
                    'mysql': 0.94,
                    'postgresql': 0.97,
                    'oracle': 0.90,
                    'sqlserver': 0.78,
                    'mongodb': 0.93
                },
                'ai_insights': {
                    'strengths': ['Latest performance optimizations', 'Enhanced security', 'Modern container support'],
                    'weaknesses': ['Commercial licensing', 'Newer platform'],
                    'migration_considerations': ['Best performance option', 'Modern Linux features']
                }
            },
            'ubuntu_20_04': {
                'name': 'Ubuntu Server 20.04 LTS',
                'cpu_efficiency': 0.91,
                'memory_efficiency': 0.89,
                'io_efficiency': 0.90,
                'network_efficiency': 0.92,
                'licensing_cost_factor': 1.0,
                'management_complexity': 0.8,
                'security_overhead': 0.02,
                'virtualization_overhead': 0.05,
                'database_optimizations': {
                    'mysql': 0.90,
                    'postgresql': 0.93,
                    'oracle': 0.85,
                    'sqlserver': 0.70,
                    'mongodb': 0.88
                },
                'ai_insights': {
                    'strengths': ['No licensing costs', 'Good community support', 'Wide compatibility'],
                    'weaknesses': ['Limited enterprise support', 'Manual management'],
                    'migration_considerations': ['Cost-effective option', 'Consider support needs']
                }
            },
            'ubuntu_22_04': {
                'name': 'Ubuntu Server 22.04 LTS',
                'cpu_efficiency': 0.93,
                'memory_efficiency': 0.91,
                'io_efficiency': 0.92,
                'network_efficiency': 0.94,
                'licensing_cost_factor': 1.0,
                'management_complexity': 0.7,
                'security_overhead': 0.01,
                'virtualization_overhead': 0.04,
                'database_optimizations': {
                    'mysql': 0.92,
                    'postgresql': 0.95,
                    'oracle': 0.87,
                    'sqlserver': 0.73,
                    'mongodb': 0.91
                },
                'ai_insights': {
                    'strengths': ['Latest Ubuntu LTS', 'No licensing costs', 'Modern features'],
                    'weaknesses': ['Limited enterprise support', 'Self-managed'],
                    'migration_considerations': ['Best free option', 'Modern platform features']
                }
            }
        }
        
        # Storage type definitions
        self.storage_types = {
            'nvme_ssd': {
                'iops': 100000,
                'throughput_mbps': 3500,
                'latency_ms': 0.1
            },
            'sata_ssd': {
                'iops': 50000,
                'throughput_mbps': 550,
                'latency_ms': 0.2
            },
            'sas_hdd': {
                'iops': 200,
                'throughput_mbps': 200,
                'latency_ms': 8.0
            }
        }
    
    def extract_database_engine(self, target_database_selection: str, ec2_database_engine: str = None) -> str:
        """Extract the actual database engine from target selection"""
        
        if target_database_selection.startswith('rds_'):
            # For RDS, extract engine from the selection (e.g., 'rds_mysql' -> 'mysql')
            return target_database_selection.replace('rds_', '')
        elif target_database_selection.startswith('ec2_'):
            # For EC2, use the separately selected database engine
            return ec2_database_engine if ec2_database_engine else 'mysql'  # Default to mysql
        else:
            # Fallback for any other format
            return target_database_selection
    
    def calculate_os_performance_impact(self, os_type: str, platform_type: str, database_engine: str, ec2_database_engine: str = None) -> Dict:
        """Enhanced OS performance calculation with AI insights"""
        
        # Extract the actual database engine
        actual_engine = self.extract_database_engine(database_engine, ec2_database_engine)
        
        os_config = self.operating_systems[os_type]
        
        # Base OS efficiency calculation (preserved original logic)
        base_efficiency = (
            os_config['cpu_efficiency'] * 0.3 +
            os_config['memory_efficiency'] * 0.25 +
            os_config['io_efficiency'] * 0.25 +
            os_config['network_efficiency'] * 0.2
        )
        
        # Database-specific optimization
        db_optimization = os_config['database_optimizations'].get(actual_engine, 0.85)
        
        # Virtualization impact
        if platform_type == 'vmware':
            virtualization_penalty = os_config['virtualization_overhead']
            total_efficiency = base_efficiency * db_optimization * (1 - virtualization_penalty)
        else:
            total_efficiency = base_efficiency * db_optimization
        
        # Platform-specific adjustments
        if platform_type == 'physical':
            if 'windows' in os_type:
                total_efficiency *= 1.02
            else:
                total_efficiency *= 1.05
        
        # Enhanced return with AI insights
        return {
            **{k: v for k, v in os_config.items() if k != 'ai_insights'},
            'total_efficiency': total_efficiency,
            'base_efficiency': base_efficiency,
            'db_optimization': db_optimization,
            'actual_database_engine': actual_engine,
            'virtualization_overhead': os_config['virtualization_overhead'] if platform_type == 'vmware' else 0,
            'ai_insights': os_config['ai_insights'],
            'platform_optimization': 1.02 if platform_type == 'physical' and 'windows' in os_type else 1.05 if platform_type == 'physical' else 1.0
        }
    


# Updated sidebar function with EC2 database engine selection


# Updated function calls to pass the EC2 database engine
def update_os_performance_calls():
    """Update all calls to calculate_os_performance_impact to include ec2_database_engine"""
    
    # Fixed OnPremPerformanceAnalyzer.calculate_ai_enhanced_performance method


# Updated comprehensive_ai_migration_analysis method signature
async def comprehensive_ai_migration_analysis(self, config: Dict) -> Dict:
    """Comprehensive AI-powered migration analysis with agent scaling optimization and FSx destination support"""
    
    # API status tracking
    api_status = APIStatus(
        anthropic_connected=self.ai_manager.connected,
        aws_pricing_connected=self.aws_api.connected,
        last_update=datetime.now()
    )
    
    # Enhanced on-premises performance analysis with FIXED parameters
    onprem_performance = self.onprem_analyzer.calculate_ai_enhanced_performance(config, self.os_manager)
    
    # Determine network path key based on config and destination storage
    network_path_key = self._get_network_path_key(config)
    
    # AI-enhanced network path analysis
    network_perf = self.network_manager.calculate_ai_enhanced_path_performance(network_path_key)
    
    # Determine migration type and tools (preserved)
    is_homogeneous = config['source_database_engine'] == config.get('ec2_database_engine', config.get('database_engine', '').replace('rds_', ''))
    migration_type = 'homogeneous' if is_homogeneous else 'heterogeneous'
    primary_tool = 'datasync' if is_homogeneous else 'dms'
    
    # Rest of the analysis continues as before...
    # Enhanced agent analysis with scaling support and destination storage
    agent_analysis = await self._analyze_ai_migration_agents_with_scaling(config, primary_tool, network_perf)
    
    # Calculate effective migration throughput with multiple agents
    agent_throughput = agent_analysis['total_effective_throughput']
    network_throughput = network_perf['effective_bandwidth_mbps']
    migration_throughput = min(agent_throughput, network_throughput)
    
    # AI-enhanced migration time calculation with agent scaling
    migration_time_hours = await self._calculate_ai_migration_time_with_agents(
        config, migration_throughput, onprem_performance, agent_analysis
    )
    
    # AI-powered AWS sizing recommendations
    aws_sizing = await self.aws_manager.ai_enhanced_aws_sizing(config)
    
    # Enhanced cost analysis with agent scaling costs and FSx costs
    cost_analysis = await self._calculate_ai_enhanced_costs_with_agents(
        config, aws_sizing, agent_analysis, network_perf
    )
    
    # Generate FSx destination comparisons
    fsx_comparisons = await self._generate_fsx_destination_comparisons(config)
    
    return {
        'api_status': api_status,
        'onprem_performance': onprem_performance,
        'network_performance': network_perf,
        'migration_type': migration_type,
        'primary_tool': primary_tool,
        'agent_analysis': agent_analysis,
        'migration_throughput_mbps': migration_throughput,
        'estimated_migration_time_hours': migration_time_hours,
        'aws_sizing_recommendations': aws_sizing,
        'cost_analysis': cost_analysis,
        'fsx_comparisons': fsx_comparisons,
        'ai_overall_assessment': await self._generate_ai_overall_assessment_with_agents(
            config, onprem_performance, aws_sizing, migration_time_hours, agent_analysis
        )
    }
    
    def calculate_os_performance_impact(self, os_type: str, platform_type: str, database_engine: str) -> Dict:
        """Enhanced OS performance calculation with AI insights"""
        
        os_config = self.operating_systems[os_type]
        
        # Base OS efficiency calculation (preserved original logic)
        base_efficiency = (
            os_config['cpu_efficiency'] * 0.3 +
            os_config['memory_efficiency'] * 0.25 +
            os_config['io_efficiency'] * 0.25 +
            os_config['network_efficiency'] * 0.2
        )
        
        # Database-specific optimization
        db_optimization = os_config['database_optimizations'][database_engine]
        
        # Virtualization impact
        if platform_type == 'vmware':
            virtualization_penalty = os_config['virtualization_overhead']
            total_efficiency = base_efficiency * db_optimization * (1 - virtualization_penalty)
        else:
            total_efficiency = base_efficiency * db_optimization
        
        # Platform-specific adjustments
        if platform_type == 'physical':
            if 'windows' in os_type:
                total_efficiency *= 1.02
            else:
                total_efficiency *= 1.05
        
        # Enhanced return with AI insights
        return {
            **{k: v for k, v in os_config.items() if k != 'ai_insights'},
            'total_efficiency': total_efficiency,
            'base_efficiency': base_efficiency,
            'db_optimization': db_optimization,
            'virtualization_overhead': os_config['virtualization_overhead'] if platform_type == 'vmware' else 0,
            'ai_insights': os_config['ai_insights'],
            'platform_optimization': 1.02 if platform_type == 'physical' and 'windows' in os_type else 1.05 if platform_type == 'physical' else 1.0
        }



class EnhancedNetworkIntelligenceManager:
    """AI-powered network path intelligence with enhanced analysis including FSx destinations"""
    
    
    def calculate_ai_enhanced_path_performance(self, path_key: str, time_of_day: int = None) -> Dict:
        """AI-enhanced network path performance calculation"""
        
        path = self.network_paths[path_key]
        
        if time_of_day is None:
            time_of_day = datetime.now().hour
        
        # Original performance calculation (preserved)
        total_latency = 0
        min_bandwidth = float('inf')
        total_reliability = 1.0
        total_cost_factor = 0
        ai_optimization_score = 1.0
        
        adjusted_segments = []
        
        for segment in path['segments']:
            # Base metrics
            segment_latency = segment['latency_ms']
            segment_bandwidth = segment['bandwidth_mbps']
            segment_reliability = segment['reliability']
            
            # Time-of-day and congestion adjustments (preserved original logic)
            if segment['connection_type'] == 'internal_lan':
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.1
                else:
                    congestion_factor = 0.95
            elif segment['connection_type'] == 'private_line':
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.2
                else:
                    congestion_factor = 0.9
            elif segment['connection_type'] == 'direct_connect':
                if 9 <= time_of_day <= 17:
                    congestion_factor = 1.05
                else:
                    congestion_factor = 0.98
            else:
                congestion_factor = 1.0
            
            # Apply congestion
            effective_bandwidth = segment_bandwidth / congestion_factor
            effective_latency = segment_latency * congestion_factor
            
            # OS-specific adjustments (preserved)
            if path['os_type'] == 'windows' and segment['connection_type'] != 'internal_lan':
                effective_bandwidth *= 0.95
                effective_latency *= 1.1
            
            # Destination storage adjustments
            if 'FSx' in path['destination_storage']:
                if path['destination_storage'] == 'FSx_Windows':
                    effective_bandwidth *= 1.1  # Better Windows integration
                    effective_latency *= 0.9    # Lower latency
                elif path['destination_storage'] == 'FSx_Lustre':
                    effective_bandwidth *= 1.3  # High performance
                    effective_latency *= 0.7    # Very low latency
            
            # AI optimization potential
            ai_optimization_score *= segment['ai_optimization_potential']
            
            # Accumulate metrics
            total_latency += effective_latency
            min_bandwidth = min(min_bandwidth, effective_bandwidth)
            total_reliability *= segment_reliability
            total_cost_factor += segment['cost_factor']
            
            adjusted_segments.append({
                **segment,
                'effective_bandwidth_mbps': effective_bandwidth,
                'effective_latency_ms': effective_latency,
                'congestion_factor': congestion_factor
            })
        
        # Calculate quality scores (preserved original logic)
        latency_score = max(0, 100 - (total_latency * 2))
        bandwidth_score = min(100, (min_bandwidth / 1000) * 20)
        reliability_score = total_reliability * 100
        
        # AI-enhanced network quality with optimization potential
        base_network_quality = (latency_score * 0.25 + bandwidth_score * 0.45 + reliability_score * 0.30)
        ai_enhanced_quality = base_network_quality * ai_optimization_score
        
        # Destination storage performance bonus
        storage_bonus = 0
        if path['destination_storage'] == 'FSx_Windows':
            storage_bonus = 10
        elif path['destination_storage'] == 'FSx_Lustre':
            storage_bonus = 20
        
        ai_enhanced_quality = min(100, ai_enhanced_quality + storage_bonus)
        
        return {
            'path_name': path['name'],
            'destination_storage': path['destination_storage'],
            'total_latency_ms': total_latency,
            'effective_bandwidth_mbps': min_bandwidth,
            'total_reliability': total_reliability,
            'network_quality_score': base_network_quality,
            'ai_enhanced_quality_score': ai_enhanced_quality,
            'ai_optimization_potential': (1 - ai_optimization_score) * 100,
            'total_cost_factor': total_cost_factor,
            'storage_performance_bonus': storage_bonus,
            'segments': adjusted_segments,
            'environment': path['environment'],
            'os_type': path['os_type'],
            'storage_type': path['storage_type'],
            'ai_insights': path['ai_insights']
        }
    
    def get_available_paths_by_storage(self, os_type: str, environment: str) -> Dict:
        """Get available network paths grouped by destination storage type"""
        
        storage_groups = {
            'S3': [],
            'FSx_Windows': [],
            'FSx_Lustre': []
        }
        
        for path_key, path_data in self.network_paths.items():
            if (path_data['os_type'] == os_type and 
                path_data['environment'] == environment):
                
                storage_type = path_data['destination_storage']
                if storage_type in storage_groups:
                    storage_groups[storage_type].append({
                        'key': path_key,
                        'name': path_data['name'],
                        'storage_type': storage_type
                    })
        
        return storage_groups

class EnhancedAgentSizingManager:
    """Enhanced agent sizing with scalable agent count and AI recommendations"""
    
    def __init__(self):
        # DataSync agent specifications
        self.datasync_agent_specs = {
            'small': {
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 250,
                'max_concurrent_tasks_per_agent': 10,
                'cost_per_hour_per_agent': 0.05,
                'ai_optimization_tips': [
                    'Suitable for small databases under 1TB',
                    'Good for development and testing',
                    'Cost-effective for non-critical workloads'
                ]
            },
            'medium': {
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps_per_agent': 500,
                'max_concurrent_tasks_per_agent': 20,
                'cost_per_hour_per_agent': 0.10,
                'ai_optimization_tips': [
                    'Balanced performance and cost',
                    'Suitable for most production workloads',
                    'Good scaling characteristics'
                ]
            },
            'large': {
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps_per_agent': 1000,
                'max_concurrent_tasks_per_agent': 40,
                'cost_per_hour_per_agent': 0.20,
                'ai_optimization_tips': [
                    'High-performance option',
                    'Suitable for large databases',
                    'Better for time-sensitive migrations'
                ]
            },
            'xlarge': {
                'vcpu': 16,
                'memory_gb': 32,
                'max_throughput_mbps_per_agent': 2000,
                'max_concurrent_tasks_per_agent': 80,
                'cost_per_hour_per_agent': 0.40,
                'ai_optimization_tips': [
                    'Maximum performance',
                    'Best for very large databases',
                    'Optimal for minimal downtime requirements'
                ]
            }
        }
        
        # DMS agent specifications
        self.dms_agent_specs = {
            'small': {
                'vcpu': 2,
                'memory_gb': 4,
                'max_throughput_mbps_per_agent': 200,
                'max_concurrent_tasks_per_agent': 8,
                'cost_per_hour_per_agent': 0.06,
                'ai_optimization_tips': [
                    'Basic DMS performance',
                    'Suitable for small heterogeneous migrations',
                    'Cost-effective for simple transformations'
                ]
            },
            'medium': {
                'vcpu': 4,
                'memory_gb': 8,
                'max_throughput_mbps_per_agent': 400,
                'max_concurrent_tasks_per_agent': 16,
                'cost_per_hour_per_agent': 0.12,
                'ai_optimization_tips': [
                    'Standard DMS performance',
                    'Good for most heterogeneous migrations',
                    'Balanced transformation capabilities'
                ]
            },
            'large': {
                'vcpu': 8,
                'memory_gb': 16,
                'max_throughput_mbps_per_agent': 800,
                'max_concurrent_tasks_per_agent': 32,
                'cost_per_hour_per_agent': 0.24,
                'ai_optimization_tips': [
                    'High-performance DMS',
                    'Suitable for complex transformations',
                    'Better for large heterogeneous migrations'
                ]
            },
            'xlarge': {
                'vcpu': 16,
                'memory_gb': 32,
                'max_throughput_mbps_per_agent': 1500,
                'max_concurrent_tasks_per_agent': 64,
                'cost_per_hour_per_agent': 0.48,
                'ai_optimization_tips': [
                    'Maximum DMS performance',
                    'Best for very complex transformations',
                    'Optimal for large-scale heterogeneous migrations'
                ]
            },
            'xxlarge': {
                'vcpu': 32,
                'memory_gb': 64,
                'max_throughput_mbps_per_agent': 2500,
                'max_concurrent_tasks_per_agent': 128,
                'cost_per_hour_per_agent': 0.96,
                'ai_optimization_tips': [
                    'Enterprise-grade DMS performance',
                    'Best for extremely large migrations',
                    'Maximum transformation capabilities'
                ]
            }
        }
    
    def calculate_agent_configuration(self, agent_type: str, agent_size: str, number_of_agents: int, destination_storage: str = 'S3') -> Dict:
        """Calculate agent configuration with corrected FSx architecture"""

        if agent_type == 'datasync':
            agent_spec = self.datasync_agent_specs[agent_size]
        else:  # dms
            agent_spec = self.dms_agent_specs[agent_size]

        # Get actual migration architecture
        architecture = self.get_actual_migration_architecture(agent_type, destination_storage, {})

        # Calculate scaling efficiency
        scaling_efficiency = self._calculate_scaling_efficiency(number_of_agents)

        # Calculate throughput based on actual architecture
        if architecture['agent_targets_destination']:
            # Direct migration (S3)
            storage_multiplier = self._get_storage_performance_multiplier(destination_storage)
            total_throughput = (agent_spec['max_throughput_mbps_per_agent'] * 
                            number_of_agents * scaling_efficiency * storage_multiplier)
            fsx_multiplier = storage_multiplier  # Use storage_multiplier for consistency
        else:
            # Split workload (FSx scenarios)
            base_throughput = agent_spec['max_throughput_mbps_per_agent'] * number_of_agents * scaling_efficiency
            
            # Database portion goes to EC2+EBS (no FSx multiplier)
            db_portion = base_throughput * (architecture.get('database_percentage', 80) / 100)
            
            # File portion gets FSx multiplier
            file_portion = base_throughput * (architecture.get('file_percentage', 20) / 100)
            fsx_multiplier = self._get_storage_performance_multiplier(destination_storage)
            file_portion_enhanced = file_portion * fsx_multiplier
            
            total_throughput = db_portion + file_portion_enhanced

        total_concurrent_tasks = (agent_spec['max_concurrent_tasks_per_agent'] * number_of_agents)
        total_cost_per_hour = agent_spec['cost_per_hour_per_agent'] * number_of_agents

        # Calculate management overhead
        management_overhead_factor = 1.0 + (number_of_agents - 1) * 0.05
        storage_overhead = self._get_storage_management_overhead(destination_storage)

        return {
            'agent_type': agent_type,
            'agent_size': agent_size,
            'number_of_agents': number_of_agents,
            'destination_storage': destination_storage,
            'migration_architecture': architecture,  # NEW: Architecture details
            'per_agent_spec': agent_spec,
            'total_vcpu': agent_spec['vcpu'] * number_of_agents,
            'total_memory_gb': agent_spec['memory_gb'] * number_of_agents,
            'max_throughput_mbps_per_agent': agent_spec['max_throughput_mbps_per_agent'],
            'total_max_throughput_mbps': total_throughput,
            'effective_throughput_mbps': total_throughput,
            'total_concurrent_tasks': total_concurrent_tasks,
            'cost_per_hour_per_agent': agent_spec['cost_per_hour_per_agent'],
            'total_cost_per_hour': total_cost_per_hour,
            'total_monthly_cost': total_cost_per_hour * 24 * 30,
            'scaling_efficiency': scaling_efficiency,
            'storage_performance_multiplier': fsx_multiplier if architecture['agent_targets_destination'] else 1.0,
            'management_overhead_factor': management_overhead_factor,
            'storage_management_overhead': storage_overhead,
            'effective_cost_per_hour': total_cost_per_hour * management_overhead_factor * storage_overhead,
            'ai_optimization_tips': agent_spec['ai_optimization_tips'],
            'scaling_recommendations': self._get_scaling_recommendations(agent_size, number_of_agents, destination_storage),
            'optimal_configuration': self._assess_configuration_optimality(agent_size, number_of_agents, destination_storage)
        }
    
    def _get_storage_performance_multiplier(self, destination_storage: str) -> float:
        """Get performance multiplier based on destination storage type"""
        multipliers = {
            'S3': 1.0,
            'FSx_Windows': 1.15,
            'FSx_Lustre': 1.4
        }
        return multipliers.get(destination_storage, 1.0)
    
    def _get_storage_management_overhead(self, destination_storage: str) -> float:
        """Get management overhead factor for destination storage"""
        overheads = {
            'S3': 1.0,
            'FSx_Windows': 1.1,
            'FSx_Lustre': 1.2
        }
        return overheads.get(destination_storage, 1.0)
    
    def _calculate_scaling_efficiency(self, number_of_agents: int) -> float:
        """Calculate scaling efficiency - diminishing returns with more agents"""
        if number_of_agents == 1:
            return 1.0
        elif number_of_agents <= 3:
            return 0.95  # 5% overhead for coordination
        elif number_of_agents <= 5:
            return 0.90  # 10% overhead
        elif number_of_agents <= 8:
            return 0.85  # 15% overhead
        else:
            return 0.80  # 20% overhead for complex coordination
    
    def _get_scaling_recommendations(self, agent_size: str, number_of_agents: int, destination_storage: str) -> List[str]:
        """Get scaling-specific recommendations"""
        recommendations = []
        
        if number_of_agents == 1:
            recommendations.append("Single agent configuration - consider scaling for larger workloads")
        elif number_of_agents <= 3:
            recommendations.append("Good balance of performance and manageability")
        else:
            recommendations.append("High-scale configuration requiring careful coordination")
        
        return recommendations
    
    def _assess_configuration_optimality(self, agent_size: str, number_of_agents: int, destination_storage: str) -> Dict:
        """Assess if the configuration is optimal"""
        
        efficiency_score = 100
        
        # Penalize for too many small agents
        if agent_size == 'small' and number_of_agents > 6:
            efficiency_score -= 20
        
        # Penalize for management complexity
        if number_of_agents > 8:
            efficiency_score -= 25
        
        # Optimal ranges
        if 2 <= number_of_agents <= 4 and agent_size in ['medium', 'large']:
            efficiency_score += 10
        
        return {
            'efficiency_score': max(0, efficiency_score),
            'management_complexity': "Low" if number_of_agents <= 2 else "Medium" if number_of_agents <= 5 else "High",
            'cost_efficiency': "Good" if efficiency_score >= 90 else "Fair" if efficiency_score >= 75 else "Poor"
        }
    
    def get_actual_migration_architecture(self, agent_type: str, destination_storage: str, config: Dict) -> Dict:
        """Determine the actual migration architecture based on destination storage"""

        if destination_storage == 'S3':
            # Direct migration to S3
            return {
                'primary_target': 'S3',
                'secondary_target': None,
                'agent_targets_destination': True,
                'architecture_type': 'direct_cloud_storage',
                'bandwidth_calculation': 'direct',
                'description': f'{agent_type.upper()} agents transfer directly to S3'
            }
        
        elif destination_storage == 'FSx_Windows':
            # Hybrid architecture
            return {
                'primary_target': 'EC2_EBS',  # Database goes to EC2 + EBS
                'secondary_target': 'FSx_Windows',  # File data goes to FSx Windows
                'agent_targets_destination': False,  # Agents don't directly target FSx
                'architecture_type': 'hybrid_storage',
                'bandwidth_calculation': 'split_workload',
                'database_percentage': 80,
                'file_percentage': 20,
                'description': f'{agent_type.upper()} for database → EC2+EBS; DataSync for files → FSx Windows'
            }
        
        elif destination_storage == 'FSx_Lustre':
            # HPC architecture
            return {
                'primary_target': 'EC2_EBS',  # Database still goes to EC2 + EBS
                'secondary_target': 'FSx_Lustre',  # HPC data goes to FSx Lustre
                'agent_targets_destination': False,
                'architecture_type': 'hpc_hybrid',
                'bandwidth_calculation': 'split_workload',
                'database_percentage': 70,
                'file_percentage': 30,
                'description': f'{agent_type.upper()} for database → EC2+EBS; DataSync for HPC data → FSx Lustre'
            }
        
        else:
            # Fallback
            return {
                'primary_target': destination_storage,
                'secondary_target': None,
                'agent_targets_destination': True,
                'architecture_type': 'standard',
                'bandwidth_calculation': 'direct',
                'description': f'{agent_type.upper()} agents transfer to {destination_storage}'
            }
    
class EnhancedAWSMigrationManager:
    """Enhanced AWS migration manager with AI and real-time pricing"""
    
    
    async def ai_enhanced_aws_sizing(self, on_prem_config: Dict) -> Dict:
        """AI-enhanced AWS sizing with real-time pricing"""
        
        # Get real-time pricing
        pricing_data = await self.aws_api.get_real_time_pricing()
        
        # Get AI workload analysis
        ai_analysis = await self.ai_manager.analyze_migration_workload(
            on_prem_config, 
            on_prem_config.get('performance_data', {})
        )
        
        # Enhanced sizing logic with AI insights
        rds_recommendations = await self._ai_calculate_rds_sizing(
            on_prem_config, pricing_data, ai_analysis
        )
        
        ec2_recommendations = await self._ai_calculate_ec2_sizing(
            on_prem_config, pricing_data, ai_analysis
        )
        
        # AI-enhanced reader/writer configuration
        reader_writer_config = await self._ai_calculate_reader_writer_config(
            on_prem_config, ai_analysis
        )
        
        # AI-powered deployment recommendation
        deployment_recommendation = await self._ai_recommend_deployment_type(
            on_prem_config, ai_analysis, rds_recommendations, ec2_recommendations
        )
        
        return {
            'rds_recommendations': rds_recommendations,
            'ec2_recommendations': ec2_recommendations,
            'reader_writer_config': reader_writer_config,
            'deployment_recommendation': deployment_recommendation,
            'ai_analysis': ai_analysis,
            'pricing_data': pricing_data
        }
    
    async def _ai_calculate_rds_sizing(self, config: Dict, pricing_data: Dict, ai_analysis: Dict) -> Dict:
        """AI-enhanced RDS sizing calculation"""
        
        # Extract configuration
        cpu_cores = config['cpu_cores']
        ram_gb = config['ram_gb']
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        database_engine = config['database_engine']
        environment = config.get('environment', 'non-production')
        
        # AI-based sizing multipliers
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        complexity_multiplier = 1.0 + (ai_complexity - 5) * 0.1  # Adjust based on AI complexity
        
        # Agent scaling impact on sizing
        num_agents = config.get('number_of_agents', 1)
        agent_scaling_factor = 1.0 + (num_agents - 1) * 0.05  # More agents may need more DB resources
        
        cpu_multiplier = (1.2 if performance_req == 'high' else 1.0) * complexity_multiplier * agent_scaling_factor
        memory_multiplier = (1.3 if database_engine in ['oracle', 'postgresql'] else 1.1) * complexity_multiplier
        
        # Required cloud resources with AI adjustment
        required_vcpu = max(2, int(cpu_cores * cpu_multiplier))
        required_memory = max(8, int(ram_gb * memory_multiplier))
        
        # Use real-time pricing data
        rds_instances = pricing_data.get('rds_instances', {})
        
        # AI-enhanced instance selection
        best_instance = None
        best_score = float('inf')
        
        for instance_type, specs in rds_instances.items():
            if specs['vcpu'] >= required_vcpu and specs['memory'] >= required_memory:
                # AI-enhanced scoring
                cpu_waste = specs['vcpu'] - required_vcpu
                memory_waste = specs['memory'] - required_memory
                cost_factor = specs['cost_per_hour']
                
                # AI complexity penalty for oversizing
                ai_penalty = 0 if ai_complexity <= 6 else (ai_complexity - 6) * 0.1
                
                score = (cpu_waste * 0.3 + memory_waste * 0.001 + cost_factor * 0.5 + ai_penalty)
                
                if score < best_score:
                    best_score = score
                    best_instance = instance_type
        
        if not best_instance:
            best_instance = 'db.r6g.8xlarge'
        
        # AI-enhanced storage recommendations
        storage_multiplier = 1.5 + (ai_complexity - 5) * 0.1  # More storage for complex migrations
        storage_size_gb = max(database_size_gb * storage_multiplier, 100)
        storage_type = 'io1' if database_size_gb > 5000 or performance_req == 'high' or ai_complexity > 7 else 'gp3'
        
        # Calculate costs with real-time pricing
        instance_cost = rds_instances[best_instance]['cost_per_hour'] * 24 * 30
        storage_cost = storage_size_gb * pricing_data.get('storage', {}).get(storage_type, {}).get('cost_per_gb_month', 0.08)
        
        return {
            'primary_instance': best_instance,
            'instance_specs': rds_instances[best_instance],
            'storage_type': storage_type,
            'storage_size_gb': storage_size_gb,
            'monthly_instance_cost': instance_cost,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': instance_cost + storage_cost,
            'multi_az': environment == 'production',
            'backup_retention_days': 30 if environment == 'production' else 7,
            'ai_sizing_factors': {
                'complexity_multiplier': complexity_multiplier,
                'agent_scaling_factor': agent_scaling_factor,
                'ai_complexity_score': ai_complexity,
                'storage_multiplier': storage_multiplier
            },
            'ai_recommendations': ai_analysis.get('performance_recommendations', [])
        }
    
    async def _ai_calculate_ec2_sizing(self, config: Dict, pricing_data: Dict, ai_analysis: Dict) -> Dict:
        """AI-enhanced EC2 sizing calculation"""
        
        # Similar to RDS but with EC2-specific adjustments
        cpu_cores = config['cpu_cores']
        ram_gb = config['ram_gb']
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        database_engine = config['database_engine']
        environment = config.get('environment', 'non-production')
        
        # AI complexity-based adjustments
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        complexity_multiplier = 1.0 + (ai_complexity - 5) * 0.15  # More aggressive for EC2
        
        # Agent scaling impact
        num_agents = config.get('number_of_agents', 1)
        agent_scaling_factor = 1.0 + (num_agents - 1) * 0.08  # More EC2 resources for multi-agent coordination
        
        # EC2 needs more overhead for OS and database management
        cpu_multiplier = (1.4 if performance_req == 'high' else 1.2) * complexity_multiplier * agent_scaling_factor
        memory_multiplier = (1.5 if database_engine in ['oracle', 'postgresql'] else 1.3) * complexity_multiplier
        
        required_vcpu = max(2, int(cpu_cores * cpu_multiplier))
        required_memory = max(8, int(ram_gb * memory_multiplier))
        
        # Use real-time pricing
        ec2_instances = pricing_data.get('ec2_instances', {})
        
        best_instance = None
        best_score = float('inf')
        
        for instance_type, specs in ec2_instances.items():
            if specs['vcpu'] >= required_vcpu and specs['memory'] >= required_memory:
                cpu_waste = specs['vcpu'] - required_vcpu
                memory_waste = specs['memory'] - required_memory
                cost_factor = specs['cost_per_hour']
                
                # AI penalty for complex workloads
                ai_penalty = 0 if ai_complexity <= 6 else (ai_complexity - 6) * 0.15
                
                score = (cpu_waste * 0.3 + memory_waste * 0.001 + cost_factor * 0.5 + ai_penalty)
                
                if score < best_score:
                    best_score = score
                    best_instance = instance_type
        
        if not best_instance:
            best_instance = 'r6i.8xlarge'
        
        # AI-enhanced storage sizing for EC2
        storage_multiplier = 2.0 + (ai_complexity - 5) * 0.2  # Even more generous for EC2
        storage_size_gb = max(database_size_gb * storage_multiplier, 100)
        storage_type = 'io2' if performance_req == 'high' or ai_complexity > 7 else 'gp3'
        
        # Calculate costs
        instance_cost = ec2_instances[best_instance]['cost_per_hour'] * 24 * 30
        storage_cost = storage_size_gb * pricing_data.get('storage', {}).get(storage_type, {}).get('cost_per_gb_month', 0.08)
        
        return {
            'primary_instance': best_instance,
            'instance_specs': ec2_instances[best_instance],
            'storage_type': storage_type,
            'storage_size_gb': storage_size_gb,
            'monthly_instance_cost': instance_cost,
            'monthly_storage_cost': storage_cost,
            'total_monthly_cost': instance_cost + storage_cost,
            'ebs_optimized': True,
            'enhanced_networking': True,
            'ai_sizing_factors': {
                'complexity_multiplier': complexity_multiplier,
                'agent_scaling_factor': agent_scaling_factor,
                'ai_complexity_score': ai_complexity,
                'storage_multiplier': storage_multiplier
            },
            'ai_recommendations': ai_analysis.get('performance_recommendations', [])
        }
    
    async def _ai_calculate_reader_writer_config(self, config: Dict, ai_analysis: Dict) -> Dict:
        """AI-enhanced reader/writer configuration with better logic"""
    
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        environment = config.get('environment', 'non-production')
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        num_agents = config.get('number_of_agents', 1)
        
        # Start with single writer
        writers = 1
        readers = 0
        
        # Enhanced reader scaling logic - ensure we always have some readers for reasonable sizes
        if database_size_gb > 500:  # Lowered threshold
            readers += 1
        if database_size_gb > 2000:
            readers += 1 + max(0, int(ai_complexity / 5))  # AI complexity adds more readers
        if database_size_gb > 10000:
            readers += 2 + max(0, int(ai_complexity / 3))
        if database_size_gb > 50000:
            readers += 3
        
        # Performance-based scaling with AI insights
        if performance_req == 'high':
            readers += 2 + max(0, int(ai_complexity / 4))
        else:  # Even standard performance should have readers for decent sizes
            if database_size_gb > 1000:
                readers += 1
        
        # Agent scaling impact - more agents may benefit from more read replicas
        if num_agents > 2:
            readers += 1 + (num_agents // 3)
        elif num_agents > 1 and database_size_gb > 1000:
            readers += 1
        
        # Environment-based scaling
        if environment == 'production':
            readers = max(readers, 2)  # Minimum 2 readers for production
            if database_size_gb > 50000 or ai_complexity > 8:
                writers = 2  # Multi-writer for very large or complex production DBs
        else:
            # Non-production should still have at least 1 reader for decent sizes
            if database_size_gb > 1000:
                readers = max(readers, 1)
        
        # AI-recommended adjustments
        ai_best_practices = ai_analysis.get('best_practices', [])
        if any('high availability' in practice.lower() for practice in ai_best_practices):
            readers += 1
        if any('read replica' in practice.lower() for practice in ai_best_practices):
            readers += 1
        
        # Ensure minimum sensible configuration
        if database_size_gb > 1000 and readers == 0:
            readers = 1  # Always have at least 1 reader for databases > 1TB
        
        # Calculate read/write distribution
        total_capacity = writers + readers
        write_capacity_percent = (writers / total_capacity) * 100 if total_capacity > 0 else 100
        read_capacity_percent = (readers / total_capacity) * 100 if total_capacity > 0 else 0
        
        return {
            'writers': writers,
            'readers': readers,
            'total_instances': total_capacity,
            'write_capacity_percent': write_capacity_percent,
            'read_capacity_percent': read_capacity_percent,
            'recommended_read_split': min(80, read_capacity_percent),
            'reasoning': f"AI-optimized for {database_size_gb}GB, complexity {ai_complexity}/10, {performance_req} performance, {environment}, {num_agents} agents",
            'ai_insights': {
                'complexity_impact': ai_complexity,
                'agent_scaling_impact': num_agents,
                'scaling_factors': [
                    f"Database size drives {readers} reader replicas",
                    f"Performance requirement: {performance_req}",
                    f"Environment: {environment} scaling applied"
                ],
                'optimization_potential': f"{max(0, (10 - ai_complexity) * 10)}% potential for further optimization"
            }
        }
    async def _ai_recommend_deployment_type(self, config: Dict, ai_analysis: Dict, 
                                          rds_rec: Dict, ec2_rec: Dict) -> Dict:
        """AI-powered deployment type recommendation"""
        
        database_size_gb = config['database_size_gb']
        performance_req = config.get('performance_requirements', 'standard')
        database_engine = config['database_engine']
        environment = config.get('environment', 'non-production')
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        num_agents = config.get('number_of_agents', 1)
        
        rds_score = 0
        ec2_score = 0
        
        # Size-based scoring (preserved original logic)
        if database_size_gb < 2000:
            rds_score += 40
        elif database_size_gb > 10000:
            ec2_score += 30
        
        # Performance scoring
        if performance_req == 'high':
            ec2_score += 30
            rds_score += 15
        else:
            rds_score += 35
        
        # Database engine scoring
        if database_engine in ['mysql', 'postgresql']:
            rds_score += 25
        elif database_engine == 'oracle':
            ec2_score += 25
        
        # Environment scoring
        if environment == 'production':
            rds_score += 20
        else:
            ec2_score += 10
        
        # AI complexity scoring
        if ai_complexity > 7:
            ec2_score += 20  # Complex workloads might need more control
            rds_score += 5   # But managed services help with complexity
        elif ai_complexity < 4:
            rds_score += 25  # Simple workloads perfect for managed services
        
        # Agent scaling considerations
        if num_agents > 3:
            ec2_score += 15  # Multi-agent setups may need more control
        elif num_agents == 1:
            rds_score += 10  # Single agent works well with managed services
        
        # AI insights-based scoring
        risk_factors = ai_analysis.get('risk_factors', [])
        if any('performance' in risk.lower() for risk in risk_factors):
            ec2_score += 15
        if any('management' in risk.lower() or 'operational' in risk.lower() for risk in risk_factors):
            rds_score += 20
        
        # Management complexity
        rds_score += 20
        
        # Cost consideration
        rds_cost = rds_rec.get('total_monthly_cost', 0)
        ec2_cost = ec2_rec.get('total_monthly_cost', 0)
        
        if ec2_cost < rds_cost * 0.8:  # EC2 significantly cheaper
            ec2_score += 15
        elif rds_cost < ec2_cost * 0.9:  # RDS competitive
            rds_score += 10
        
        recommendation = 'rds' if rds_score > ec2_score else 'ec2'
        confidence = abs(rds_score - ec2_score) / max(rds_score, ec2_score, 1)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'rds_score': rds_score,
            'ec2_score': ec2_score,
            'ai_complexity_factor': ai_complexity,
            'agent_scaling_factor': num_agents,
            'primary_reasons': self._get_ai_deployment_reasons(
                recommendation, rds_score, ec2_score, ai_analysis, num_agents
            ),
            'ai_insights': {
                'complexity_impact': f"AI complexity score {ai_complexity}/10 influenced recommendation",
                'agent_impact': f"{num_agents} agents affected scoring",
                'risk_mitigation': ai_analysis.get('mitigation_strategies', [])[:3],
                'cost_factors': {
                    'rds_monthly': rds_cost,
                    'ec2_monthly': ec2_cost,
                    'cost_difference_percent': abs(rds_cost - ec2_cost) / max(rds_cost, ec2_cost, 1) * 100
                }
            }
        }
    
    def _get_ai_deployment_reasons(self, recommendation: str, rds_score: int, 
                                 ec2_score: int, ai_analysis: Dict, num_agents: int) -> List[str]:
        """Get AI-enhanced reasons for deployment recommendation"""
        
        base_reasons = []
        ai_complexity = ai_analysis.get('ai_complexity_score', 6)
        
        if recommendation == 'rds':
            base_reasons = [
                "AI-optimized managed service reduces operational complexity",
                "Automated backups and patching aligned with AI recommendations",
                "Built-in monitoring supports AI-driven optimization",
                "Easy scaling matches AI-predicted growth patterns",
                f"AI complexity score ({ai_complexity}/10) suggests managed service benefits"
            ]
            
            if num_agents <= 2:
                base_reasons.append(f"Simple {num_agents}-agent setup works well with managed RDS")
        else:
            base_reasons = [
                "Maximum control needed for AI-identified performance requirements",
                "Complex configurations support AI optimization strategies",
                "Custom tuning capabilities for AI-recommended improvements",
                "Full control enables AI-driven performance optimization",
                f"AI complexity score ({ai_complexity}/10) suggests need for advanced control"
            ]
            
            if num_agents > 3:
                base_reasons.append(f"{num_agents}-agent coordination benefits from EC2 flexibility")
        
        # Add AI-specific insights
        ai_recommendations = ai_analysis.get('performance_recommendations', [])
        if ai_recommendations:
            base_reasons.append(f"Supports AI recommendation: {ai_recommendations[0][:50]}...")
        
        return base_reasons[:6]

class OnPremPerformanceAnalyzer:
    """Enhanced on-premises performance analyzer with AI insights"""
    
    
    def calculate_ai_enhanced_performance(self, config: Dict, os_manager: OSPerformanceManager) -> Dict:
        """AI-enhanced on-premises performance calculation"""
        
        # Get original OS impact
        os_impact = os_manager.calculate_os_performance_impact(
            config['operating_system'], 
            config['server_type'], 
            config['database_engine']
        )
        
        # Original performance calculations (preserved)
        cpu_performance = self._calculate_cpu_performance(config, os_impact)
        memory_performance = self._calculate_memory_performance(config, os_impact)
        storage_performance = self._calculate_storage_performance(config, os_impact)
        network_performance = self._calculate_network_performance(config, os_impact)
        database_performance = self._calculate_database_performance(config, os_impact)
        
        # Enhanced performance analysis using new metrics
        current_performance_analysis = self._analyze_current_performance_metrics(config)
        
        # AI-enhanced overall performance analysis
        overall_performance = self._calculate_ai_enhanced_overall_performance(
            cpu_performance, memory_performance, storage_performance, 
            network_performance, database_performance, os_impact, config
        )
        
        # AI bottleneck analysis
        ai_bottlenecks = self._ai_identify_bottlenecks(
            cpu_performance, memory_performance, storage_performance, 
            network_performance, config
        )
        
        # Resource adequacy analysis
        resource_adequacy = self._analyze_resource_adequacy(config)
        
        return {
            'cpu_performance': cpu_performance,
            'memory_performance': memory_performance,
            'storage_performance': storage_performance,
            'network_performance': network_performance,
            'database_performance': database_performance,
            'current_performance_analysis': current_performance_analysis,
            'resource_adequacy': resource_adequacy,
            'overall_performance': overall_performance,
            'os_impact': os_impact,
            'bottlenecks': ai_bottlenecks['bottlenecks'],
            'ai_insights': ai_bottlenecks['ai_insights'],
            'performance_score': overall_performance['composite_score'],
            'ai_optimization_recommendations': self._generate_ai_optimization_recommendations(
                overall_performance, os_impact, config
            )
        }
    
    def _analyze_current_performance_metrics(self, config: Dict) -> Dict:
        """Analyze current performance metrics provided by user"""
        
        current_storage = config.get('current_storage_gb', 0)
        peak_iops = config.get('peak_iops', 0)
        max_throughput = config.get('max_throughput_mbps', 0)
        database_size = config.get('database_size_gb', 0)
        
        # Storage utilization analysis
        storage_utilization = (database_size / current_storage) * 100 if current_storage > 0 else 0
        
        # IOPS intensity analysis
        iops_per_gb = peak_iops / database_size if database_size > 0 else 0
        
        # Throughput efficiency
        throughput_per_gb = max_throughput / database_size if database_size > 0 else 0
        
        # Performance classification
        if iops_per_gb > 50:
            workload_type = "High IOPS (OLTP-intensive)"
        elif throughput_per_gb > 1:
            workload_type = "High Throughput (Analytics/Batch)"
        else:
            workload_type = "Balanced Workload"
        
        return {
            'storage_utilization_percent': storage_utilization,
            'iops_per_gb': iops_per_gb,
            'throughput_per_gb_mbps': throughput_per_gb,
            'workload_classification': workload_type,
            'storage_efficiency': min(100, (100 - storage_utilization) + 50),  # Higher efficiency for lower utilization
            'performance_intensity': min(100, (iops_per_gb * 2) + (throughput_per_gb * 10)),
            'optimization_priority': "High" if storage_utilization > 80 or iops_per_gb > 100 else "Medium" if storage_utilization > 60 else "Low"
        }
    
    def _analyze_resource_adequacy(self, config: Dict) -> Dict:
        """Analyze resource adequacy comparing current vs anticipated needs"""
        
        current_memory = config.get('ram_gb', 0)
        anticipated_memory = config.get('anticipated_max_memory_gb', 0)
        current_cpu = config.get('cpu_cores', 0)
        anticipated_cpu = config.get('anticipated_max_cpu_cores', 0)
        
        # Memory adequacy
        memory_gap = anticipated_memory - current_memory
        memory_adequacy_score = (current_memory / anticipated_memory) * 100 if anticipated_memory > 0 else 100
        
        # CPU adequacy
        cpu_gap = anticipated_cpu - current_cpu
        cpu_adequacy_score = (current_cpu / anticipated_cpu) * 100 if anticipated_cpu > 0 else 100
        
        # Overall readiness
        overall_adequacy = (memory_adequacy_score + cpu_adequacy_score) / 2
        
        # Readiness classification
        if overall_adequacy >= 90:
            readiness_level = "Excellent - Current resources meet anticipated needs"
        elif overall_adequacy >= 75:
            readiness_level = "Good - Minor upgrades may be beneficial"
        elif overall_adequacy >= 60:
            readiness_level = "Fair - Moderate upgrades recommended"
        else:
            readiness_level = "Poor - Significant upgrades required"
        
        return {
            'memory_gap_gb': memory_gap,
            'cpu_gap_cores': cpu_gap,
            'memory_adequacy_score': memory_adequacy_score,
            'cpu_adequacy_score': cpu_adequacy_score,
            'overall_adequacy_score': overall_adequacy,
            'readiness_level': readiness_level,
            'upgrade_priority': "Immediate" if overall_adequacy < 60 else "Short-term" if overall_adequacy < 75 else "Long-term" if overall_adequacy < 90 else "Optional"
        }
    
    # Keep all original calculation methods but add AI enhancements
    def _calculate_cpu_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate CPU performance metrics with AI insights"""
        
        # Original calculation (preserved)
        base_performance = config['cpu_cores'] * config['cpu_ghz']
        os_adjusted = base_performance * os_impact['cpu_efficiency']
        
        if config['server_type'] == 'vmware':
            virtualization_penalty = 1 - os_impact['virtualization_overhead']
            final_performance = os_adjusted * virtualization_penalty
        else:
            final_performance = os_adjusted * 1.05
        
        single_thread_perf = config['cpu_ghz'] * os_impact['cpu_efficiency']
        multi_thread_perf = final_performance
        
        # AI enhancement: predict scaling characteristics
        ai_scaling_prediction = self._predict_cpu_scaling(config, final_performance)
        
        return {
            'base_performance': base_performance,
            'os_adjusted_performance': os_adjusted,
            'final_performance': final_performance,
            'single_thread_performance': single_thread_perf,
            'multi_thread_performance': multi_thread_perf,
            'utilization_estimate': 0.7,
            'efficiency_factor': os_impact['cpu_efficiency'],
            'ai_scaling_prediction': ai_scaling_prediction,
            'ai_bottleneck_risk': 'high' if final_performance < 30 else 'medium' if final_performance < 60 else 'low'
        }
    
    def _calculate_memory_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate memory performance with AI insights"""
        
        # Original calculation (preserved)
        base_memory = config['ram_gb']
        
        if 'windows' in config['operating_system']:
            os_overhead = 4
        else:
            os_overhead = 2
        
        available_memory = base_memory - os_overhead
        db_memory = available_memory * 0.8
        buffer_pool = db_memory * 0.7
        memory_efficiency = os_impact['memory_efficiency']
        effective_memory = available_memory * memory_efficiency
        
        # AI enhancement: memory pressure prediction
        ai_memory_analysis = self._analyze_memory_requirements(config, effective_memory)
        
        return {
            'total_memory_gb': base_memory,
            'os_overhead_gb': os_overhead,
            'available_memory_gb': available_memory,
            'database_memory_gb': db_memory,
            'buffer_pool_gb': buffer_pool,
            'effective_memory_gb': effective_memory,
            'memory_efficiency': memory_efficiency,
            'memory_pressure': 'low' if available_memory > 32 else 'medium' if available_memory > 16 else 'high',
            'ai_memory_analysis': ai_memory_analysis,
            'ai_optimization_potential': ai_memory_analysis['optimization_potential']
        }
    
    def _calculate_storage_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate storage performance with AI insights"""
        
        # Original calculation (preserved)
        if config['cpu_cores'] >= 8:
            storage_type = 'nvme_ssd'
        elif config['cpu_cores'] >= 4:
            storage_type = 'sata_ssd'
        else:
            storage_type = 'sas_hdd'
        
        storage_specs = self.storage_types[storage_type]
        
        effective_iops = storage_specs['iops'] * os_impact['io_efficiency']
        effective_throughput = storage_specs['throughput_mbps'] * os_impact['io_efficiency']
        effective_latency = storage_specs['latency_ms'] / os_impact['io_efficiency']
        
        # AI enhancement: storage optimization analysis
        ai_storage_analysis = self._analyze_storage_optimization(config, storage_type, effective_iops)
        
        return {
            'storage_type': storage_type,
            'base_iops': storage_specs['iops'],
            'effective_iops': effective_iops,
            'base_throughput_mbps': storage_specs['throughput_mbps'],
            'effective_throughput_mbps': effective_throughput,
            'base_latency_ms': storage_specs['latency_ms'],
            'effective_latency_ms': effective_latency,
            'io_efficiency': os_impact['io_efficiency'],
            'ai_storage_analysis': ai_storage_analysis,
            'ai_upgrade_recommendation': ai_storage_analysis['upgrade_recommendation']
        }
    
    def _calculate_network_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate network performance with AI insights"""
        
        # Original calculation (preserved)
        base_bandwidth = config['nic_speed']
        effective_bandwidth = base_bandwidth * os_impact['network_efficiency']
        
        if config['server_type'] == 'vmware':
            effective_bandwidth *= 0.92
        
        # AI enhancement: network optimization analysis
        ai_network_analysis = self._analyze_network_optimization(config, effective_bandwidth)
        
        return {
            'nic_type': config['nic_type'],
            'base_bandwidth_mbps': base_bandwidth,
            'effective_bandwidth_mbps': effective_bandwidth,
            'network_efficiency': os_impact['network_efficiency'],
            'estimated_latency_ms': 0.1 if 'fiber' in config['nic_type'] else 0.2,
            'ai_network_analysis': ai_network_analysis,
            'ai_optimization_score': ai_network_analysis['optimization_score']
        }
    
    def _calculate_database_performance(self, config: Dict, os_impact: Dict) -> Dict:
        """Calculate database performance with AI insights"""
        
        # Original calculation (preserved)
        db_optimization = os_impact['db_optimization']
        
        if config['database_engine'] == 'mysql':
            base_tps = 5000
            connection_limit = 1000
        elif config['database_engine'] == 'postgresql':
            base_tps = 4500
            connection_limit = 500
        elif config['database_engine'] == 'oracle':
            base_tps = 6000
            connection_limit = 2000
        elif config['database_engine'] == 'sqlserver':
            base_tps = 5500
            connection_limit = 1500
        else:
            base_tps = 4000
            connection_limit = 800
        
        hardware_factor = min(2.0, (config['cpu_cores'] / 4) * (config['ram_gb'] / 16))
        effective_tps = base_tps * hardware_factor * db_optimization
        
        # AI enhancement: database optimization analysis
        ai_db_analysis = self._analyze_database_optimization(config, effective_tps, db_optimization)
        
        return {
            'database_engine': config['database_engine'],
            'base_tps': base_tps,
            'hardware_factor': hardware_factor,
            'db_optimization': db_optimization,
            'effective_tps': effective_tps,
            'max_connections': connection_limit,
            'query_cache_efficiency': db_optimization * 0.9,
            'ai_db_analysis': ai_db_analysis,
            'ai_tuning_potential': ai_db_analysis['tuning_potential']
        }
    
    def _calculate_ai_enhanced_overall_performance(self, cpu_perf: Dict, mem_perf: Dict, 
                                                 storage_perf: Dict, net_perf: Dict, 
                                                 db_perf: Dict, os_impact: Dict, config: Dict) -> Dict:
        """AI-enhanced overall performance calculation"""
        
        # Original scoring (preserved)
        cpu_score = min(100, (cpu_perf['final_performance'] / 50) * 100)
        memory_score = min(100, (mem_perf['effective_memory_gb'] / 64) * 100)
        storage_score = min(100, (storage_perf['effective_iops'] / 100000) * 100)
        network_score = min(100, (net_perf['effective_bandwidth_mbps'] / 10000) * 100)
        database_score = min(100, (db_perf['effective_tps'] / 10000) * 100)
        
        # Original composite score
        composite_score = (
            cpu_score * 0.25 +
            memory_score * 0.2 +
            storage_score * 0.25 +
            network_score * 0.15 +
            database_score * 0.15
        )
        
        # AI enhancement: workload-specific optimization
        ai_workload_optimization = self._analyze_workload_optimization(config, {
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score,
            'network_score': network_score,
            'database_score': database_score
        })
        
        # AI-adjusted composite score
        ai_adjusted_score = composite_score * ai_workload_optimization['optimization_factor']
        
        return {
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'storage_score': storage_score,
            'network_score': network_score,
            'database_score': database_score,
            'composite_score': composite_score,
            'ai_adjusted_score': ai_adjusted_score,
            'performance_tier': self._get_performance_tier(ai_adjusted_score),
            'scaling_recommendation': self._get_ai_scaling_recommendation(cpu_score, memory_score, storage_score),
            'ai_workload_optimization': ai_workload_optimization
        }
    
    # AI helper methods
    def _predict_cpu_scaling(self, config: Dict, performance: float) -> Dict:
        """AI-powered CPU scaling prediction"""
        database_size = config['database_size_gb']
        
        scaling_factor = 1.0
        if database_size > 10000:
            scaling_factor = 1.3
        elif database_size > 5000:
            scaling_factor = 1.15
        
        return {
            'predicted_scaling_needs': scaling_factor,
            'bottleneck_prediction': 'likely' if performance < 40 else 'possible' if performance < 70 else 'unlikely',
            'recommendation': 'Upgrade CPU' if performance < 40 else 'Monitor performance' if performance < 70 else 'Current CPU sufficient'
        }
    
    def _analyze_memory_requirements(self, config: Dict, effective_memory: float) -> Dict:
        """AI memory requirement analysis"""
        database_size = config['database_size_gb']
        
        recommended_memory = database_size * 0.1  # 10% of database size as baseline
        if config['database_engine'] in ['oracle', 'postgresql']:
            recommended_memory *= 1.5
        
        optimization_potential = max(0, (recommended_memory - effective_memory) / recommended_memory)
        
        return {
            'recommended_memory_gb': recommended_memory,
            'current_vs_recommended': effective_memory / recommended_memory,
            'optimization_potential': optimization_potential,
            'memory_adequacy': 'sufficient' if effective_memory >= recommended_memory else 'insufficient'
        }
    
    def _analyze_storage_optimization(self, config: Dict, storage_type: str, effective_iops: float) -> Dict:
        """AI storage optimization analysis"""
        database_size = config['database_size_gb']
        
        # Predict IOPS requirements based on database size and type
        if config['database_engine'] in ['oracle', 'sqlserver']:
            required_iops = database_size * 5  # Higher IOPS requirement
        else:
            required_iops = database_size * 3
        
        upgrade_needed = effective_iops < required_iops
        
        return {
            'required_iops': required_iops,
            'current_vs_required': effective_iops / required_iops,
            'upgrade_recommendation': 'Upgrade to NVMe SSD' if upgrade_needed and storage_type != 'nvme_ssd' else 'Current storage adequate',
            'performance_impact': 'high' if upgrade_needed else 'low'
        }
    
    def _analyze_network_optimization(self, config: Dict, effective_bandwidth: float) -> Dict:
        """AI network optimization analysis"""
        database_size = config['database_size_gb']
        
        # Predict network requirements
        required_bandwidth = min(10000, database_size * 10)  # 10 Mbps per GB, max 10 Gbps
        
        optimization_score = min(100, (effective_bandwidth / required_bandwidth) * 100)
        
        return {
            'required_bandwidth_mbps': required_bandwidth,
            'optimization_score': optimization_score,
            'bottleneck_risk': 'high' if optimization_score < 50 else 'medium' if optimization_score < 80 else 'low'
        }
    
    def _analyze_database_optimization(self, config: Dict, effective_tps: float, db_optimization: float) -> Dict:
        """AI database optimization analysis"""
        
        # Predict TPS requirements based on database characteristics
        base_requirement = 2000  # Base TPS requirement
        if config['performance_requirements'] == 'high':
            base_requirement *= 2
        if config['database_size_gb'] > 10000:
            base_requirement *= 1.5
        
        tuning_potential = (1 - db_optimization) * 100
        
        return {
            'required_tps': base_requirement,
            'current_vs_required': effective_tps / base_requirement,
            'tuning_potential': tuning_potential,
            'optimization_priority': 'high' if tuning_potential > 20 else 'medium' if tuning_potential > 10 else 'low'
        }
    
    def _analyze_workload_optimization(self, config: Dict, scores: Dict) -> Dict:
        """AI workload-specific optimization analysis"""
        
        # Identify workload pattern
        if config['database_engine'] in ['mysql', 'postgresql']:
            workload_type = 'oltp'
            cpu_weight = 0.3
            memory_weight = 0.25
            storage_weight = 0.3
        elif config['database_engine'] == 'oracle':
            workload_type = 'mixed'
            cpu_weight = 0.25
            memory_weight = 0.3
            storage_weight = 0.25
        else:
            workload_type = 'general'
            cpu_weight = 0.25
            memory_weight = 0.2
            storage_weight = 0.25
        
        # Calculate workload-specific optimization factor
        weighted_score = (
            scores['cpu_score'] * cpu_weight +
            scores['memory_score'] * memory_weight +
            scores['storage_score'] * storage_weight +
            scores['network_score'] * 0.1 +
            scores['database_score'] * 0.1
        )
        
        optimization_factor = weighted_score / 100
        
        return {
            'workload_type': workload_type,
            'optimization_factor': optimization_factor,
            'primary_bottleneck': max(scores.items(), key=lambda x: x[1])[0].replace('_score', ''),
            'recommended_focus': self._get_optimization_focus(scores)
        }
    
    def _get_optimization_focus(self, scores: Dict) -> str:
        """Get primary optimization focus"""
        min_score = min(scores.values())
        min_component = [k for k, v in scores.items() if v == min_score][0]
        
        focus_map = {
            'cpu_score': 'CPU and processing optimization',
            'memory_score': 'Memory allocation and caching',
            'storage_score': 'Storage performance and I/O optimization',
            'network_score': 'Network bandwidth and latency',
            'database_score': 'Database configuration and tuning'
        }
        
        return focus_map.get(min_component, 'General performance optimization')
    
    def _ai_identify_bottlenecks(self, cpu_perf: Dict, mem_perf: Dict, 
                               storage_perf: Dict, net_perf: Dict, config: Dict) -> Dict:
        """AI-powered bottleneck identification"""
        
        bottlenecks = []
        ai_insights = []
        
        # CPU analysis
        if cpu_perf.get('ai_bottleneck_risk') == 'high':
            bottlenecks.append("CPU performance insufficient for workload")
            ai_insights.append("AI predicts CPU will be primary bottleneck during peak loads")
        
        # Memory analysis
        if mem_perf.get('memory_pressure') == 'high':
            bottlenecks.append("Memory pressure detected")
            ai_insights.append(f"AI recommends {mem_perf['ai_memory_analysis']['recommended_memory_gb']:.0f}GB for optimal performance")
        
        # Storage analysis
        if storage_perf.get('ai_storage_analysis', {}).get('performance_impact') == 'high':
            bottlenecks.append("Storage IOPS insufficient")
            ai_insights.append("AI suggests storage upgrade will provide significant performance improvement")
        
        # Network analysis
        if net_perf.get('ai_network_analysis', {}).get('bottleneck_risk') == 'high':
            bottlenecks.append("Network bandwidth limited")
            ai_insights.append("AI identifies network as migration performance constraint")
        
        if not bottlenecks:
            bottlenecks.append("No significant bottlenecks detected by AI analysis")
            ai_insights.append("AI assessment indicates well-balanced system configuration")
        
        return {
            'bottlenecks': bottlenecks,
            'ai_insights': ai_insights
        }
    
    def _generate_ai_optimization_recommendations(self, overall_perf: Dict, 
                                                os_impact: Dict, config: Dict) -> List[str]:
        """Generate AI-powered optimization recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if overall_perf['ai_adjusted_score'] < 60:
            recommendations.append("AI recommends comprehensive performance review and hardware upgrade")
        elif overall_perf['ai_adjusted_score'] < 80:
            recommendations.append("AI suggests targeted optimization of lowest-performing components")
        
        # OS-specific recommendations
        os_insights = os_impact.get('ai_insights', {})
        if os_insights:
            recommendations.extend([
                f"OS Consideration: {insight}" for insight in os_insights.get('migration_considerations', [])[:2]
            ])
        
        # Workload-specific recommendations
        workload_opt = overall_perf.get('ai_workload_optimization', {})
        if workload_opt.get('recommended_focus'):
            recommendations.append(f"AI Priority: Focus on {workload_opt['recommended_focus']}")
        
        # Database-specific recommendations
        if config['database_size_gb'] > 10000:
            recommendations.append("AI recommends staged migration approach for large database")
        
        if config['performance_requirements'] == 'high':
            recommendations.append("AI suggests performance testing in AWS environment before full migration")
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    # Keep original helper methods
    def _get_performance_tier(self, score: float) -> str:
        if score >= 80:
            return "High Performance"
        elif score >= 60:
            return "Standard Performance"
        elif score >= 40:
            return "Basic Performance"
        else:
            return "Limited Performance"
    
    def _get_ai_scaling_recommendation(self, cpu_score: float, memory_score: float, storage_score: float) -> List[str]:
        """AI-enhanced scaling recommendations"""
        recommendations = []
        
        if cpu_score < 60:
            recommendations.append("AI Priority: CPU upgrade or more cores for improved performance")
        if memory_score < 60:
            recommendations.append("AI Priority: Memory expansion for better caching efficiency")
        if storage_score < 60:
            recommendations.append("AI Priority: Storage upgrade to NVMe SSD for IOPS improvement")
        
        if not recommendations:
            recommendations.append("AI Assessment: System is well-balanced for current workload")
        
        return recommendations

class EnhancedMigrationAnalyzer:
    """Enhanced migration analyzer with AI and AWS API integration plus FSx support"""
    
    
    async def comprehensive_ai_migration_analysis(self, config: Dict) -> Dict:
        """Comprehensive AI-powered migration analysis with agent scaling and FSx support"""
        
        # API status tracking
        api_status = APIStatus(
            anthropic_connected=self.ai_manager.connected,
            aws_pricing_connected=self.aws_api.connected,
            last_update=datetime.now()
        )
        
        # Enhanced on-premises performance analysis
        onprem_performance = self.onprem_analyzer.calculate_ai_enhanced_performance(config, self.os_manager)
        
        # Determine network path key based on config and destination storage
        network_path_key = self._get_network_path_key(config)
        
        # AI-enhanced network path analysis
        network_perf = self.network_manager.calculate_ai_enhanced_path_performance(network_path_key)
        
        # Determine migration type and tools (preserved)
        is_homogeneous = config['source_database_engine'] == config['database_engine']
        migration_type = 'homogeneous' if is_homogeneous else 'heterogeneous'
        primary_tool = 'datasync' if is_homogeneous else 'dms'
        
        # Enhanced agent analysis with scaling support and destination storage
        agent_analysis = await self._analyze_ai_migration_agents_with_scaling(config, primary_tool, network_perf)
        
        # Calculate effective migration throughput with multiple agents
        agent_throughput = agent_analysis['total_effective_throughput']
        network_throughput = network_perf['effective_bandwidth_mbps']
        migration_throughput = min(agent_throughput, network_throughput)
        
        # AI-enhanced migration time calculation with agent scaling
        migration_time_hours = await self._calculate_ai_migration_time_with_agents(
            config, migration_throughput, onprem_performance, agent_analysis
        )
        
        # AI-powered AWS sizing recommendations
        aws_sizing = await self.aws_manager.ai_enhanced_aws_sizing(config)
        
        # Enhanced cost analysis with agent scaling costs and FSx costs
        cost_analysis = await self._calculate_ai_enhanced_costs_with_agents(
            config, aws_sizing, agent_analysis, network_perf
        )
        
        # Generate FSx destination comparisons
        fsx_comparisons = await self._generate_fsx_destination_comparisons(config)
        
        return {
            'api_status': api_status,
            'onprem_performance': onprem_performance,
            'network_performance': network_perf,
            'migration_type': migration_type,
            'primary_tool': primary_tool,
            'agent_analysis': agent_analysis,
            'migration_throughput_mbps': migration_throughput,
            'estimated_migration_time_hours': migration_time_hours,
            'aws_sizing_recommendations': aws_sizing,
            'cost_analysis': cost_analysis,
            'fsx_comparisons': fsx_comparisons,
            'ai_overall_assessment': await self._generate_ai_overall_assessment_with_agents(
                config, onprem_performance, aws_sizing, migration_time_hours, agent_analysis
            )
        }
    
    def _get_network_path_key(self, config: Dict) -> str:
        """Get network path key with corrected FSx routing"""
        
        # Validate required config keys
        if 'operating_system' not in config:
            raise ValueError("Missing required 'operating_system' in config")
        if 'environment' not in config:
            raise ValueError("Missing required 'environment' in config")
        
        # Determine OS type with more precise matching
        os_lower = config['operating_system'].lower()
        if any(os_name in os_lower for os_name in ['linux', 'ubuntu', 'rhel', 'centos', 'debian']):
            os_type = 'linux'
        elif 'windows' in os_lower:
            os_type = 'windows'
        else:
            # Default to linux if unknown
            os_type = 'linux'
        
        # Clean environment name
        environment = config['environment'].replace('-', '_').lower()
        
        # Get destination storage type
        destination_storage = config.get('destination_storage_type', 'S3').lower()
        
        # FIXED: Proper FSx destination path mapping
        if environment in ['non_production', 'nonprod', 'dev', 'test', 'staging']:
            if destination_storage == 's3':
                return f"nonprod_sj_{os_type}_{'nas' if os_type == 'linux' else 'share'}_s3"
            elif destination_storage == 'fsx_windows':
                return f"nonprod_sj_{os_type}_{'nas' if os_type == 'linux' else 'share'}_fsx_windows"
            elif destination_storage == 'fsx_lustre':
                return f"nonprod_sj_{os_type}_{'nas' if os_type == 'linux' else 'share'}_fsx_lustre"
            else:
                # Fallback for unknown FSx types
                return f"nonprod_sj_{os_type}_{'nas' if os_type == 'linux' else 'share'}_s3"
        
        elif environment in ['production', 'prod']:
            if destination_storage == 's3':
                return f"prod_sa_{os_type}_{'nas' if os_type == 'linux' else 'share'}_s3"
            elif destination_storage == 'fsx_windows':
                return f"prod_sa_{os_type}_{'nas' if os_type == 'linux' else 'share'}_fsx_windows"
            elif destination_storage == 'fsx_lustre':
                return f"prod_sa_{os_type}_{'nas' if os_type == 'linux' else 'share'}_fsx_lustre"
            else:
                # Fallback for unknown FSx types
                return f"prod_sa_{os_type}_{'nas' if os_type == 'linux' else 'share'}_s3"
        
        # Default fallback
        return f"nonprod_sj_{os_type}_{'nas' if os_type == 'linux' else 'share'}_s3"
    
    
    async def _generate_corrected_fsx_destination_comparisons(self, config: Dict) -> Dict:
        """Generate FSx comparisons with corrected architecture understanding"""

        comparisons = {}
        destination_types = ['S3', 'FSx_Windows', 'FSx_Lustre']
        
        for dest_type in destination_types:
            temp_config = config.copy()
            temp_config['destination_storage_type'] = dest_type
            
            # Get corrected architecture
            agent_manager = EnhancedAgentSizingManager()
            architecture = agent_manager.get_actual_migration_architecture(
                'datasync' if config['source_database_engine'] == config['database_engine'] else 'dms',
                dest_type,
                config
            )
            
            # Calculate migration specifics based on architecture
            if dest_type == 'S3':
                # Direct migration
                migration_description = "Direct database migration to S3"
                complexity_factor = 1.0
                setup_complexity = "Low"
            
            elif dest_type == 'FSx_Windows':
                # Hybrid architecture
                migration_description = "Database to EC2+EBS + File shares to FSx Windows"
                complexity_factor = 1.3
                setup_complexity = "Medium-High"
            
            elif dest_type == 'FSx_Lustre':
                # HPC hybrid
                migration_description = "Database to EC2+EBS + HPC data to FSx Lustre"
                complexity_factor = 1.5
                setup_complexity = "High"
            
            else:
                # Fallback for unknown destination types
                migration_description = f"Migration to {dest_type}"
                complexity_factor = 1.2
                setup_complexity = "Medium"
            
            # Network path calculation
            network_path_key = self._get_network_path_key(temp_config)
            network_perf = self.network_manager.calculate_ai_enhanced_path_performance(network_path_key)
            
            # Agent configuration
            is_homogeneous = config['source_database_engine'] == config['database_engine']
            primary_tool = 'datasync' if is_homogeneous else 'dms'
            agent_size = config.get('datasync_agent_size' if is_homogeneous else 'dms_agent_size', 'medium')
            num_agents = config.get('number_of_agents', 1)
            
            agent_config = agent_manager.calculate_agent_configuration(
                primary_tool, agent_size, num_agents, dest_type
            )
            
            # Migration time calculation with architecture consideration
            database_size_gb = config['database_size_gb']
            
            # Initialize variables
            migration_throughput = 0
            migration_time_hours = 0
            
            if architecture['agent_targets_destination']:
                # Direct migration
                migration_throughput = min(agent_config['total_max_throughput_mbps'], 
                                        network_perf['effective_bandwidth_mbps'])
                
                # Prevent division by zero
                if migration_throughput > 0:
                    migration_time_hours = (database_size_gb * 8 * 1000) / (migration_throughput * 3600)
                else:
                    migration_time_hours = float('inf')  # Infinite time if no throughput
            else:
                # Split workload
                db_size = database_size_gb * (architecture.get('database_percentage', 80) / 100)
                file_size = database_size_gb * (architecture.get('file_percentage', 20) / 100)
                
                # Calculate throughput for each portion
                base_throughput = min(agent_config['total_max_throughput_mbps'], 
                                    network_perf['effective_bandwidth_mbps'])
                
                if base_throughput > 0:
                    # Database migration throughput (no FSx multiplier)
                    db_throughput = base_throughput
                    db_migration_time = (db_size * 8 * 1000) / (db_throughput * 3600)
                    
                    # File migration throughput (with FSx multiplier if applicable)
                    file_throughput = base_throughput * agent_config.get('storage_performance_multiplier', 1.0)
                    file_migration_time = (file_size * 8 * 1000) / (file_throughput * 3600)
                    
                    # Total time (can run in parallel, so take the max)
                    migration_time_hours = max(db_migration_time, file_migration_time) * complexity_factor
                    
                    # Use the effective combined throughput for reporting
                    if migration_time_hours > 0:
                        migration_throughput = (database_size_gb * 8 * 1000) / (migration_time_hours * 3600)
                    else:
                        migration_throughput = base_throughput
                else:
                    migration_time_hours = float('inf')
                    migration_throughput = 0
            
            # Cost calculation
            base_storage_cost = database_size_gb * {
                'S3': 0.023,
                'FSx_Windows': 0.13,
                'FSx_Lustre': 0.14
            }.get(dest_type, 0.023)
            
            # Add EC2+EBS costs for FSx scenarios
            if dest_type in ['FSx_Windows', 'FSx_Lustre']:
                ec2_storage_cost = database_size_gb * 0.08  # EBS GP3 for database
                total_storage_cost = base_storage_cost + ec2_storage_cost
            else:
                total_storage_cost = base_storage_cost
            
            comparisons[dest_type] = {
                'destination_type': dest_type,
                'migration_architecture': architecture,
                'migration_description': migration_description,
                'network_performance': network_perf,
                'agent_configuration': agent_config,
                'migration_throughput_mbps': migration_throughput,
                'estimated_migration_time_hours': migration_time_hours,
                'estimated_monthly_storage_cost': total_storage_cost,
                'setup_complexity': setup_complexity,
                'performance_rating': self._calculate_destination_performance_rating(dest_type, network_perf, agent_config),
                'cost_rating': self._calculate_destination_cost_rating(dest_type, total_storage_cost),
                'complexity_rating': setup_complexity,
                'recommendations': self._get_corrected_destination_recommendations(dest_type, architecture, config),
                'architecture_notes': [
                    f"Primary tool: {agent_config['migration_architecture']['description']}",
                    f"Architecture: {agent_config['migration_architecture']['architecture_type']}",
                    f"Bandwidth calculation: {agent_config['migration_architecture']['bandwidth_calculation']}"
                ]
            }
        
        return comparisons
    def _calculate_destination_performance_rating(self, dest_type: str, network_perf: Dict, agent_config: Dict) -> str:
        """Calculate performance rating for destination type"""
        
        throughput = agent_config.get('total_max_throughput_mbps', 0)
        network_quality = network_perf.get('ai_enhanced_quality_score', 0)
        
        # Base performance scoring
        performance_score = 0
        
        if dest_type == 'S3':
            performance_score = 70 + (network_quality * 0.3)
        elif dest_type == 'FSx_Windows':
            performance_score = 80 + (network_quality * 0.2)
        elif dest_type == 'FSx_Lustre':
            performance_score = 95 + (network_quality * 0.05)
        
        # Adjust for throughput
        if throughput > 2000:
            performance_score += 10
        elif throughput > 1000:
            performance_score += 5
        
        if performance_score >= 90:
            return "Excellent"
        elif performance_score >= 80:
            return "Very Good"
        elif performance_score >= 70:
            return "Good"
        elif performance_score >= 60:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_destination_cost_rating(self, dest_type: str, estimated_cost: float) -> str:
        """Calculate cost rating for destination type"""
        
        if dest_type == 'S3':
            return "Excellent"
        elif dest_type == 'FSx_Windows':
            return "Good"
        elif dest_type == 'FSx_Lustre':
            return "Fair"
        else:
            return "Unknown"
    
    def _calculate_destination_complexity_rating(self, dest_type: str, config: Dict) -> str:
        """Calculate complexity rating for destination type"""
        
        base_complexity = 1
        
        if config['source_database_engine'] != config['database_engine']:
            base_complexity += 1
        if config['database_size_gb'] > 10000:
            base_complexity += 1
        if config.get('number_of_agents', 1) > 3:
            base_complexity += 1
        
        # Destination-specific complexity
        if dest_type == 'S3':
            dest_complexity = 1
        elif dest_type == 'FSx_Windows':
            dest_complexity = 2
        elif dest_type == 'FSx_Lustre':
            dest_complexity = 3
        else:
            dest_complexity = 1
        
        total_complexity = base_complexity + dest_complexity
        
        if total_complexity <= 2:
            return "Low"
        elif total_complexity <= 4:
            return "Medium"
        else:
            return "High"
    
    def _get_destination_recommendations(self, dest_type: str, config: Dict, network_perf: Dict) -> List[str]:
        """Get recommendations for specific destination type"""
        
        recommendations = []
        
        if dest_type == 'S3':
            recommendations.extend([
                "Ideal for cost-effective cloud storage",
                "Simple integration with AWS services",
                "Excellent durability and availability",
                "Consider S3 Intelligent Tiering for cost optimization"
            ])
        elif dest_type == 'FSx_Windows':
            recommendations.extend([
                "Perfect for Windows-based applications",
                "Native Windows file system features",
                "Better performance than S3 for file-based workloads",
                "Requires Active Directory integration planning"
            ])
        elif dest_type == 'FSx_Lustre':
            recommendations.extend([
                "Best choice for high-performance computing",
                "Extremely high throughput and low latency",
                "Ideal for analytics and machine learning workloads",
                "Requires Lustre expertise and careful configuration"
            ])
        
        # Add configuration-specific recommendations
        if config['database_size_gb'] > 20000:
            if dest_type == 'FSx_Lustre':
                recommendations.append("Large database size benefits from Lustre's parallel performance")
            elif dest_type == 'S3':
                recommendations.append("Consider multipart uploads for large database migration")
        
        if config['performance_requirements'] == 'high':
            if dest_type == 'FSx_Lustre':
                recommendations.append("High performance requirements perfectly match Lustre capabilities")
            elif dest_type == 'S3':
                recommendations.append("Consider S3 Transfer Acceleration for better performance")
        
        return recommendations[:4]  # Limit to top 4 recommendations
    
    async def _analyze_ai_migration_agents_with_scaling(self, config: Dict, primary_tool: str, network_perf: Dict) -> Dict:
        """Enhanced migration agent analysis with scaling support and destination storage"""
        
        num_agents = config.get('number_of_agents', 1)
        destination_storage = config.get('destination_storage_type', 'S3')
        
        if primary_tool == 'datasync':
            agent_size = config['datasync_agent_size']
            agent_config = self.agent_manager.calculate_agent_configuration('datasync', agent_size, num_agents, destination_storage)
        else:
            agent_size = config['dms_agent_size']
            agent_config = self.agent_manager.calculate_agent_configuration('dms', agent_size, num_agents, destination_storage)
        
        # Calculate throughput impact with scaling and destination storage
        total_max_throughput = agent_config['total_max_throughput_mbps']
        network_bandwidth = network_perf['effective_bandwidth_mbps']
        total_effective_throughput = min(total_max_throughput, network_bandwidth)
        throughput_impact = total_effective_throughput / total_max_throughput
        
        # Determine bottleneck with agent scaling considerations
        if total_max_throughput < network_bandwidth:
            bottleneck = f'agents ({num_agents} agents)'
            bottleneck_severity = 'high' if throughput_impact < 0.7 else 'medium'
        else:
            bottleneck = 'network'
            bottleneck_severity = 'medium' if throughput_impact > 0.8 else 'high'
        
        # AI enhancement: optimization recommendations with scaling and destination storage
        ai_agent_optimization = await self._get_ai_agent_optimization_with_scaling(
            agent_config, network_perf, config, num_agents
        )
        
        # Get optimal agent recommendations for this destination
        optimal_recommendations = self.agent_manager.recommend_optimal_agents(
            config['database_size_gb'],
            network_bandwidth,
            config.get('target_migration_hours', 24),
            destination_storage
        )
        
        return {
            'primary_tool': primary_tool,
            'agent_size': agent_size,
            'number_of_agents': num_agents,
            'destination_storage': destination_storage,
            'agent_configuration': agent_config,
            'total_max_throughput_mbps': total_max_throughput,
            'total_effective_throughput': total_effective_throughput,
            'throughput_impact': throughput_impact,
            'bottleneck': bottleneck,
            'bottleneck_severity': bottleneck_severity,
            'scaling_efficiency': agent_config['scaling_efficiency'],
            'management_overhead': agent_config['management_overhead_factor'],
            'storage_performance_multiplier': agent_config.get('storage_performance_multiplier', 1.0),
            'ai_optimization': ai_agent_optimization,
            'optimal_recommendations': optimal_recommendations,
            'cost_per_hour': agent_config['effective_cost_per_hour'],
            'monthly_cost': agent_config['total_monthly_cost']
        }
    
    async def _calculate_ai_migration_time_with_agents(self, config: Dict, migration_throughput: float, 
                                                     onprem_performance: Dict, agent_analysis: Dict) -> float:
        """AI-enhanced migration time calculation with agent scaling and destination storage"""
        
        database_size_gb = config['database_size_gb']
        num_agents = config.get('number_of_agents', 1)
        destination_storage = config.get('destination_storage_type', 'S3')
        
        # Base calculation (preserved)
        base_time_hours = (database_size_gb * 8 * 1000) / (migration_throughput * 3600)
        
        # AI adjustments
        ai_complexity = onprem_performance.get('ai_optimization_recommendations', [])
        complexity_factor = 1.0
        
        # Increase time for complex scenarios
        if config['source_database_engine'] != config['database_engine']:
            complexity_factor *= 1.3  # Heterogeneous migration
        
        if 'windows' in config['operating_system']:
            complexity_factor *= 1.1  # Windows overhead
        
        if config['server_type'] == 'vmware':
            complexity_factor *= 1.05  # Virtualization overhead
        
        # Destination storage adjustments
        if destination_storage == 'FSx_Windows':
            complexity_factor *= 0.9  # Better performance, less time
        elif destination_storage == 'FSx_Lustre':
            complexity_factor *= 0.7  # Much better performance, significantly less time
        
        # Agent scaling adjustments
        scaling_efficiency = agent_analysis.get('scaling_efficiency', 1.0)
        storage_multiplier = agent_analysis.get('storage_performance_multiplier', 1.0)
        
        if num_agents > 1:
            # More agents can reduce time but with coordination overhead
            agent_time_factor = (1 / min(num_agents * scaling_efficiency * storage_multiplier, 6.0))  # Diminishing returns
            complexity_factor *= agent_time_factor
            
            # Add coordination overhead for many agents
            if num_agents > 5:
                complexity_factor *= 1.1  # 10% coordination overhead
        
        # AI insights factor
        if len(ai_complexity) > 3:
            complexity_factor *= 1.2  # Many optimization needs
        
        return base_time_hours * complexity_factor
    
    async def _calculate_ai_enhanced_costs_with_agents(self, config: Dict, aws_sizing: Dict, 
                                                     agent_analysis: Dict, network_perf: Dict) -> Dict:
        """AI-enhanced cost calculation with agent scaling costs and FSx storage costs"""
        
        # Get deployment recommendation
        deployment_rec = aws_sizing['deployment_recommendation']['recommendation']
        
        # Use real-time pricing data
        if deployment_rec == 'rds':
            aws_compute_cost = aws_sizing['rds_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['rds_recommendations']['monthly_storage_cost']
        else:
            aws_compute_cost = aws_sizing['ec2_recommendations']['monthly_instance_cost']
            aws_storage_cost = aws_sizing['ec2_recommendations']['monthly_storage_cost']
        
        # Enhanced agent costs with scaling
        agent_monthly_cost = agent_analysis.get('monthly_cost', 0)
        management_overhead = agent_analysis.get('management_overhead', 1.0)
        storage_management_overhead = agent_analysis.get('storage_management_overhead', 1.0)
        total_agent_cost = agent_monthly_cost * management_overhead * storage_management_overhead
        
        # Destination storage costs
        destination_storage = config.get('destination_storage_type', 'S3')
        destination_storage_cost = self._calculate_destination_storage_cost(config, destination_storage)
        
        # Network costs with AI optimization
        base_network_cost = 800 if 'prod' in config.get('network_path', '') else 400
        ai_optimization_factor = network_perf.get('ai_optimization_potential', 0) / 100
        optimized_network_cost = base_network_cost * (1 - ai_optimization_factor * 0.2)
        
        # OS licensing with AI insights
        os_licensing_cost = self.os_manager.operating_systems[config['operating_system']]['licensing_cost_factor'] * 150
        
        # Management costs (AI reduces if RDS, increases with more agents and complex storage)
        base_management_cost = 200 if deployment_rec == 'ec2' else 50
        ai_management_reduction = 0.15 if aws_sizing.get('ai_analysis', {}).get('ai_complexity_score', 6) < 5 else 0
        agent_management_increase = (config.get('number_of_agents', 1) - 1) * 50  # $50 per additional agent management
        storage_management_increase = {
            'S3': 0,
            'FSx_Windows': 100,
            'FSx_Lustre': 200
        }.get(destination_storage, 0)
        
        management_cost = ((base_management_cost * (1 - ai_management_reduction)) + 
                          agent_management_increase + storage_management_increase)
        
        # One-time migration costs with AI complexity, agent scaling, and storage complexity
        ai_complexity = aws_sizing.get('ai_analysis', {}).get('ai_complexity_score', 6)
        complexity_multiplier = 1.0 + (ai_complexity - 5) * 0.1
        
        # Agent scaling impact on migration cost
        num_agents = config.get('number_of_agents', 1)
        agent_setup_cost = num_agents * 500  # $500 setup cost per agent
        agent_coordination_cost = max(0, (num_agents - 1) * 200)  # $200 coordination cost per additional agent
        
        # Destination storage setup costs
        storage_setup_costs = {
            'S3': 100,
            'FSx_Windows': 1000,
            'FSx_Lustre': 2000
        }
        storage_setup_cost = storage_setup_costs.get(destination_storage, 100)
        
        base_migration_cost = config['database_size_gb'] * 0.1
        one_time_migration_cost = (base_migration_cost * complexity_multiplier + 
                                 agent_setup_cost + agent_coordination_cost + storage_setup_cost)
        
        total_monthly_cost = (aws_compute_cost + aws_storage_cost + total_agent_cost + 
                            destination_storage_cost + optimized_network_cost + 
                            os_licensing_cost + management_cost)
        
        # AI-predicted savings with agent optimization and storage efficiency
        ai_optimization_potential = aws_sizing.get('ai_analysis', {}).get('performance_recommendations', [])
        agent_efficiency_bonus = agent_analysis.get('scaling_efficiency', 1.0) * 0.1  # Up to 10% bonus for efficient scaling
        storage_efficiency_bonus = {
            'S3': 0.15,  # High efficiency
            'FSx_Windows': 0.10,  # Medium efficiency
            'FSx_Lustre': 0.05   # Lower efficiency due to high performance focus
        }.get(destination_storage, 0.10)
        
        estimated_monthly_savings = total_monthly_cost * (0.1 + len(ai_optimization_potential) * 0.02 + 
                                                        agent_efficiency_bonus + storage_efficiency_bonus)
        roi_months = int(one_time_migration_cost / estimated_monthly_savings) if estimated_monthly_savings > 0 else None
        
        return {
            'aws_compute_cost': aws_compute_cost,
            'aws_storage_cost': aws_storage_cost,
            'agent_cost': total_agent_cost,
            'agent_base_cost': agent_monthly_cost,
            'agent_management_overhead': management_overhead,
            'destination_storage_cost': destination_storage_cost,
            'destination_storage_type': destination_storage,
            'network_cost': optimized_network_cost,
            'os_licensing_cost': os_licensing_cost,
            'management_cost': management_cost,
            'total_monthly_cost': total_monthly_cost,
            'one_time_migration_cost': one_time_migration_cost,
            'agent_setup_cost': agent_setup_cost,
            'agent_coordination_cost': agent_coordination_cost,
            'storage_setup_cost': storage_setup_cost,
            'estimated_monthly_savings': estimated_monthly_savings,
            'roi_months': roi_months,
            'ai_cost_insights': {
                'ai_optimization_factor': ai_optimization_factor,
                'complexity_multiplier': complexity_multiplier,
                'management_reduction': ai_management_reduction,
                'agent_efficiency_bonus': agent_efficiency_bonus,
                'storage_efficiency_bonus': storage_efficiency_bonus,
                'potential_additional_savings': f"{len(ai_optimization_potential) * 2 + int(agent_efficiency_bonus * 10) + int(storage_efficiency_bonus * 10)}% through AI, agent, and storage optimization"
            }
        }
    
    def _calculate_destination_storage_cost(self, config: Dict, destination_storage: str) -> float:
        """Calculate monthly cost for destination storage"""
        
        database_size_gb = config['database_size_gb']
        
        # Base cost per GB per month
        storage_costs = {
            'S3': 0.023,  # S3 Standard
            'FSx_Windows': 0.13,  # FSx for Windows
            'FSx_Lustre': 0.14   # FSx for Lustre
        }
        
        base_cost_per_gb = storage_costs.get(destination_storage, 0.023)
        
        # Storage multiplier based on migration requirements
        storage_multiplier = 1.5  # Extra space for migration
        
        return database_size_gb * storage_multiplier * base_cost_per_gb
    
    async def _get_ai_agent_optimization_with_scaling(self, agent_config: Dict, network_perf: Dict, 
                                                    config: Dict, num_agents: int) -> Dict:
        """Get AI-powered agent optimization recommendations with scaling and destination storage considerations"""
        
        # Enhanced analysis with scaling and destination storage
        network_bandwidth = network_perf['effective_bandwidth_mbps']
        total_agent_capacity = agent_config['total_max_throughput_mbps']
        destination_storage = config.get('destination_storage_type', 'S3')
        
        if network_bandwidth < total_agent_capacity:
            bottleneck_type = 'network'
            optimization_potential = network_perf.get('ai_optimization_potential', 0)
            recommendations = network_perf.get('ai_insights', {}).get('optimization_opportunities', [])
        else:
            bottleneck_type = 'agents'
            optimization_potential = max(0, 20 - (agent_config['scaling_efficiency'] * 20))  # Efficiency-based optimization
            recommendations = agent_config.get('scaling_recommendations', [])
        
        # Agent-specific optimizations
        agent_recommendations = [
            f"Optimize {num_agents}-agent coordination and load balancing",
            "Implement intelligent retry mechanisms across agents",
            "Configure bandwidth throttling during peak hours"
        ]
        
        if num_agents > 3:
            agent_recommendations.append("Consider agent consolidation to reduce complexity")
        elif num_agents == 1 and config['database_size_gb'] > 5000:
            agent_recommendations.append("Scale to multiple agents for better throughput")
        
        # Destination storage-specific optimizations
        if destination_storage == 'FSx_Windows':
            agent_recommendations.append("Configure agents for SMB protocol optimization")
            agent_recommendations.append("Implement Windows file sharing best practices")
        elif destination_storage == 'FSx_Lustre':
            agent_recommendations.append("Optimize agents for Lustre parallel I/O")
            agent_recommendations.append("Configure Lustre striping for maximum performance")
        
        recommendations.extend(agent_recommendations)
        
        return {
            'bottleneck_type': bottleneck_type,
            'optimization_potential_percent': optimization_potential,
            'recommendations': recommendations[:6],
            'estimated_improvement': f"{optimization_potential}% throughput improvement possible",
            'scaling_assessment': agent_config.get('optimal_configuration', {}),
            'current_efficiency': agent_config.get('scaling_efficiency', 1.0) * 100,
            'destination_storage_impact': f"{destination_storage} provides {agent_config.get('storage_performance_multiplier', 1.0):.1f}x performance multiplier"
        }
    
    async def _generate_ai_overall_assessment_with_agents(self, config: Dict, onprem_performance: Dict, 
                                                        aws_sizing: Dict, migration_time: float, 
                                                        agent_analysis: Dict) -> Dict:
        """Generate AI-powered overall migration assessment with agent scaling and destination storage considerations"""
        
        # Migration readiness score with agent and destination storage considerations
        readiness_factors = []
        readiness_score = 100
        
        # Performance readiness
        perf_score = onprem_performance.get('performance_score', 0)
        if perf_score < 50:
            readiness_score -= 20
            readiness_factors.append("Performance optimization needed before migration")
        elif perf_score < 70:
            readiness_score -= 10
            readiness_factors.append("Minor performance improvements recommended")
        
        # Complexity assessment
        ai_complexity = aws_sizing.get('ai_analysis', {}).get('ai_complexity_score', 6)
        if ai_complexity > 8:
            readiness_score -= 25
            readiness_factors.append("High complexity migration requires extensive planning")
        elif ai_complexity > 6:
            readiness_score -= 10
            readiness_factors.append("Moderate complexity migration needs careful execution")
        
        # Agent scaling assessment
        num_agents = config.get('number_of_agents', 1)
        scaling_efficiency = agent_analysis.get('scaling_efficiency', 1.0)
        optimal_config = agent_analysis.get('optimal_recommendations', {}).get('optimal_configuration')
        
        if optimal_config and num_agents != optimal_config['configuration']['number_of_agents']:
            readiness_score -= 10
            readiness_factors.append(f"Agent configuration not optimal - consider {optimal_config['configuration']['number_of_agents']} agents")
        
        if scaling_efficiency < 0.85:
            readiness_score -= 15
            readiness_factors.append("Agent scaling efficiency below optimal - coordination overhead detected")
        
        # Destination storage assessment
        destination_storage = config.get('destination_storage_type', 'S3')
        if destination_storage == 'FSx_Lustre':
            readiness_score += 5  # Bonus for high-performance storage
            readiness_factors.append("FSx for Lustre provides excellent performance capabilities")
        elif destination_storage == 'FSx_Windows':
            readiness_score += 3  # Small bonus for enhanced performance
            readiness_factors.append("FSx for Windows provides good performance and integration")
        
        # Time assessment
        if migration_time > 48:
            readiness_score -= 15
            readiness_factors.append("Long migration time requires extensive downtime planning")
        elif migration_time > 24:
            readiness_score -= 5
            readiness_factors.append("Extended migration window needed")
        
        # Network readiness
        network_quality = onprem_performance.get('network_performance', {}).get('ai_optimization_score', 80)
        if network_quality < 60:
            readiness_score -= 15
            readiness_factors.append("Network optimization required for successful migration")
        
        # Agent bottleneck assessment
        bottleneck_severity = agent_analysis.get('bottleneck_severity', 'medium')
        if bottleneck_severity == 'high':
            readiness_score -= 20
            readiness_factors.append("Significant agent or network bottleneck detected")
        elif bottleneck_severity == 'medium':
            readiness_score -= 5
            readiness_factors.append("Minor bottleneck may impact migration performance")
        
        # Success probability with agent scaling bonus and destination storage bonus
        base_success_probability = max(60, min(95, readiness_score))
        agent_efficiency_bonus = (scaling_efficiency - 0.8) * 25 if scaling_efficiency > 0.8 else 0  # Up to 5% bonus
        storage_performance_bonus = {
            'S3': 0,
            'FSx_Windows': 2,
            'FSx_Lustre': 5
        }.get(destination_storage, 0)
        
        success_probability = min(95, base_success_probability + agent_efficiency_bonus + storage_performance_bonus)
        
        # Risk level
        if readiness_score >= 80:
            risk_level = "Low"
        elif readiness_score >= 65:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'migration_readiness_score': readiness_score,
            'success_probability': success_probability,
            'risk_level': risk_level,
            'readiness_factors': readiness_factors,
            'ai_confidence': aws_sizing.get('deployment_recommendation', {}).get('confidence', 0.5),
            'agent_scaling_impact': {
                'scaling_efficiency': scaling_efficiency * 100,
                'optimal_agents': optimal_config['configuration']['number_of_agents'] if optimal_config else num_agents,
                'current_agents': num_agents,
                'efficiency_bonus': agent_efficiency_bonus
            },
            'destination_storage_impact': {
                'storage_type': destination_storage,
                'performance_bonus': storage_performance_bonus,
                'storage_performance_multiplier': agent_analysis.get('storage_performance_multiplier', 1.0)
            },
            'recommended_next_steps': self._get_next_steps_with_agents(readiness_score, ai_complexity, agent_analysis, destination_storage),
            'timeline_recommendation': self._get_timeline_recommendation_with_agents(migration_time, ai_complexity, num_agents, destination_storage)
        }
    
    def _get_next_steps_with_agents(self, readiness_score: float, ai_complexity: int, 
                                  agent_analysis: Dict, destination_storage: str) -> List[str]:
        """Get recommended next steps with agent scaling and destination storage considerations"""
        
        steps = []
        
        if readiness_score < 70:
            steps.append("Conduct detailed performance baseline and optimization")
            steps.append("Address identified bottlenecks before migration")
        
        if ai_complexity > 7:
            steps.append("Develop comprehensive migration strategy and testing plan")
            steps.append("Consider engaging AWS migration specialists")
        
        # Agent-specific steps
        num_agents = agent_analysis.get('number_of_agents', 1)
        optimal_config = agent_analysis.get('optimal_recommendations', {}).get('optimal_configuration')
        
        if optimal_config and num_agents != optimal_config['configuration']['number_of_agents']:
            steps.append(f"Optimize agent configuration to {optimal_config['configuration']['number_of_agents']} agents")
        
        if num_agents > 3:
            steps.append(f"Set up centralized monitoring for {num_agents}-agent coordination")
        
        # Destination storage-specific steps
        if destination_storage == 'FSx_Windows':
            steps.append("Plan Active Directory integration for FSx for Windows")
            steps.append("Test SMB protocol performance and optimization")
        elif destination_storage == 'FSx_Lustre':
            steps.append("Design Lustre file system layout and striping strategy")
            steps.append("Plan Lustre client configuration and optimization")
        
        steps.extend([
            "Set up AWS environment and conduct connectivity tests",
            f"Perform proof-of-concept migration with {destination_storage} destination",
            "Develop detailed cutover and rollback procedures"
        ])
        
        return steps[:6]
    
    def _get_timeline_recommendation_with_agents(self, migration_time: float, ai_complexity: int, 
                                               num_agents: int, destination_storage: str) -> Dict:
        """Get AI-recommended timeline with agent scaling and destination storage considerations"""
        
        # Base phases
        planning_weeks = 2 + (ai_complexity - 5) * 0.5
        testing_weeks = 3 + (ai_complexity - 5) * 0.5
        migration_hours = migration_time
        
        # Agent scaling adjustments
        if num_agents > 3:
            planning_weeks += 1  # More planning for complex agent setups
            testing_weeks += 0.5  # Additional testing for coordination
        
        # Destination storage adjustments
        if destination_storage == 'FSx_Windows':
            planning_weeks += 0.5  # AD integration planning
            testing_weeks += 0.5   # SMB testing
        elif destination_storage == 'FSx_Lustre':
            planning_weeks += 1.0  # Lustre design and planning
            testing_weeks += 1.0   # Lustre performance testing
        
        return {
            'planning_phase_weeks': max(1, planning_weeks),
            'testing_phase_weeks': max(2, testing_weeks),
            'migration_window_hours': migration_hours,
            'total_project_weeks': max(6, planning_weeks + testing_weeks + 1),
            'recommended_approach': 'staged' if ai_complexity > 7 or migration_time > 24 or num_agents > 5 else 'direct',
            'agent_coordination_time': f"{num_agents * 2} hours for agent setup and coordination" if num_agents > 1 else "N/A",
            'storage_setup_time': {
                'S3': "Minimal setup time",
                'FSx_Windows': f"{planning_weeks * 2} hours for AD integration and FSx setup",
                'FSx_Lustre': f"{planning_weeks * 4} hours for Lustre configuration and optimization"
            }.get(destination_storage, "Standard setup time")
        }

# PDF Report Generation Class
class PDFReportGenerator:
    """Generate executive PDF reports for migration analysis"""
    
    def __init__(self):
        self.report_style = {
            'title_color': '#1e3c72',
            'header_color': '#2a5298', 
            'accent_color': '#3498db',
            'text_color': '#2c3e50',
            'background_color': '#f8f9fa'
        }
    
    def generate_executive_report(self, analysis: Dict, config: Dict) -> bytes:
        """Generate comprehensive executive PDF report"""
        
        # Create matplotlib figure for PDF
        plt.style.use('default')
        
        # Create PDF buffer
        buffer = BytesIO()
        
        with PdfPages(buffer) as pdf:
            # Page 1: Executive Summary
            self._create_executive_summary_page(pdf, analysis, config)
            
            # Page 2: Technical Analysis
            self._create_technical_analysis_page(pdf, analysis, config)
            
            # Page 3: AWS Sizing Recommendations
            self._create_aws_sizing_page(pdf, analysis, config)
            
            # Page 4: Cost Analysis
            self._create_cost_analysis_page(pdf, analysis, config)
            
            # Page 5: Risk Assessment & Timeline
            self._create_risk_timeline_page(pdf, analysis, config)
            
            # Page 6: Agent Scaling Analysis
            self._create_agent_scaling_page(pdf, analysis, config)
            
            # Page 7: FSx Destination Comparison (New)
            self._create_fsx_comparison_page(pdf, analysis, config)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def _create_executive_summary_page(self, pdf, analysis: Dict, config: Dict):
        """Create executive summary page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('AWS Database Migration - Executive Summary', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        # Migration Overview
        ax1.axis('off')
        ax1.text(0.05, 0.95, 'Migration Overview', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax1.transAxes)
        
        overview_text = f"""
        Source: {config['source_database_engine'].upper()} ({config['database_size_gb']:,} GB)
        Target: AWS {config['database_engine'].upper()}
        Type: {'Homogeneous' if config['source_database_engine'] == config['database_engine'] else 'Heterogeneous'}
        Environment: {config['environment'].title()}
        Destination: {config.get('destination_storage_type', 'S3')}
        Migration Time: {analysis.get('estimated_migration_time_hours', 0):.1f} hours
        Downtime Tolerance: {config['downtime_tolerance_minutes']} minutes
        Agents: {config.get('number_of_agents', 1)} {analysis.get('primary_tool', 'DataSync').upper()} agents
        """
        
        ax1.text(0.05, 0.75, overview_text, fontsize=10, transform=ax1.transAxes, verticalalignment='top')
        
        # AI Readiness Score (Gauge chart)
        readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
        self._create_gauge_chart(ax2, readiness_score, 'Migration Readiness', 'AI Assessment')
        
        # Cost Summary
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Cost Summary', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        cost_analysis = analysis.get('cost_analysis', {})
        cost_text = f"""
        Monthly AWS Cost: ${cost_analysis.get('total_monthly_cost', 0):,.0f}
        One-time Migration: ${cost_analysis.get('one_time_migration_cost', 0):,.0f}
        Annual Cost: ${cost_analysis.get('total_monthly_cost', 0) * 12:,.0f}
        Agent Costs: ${cost_analysis.get('agent_cost', 0):,.0f}/month
        Storage Destination: ${cost_analysis.get('destination_storage_cost', 0):,.0f}/month
        Potential Savings: ${cost_analysis.get('estimated_monthly_savings', 0):,.0f}/month
        ROI Timeline: {cost_analysis.get('roi_months', 'TBD')} months
        """
        
        ax3.text(0.05, 0.75, cost_text, fontsize=10, transform=ax3.transAxes, verticalalignment='top')
        
        # Performance Score
        perf_score = analysis.get('onprem_performance', {}).get('performance_score', 0)
        self._create_gauge_chart(ax4, perf_score, 'Performance Score', 'Current System')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_fsx_comparison_page(self, pdf, analysis: Dict, config: Dict):
        """Create FSx destination comparison page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('FSx Destination Storage Comparison Analysis', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        fsx_comparisons = analysis.get('fsx_comparisons', {})
        
        # Performance Comparison
        destinations = list(fsx_comparisons.keys())
        migration_times = [fsx_comparisons[dest].get('estimated_migration_time_hours', 0) for dest in destinations]
        
        bars = ax1.bar(destinations, migration_times, color=['#3498db', '#e74c3c', '#f39c12'])
        ax1.set_title('Migration Time Comparison', fontweight='bold')
        ax1.set_ylabel('Hours')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(migration_times) * 0.01,
                   f'{height:.1f}h', ha='center', va='bottom', fontsize=8)
        
        # Cost Comparison
        storage_costs = [fsx_comparisons[dest].get('estimated_monthly_storage_cost', 0) for dest in destinations]
        
        bars = ax2.bar(destinations, storage_costs, color=['#27ae60', '#e67e22', '#9b59b6'])
        ax2.set_title('Monthly Storage Cost Comparison', fontweight='bold')
        ax2.set_ylabel('Cost ($)')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(storage_costs) * 0.01,
                   f'${height:,.0f}', ha='center', va='bottom', fontsize=8)
        
        # Performance vs Cost Matrix
        performance_ratings = [fsx_comparisons[dest].get('performance_rating', 'Unknown') for dest in destinations]
        cost_ratings = [fsx_comparisons[dest].get('cost_rating', 'Unknown') for dest in destinations]
        
        # Convert ratings to numeric values for plotting
        rating_map = {'Excellent': 5, 'Very Good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1, 'Unknown': 0}
        perf_numeric = [rating_map.get(rating, 0) for rating in performance_ratings]
        cost_numeric = [rating_map.get(rating, 0) for rating in cost_ratings]
        
        colors = ['#3498db', '#e74c3c', '#f39c12']
        for i, dest in enumerate(destinations):
            ax3.scatter(cost_numeric[i], perf_numeric[i], s=200, c=colors[i], alpha=0.7, label=dest)
            ax3.annotate(dest, (cost_numeric[i], perf_numeric[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax3.set_xlabel('Cost Rating')
        ax3.set_ylabel('Performance Rating')
        ax3.set_title('Performance vs Cost Matrix', fontweight='bold')
        ax3.set_xticks(range(1, 6))
        ax3.set_xticklabels(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        ax3.set_yticks(range(1, 6))
        ax3.set_yticklabels(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        ax3.grid(True, alpha=0.3)
        
        # Recommendation Summary
        ax4.axis('off')
        ax4.text(0.05, 0.95, 'Destination Recommendations', fontsize=14, fontweight='bold', 
                color=self.report_style['header_color'], transform=ax4.transAxes)
        
        # Current destination
        current_dest = config.get('destination_storage_type', 'S3')
        current_comparison = fsx_comparisons.get(current_dest, {})
        
        rec_text = f"""
        Current Selection: {current_dest}
        Performance Rating: {current_comparison.get('performance_rating', 'Unknown')}
        Cost Rating: {current_comparison.get('cost_rating', 'Unknown')}
        Complexity: {current_comparison.get('complexity_rating', 'Unknown')}
        Migration Time: {current_comparison.get('estimated_migration_time_hours', 0):.1f} hours
        
        Alternative Options:
        """
        
        # Add alternatives
        for dest in destinations:
            if dest != current_dest:
                comp = fsx_comparisons.get(dest, {})
                rec_text += f"\n{dest}: {comp.get('performance_rating', 'Unknown')} perf, {comp.get('cost_rating', 'Unknown')} cost"
        
        ax4.text(0.05, 0.85, rec_text, fontsize=9, transform=ax4.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_agent_scaling_page(self, pdf, analysis: Dict, config: Dict):
        """Create agent scaling analysis page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Agent Scaling Analysis & Optimization', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        agent_analysis = analysis.get('agent_analysis', {})
        
        # Agent Configuration Overview
        ax1.axis('off')
        ax1.text(0.05, 0.95, 'Agent Configuration', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax1.transAxes)
        
        agent_text = f"""
        Agent Type: {agent_analysis.get('primary_tool', 'Unknown').upper()}
        Agent Size: {agent_analysis.get('agent_size', 'Unknown')}
        Number of Agents: {agent_analysis.get('number_of_agents', 1)}
        Destination: {agent_analysis.get('destination_storage', 'Unknown')}
        Total Throughput: {agent_analysis.get('total_max_throughput_mbps', 0):,.0f} Mbps
        Effective Throughput: {agent_analysis.get('total_effective_throughput', 0):,.0f} Mbps
        Scaling Efficiency: {agent_analysis.get('scaling_efficiency', 1.0)*100:.1f}%
        Storage Multiplier: {agent_analysis.get('storage_performance_multiplier', 1.0):.1f}x
        Bottleneck: {agent_analysis.get('bottleneck', 'Unknown')}
        Monthly Cost: ${agent_analysis.get('monthly_cost', 0):,.0f}
        """
        
        ax1.text(0.05, 0.75, agent_text, fontsize=10, transform=ax1.transAxes, verticalalignment='top')
        
        # Throughput Comparison
        throughput_data = {
            'Max Capacity': agent_analysis.get('total_max_throughput_mbps', 0),
            'Effective Throughput': agent_analysis.get('total_effective_throughput', 0),
            'Network Limit': analysis.get('network_performance', {}).get('effective_bandwidth_mbps', 0)
        }
        
        self._create_bar_chart(ax2, throughput_data, 'Throughput Analysis', 'Mbps')
        
        # Scaling Efficiency
        scaling_efficiency = agent_analysis.get('scaling_efficiency', 1.0) * 100
        self._create_gauge_chart(ax3, scaling_efficiency, 'Scaling Efficiency', f"{agent_analysis.get('number_of_agents', 1)} Agents")
        
        # Cost vs Performance
        optimal_config = agent_analysis.get('optimal_recommendations', {}).get('optimal_configuration')
        if optimal_config:
            cost_perf_data = {
                'Current Config': agent_analysis.get('monthly_cost', 0),
                'Optimal Config': optimal_config.get('total_cost_per_hour', 0) * 24 * 30
            }
            self._create_bar_chart(ax4, cost_perf_data, 'Cost Comparison', 'Monthly Cost ($)')
        else:
            ax4.text(0.5, 0.5, 'Optimal Configuration\nData Not Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # Include all other PDF generation methods from original code...
    def _create_technical_analysis_page(self, pdf, analysis: Dict, config: Dict):
        """Create technical analysis page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Technical Analysis & Performance Assessment', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        # Performance Breakdown
        onprem_perf = analysis.get('onprem_performance', {}).get('overall_performance', {})
        
        performance_metrics = {
            'CPU': onprem_perf.get('cpu_score', 0),
            'Memory': onprem_perf.get('memory_score', 0),
            'Storage': onprem_perf.get('storage_score', 0),
            'Network': onprem_perf.get('network_score', 0),
            'Database': onprem_perf.get('database_score', 0)
        }
        
        # Performance radar chart
        self._create_radar_chart(ax1, performance_metrics, 'Current Performance Profile')
        
        # Network Analysis
        network_perf = analysis.get('network_performance', {})
        network_data = {
            'Quality Score': network_perf.get('network_quality_score', 0),
            'AI Enhanced': network_perf.get('ai_enhanced_quality_score', 0),
            'Bandwidth (%)': min(100, network_perf.get('effective_bandwidth_mbps', 0) / 100),
            'Reliability (%)': network_perf.get('total_reliability', 0) * 100
        }
        
        self._create_bar_chart(ax2, network_data, 'Network Performance Analysis', 'Score/Percentage')
        
        # OS Performance Impact
        os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
        os_data = {
            'CPU Efficiency': os_impact.get('cpu_efficiency', 0) * 100,
            'Memory Efficiency': os_impact.get('memory_efficiency', 0) * 100,
            'I/O Efficiency': os_impact.get('io_efficiency', 0) * 100,
            'Network Efficiency': os_impact.get('network_efficiency', 0) * 100,
            'DB Optimization': os_impact.get('db_optimization', 0) * 100
        }
        
        self._create_bar_chart(ax3, os_data, 'OS Performance Impact', 'Efficiency (%)')
        
        # Bottleneck Analysis
        ax4.axis('off')
        ax4.text(0.05, 0.95, 'Identified Bottlenecks', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax4.transAxes)
        
        bottlenecks = analysis.get('onprem_performance', {}).get('bottlenecks', [])
        ai_insights = analysis.get('onprem_performance', {}).get('ai_insights', [])
        
        bottleneck_text = "Current Bottlenecks:\n"
        for i, bottleneck in enumerate(bottlenecks[:3], 1):
            bottleneck_text += f"{i}. {bottleneck}\n"
        
        bottleneck_text += "\nAI Insights:\n"
        for i, insight in enumerate(ai_insights[:2], 1):
            bottleneck_text += f"{i}. {insight}\n"
        
        ax4.text(0.05, 0.85, bottleneck_text, fontsize=9, transform=ax4.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_aws_sizing_page(self, pdf, analysis: Dict, config: Dict):
        """Create AWS sizing recommendations page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('AWS Sizing & Configuration Recommendations', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        deployment_rec = aws_sizing.get('deployment_recommendation', {})
        
        # Deployment Recommendation
        ax1.axis('off')
        ax1.text(0.05, 0.95, 'Recommended Deployment', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax1.transAxes)
        
        recommendation = deployment_rec.get('recommendation', 'unknown').upper()
        confidence = deployment_rec.get('confidence', 0) * 100
        
        if recommendation == 'RDS':
            rds_rec = aws_sizing.get('rds_recommendations', {})
            deploy_text = f"""
            Deployment: Amazon RDS Managed Service
            Instance: {rds_rec.get('primary_instance', 'N/A')}
            Storage: {rds_rec.get('storage_size_gb', 0):,.0f} GB ({rds_rec.get('storage_type', 'gp3').upper()})
            Multi-AZ: {'Yes' if rds_rec.get('multi_az', False) else 'No'}
            Monthly Cost: ${rds_rec.get('total_monthly_cost', 0):,.0f}
            AI Confidence: {confidence:.1f}%
            """
        else:
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            deploy_text = f"""
            Deployment: Amazon EC2 Self-Managed
            Instance: {ec2_rec.get('primary_instance', 'N/A')}
            Storage: {ec2_rec.get('storage_size_gb', 0):,.0f} GB ({ec2_rec.get('storage_type', 'gp3').upper()})
            EBS Optimized: {'Yes' if ec2_rec.get('ebs_optimized', False) else 'No'}
            Monthly Cost: ${ec2_rec.get('total_monthly_cost', 0):,.0f}
            AI Confidence: {confidence:.1f}%
            """
        
        ax1.text(0.05, 0.75, deploy_text, fontsize=10, transform=ax1.transAxes, verticalalignment='top')
        
        # RDS vs EC2 Comparison
        rds_score = deployment_rec.get('rds_score', 0)
        ec2_score = deployment_rec.get('ec2_score', 0)
        
        comparison_data = {'RDS Score': rds_score, 'EC2 Score': ec2_score}
        self._create_bar_chart(ax2, comparison_data, 'Deployment Scoring', 'Score')
        
        # Reader/Writer Configuration
        reader_writer = aws_sizing.get('reader_writer_config', {})
        
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Instance Configuration', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        instance_text = f"""
        Writer Instances: {reader_writer.get('writers', 1)}
        Reader Instances: {reader_writer.get('readers', 0)}
        Total Instances: {reader_writer.get('total_instances', 1)}
        Read Capacity: {reader_writer.get('read_capacity_percent', 0):.1f}%
        Write Capacity: {reader_writer.get('write_capacity_percent', 100):.1f}%
        Recommended Read Split: {reader_writer.get('recommended_read_split', 0):.0f}%
        
        AI Reasoning:
        {reader_writer.get('reasoning', 'Standard configuration')[:100]}...
        """
        
        ax3.text(0.05, 0.75, instance_text, fontsize=9, transform=ax3.transAxes, verticalalignment='top')
        
        # AI Complexity Factors
        ai_analysis = aws_sizing.get('ai_analysis', {})
        complexity_score = ai_analysis.get('ai_complexity_score', 6)
        
        self._create_gauge_chart(ax4, complexity_score * 10, 'AI Complexity Score', f'{complexity_score}/10')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_cost_analysis_page(self, pdf, analysis: Dict, config: Dict):
        """Create cost analysis page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Cost Analysis & Financial Projections', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        cost_analysis = analysis.get('cost_analysis', {})
        
        # Monthly Cost Breakdown (Pie Chart)
        cost_breakdown = {
            'Compute': cost_analysis.get('aws_compute_cost', 0),
            'Storage': cost_analysis.get('aws_storage_cost', 0),
            'Network': cost_analysis.get('network_cost', 0),
            'Agents': cost_analysis.get('agent_cost', 0),
            'Destination Storage': cost_analysis.get('destination_storage_cost', 0),
            'OS Licensing': cost_analysis.get('os_licensing_cost', 0),
            'Management': cost_analysis.get('management_cost', 0)
        }
        
        # Filter out zero values
        cost_breakdown = {k: v for k, v in cost_breakdown.items() if v > 0}
        
        self._create_pie_chart(ax1, cost_breakdown, 'Monthly Cost Breakdown')
        
        # Cost Projections
        monthly_cost = cost_analysis.get('total_monthly_cost', 0)
        one_time_cost = cost_analysis.get('one_time_migration_cost', 0)
        
        projections = {
            '1 Year': monthly_cost * 12 + one_time_cost,
            '2 Years': monthly_cost * 24 + one_time_cost,
            '3 Years': monthly_cost * 36 + one_time_cost,
            '5 Years': monthly_cost * 60 + one_time_cost
        }
        
        self._create_bar_chart(ax2, projections, 'Total Cost Projections', 'Cost ($)')
        
        # Savings Analysis
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Savings & ROI Analysis', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        savings_text = f"""
        Monthly Savings: ${cost_analysis.get('estimated_monthly_savings', 0):,.0f}
        Annual Savings: ${cost_analysis.get('estimated_monthly_savings', 0) * 12:,.0f}
        ROI Timeline: {cost_analysis.get('roi_months', 'TBD')} months
        Break-even Point: Year {cost_analysis.get('roi_months', 12) / 12:.1f}
        
        Destination Storage: {cost_analysis.get('destination_storage_type', 'S3')}
        Storage Cost: ${cost_analysis.get('destination_storage_cost', 0):,.0f}/month
        
        AI Cost Insights:
        - Optimization Factor: {cost_analysis.get('ai_cost_insights', {}).get('ai_optimization_factor', 0)*100:.1f}%
        - Complexity Multiplier: {cost_analysis.get('ai_cost_insights', {}).get('complexity_multiplier', 1.0):.2f}x
        - Additional Savings: {cost_analysis.get('ai_cost_insights', {}).get('potential_additional_savings', '0%')}
        """
        
        ax3.text(0.05, 0.75, savings_text, fontsize=9, transform=ax3.transAxes, verticalalignment='top')
        
        # Cost Comparison
        current_cost_estimate = monthly_cost * 0.8  # Assume current is 20% higher
        cost_comparison = {
            'Current (Est.)': current_cost_estimate,
            'AWS Monthly': monthly_cost
        }
        
        self._create_bar_chart(ax4, cost_comparison, 'Cost Comparison', 'Monthly Cost ($)')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_risk_timeline_page(self, pdf, analysis: Dict, config: Dict):
        """Create risk assessment and timeline page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Risk Assessment & Project Timeline', fontsize=16, fontweight='bold', color=self.report_style['title_color'])
        
        ai_assessment = analysis.get('ai_overall_assessment', {})
        
        # Risk Assessment
        readiness_score = ai_assessment.get('migration_readiness_score', 0)
        success_prob = ai_assessment.get('success_probability', 0)
        risk_level = ai_assessment.get('risk_level', 'Medium')
        
        risk_data = {
            'Success Probability': success_prob,
            'Risk Mitigation': 100 - (100 - readiness_score) * 0.8
        }
        
        self._create_bar_chart(ax1, risk_data, 'Risk Assessment', 'Percentage (%)')
        
        # Timeline Visualization
        timeline = ai_assessment.get('timeline_recommendation', {})
        
        timeline_data = {
            'Planning': timeline.get('planning_phase_weeks', 2),
            'Testing': timeline.get('testing_phase_weeks', 3),
            'Migration': timeline.get('migration_window_hours', 24) / (7 * 24),  # Convert to weeks
            'Validation': 1
        }
        
        self._create_bar_chart(ax2, timeline_data, 'Project Timeline', 'Duration (Weeks)')
        
        # Risk Factors
        ax3.axis('off')
        ax3.text(0.05, 0.95, 'Key Risk Factors', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax3.transAxes)
        
        risk_factors = ai_assessment.get('readiness_factors', [])
        aws_sizing = analysis.get('aws_sizing_recommendations', {})
        ai_analysis = aws_sizing.get('ai_analysis', {})
        risk_percentages = ai_analysis.get('risk_percentages', {})
        
        risk_text = "Identified Risk Factors:\n"
        for i, factor in enumerate(risk_factors[:4], 1):
            risk_text += f"{i}. {factor}\n"
        
        if risk_percentages:
            risk_text += "\nQuantified Risks:\n"
            for risk, percentage in list(risk_percentages.items())[:3]:
                risk_text += f"• {risk.replace('_', ' ').title()}: {percentage}%\n"
        
        ax3.text(0.05, 0.85, risk_text, fontsize=9, transform=ax3.transAxes, verticalalignment='top')
        
        # Next Steps
        ax4.axis('off')
        ax4.text(0.05, 0.95, 'Recommended Next Steps', fontsize=14, fontweight='bold', color=self.report_style['header_color'], transform=ax4.transAxes)
        
        next_steps = ai_assessment.get('recommended_next_steps', [])
        
        steps_text = ""
        for i, step in enumerate(next_steps, 1):
            steps_text += f"{i}. {step}\n"
        
        ax4.text(0.05, 0.85, steps_text, fontsize=9, transform=ax4.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_gauge_chart(self, ax, value, title, subtitle):
        """Create a gauge chart"""
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background arc
        ax.plot(theta, r, color='lightgray', linewidth=8)
        
        # Value arc
        value_theta = np.linspace(0, (value/100) * np.pi, int(value))
        value_r = np.ones_like(value_theta)
        
        color = '#e74c3c' if value < 50 else '#f39c12' if value < 75 else '#27ae60'
        ax.plot(value_theta, value_r, color=color, linewidth=8)
        
        # Add value text
        ax.text(0, -0.3, f'{value:.1f}', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(0, -0.5, title, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0, -0.65, subtitle, ha='center', va='center', fontsize=10)
        
        ax.set_ylim(-0.8, 1.2)
        ax.set_xlim(-1.2, 1.2)
        ax.axis('off')
    
    def _create_radar_chart(self, ax, data, title):
        """Create a radar chart"""
        
        labels = list(data.keys())
        values = list(data.values())
        
        # Number of variables
        num_vars = len(labels)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle
        
        # Add values
        values += values[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color=self.report_style['accent_color'])
        ax.fill(angles, values, alpha=0.25, color=self.report_style['accent_color'])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(True)
    
    def _create_bar_chart(self, ax, data, title, ylabel):
        """Create a bar chart"""
        
        labels = list(data.keys())
        values = list(data.values())
        
        bars = ax.bar(labels, values, color=self.report_style['accent_color'], alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        
        # Format y-axis for currency if needed
        if '$' in ylabel or 'Cost' in ylabel:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    def _create_pie_chart(self, ax, data, title):
        """Create a pie chart"""
        
        labels = list(data.keys())
        values = list(data.values())
        
        # Filter out zero values
        filtered_data = [(label, value) for label, value in zip(labels, values) if value > 0]
        if filtered_data:
            labels, values = zip(*filtered_data)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        
        ax.set_title(title, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

# Network path diagram function
def create_network_path_diagram(network_perf: Dict) -> go.Figure:
    """Create an interactive network path diagram"""
    
    segments = network_perf.get('segments', [])
    if not segments:
        return go.Figure()
    
    # Create network diagram using plotly
    fig = go.Figure()
    
    # Define positions for network nodes
    num_segments = len(segments)
    x_positions = [i * 100 for i in range(num_segments + 1)]
    y_positions = [50] * (num_segments + 1)
    
    # Add network segments as lines
    for i, segment in enumerate(segments):
        # Calculate line properties based on performance
        line_width = max(2, min(10, segment['effective_bandwidth_mbps'] / 200))
        
        # Color based on performance (green = good, yellow = ok, red = poor)
        reliability = segment['reliability']
        if reliability > 0.999:
            line_color = '#27ae60'  # Green
        elif reliability > 0.995:
            line_color = '#f39c12'  # Orange
        else:
            line_color = '#e74c3c'  # Red
        
        # Add line for network segment
        fig.add_trace(go.Scatter(
            x=[x_positions[i], x_positions[i+1]],
            y=[y_positions[i], y_positions[i+1]],
            mode='lines+markers',
            line=dict(
                width=line_width,
                color=line_color
            ),
            marker=dict(size=15, color='#2c3e50', symbol='square'),
            name=segment['name'],
            hovertemplate=f"""
            <b>{segment['name']}</b><br>
            Type: {segment['connection_type'].replace('_', ' ').title()}<br>
            Bandwidth: {segment['effective_bandwidth_mbps']:.0f} Mbps<br>
            Latency: {segment['effective_latency_ms']:.1f} ms<br>
            Reliability: {segment['reliability']*100:.3f}%<br>
            Cost Factor: {segment['cost_factor']:.1f}x<br>
            AI Optimization: {segment.get('ai_optimization_potential', 0)*100:.1f}%<br>
            <extra></extra>
            """
        ))
        
        # Add bandwidth and latency annotations
        mid_x = (x_positions[i] + x_positions[i+1]) / 2
        mid_y = (y_positions[i] + y_positions[i+1]) / 2 + 20
        
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            text=f"<b>{segment['effective_bandwidth_mbps']:.0f} Mbps</b><br>{segment['effective_latency_ms']:.1f} ms",
            showarrow=False,
            font=dict(size=10, color='#2c3e50'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#bdc3c7',
            borderwidth=1,
            borderpad=4
        )
        
        # Add connection type label
        fig.add_annotation(
            x=mid_x,
            y=mid_y - 25,
            text=f"<i>{segment['connection_type'].replace('_', ' ').title()}</i>",
            showarrow=False,
            font=dict(size=8, color='#7f8c8d'),
            bgcolor='rgba(248,249,250,0.8)',
            bordercolor='#dee2e6',
            borderwidth=1,
            borderpad=2
        )
    
    # Add source and destination nodes
    fig.add_trace(go.Scatter(
        x=[x_positions[0]],
        y=[y_positions[0]],
        mode='markers+text',
        marker=dict(size=25, color='#27ae60', symbol='circle'),
        text=['SOURCE'],
        textposition='bottom center',
        name='Source System',
        hovertemplate="<b>Source System</b><br>On-Premises Database<extra></extra>"
    ))
    
    destination_storage = network_perf.get('destination_storage', 'S3')
    destination_color = {
        'S3': '#3498db',
        'FSx_Windows': '#e74c3c', 
        'FSx_Lustre': '#9b59b6'
    }.get(destination_storage, '#3498db')
    
    fig.add_trace(go.Scatter(
        x=[x_positions[-1]],
        y=[y_positions[-1]],
        mode='markers+text',
        marker=dict(size=25, color=destination_color, symbol='circle'),
        text=[destination_storage],
        textposition='bottom center',
        name=f'AWS {destination_storage}',
        hovertemplate=f"<b>AWS Destination</b><br>{destination_storage} Storage Service<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Network Migration Path: {network_perf.get('path_name', 'Unknown')} → {destination_storage}",
            font=dict(size=16, color='#2c3e50'),
            x=0.5
        ),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[-20, max(x_positions) + 20]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False, 
            range=[0, 100]
        ),
        showlegend=False,
        height=350,
        plot_bgcolor='rgba(248,249,250,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# Enhanced rendering functions

def render_enhanced_header():
    """Enhanced header with professional styling"""
    
    # Initialize managers to check status
    ai_manager = AnthropicAIManager()
    aws_api = AWSAPIManager()
    
    ai_status = "🟢" if ai_manager.connected else "🔴"
    aws_status = "🟢" if aws_api.connected else "🔴"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AWS Enterprise Database Migration Analyzer AI v3.0</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Professional-Grade Migration Analysis • AI-Powered Insights • Real-time AWS Integration • Agent Scaling Optimization • FSx Destination Analysis
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
            Comprehensive Network Path Analysis • OS Performance Optimization • Enterprise-Ready Migration Planning • Multi-Agent Coordination • S3/FSx Comparisons
        </p>
        <div style="margin-top: 1rem; font-size: 0.8rem;">
            <span style="margin-right: 20px;">{ai_status} Anthropic Claude AI</span>
            <span style="margin-right: 20px;">{aws_status} AWS Pricing APIs</span>
            <span style="margin-right: 20px;">🟢 Network Intelligence Engine</span>
            <span style="margin-right: 20px;">🟢 Agent Scaling Optimizer</span>
            <span>🟢 FSx Destination Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_api_status_sidebar():
    """Enhanced API status sidebar"""
    
    st.sidebar.markdown("### 🔌 System Status")
    
    # Check API status
    ai_manager = AnthropicAIManager()
    aws_api = AWSAPIManager()
    
    # Anthropic AI Status
    ai_status_class = "status-online" if ai_manager.connected else "status-offline"
    ai_status_text = "Connected" if ai_manager.connected else "Disconnected"
    
    st.sidebar.markdown(f"""
    <div class="api-status-card">
        <span class="status-indicator {ai_status_class}"></span>
        <strong>Anthropic Claude AI:</strong> {ai_status_text}
        {f"<br><small>Error: {ai_manager.error_message[:50]}...</small>" if ai_manager.error_message else ""}
    </div>
    """, unsafe_allow_html=True)
    
    # AWS API Status
    aws_status_class = "status-online" if aws_api.connected else "status-warning"
    aws_status_text = "Connected" if aws_api.connected else "Using Fallback Data"
    
    st.sidebar.markdown(f"""
    <div class="api-status-card">
        <span class="status-indicator {aws_status_class}"></span>
        <strong>AWS Pricing API:</strong> {aws_status_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration instructions
    if not ai_manager.connected or not aws_api.connected:
        st.sidebar.markdown("### ⚙️ Configuration")
        
        if not ai_manager.connected:
            st.sidebar.info("💡 Add ANTHROPIC_API_KEY to Streamlit secrets for enhanced AI analysis")
        
        if not aws_api.connected:
            st.sidebar.info("💡 Configure AWS credentials for real-time pricing data")

def render_enhanced_sidebar_controls():
    """Enhanced sidebar with AI-powered recommendations, agent scaling, and FSx destination selection"""
    
    st.sidebar.header("🤖 AI-Powered Migration Configuration v3.0 with FSx Analysis")
    
    # Render API status
    render_api_status_sidebar()
    
    st.sidebar.markdown("---")
    
    # Operating System Selection with AI insights
    st.sidebar.subheader("💻 Operating System (AI-Enhanced)")
    operating_system = st.sidebar.selectbox(
        "OS Selection",
        ["windows_server_2019", "windows_server_2022", "rhel_8", "rhel_9", "ubuntu_20_04", "ubuntu_22_04"],
        index=3,
        format_func=lambda x: {
            'windows_server_2019': '🔵 Windows Server 2019',
            'windows_server_2022': '🔵 Windows Server 2022 (Latest)',
            'rhel_8': '🔴 Red Hat Enterprise Linux 8',
            'rhel_9': '🔴 Red Hat Enterprise Linux 9 (Latest)',
            'ubuntu_20_04': '🟠 Ubuntu Server 20.04 LTS',
            'ubuntu_22_04': '🟠 Ubuntu Server 22.04 LTS (Latest)'
        }[x],
        help="AI analyzes OS performance characteristics and migration impact"
    )
    
    # Show AI OS insights
    os_manager = OSPerformanceManager()
    os_config = os_manager.operating_systems[operating_system]
    
    with st.sidebar.expander("🤖 AI OS Insights"):
        st.markdown(f"**Strengths:** {', '.join(os_config['ai_insights']['strengths'][:2])}")
        st.markdown(f"**Key Consideration:** {os_config['ai_insights']['weaknesses'][0]}")
    
    # Platform Configuration
    st.sidebar.subheader("🖥️ Server Platform")
    server_type = st.sidebar.selectbox(
        "Platform Type",
        ["physical", "vmware"],
        format_func=lambda x: "🏢 Physical Server" if x == "physical" else "☁️ VMware Virtual Machine",
        help="Physical vs Virtual performance analysis with AI optimization"
    )
    
    # Hardware Configuration with AI recommendations
    st.sidebar.subheader("⚙️ Hardware Configuration")
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256, 512], index=2, 
                                 help="AI calculates optimal memory for database workload")
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24, 32, 48, 64], index=2,
                                   help="AI analyzes CPU requirements for migration performance")
    cpu_ghz = st.sidebar.selectbox("CPU GHz", [2.0, 2.4, 2.8, 3.2, 3.6, 4.0], index=3)
    
    # Enhanced Performance Metrics
    st.sidebar.subheader("📊 Current Performance Metrics")
    current_storage_gb = st.sidebar.number_input("Current Storage (GB)", 
                                                min_value=100, max_value=500000, value=2000, step=100,
                                                help="Current storage capacity in use")
    peak_iops = st.sidebar.number_input("Peak IOPS", 
                                       min_value=100, max_value=1000000, value=10000, step=500,
                                       help="Maximum IOPS observed during peak usage")
    max_throughput_mbps = st.sidebar.number_input("Max Throughput (MB/s)", 
                                                 min_value=10, max_value=10000, value=500, step=50,
                                                 help="Maximum storage throughput observed")
    anticipated_max_memory_gb = st.sidebar.number_input("Anticipated Max Memory (GB)", 
                                                       min_value=4, max_value=1024, value=64, step=8,
                                                       help="Maximum memory usage anticipated for workload")
    anticipated_max_cpu_cores = st.sidebar.number_input("Anticipated Max CPU Cores", 
                                                       min_value=1, max_value=128, value=16, step=2,
                                                       help="Maximum CPU cores anticipated for workload")
    
    # Network Interface with AI insights
    nic_type = st.sidebar.selectbox(
        "NIC Type",
        ["gigabit_copper", "gigabit_fiber", "10g_copper", "10g_fiber", "25g_fiber", "40g_fiber"],
        index=3,
        format_func=lambda x: {
            'gigabit_copper': '🔶 1Gbps Copper',
            'gigabit_fiber': '🟡 1Gbps Fiber',
            '10g_copper': '🔵 10Gbps Copper',
            '10g_fiber': '🟢 10Gbps Fiber',
            '25g_fiber': '🟣 25Gbps Fiber',
            '40g_fiber': '🔴 40Gbps Fiber'
        }[x],
        help="AI analyzes network impact on migration throughput"
    )
    
    nic_speeds = {
        'gigabit_copper': 1000, 'gigabit_fiber': 1000,
        '10g_copper': 10000, '10g_fiber': 10000, 
        '25g_fiber': 25000, '40g_fiber': 40000
    }
    nic_speed = nic_speeds[nic_type]
    
    # Migration Configuration with AI analysis
    st.sidebar.subheader("🔄 Migration Setup (AI-Optimized)")
    
    source_database_engine = st.sidebar.selectbox(
        "Source Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': '🐬 MySQL', 'postgresql': '🐘 PostgreSQL', 'oracle': '🏛️ Oracle',
            'sqlserver': '🪟 SQL Server', 'mongodb': '🍃 MongoDB'
        }[x],
        help="AI determines migration complexity based on source engine"
    )
    
    # FIXED: Target Database Selection with SQL Server Logic
    if source_database_engine == "sqlserver":
        # Force SQL Server to EC2 due to licensing and compatibility
        st.sidebar.warning("🚨 **SQL Server Auto-Configuration**")
        st.sidebar.info("""
        **SQL Server → EC2 (Self-Managed)**
        
        ✅ **Why EC2 for SQL Server:**
        • Better licensing control (BYOL)
        • Advanced SQL Server features support
        • Custom configuration capabilities
        • Better performance optimization options
        """)
        
        database_engine = st.sidebar.selectbox(
            "Target SQL Server on EC2",
            [
                "ec2_r6i_large", "ec2_r6i_xlarge", "ec2_r6i_2xlarge", "ec2_r6i_4xlarge",
                "ec2_m5_large", "ec2_m5_xlarge", "ec2_m5_2xlarge", "ec2_m5_4xlarge",
                "ec2_c5_xlarge", "ec2_c5_2xlarge", "ec2_c5_4xlarge"
            ],
            index=1,  # Default to r6i.xlarge
            format_func=lambda x: {
                'ec2_r6i_large': '🧠 EC2 r6i.large (2 vCPU, 16GB) - Memory Optimized',
                'ec2_r6i_xlarge': '🧠 EC2 r6i.xlarge (4 vCPU, 32GB) - Memory Optimized ⭐',
                'ec2_r6i_2xlarge': '🧠 EC2 r6i.2xlarge (8 vCPU, 64GB) - Memory Optimized',
                'ec2_r6i_4xlarge': '🧠 EC2 r6i.4xlarge (16 vCPU, 128GB) - Memory Optimized',
                'ec2_m5_large': '🔵 EC2 m5.large (2 vCPU, 8GB) - General Purpose',
                'ec2_m5_xlarge': '🔵 EC2 m5.xlarge (4 vCPU, 16GB) - General Purpose',
                'ec2_m5_2xlarge': '🔵 EC2 m5.2xlarge (8 vCPU, 32GB) - General Purpose',
                'ec2_m5_4xlarge': '🔵 EC2 m5.4xlarge (16 vCPU, 64GB) - General Purpose',
                'ec2_c5_xlarge': '⚡ EC2 c5.xlarge (4 vCPU, 8GB) - Compute Optimized',
                'ec2_c5_2xlarge': '⚡ EC2 c5.2xlarge (8 vCPU, 16GB) - Compute Optimized',
                'ec2_c5_4xlarge': '⚡ EC2 c5.4xlarge (16 vCPU, 32GB) - Compute Optimized'
            }[x],
            help="SQL Server on EC2 for optimal licensing and performance control"
        )
        
        # Set the database engine for migration logic
        ec2_database_engine = "sqlserver"
        deployment_type = "ec2_mandatory"
        
    else:
        # Regular database selection for non-SQL Server
        database_engine = st.sidebar.selectbox(
            "Target Database (AWS)",
            [
                # RDS Managed Services
                "rds_mysql", "rds_postgresql", "rds_oracle", "rds_mongodb",
                
                # EC2 Instance Types for Self-Managed Databases
                "ec2_t3_medium", "ec2_t3_large", "ec2_t3_xlarge", "ec2_t3_2xlarge",
                "ec2_c5_large", "ec2_c5_xlarge", "ec2_c5_2xlarge", "ec2_c5_4xlarge",
                "ec2_r6i_large", "ec2_r6i_xlarge", "ec2_r6i_2xlarge", "ec2_r6i_4xlarge", 
                "ec2_m5_large", "ec2_m5_xlarge", "ec2_m5_2xlarge", "ec2_m5_4xlarge"
            ],
            index=0,  # Default to RDS MySQL
            format_func=lambda x: {
                # RDS Managed Services (NO SQL Server in RDS options)
                'rds_mysql': '☁️ RDS MySQL (Managed)',
                'rds_postgresql': '☁️ RDS PostgreSQL (Managed)', 
                'rds_oracle': '☁️ RDS Oracle (Managed)',
                'rds_mongodb': '☁️ DocumentDB (Managed)',
                
                # EC2 options
                'ec2_t3_medium': '🖥️ EC2 t3.medium (2 vCPU, 4GB) - Burstable',
                'ec2_t3_large': '🖥️ EC2 t3.large (2 vCPU, 8GB) - Burstable',
                'ec2_t3_xlarge': '🖥️ EC2 t3.xlarge (4 vCPU, 16GB) - Burstable',
                'ec2_t3_2xlarge': '🖥️ EC2 t3.2xlarge (8 vCPU, 32GB) - Burstable',
                'ec2_c5_large': '⚡ EC2 c5.large (2 vCPU, 4GB) - Compute Optimized',
                'ec2_c5_xlarge': '⚡ EC2 c5.xlarge (4 vCPU, 8GB) - Compute Optimized',
                'ec2_c5_2xlarge': '⚡ EC2 c5.2xlarge (8 vCPU, 16GB) - Compute Optimized',
                'ec2_c5_4xlarge': '⚡ EC2 c5.4xlarge (16 vCPU, 32GB) - Compute Optimized',
                'ec2_r6i_large': '🧠 EC2 r6i.large (2 vCPU, 16GB) - Memory Optimized',
                'ec2_r6i_xlarge': '🧠 EC2 r6i.xlarge (4 vCPU, 32GB) - Memory Optimized',
                'ec2_r6i_2xlarge': '🧠 EC2 r6i.2xlarge (8 vCPU, 64GB) - Memory Optimized',
                'ec2_r6i_4xlarge': '🧠 EC2 r6i.4xlarge (16 vCPU, 128GB) - Memory Optimized',
                'ec2_m5_large': '🔵 EC2 m5.large (2 vCPU, 8GB) - General Purpose',
                'ec2_m5_xlarge': '🔵 EC2 m5.xlarge (4 vCPU, 16GB) - General Purpose',
                'ec2_m5_2xlarge': '🔵 EC2 m5.2xlarge (8 vCPU, 32GB) - General Purpose',
                'ec2_m5_4xlarge': '🔵 EC2 m5.4xlarge (16 vCPU, 64GB) - General Purpose'
            }[x],
            help="Select target AWS service: RDS (managed) or EC2 instance type (self-managed)"
        )
        
        ec2_database_engine = source_database_engine  # Use source engine for EC2 deployments
        deployment_type = "flexible"
    
    # Show deployment and migration type indicators
    if source_database_engine == "sqlserver":
        st.sidebar.success("🎯 **SQL Server → EC2 (Optimized Path)**")
        st.sidebar.info("✅ Licensing control\n✅ Advanced features\n✅ Custom optimization")
    else:
        # Regular deployment type indicator
        if database_engine.startswith('rds_'):
            deployment_type_display = "🟢 Managed Service (RDS/DocumentDB)"
            management_complexity = "Low"
        else:
            deployment_type_display = "🟡 Self-Managed (EC2)"
            management_complexity = "High"
        
        st.sidebar.info(f"**Deployment:** {deployment_type_display}")
        st.sidebar.info(f"**Management:** {management_complexity}")
            
    # Show migration type with enhanced logic (FIXED - using variables instead of config)
    target_db_type = database_engine.split('_')[1] if '_' in database_engine else database_engine
    
    # Check if source and target database engines match
    is_homogeneous = source_database_engine == target_db_type or (
        source_database_engine == 'mongodb' and 'mongodb' in database_engine
    )
    
    migration_type_indicator = "🟢 Homogeneous" if is_homogeneous else "🟡 Heterogeneous"
    st.sidebar.info(f"**Migration Type:** {migration_type_indicator}")
    
    # Enhanced migration complexity assessment
    complexity_factors = []
    base_complexity = 1
    
    # Database engine complexity
    if not is_homogeneous:
        complexity_factors.append("Different database engines")
        base_complexity += 2
    
    # Deployment complexity
    if database_engine.startswith('ec2_'):
        complexity_factors.append("Self-managed deployment")
        base_complexity += 1
    
    # Instance type complexity
    if 'x1e' in database_engine:
        complexity_factors.append("High-memory instance configuration")
        base_complexity += 0.5
    elif 'i3' in database_engine:
        complexity_factors.append("Storage-optimized instance setup")
        base_complexity += 0.5
    
    migration_complexity = min(10, base_complexity)
    
    with st.sidebar.expander("📊 Migration Complexity Analysis"):
        st.write(f"**Complexity Score:** {migration_complexity:.1f}/10")
        st.write(f"**Factors:**")
        for factor in complexity_factors:
            st.write(f"• {factor}")
        if not complexity_factors:
            st.write("• Standard migration complexity")
    
    # Enhanced instance recommendations based on selection
    if database_engine.startswith('ec2_'):
        with st.sidebar.expander("💡 EC2 Instance Insights"):
            instance_type = database_engine.replace('ec2_', '')
            
            instance_recommendations = {
                't3_medium': {
                    'best_for': 'Development/testing environments',
                    'workload': 'Light to moderate workloads',
                    'cost_efficiency': 'Excellent',
                    'performance': 'Burstable - good for variable workloads'
                },
                't3_large': {
                    'best_for': 'Small production databases',
                    'workload': 'Moderate workloads with burst capability',
                    'cost_efficiency': 'Very Good',
                    'performance': 'Burstable - handles traffic spikes well'
                },
                'c5_large': {
                    'best_for': 'CPU-intensive database operations',
                    'workload': 'High-frequency transactions',
                    'cost_efficiency': 'Good',
                    'performance': 'Consistent high CPU performance'
                },
                'r6i_large': {
                    'best_for': 'Memory-intensive databases',
                    'workload': 'Large datasets, caching',
                    'cost_efficiency': 'Good',
                    'performance': 'Excellent for memory-bound workloads'
                },
                'x1e_xlarge': {
                    'best_for': 'Very large in-memory databases',
                    'workload': 'Analytics, big data processing',
                    'cost_efficiency': 'Specialized use case',
                    'performance': 'Exceptional memory capacity'
                },
                'i3_large': {
                    'best_for': 'High I/O database applications',
                    'workload': 'Database workloads requiring high IOPS',
                    'cost_efficiency': 'Good for storage-intensive apps',
                    'performance': 'Very high storage performance'
                }
            }
            
            # Find matching recommendation
            rec = None
            for key, value in instance_recommendations.items():
                if key in instance_type:
                    rec = value
                    break
            
            if rec:
                st.write(f"**Best For:** {rec['best_for']}")
                st.write(f"**Workload:** {rec['workload']}")
                st.write(f"**Cost Efficiency:** {rec['cost_efficiency']}")
                st.write(f"**Performance:** {rec['performance']}")
            else:
                st.write("**General Purpose:** Suitable for most database workloads")
                st.write("**Performance:** Depends on specific instance specifications")
    
    # Continue with the rest of the original function...
    database_size_gb = st.sidebar.number_input("Database Size (GB)", 
                                              min_value=100, max_value=100000, value=1000, step=100,
                                              help="AI calculates migration time and resource requirements")
    
    # Migration Parameters
    downtime_tolerance_minutes = st.sidebar.number_input("Max Downtime (minutes)", 
                                                        min_value=1, max_value=480, value=60,
                                                        help="AI optimizes migration strategy for downtime window")
    performance_requirements = st.sidebar.selectbox("Performance Requirement", ["standard", "high"],
                                                   help="AI adjusts AWS sizing recommendations")
    
    # NEW: Destination Storage Selection
    st.sidebar.subheader("🗄️ Destination Storage (Enhanced Analysis)")
    destination_storage_type = st.sidebar.selectbox(
        "AWS Destination Storage",
        ["S3", "FSx_Windows", "FSx_Lustre"],
        format_func=lambda x: {
            'S3': '☁️ Amazon S3 (Standard)',
            'FSx_Windows': '🪟 Amazon FSx for Windows File Server',
            'FSx_Lustre': '⚡ Amazon FSx for Lustre (High Performance)'
        }[x],
        help="AI analyzes performance, cost, and complexity for each destination type"
    )
    
    # Show destination storage insights
    with st.sidebar.expander("🎯 Storage Destination Insights"):
        if destination_storage_type == "S3":
            st.markdown("""
            **S3 Advantages:**
            • Cost-effective and scalable
            • Simple migration integration
            • Excellent durability (99.999999999%)
            
            **Best For:** General-purpose storage, cost optimization
            """)
        elif destination_storage_type == "FSx_Windows":
            st.markdown("""
            **FSx Windows Advantages:**
            • Native Windows file system features
            • Better performance than S3
            • Active Directory integration
            
            **Best For:** Windows-based applications, file shares
            """)
        elif destination_storage_type == "FSx_Lustre":
            st.markdown("""
            **FSx Lustre Advantages:**
            • Extremely high performance (sub-ms latency)
            • Parallel processing optimized
            • Perfect for HPC and analytics
            
            **Best For:** High-performance computing, ML workloads
            """)
    
    # Environment
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"],
                                     help="AI adjusts reliability and performance requirements")
    
    # Enhanced Agent Sizing Section with AI recommendations
    st.sidebar.subheader("🤖 Migration Agent Configuration (AI-Optimized)")
    
    # Determine migration type for tool selection
    primary_tool = "DataSync" if is_homogeneous else "DMS"
    
    st.sidebar.success(f"**Primary Tool:** AWS {primary_tool}")
    
    # Agent Count Configuration
    st.sidebar.markdown("**📊 Agent Scaling Configuration:**")
    
    number_of_agents = st.sidebar.number_input(
        "Number of Migration Agents",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Configure the number of agents for parallel migration processing"
    )
    
    # Show agent scaling insights
    if number_of_agents == 1:
        st.sidebar.info("💡 Single agent - simple but may limit throughput")
    elif number_of_agents <= 3:
        st.sidebar.success("✅ Optimal range for most workloads")
    elif number_of_agents <= 5:
        st.sidebar.warning("⚠️ High agent count - ensure proper coordination")
    else:
        st.sidebar.error("🔴 Very high agent count - diminishing returns likely")
    
    if is_homogeneous:
        datasync_agent_size = st.sidebar.selectbox(
            "DataSync Agent Size",
            ["small", "medium", "large", "xlarge"],
            index=1,
            format_func=lambda x: {
                'small': '📦 Small (t3.medium) - 250 Mbps/agent',
                'medium': '📦 Medium (c5.large) - 500 Mbps/agent',
                'large': '📦 Large (c5.xlarge) - 1000 Mbps/agent',
                'xlarge': '📦 XLarge (c5.2xlarge) - 2000 Mbps/agent'
            }[x],
            help=f"AI recommends optimal agent size for {number_of_agents} agents"
        )
        dms_agent_size = None
    else:
        dms_agent_size = st.sidebar.selectbox(
            "DMS Instance Size",
            ["small", "medium", "large", "xlarge", "xxlarge"],
            index=1,
            format_func=lambda x: {
                'small': '🔄 Small (t3.medium) - 200 Mbps/agent',
                'medium': '🔄 Medium (c5.large) - 400 Mbps/agent',
                'large': '🔄 Large (c5.xlarge) - 800 Mbps/agent',
                'xlarge': '🔄 XLarge (c5.2xlarge) - 1500 Mbps/agent',
                'xxlarge': '🔄 XXLarge (c5.4xlarge) - 2500 Mbps/agent'
            }[x],
            help=f"AI recommends optimal instance size for {number_of_agents} agents"
        )
        datasync_agent_size = None
    
    # Show estimated throughput with current configuration including destination storage impact
    agent_manager = EnhancedAgentSizingManager()
    if is_homogeneous:
        test_config = agent_manager.calculate_agent_configuration('datasync', datasync_agent_size, number_of_agents, destination_storage_type)
    else:
        test_config = agent_manager.calculate_agent_configuration('dms', dms_agent_size, number_of_agents, destination_storage_type)
    
    st.sidebar.markdown(f"""
    <div class="agent-scaling-card">
        <h4>🚀 Current Configuration Impact</h4>
        <p><strong>Total Throughput:</strong> {test_config['total_max_throughput_mbps']:,.0f} Mbps</p>
        <p><strong>Scaling Efficiency:</strong> {test_config['scaling_efficiency']*100:.1f}%</p>
        <p><strong>Storage Multiplier:</strong> {test_config['storage_performance_multiplier']:.1f}x</p>
        <p><strong>Monthly Cost:</strong> ${test_config['total_monthly_cost']:,.0f}</p>
        <p><strong>Config Rating:</strong> {test_config['optimal_configuration']['efficiency_score']:.0f}/100</p>
        <p><strong>Destination:</strong> {destination_storage_type}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Configuration Section
    st.sidebar.subheader("🧠 AI Configuration")
    
    enable_ai_analysis = st.sidebar.checkbox("Enable AI Analysis", value=True,
                                           help="Use Anthropic AI for intelligent recommendations")
    
    if enable_ai_analysis:
        ai_analysis_depth = st.sidebar.selectbox(
            "AI Analysis Depth",
            ["standard", "comprehensive"],
            help="Comprehensive analysis provides more detailed AI insights"
        )
    else:
        ai_analysis_depth = "standard"
    
    # Real-time AWS Pricing
    use_realtime_pricing = st.sidebar.checkbox("Real-time AWS Pricing", value=True,
                                             help="Fetch current AWS pricing via API")
    
    if st.sidebar.button("🔄 Refresh AI Analysis", type="primary"):
        st.rerun()
    
    return {
        'operating_system': operating_system,
        'server_type': server_type,
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'cpu_ghz': cpu_ghz,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'source_database_engine': source_database_engine,
        'database_engine': database_engine,
        'ec2_database_engine': ec2_database_engine,  # NEW: Separate EC2 engine tracking
        'database_size_gb': database_size_gb,
        'downtime_tolerance_minutes': downtime_tolerance_minutes,
        'performance_requirements': performance_requirements,
        'destination_storage_type': destination_storage_type,
        'environment': environment,
        'datasync_agent_size': datasync_agent_size,
        'dms_agent_size': dms_agent_size,
        'number_of_agents': number_of_agents,
        'enable_ai_analysis': enable_ai_analysis,
        'ai_analysis_depth': ai_analysis_depth,
        'use_realtime_pricing': use_realtime_pricing,
        'current_storage_gb': current_storage_gb,
        'peak_iops': peak_iops,
        'max_throughput_mbps': max_throughput_mbps,
        'anticipated_max_memory_gb': anticipated_max_memory_gb,
        'anticipated_max_cpu_cores': anticipated_max_cpu_cores,
        'deployment_type': deployment_type,
        'is_sql_server': source_database_engine == "sqlserver"  # NEW: SQL Server flag
    }
    
    # Render API status
    render_api_status_sidebar()
    
    st.sidebar.markdown("---")
    
    # Operating System Selection with AI insights
    st.sidebar.subheader("💻 Operating System (AI-Enhanced)")
    operating_system = st.sidebar.selectbox(
        "OS Selection",
        ["windows_server_2019", "windows_server_2022", "rhel_8", "rhel_9", "ubuntu_20_04", "ubuntu_22_04"],
        index=3,
        format_func=lambda x: {
            'windows_server_2019': '🔵 Windows Server 2019',
            'windows_server_2022': '🔵 Windows Server 2022 (Latest)',
            'rhel_8': '🔴 Red Hat Enterprise Linux 8',
            'rhel_9': '🔴 Red Hat Enterprise Linux 9 (Latest)',
            'ubuntu_20_04': '🟠 Ubuntu Server 20.04 LTS',
            'ubuntu_22_04': '🟠 Ubuntu Server 22.04 LTS (Latest)'
        }[x],
        help="AI analyzes OS performance characteristics and migration impact"
    )
    
    # Show AI OS insights
    os_manager = OSPerformanceManager()
    os_config = os_manager.operating_systems[operating_system]
    
    with st.sidebar.expander("🤖 AI OS Insights"):
        st.markdown(f"**Strengths:** {', '.join(os_config['ai_insights']['strengths'][:2])}")
        st.markdown(f"**Key Consideration:** {os_config['ai_insights']['weaknesses'][0]}")
    
    # Platform Configuration
    st.sidebar.subheader("🖥️ Server Platform")
    server_type = st.sidebar.selectbox(
        "Platform Type",
        ["physical", "vmware"],
        format_func=lambda x: "🏢 Physical Server" if x == "physical" else "☁️ VMware Virtual Machine",
        help="Physical vs Virtual performance analysis with AI optimization"
    )
    
    # Hardware Configuration with AI recommendations
    st.sidebar.subheader("⚙️ Hardware Configuration")
    ram_gb = st.sidebar.selectbox("RAM (GB)", [8, 16, 32, 64, 128, 256, 512], index=2, 
                                 help="AI calculates optimal memory for database workload")
    cpu_cores = st.sidebar.selectbox("CPU Cores", [2, 4, 8, 16, 24, 32, 48, 64], index=2,
                                   help="AI analyzes CPU requirements for migration performance")
    cpu_ghz = st.sidebar.selectbox("CPU GHz", [2.0, 2.4, 2.8, 3.2, 3.6, 4.0], index=3)
    
    # Enhanced Performance Metrics
    st.sidebar.subheader("📊 Current Performance Metrics")
    current_storage_gb = st.sidebar.number_input("Current Storage (GB)", 
                                                min_value=100, max_value=500000, value=2000, step=100,
                                                help="Current storage capacity in use")
    peak_iops = st.sidebar.number_input("Peak IOPS", 
                                       min_value=100, max_value=1000000, value=10000, step=500,
                                       help="Maximum IOPS observed during peak usage")
    max_throughput_mbps = st.sidebar.number_input("Max Throughput (MB/s)", 
                                                 min_value=10, max_value=10000, value=500, step=50,
                                                 help="Maximum storage throughput observed")
    anticipated_max_memory_gb = st.sidebar.number_input("Anticipated Max Memory (GB)", 
                                                       min_value=4, max_value=1024, value=64, step=8,
                                                       help="Maximum memory usage anticipated for workload")
    anticipated_max_cpu_cores = st.sidebar.number_input("Anticipated Max CPU Cores", 
                                                       min_value=1, max_value=128, value=16, step=2,
                                                       help="Maximum CPU cores anticipated for workload")
    
    # Network Interface with AI insights
    nic_type = st.sidebar.selectbox(
        "NIC Type",
        ["gigabit_copper", "gigabit_fiber", "10g_copper", "10g_fiber", "25g_fiber", "40g_fiber"],
        index=3,
        format_func=lambda x: {
            'gigabit_copper': '🔶 1Gbps Copper',
            'gigabit_fiber': '🟡 1Gbps Fiber',
            '10g_copper': '🔵 10Gbps Copper',
            '10g_fiber': '🟢 10Gbps Fiber',
            '25g_fiber': '🟣 25Gbps Fiber',
            '40g_fiber': '🔴 40Gbps Fiber'
        }[x],
        help="AI analyzes network impact on migration throughput"
    )
    
    nic_speeds = {
        'gigabit_copper': 1000, 'gigabit_fiber': 1000,
        '10g_copper': 10000, '10g_fiber': 10000, 
        '25g_fiber': 25000, '40g_fiber': 40000
    }
    nic_speed = nic_speeds[nic_type]
    
    # Migration Configuration with AI analysis
    st.sidebar.subheader("🔄 Migration Setup (AI-Optimized)")
    
    # Source and Target Databases
    source_database_engine = st.sidebar.selectbox(
        "Source Database",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': '🐬 MySQL', 'postgresql': '🐘 PostgreSQL', 'oracle': '🏛️ Oracle',
            'sqlserver': '🪟 SQL Server', 'mongodb': '🍃 MongoDB'
        }[x],
        help="AI determines migration complexity based on source engine"
    )
    
    database_engine = st.sidebar.selectbox(
        "Target Database (AWS)",
        ["mysql", "postgresql", "oracle", "sqlserver", "mongodb"],
        format_func=lambda x: {
            'mysql': '☁️ RDS MySQL', 'postgresql': '☁️ RDS PostgreSQL', 'oracle': '☁️ RDS Oracle',
            'sqlserver': '☁️ RDS SQL Server', 'mongodb': '☁️ DocumentDB'
        }[x],
        help="AI recommends optimal AWS database service"
    )
    
    # Show migration type indicator
    is_homogeneous = source_database_engine == database_engine
    migration_type_indicator = "🟢 Homogeneous" if is_homogeneous else "🟡 Heterogeneous"
    st.sidebar.info(f"**Migration Type:** {migration_type_indicator}")
    
    database_size_gb = st.sidebar.number_input("Database Size (GB)", 
                                              min_value=100, max_value=100000, value=1000, step=100,
                                              help="AI calculates migration time and resource requirements")
    
    # Migration Parameters
    downtime_tolerance_minutes = st.sidebar.number_input("Max Downtime (minutes)", 
                                                        min_value=1, max_value=480, value=60,
                                                        help="AI optimizes migration strategy for downtime window")
    performance_requirements = st.sidebar.selectbox("Performance Requirement", ["standard", "high"],
                                                   help="AI adjusts AWS sizing recommendations")
    
    # NEW: Destination Storage Selection
    st.sidebar.subheader("🗄️ Destination Storage (Enhanced Analysis)")
    destination_storage_type = st.sidebar.selectbox(
        "AWS Destination Storage",
        ["S3", "FSx_Windows", "FSx_Lustre"],
        format_func=lambda x: {
            'S3': '☁️ Amazon S3 (Standard)',
            'FSx_Windows': '🪟 Amazon FSx for Windows File Server',
            'FSx_Lustre': '⚡ Amazon FSx for Lustre (High Performance)'
        }[x],
        help="AI analyzes performance, cost, and complexity for each destination type"
    )
    
    # Show destination storage insights
    with st.sidebar.expander("🎯 Storage Destination Insights"):
        if destination_storage_type == "S3":
            st.markdown("""
            **S3 Advantages:**
            • Cost-effective and scalable
            • Simple migration integration
            • Excellent durability (99.999999999%)
            
            **Best For:** General-purpose storage, cost optimization
            """)
        elif destination_storage_type == "FSx_Windows":
            st.markdown("""
            **FSx Windows Advantages:**
            • Native Windows file system features
            • Better performance than S3
            • Active Directory integration
            
            **Best For:** Windows-based applications, file shares
            """)
        elif destination_storage_type == "FSx_Lustre":
            st.markdown("""
            **FSx Lustre Advantages:**
            • Extremely high performance (sub-ms latency)
            • Parallel processing optimized
            • Perfect for HPC and analytics
            
            **Best For:** High-performance computing, ML workloads
            """)
    
    # Environment
    environment = st.sidebar.selectbox("Environment", ["non-production", "production"],
                                     help="AI adjusts reliability and performance requirements")
    
    # Enhanced Agent Sizing Section with AI recommendations
    st.sidebar.subheader("🤖 Migration Agent Configuration (AI-Optimized)")
    
    # Determine migration type for tool selection
    primary_tool = "DataSync" if is_homogeneous else "DMS"
    
    st.sidebar.success(f"**Primary Tool:** AWS {primary_tool}")
    
    # Agent Count Configuration
    st.sidebar.markdown("**📊 Agent Scaling Configuration:**")
    
    number_of_agents = st.sidebar.number_input(
        "Number of Migration Agents",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Configure the number of agents for parallel migration processing"
    )
    
    # Show agent scaling insights
    if number_of_agents == 1:
        st.sidebar.info("💡 Single agent - simple but may limit throughput")
    elif number_of_agents <= 3:
        st.sidebar.success("✅ Optimal range for most workloads")
    elif number_of_agents <= 5:
        st.sidebar.warning("⚠️ High agent count - ensure proper coordination")
    else:
        st.sidebar.error("🔴 Very high agent count - diminishing returns likely")
    
    if is_homogeneous:
        datasync_agent_size = st.sidebar.selectbox(
            "DataSync Agent Size",
            ["small", "medium", "large", "xlarge"],
            index=1,
            format_func=lambda x: {
                'small': '📦 Small (t3.medium) - 250 Mbps/agent',
                'medium': '📦 Medium (c5.large) - 500 Mbps/agent',
                'large': '📦 Large (c5.xlarge) - 1000 Mbps/agent',
                'xlarge': '📦 XLarge (c5.2xlarge) - 2000 Mbps/agent'
            }[x],
            help=f"AI recommends optimal agent size for {number_of_agents} agents"
        )
        dms_agent_size = None
    else:
        dms_agent_size = st.sidebar.selectbox(
            "DMS Instance Size",
            ["small", "medium", "large", "xlarge", "xxlarge"],
            index=1,
            format_func=lambda x: {
                'small': '🔄 Small (t3.medium) - 200 Mbps/agent',
                'medium': '🔄 Medium (c5.large) - 400 Mbps/agent',
                'large': '🔄 Large (c5.xlarge) - 800 Mbps/agent',
                'xlarge': '🔄 XLarge (c5.2xlarge) - 1500 Mbps/agent',
                'xxlarge': '🔄 XXLarge (c5.4xlarge) - 2500 Mbps/agent'
            }[x],
            help=f"AI recommends optimal instance size for {number_of_agents} agents"
        )
        datasync_agent_size = None
    
    # Show estimated throughput with current configuration including destination storage impact
    agent_manager = EnhancedAgentSizingManager()
    if is_homogeneous:
        test_config = agent_manager.calculate_agent_configuration('datasync', datasync_agent_size, number_of_agents, destination_storage_type)
    else:
        test_config = agent_manager.calculate_agent_configuration('dms', dms_agent_size, number_of_agents, destination_storage_type)
    
    st.sidebar.markdown(f"""
    <div class="agent-scaling-card">
        <h4>🚀 Current Configuration Impact</h4>
        <p><strong>Total Throughput:</strong> {test_config['total_max_throughput_mbps']:,.0f} Mbps</p>
        <p><strong>Scaling Efficiency:</strong> {test_config['scaling_efficiency']*100:.1f}%</p>
        <p><strong>Storage Multiplier:</strong> {test_config['storage_performance_multiplier']:.1f}x</p>
        <p><strong>Monthly Cost:</strong> ${test_config['total_monthly_cost']:,.0f}</p>
        <p><strong>Config Rating:</strong> {test_config['optimal_configuration']['efficiency_score']:.0f}/100</p>
        <p><strong>Destination:</strong> {destination_storage_type}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Configuration Section
    st.sidebar.subheader("🧠 AI Configuration")
    
    enable_ai_analysis = st.sidebar.checkbox("Enable AI Analysis", value=True,
                                           help="Use Anthropic AI for intelligent recommendations")
    
    if enable_ai_analysis:
        ai_analysis_depth = st.sidebar.selectbox(
            "AI Analysis Depth",
            ["standard", "comprehensive"],
            help="Comprehensive analysis provides more detailed AI insights"
        )
    else:
        ai_analysis_depth = "standard"
    
    # Real-time AWS Pricing
    use_realtime_pricing = st.sidebar.checkbox("Real-time AWS Pricing", value=True,
                                             help="Fetch current AWS pricing via API")
    
    if st.sidebar.button("🔄 Refresh AI Analysis", type="primary"):
        st.rerun()
    
    return {
        'operating_system': operating_system,
        'server_type': server_type,
        'ram_gb': ram_gb,
        'cpu_cores': cpu_cores,
        'cpu_ghz': cpu_ghz,
        'nic_type': nic_type,
        'nic_speed': nic_speed,
        'source_database_engine': source_database_engine,
        'database_engine': database_engine,
        'database_size_gb': database_size_gb,
        'downtime_tolerance_minutes': downtime_tolerance_minutes,
        'performance_requirements': performance_requirements,
        'destination_storage_type': destination_storage_type,
        'environment': environment,
        'datasync_agent_size': datasync_agent_size,
        'dms_agent_size': dms_agent_size,
        'number_of_agents': number_of_agents,
        'enable_ai_analysis': enable_ai_analysis,
        'ai_analysis_depth': ai_analysis_depth,
        'use_realtime_pricing': use_realtime_pricing,
        'current_storage_gb': current_storage_gb,
        'peak_iops': peak_iops,
        'max_throughput_mbps': max_throughput_mbps,
        'anticipated_max_memory_gb': anticipated_max_memory_gb,
        'anticipated_max_cpu_cores': anticipated_max_cpu_cores
    }

# Enhanced AWS sizing function to handle EC2 instance selections
def enhanced_aws_sizing_with_ec2_support(config: Dict):
    """Enhanced AWS sizing that properly handles both RDS and specific EC2 instance selections"""
    
    database_engine = config['database_engine']
    
    if database_engine.startswith('rds_'):
        # Use existing RDS sizing logic
        return calculate_rds_sizing(config)
    
    elif database_engine.startswith('ec2_'):
        # Enhanced EC2 sizing for specific instance types
        return calculate_ec2_sizing_with_instance_type(config)
    
    else:
        # Fallback to general EC2 sizing
        return calculate_general_ec2_sizing(config)


def calculate_ec2_sizing_with_instance_type(config: Dict):
    """Calculate sizing for specific EC2 instance types"""
    
    selected_instance = config['database_engine']
    instance_type = selected_instance.replace('ec2_', '')
    
    # Define instance specifications
    ec2_specs = {
        't3.medium': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.0416, 'category': 'burstable'},
        't3.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.0832, 'category': 'burstable'},
        't3.xlarge': {'vcpu': 4, 'memory': 16, 'cost_per_hour': 0.1664, 'category': 'burstable'},
        't3.2xlarge': {'vcpu': 8, 'memory': 32, 'cost_per_hour': 0.3328, 'category': 'burstable'},
        
        'c5.large': {'vcpu': 2, 'memory': 4, 'cost_per_hour': 0.085, 'category': 'compute_optimized'},
        'c5.xlarge': {'vcpu': 4, 'memory': 8, 'cost_per_hour': 0.17, 'category': 'compute_optimized'},
        'c5.2xlarge': {'vcpu': 8, 'memory': 16, 'cost_per_hour': 0.34, 'category': 'compute_optimized'},
        'c5.4xlarge': {'vcpu': 16, 'memory': 32, 'cost_per_hour': 0.68, 'category': 'compute_optimized'},
        
        'r6i.large': {'vcpu': 2, 'memory': 16, 'cost_per_hour': 0.252, 'category': 'memory_optimized'},
        'r6i.xlarge': {'vcpu': 4, 'memory': 32, 'cost_per_hour': 0.504, 'category': 'memory_optimized'},
        'r6i.2xlarge': {'vcpu': 8, 'memory': 64, 'cost_per_hour': 1.008, 'category': 'memory_optimized'},
        'r6i.4xlarge': {'vcpu': 16, 'memory': 128, 'cost_per_hour': 2.016, 'category': 'memory_optimized'},
        'r6i.8xlarge': {'vcpu': 32, 'memory': 256, 'cost_per_hour': 4.032, 'category': 'memory_optimized'},
        
        'm5.large': {'vcpu': 2, 'memory': 8, 'cost_per_hour': 0.096, 'category': 'general_purpose'},
        'm5.xlarge': {'vcpu': 4, 'memory': 16, 'cost_per_hour': 0.192, 'category': 'general_purpose'},
        'm5.2xlarge': {'vcpu': 8, 'memory': 32, 'cost_per_hour': 0.384, 'category': 'general_purpose'},
        'm5.4xlarge': {'vcpu': 16, 'memory': 64, 'cost_per_hour': 0.768, 'category': 'general_purpose'},
        
        'x1e.xlarge': {'vcpu': 4, 'memory': 122, 'cost_per_hour': 0.834, 'category': 'high_memory'},
        'x1e.2xlarge': {'vcpu': 8, 'memory': 244, 'cost_per_hour': 1.668, 'category': 'high_memory'},
        'x1e.4xlarge': {'vcpu': 16, 'memory': 488, 'cost_per_hour': 3.336, 'category': 'high_memory'},
        
        'i3.large': {'vcpu': 2, 'memory': 15.25, 'cost_per_hour': 0.156, 'category': 'storage_optimized'},
        'i3.xlarge': {'vcpu': 4, 'memory': 30.5, 'cost_per_hour': 0.312, 'category': 'storage_optimized'},
        'i3.2xlarge': {'vcpu': 8, 'memory': 61, 'cost_per_hour': 0.624, 'category': 'storage_optimized'}
    }
    
    instance_spec = ec2_specs.get(instance_type, ec2_specs['t3.medium'])
    
    # Calculate storage requirements
    database_size_gb = config['database_size_gb']
    storage_multiplier = 2.0  # Extra space for EC2 self-managed
    storage_size_gb = max(database_size_gb * storage_multiplier, 100)
    
    # Storage type recommendation based on instance category
    if instance_spec['category'] == 'storage_optimized':
        storage_type = 'io2'
    elif instance_spec['category'] in ['compute_optimized', 'high_memory']:
        storage_type = 'gp3'
    else:
        storage_type = 'gp3'
    
    # Calculate costs
    instance_cost = instance_spec['cost_per_hour'] * 24 * 30
    storage_cost = storage_size_gb * 0.08  # GP3 pricing
    
    return {
        'deployment_type': 'ec2',
        'selected_instance': instance_type,
        'instance_specs': instance_spec,
        'storage_type': storage_type,
        'storage_size_gb': storage_size_gb,
        'monthly_instance_cost': instance_cost,
        'monthly_storage_cost': storage_cost,
        'total_monthly_cost': instance_cost + storage_cost,
        'instance_category': instance_spec['category'],
        'management_complexity': 'high',
        'scaling_capability': 'manual',
        'backup_strategy': 'custom_implementation_required',
        'monitoring_setup': 'cloudwatch_custom_metrics',
        'os_licensing_required': True,
        'database_software_licensing': 'customer_managed'
    }


# Enhanced display function for the new options
def display_enhanced_target_selection(config: Dict):
    """Display enhanced information about the selected target"""
    
    database_engine = config['database_engine']
    
    if database_engine.startswith('rds_'):
        st.success("🌟 **Managed Service Selected**")
        st.write("✅ Automated backups and patching")
        st.write("✅ Built-in monitoring and alerting") 
        st.write("✅ Automatic scaling capabilities")
        st.write("✅ Managed security and compliance")
        
    elif database_engine.startswith('ec2_'):
        instance_type = database_engine.replace('ec2_', '')
        st.warning("🔧 **Self-Managed Instance Selected**")
        st.write(f"🖥️ Instance Type: {instance_type}")
        st.write("⚠️ Manual backup configuration required")
        st.write("⚠️ Custom monitoring setup needed")
        st.write("⚠️ Manual scaling and maintenance")
        st.write("⚠️ OS and database licensing considerations")
        
        # Show instance-specific considerations
        if 't3' in instance_type:
            st.info("💡 **Burstable Performance**: Great for variable workloads")
        elif 'c5' in instance_type:
            st.info("⚡ **Compute Optimized**: Ideal for CPU-intensive databases")
        elif 'r6i' in instance_type:
            st.info("🧠 **Memory Optimized**: Perfect for memory-intensive workloads")
        elif 'x1e' in instance_type:
            st.info("🔥 **High Memory**: Designed for very large in-memory databases")
        elif 'i3' in instance_type:
            st.info("💾 **Storage Optimized**: High IOPS for storage-intensive applications")




def render_fsx_destination_comparison_tab(analysis: Dict, config: Dict):
    """Render FSx destination comparison analysis tab using native Streamlit components"""
    st.subheader("🗄️ FSx Destination Storage Comparison & Performance Analysis")
    
    fsx_comparisons = analysis.get('fsx_comparisons', {})
    current_destination = config.get('destination_storage_type', 'S3')
    
    if not fsx_comparisons:
        st.error("FSx comparison data not available. Please run the analysis first.")
        return
    
    # Architecture Overview
    st.markdown("**🏗️ Migration Architecture Overview:**")
    
    for destination in fsx_comparisons.keys():
        comparison = fsx_comparisons[destination]
        architecture = comparison.get('migration_architecture', {})
        
        is_current = destination == current_destination
        
        with st.expander(f"{'🎯 ' if is_current else ''}**{destination} Architecture**", expanded=is_current):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Primary Target:** {architecture.get('primary_target', 'Unknown')}")
                st.write(f"**Secondary Target:** {architecture.get('secondary_target', 'None')}")
                st.write(f"**Architecture Type:** {architecture.get('architecture_type', 'Unknown')}")
                st.write(f"**Agent Direct Access:** {'Yes' if architecture.get('agent_targets_destination') else 'No'}")
            
            with col2:
                st.write(f"**Migration Description:** {comparison.get('migration_description', 'Standard migration')}")
                st.write(f"**Setup Complexity:** {comparison.get('setup_complexity', 'Unknown')}")
                st.write(f"**Migration Time:** {comparison.get('estimated_migration_time_hours', 0):.1f} hours")
                st.write(f"**Monthly Storage Cost:** ${comparison.get('estimated_monthly_storage_cost', 0):,.0f}")
            
            # Architecture notes
            if 'architecture_notes' in comparison:
                st.write("**Architecture Notes:**")
                for note in comparison['architecture_notes']:
                    st.write(f"• {note}")
    
    
    # Destination Overview Cards using native components
    st.markdown("**📊 Destination Storage Overview:**")
    
    col1, col2, col3 = st.columns(3)
    
    destinations = list(fsx_comparisons.keys())
    for i, (col, destination) in enumerate(zip([col1, col2, col3], destinations)):
        comparison = fsx_comparisons.get(destination, {})
        
        # Determine if this is the current selection
        is_current = destination == current_destination
        
        with col:
            if is_current:
                st.success(f"🎯 **{destination} (Selected)**")
            else:
                st.info(f"**{destination}**")
            
            st.write(f"**Performance:** {comparison.get('performance_rating', 'Unknown')}")
            st.write(f"**Cost Rating:** {comparison.get('cost_rating', 'Unknown')}")
            st.write(f"**Complexity:** {comparison.get('complexity_rating', 'Unknown')}")
            st.write(f"**Migration Time:** {comparison.get('estimated_migration_time_hours', 0):.1f} hours")
            st.write(f"**Monthly Storage Cost:** ${comparison.get('estimated_monthly_storage_cost', 0):,.0f}")
    
    # Performance Comparison Charts
    st.markdown("**📈 Performance & Cost Comparison:**")
    
    # Create comparison visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Migration Time Comparison
        migration_times = {dest: fsx_comparisons[dest].get('estimated_migration_time_hours', 0) 
                          for dest in destinations}
        
        fig_time = px.bar(
            x=list(migration_times.keys()),
            y=list(migration_times.values()),
            title="Migration Time Comparison",
            labels={'x': 'Destination Storage', 'y': 'Migration Time (Hours)'},
            color=list(migration_times.values()),
            color_continuous_scale='RdYlGn_r'
        )
        
        # Highlight current selection
        colors = ['red' if dest == current_destination else 'lightblue' for dest in destinations]
        fig_time.update_traces(marker_color=colors)
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Cost Comparison
        storage_costs = {dest: fsx_comparisons[dest].get('estimated_monthly_storage_cost', 0) 
                        for dest in destinations}
        
        fig_cost = px.bar(
            x=list(storage_costs.keys()),
            y=list(storage_costs.values()),
            title="Monthly Storage Cost Comparison",
            labels={'x': 'Destination Storage', 'y': 'Monthly Cost ($)'},
            color=list(storage_costs.values()),
            color_continuous_scale='RdYlBu_r'
        )
        
        # Highlight current selection
        colors = ['red' if dest == current_destination else 'lightgreen' for dest in destinations]
        fig_cost.update_traces(marker_color=colors)
        
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Detailed Analysis per Destination using tabs
    st.markdown("**🔍 Detailed Destination Analysis:**")
    
    destination_tabs = st.tabs([f"{dest}" for dest in destinations])
    
    for tab, destination in zip(destination_tabs, destinations):
        with tab:
            comparison = fsx_comparisons.get(destination, {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**{destination} Performance Metrics**")
                st.write(f"**Migration Time:** {comparison.get('estimated_migration_time_hours', 0):.1f} hours")
                st.write(f"**Throughput:** {comparison.get('migration_throughput_mbps', 0):,.0f} Mbps")
                st.write(f"**Performance Rating:** {comparison.get('performance_rating', 'Unknown')}")
                st.write(f"**Cost Rating:** {comparison.get('cost_rating', 'Unknown')}")
                st.write(f"**Complexity Rating:** {comparison.get('complexity_rating', 'Unknown')}")
            
            with col2:
                st.info(f"**{destination} Cost Analysis**")
                st.write(f"**Monthly Storage:** ${comparison.get('estimated_monthly_storage_cost', 0):,.0f}")
                st.write(f"**Agent Configuration:** {comparison.get('agent_configuration', {}).get('number_of_agents', 1)} agents")
                st.write(f"**Agent Monthly Cost:** ${comparison.get('agent_configuration', {}).get('total_monthly_cost', 0):,.0f}")
                st.write(f"**Storage Multiplier:** {comparison.get('agent_configuration', {}).get('storage_performance_multiplier', 1.0):.1f}x")
            
            # Recommendations for this destination
            recommendations = comparison.get('recommendations', [])
            if recommendations:
                with st.expander(f"💡 {destination} Recommendations", expanded=True):
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
            
            # Network performance for this destination
            network_perf = comparison.get('network_performance', {})
            if network_perf:
                with st.expander(f"🌐 {destination} Network Performance", expanded=False):
                    network_col1, network_col2 = st.columns(2)
                    
                    with network_col1:
                        st.metric("Network Quality", f"{network_perf.get('network_quality_score', 0):.1f}/100")
                        st.metric("AI Enhanced Quality", f"{network_perf.get('ai_enhanced_quality_score', 0):.1f}/100")
                    
                    with network_col2:
                        st.metric("Effective Bandwidth", f"{network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps")
                        st.metric("Total Latency", f"{network_perf.get('total_latency_ms', 0):.1f} ms")
    
    # Summary Recommendations using native components
    st.markdown("**🎯 Overall Destination Recommendations:**")
    
    # Find the best option for different criteria
    best_performance = min(destinations, key=lambda x: fsx_comparisons[x].get('estimated_migration_time_hours', float('inf')), default='S3')
    best_cost = min(destinations, key=lambda x: fsx_comparisons[x].get('estimated_monthly_storage_cost', float('inf')), default='S3')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("🚀 **Best Performance**")
        st.write(f"**Winner:** {best_performance}")
        st.write(f"**Migration Time:** {fsx_comparisons[best_performance].get('estimated_migration_time_hours', 0):.1f} hours")
        st.write(f"**Throughput:** {fsx_comparisons[best_performance].get('migration_throughput_mbps', 0):,.0f} Mbps")
    
    with col2:
        st.info("💰 **Best Cost Efficiency**")
        st.write(f"**Winner:** {best_cost}")
        st.write(f"**Monthly Cost:** ${fsx_comparisons[best_cost].get('estimated_monthly_storage_cost', 0):,.0f}")
        st.write(f"**Cost Rating:** {fsx_comparisons[best_cost].get('cost_rating', 'Unknown')}")
    
    with col3:
        current_comparison = fsx_comparisons.get(current_destination, {})
        st.warning("🎯 **Current Selection**")
        st.write(f"**Choice:** {current_destination}")
        st.write(f"**Performance:** {current_comparison.get('performance_rating', 'Unknown')}")
        st.write(f"**Complexity:** {current_comparison.get('complexity_rating', 'Unknown')}")
    
    # Decision Matrix
    st.markdown("**📊 Decision Matrix:**")
    
    # Create decision matrix data
    matrix_data = []
    for dest in destinations:
        comp = fsx_comparisons[dest]
        matrix_data.append({
            'Destination': dest,
            'Performance Rating': comp.get('performance_rating', 'Unknown'),
            'Cost Rating': comp.get('cost_rating', 'Unknown'),
            'Complexity Rating': comp.get('complexity_rating', 'Unknown'),
            'Migration Time (Hours)': f"{comp.get('estimated_migration_time_hours', 0):.1f}",
            'Monthly Cost ($)': f"${comp.get('estimated_monthly_storage_cost', 0):,.0f}",
            'Current Selection': '✅' if dest == current_destination else ''
        })
    
    df_matrix = pd.DataFrame(matrix_data)
    st.dataframe(df_matrix, use_container_width=True)
def render_agent_scaling_tab(analysis, config):
    """FIXED: Agent scaling tab with proper null checking"""
    
    # Guard clause to handle None or invalid analysis
    if analysis is None:
        analysis = {}
    if config is None:
        config = {}
    
    # Ensure analysis is a dictionary
    if not isinstance(analysis, dict):
        st.error("Invalid analysis data provided. Please check your data source.")
        return
    
    st.subheader("🤖 Agent Scaling Analysis & Optimization")
    
    agent_analysis = analysis.get('agent_analysis', {})
    optimal_recommendations = agent_analysis.get('optimal_recommendations', {})
    
    # Agent Configuration Overview with Destination Storage
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🔧 Current Agents",
            f"{config.get('number_of_agents', 1)} agents",
            delta=f"{agent_analysis.get('primary_tool', 'Unknown').upper()} {agent_analysis.get('agent_size', 'Unknown')}"
        )
    
    with col2:
        st.metric(
            "🗄️ Destination Storage",
            config.get('destination_storage_type', 'S3'),
            delta=f"Multiplier: {agent_analysis.get('storage_performance_multiplier', 1.0):.1f}x"
        )
    
    with col3:
        st.metric(
            "⚡ Actual Migration Throughput",
            f"{agent_analysis.get('total_effective_throughput', 0):,.0f} Mbps",
            delta=f"vs {analysis.get('network_performance', {}).get('effective_bandwidth_mbps', 0):,.0f} Mbps network capacity"
        )
    
    with col4:
        st.metric(
            "🎯 Scaling Efficiency",
            f"{agent_analysis.get('scaling_efficiency', 1.0)*100:.1f}%",
            delta=f"Overhead: {(agent_analysis.get('management_overhead', 1.0)-1)*100:.1f}%"
        )
    
    with col5:
        st.metric(
            "💰 Monthly Cost",
            f"${agent_analysis.get('monthly_cost', 0):,.0f}",
            delta=f"${agent_analysis.get('cost_per_hour', 0):.2f}/hour"
        )
    
    # Enhanced DYNAMIC warning about bandwidth vs throughput
    network_perf = analysis.get('network_performance', {})
    
    network_bw = network_perf.get('effective_bandwidth_mbps', 0)
    actual_throughput = agent_analysis.get('total_effective_throughput', 0)

    if network_bw > actual_throughput:
        # Get ACTUAL user configuration values
        nic_speed = config.get('nic_speed', 1000)
        nic_type = config.get('nic_type', 'gigabit_fiber')
        os_name = config.get('operating_system', 'Unknown').replace('_', ' ').title()
        server_type = config.get('server_type', 'physical')
        num_agents = config.get('number_of_agents', 1)
        tool_name = analysis.get('primary_tool', 'DMS').upper()
        
        st.warning(f"""
        ⚠️ **Why Your {nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} Shows {network_bw:,.0f} Mbps But Agents Only Achieve {actual_throughput:,.0f} Mbps**

        🔍 **Your Actual Configuration Performance Stack:**
        1. **Your Network:** {nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} connection
        2. **Your Platform ({server_type.title()}):** {'-8%' if server_type == 'vmware' else 'No virtualization overhead'}
        3. **Protocol Overhead:** -{15 if config.get('environment') != 'production' else 18}% (TCP/IP, encryption, compression)
        4. **Your Migration Setup:** {num_agents}x {tool_name} agents = {actual_throughput:,.0f} Mbps final throughput

        💡 **For Your Migration:** Plan timelines using **{actual_throughput:,.0f} Mbps actual throughput**, not the {nic_speed:,.0f} Mbps network capacity.
        
        📊 **See the Network Intelligence tab for your complete bandwidth waterfall analysis.**
        """)
    
    # Agent Configuration Details with Storage Impact
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Current Agent Configuration:**")
        
        agent_config = agent_analysis.get('agent_configuration', {})
        destination_storage = agent_analysis.get('destination_storage', 'S3')
        
        with st.container():
            st.success("Current Setup Details")
            st.write(f"**Agent Type:** {agent_analysis.get('primary_tool', 'Unknown').upper()}")
            
            # FIXED: Proper null checking for agent_size
            agent_size = agent_analysis.get('agent_size')
            if agent_size and agent_size != 'Unknown':
                st.write(f"**Agent Size:** {agent_size.title()}")
            else:
                st.write(f"**Agent Size:** Unknown")
            
            st.write(f"**Number of Agents:** {agent_analysis.get('number_of_agents', 1)}")
            st.write(f"**Destination Storage:** {destination_storage}")
            st.write(f"**Per-Agent vCPU:** {agent_config.get('per_agent_spec', {}).get('vcpu', 'N/A')}")
            st.write(f"**Per-Agent Memory:** {agent_config.get('per_agent_spec', {}).get('memory_gb', 'N/A')} GB")
            st.write(f"**Per-Agent Throughput:** {agent_config.get('max_throughput_mbps_per_agent', 0):,.0f} Mbps")
            st.write(f"**Storage Performance Bonus:** {agent_config.get('storage_performance_multiplier', 1.0):.1f}x")
            st.write(f"**Total Concurrent Tasks:** {agent_config.get('total_concurrent_tasks', 0)}")
    
    with col2:
        st.markdown("**🎯 Optimal Configuration Recommendation:**")
        
        if optimal_recommendations.get('optimal_configuration'):
            optimal_config = optimal_recommendations['optimal_configuration']
            with st.container():
                st.info("AI-Recommended Optimal Setup")
                st.write(f"**Optimal Agents:** {optimal_config['configuration']['number_of_agents']}")
                
                # FIXED: Proper null checking for optimal agent size
                optimal_agent_size = optimal_config['configuration'].get('agent_size')
                if optimal_agent_size and optimal_agent_size != 'Unknown':
                    st.write(f"**Optimal Size:** {optimal_agent_size.title()}")
                else:
                    st.write(f"**Optimal Size:** Unknown")
                
                st.write(f"**Recommended Destination:** {optimal_config['configuration']['destination_storage']}")
                st.write(f"**Total Throughput:** {optimal_config['total_throughput_mbps']:,.0f} Mbps")
                st.write(f"**Storage Performance:** {optimal_config.get('storage_performance_multiplier', 1.0):.1f}x")
                st.write(f"**Monthly Cost:** ${optimal_config['total_cost_per_hour'] * 24 * 30:,.0f}")
                st.write(f"**Overall Score:** {optimal_config['overall_score']:.1f}/100")
                
                efficiency_gain = optimal_config['overall_score'] - agent_config.get('optimal_configuration', {}).get('efficiency_score', 0)
                st.write(f"**Efficiency Gain:** {efficiency_gain:.1f} points")
        else:
            with st.container():
                st.warning("Configuration Analysis")
                st.write(f"Current configuration appears optimal for {destination_storage}")
                st.write(f"**Efficiency Score:** {agent_config.get('optimal_configuration', {}).get('efficiency_score', 0):.1f}/100")
                st.write(f"**Management Complexity:** {agent_config.get('optimal_configuration', {}).get('management_complexity', 'Unknown')}")
                st.write(f"**Storage Optimization:** {agent_config.get('optimal_configuration', {}).get('storage_optimization', 'Unknown')}")
    
    # Throughput Analysis Chart
    st.markdown("**📊 Throughput Analysis with Storage Impact:**")
    
    throughput_data = {
        'Component': ['Per-Agent Base', 'Per-Agent with Storage', 'Total Agent Capacity', 'Network Limit', 'Effective Throughput'],
        'Throughput (Mbps)': [
            agent_config.get('max_throughput_mbps_per_agent', 0),
            agent_config.get('max_throughput_mbps_per_agent', 0) * agent_config.get('storage_performance_multiplier', 1.0),
            agent_analysis.get('total_max_throughput_mbps', 0),
            analysis.get('network_performance', {}).get('effective_bandwidth_mbps', 0),
            agent_analysis.get('total_effective_throughput', 0)
        ],
        'Type': ['Base', 'Enhanced', 'Aggregate', 'Network', 'Effective']
    }
    
    fig_throughput = px.bar(
        throughput_data, 
        x='Component', 
        y='Throughput (Mbps)',
        color='Type',
        title=f"Agent vs Network Throughput Analysis ({destination_storage} Destination)",
        color_discrete_map={'Base': '#95a5a6', 'Enhanced': '#3498db', 'Aggregate': '#2ecc71', 'Network': '#e74c3c', 'Effective': '#9b59b6'}
    )
    
    st.plotly_chart(fig_throughput, use_container_width=True)
    
    # Scaling Efficiency Analysis using native components
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Scaling Efficiency Analysis:**")
        
        scaling_recommendations = agent_config.get('scaling_recommendations', [])
        
        with st.container():
            st.info("Scaling Assessment")
            st.write(f"**Current Efficiency:** {agent_analysis.get('scaling_efficiency', 1.0)*100:.1f}%")
            st.write(f"**Management Overhead:** {(agent_analysis.get('management_overhead', 1.0)-1)*100:.1f}%")
            st.write(f"**Storage Overhead:** {(agent_analysis.get('storage_management_overhead', 1.0)-1)*100:.1f}%")
            st.write(f"**Coordination Complexity:** {agent_config.get('optimal_configuration', {}).get('management_complexity', 'Unknown')}")
            
            if scaling_recommendations:
                st.write("**Key Recommendations:**")
                for rec in scaling_recommendations[:3]:
                    st.write(f"• {rec}")
    
    with col2:
        st.markdown("**🔍 Bottleneck Analysis:**")
        
        bottleneck = agent_analysis.get('bottleneck', 'unknown')
        bottleneck_severity = agent_analysis.get('bottleneck_severity', 'medium')
        
        with st.container():
            if bottleneck_severity == 'high':
                st.error("Performance Constraints")
            elif bottleneck_severity == 'medium':
                st.warning("Performance Constraints")
            else:
                st.success("Performance Constraints")
                
            st.write(f"**Primary Bottleneck:** {bottleneck.title()}")
            st.write(f"**Severity:** {bottleneck_severity.title()}")
            st.write(f"**Throughput Impact:** {agent_analysis.get('throughput_impact', 0)*100:.1f}%")
            st.write(f"**Storage Impact:** {destination_storage} provides {agent_analysis.get('storage_performance_multiplier', 1.0):.1f}x multiplier")
            
            recommendation = "Scale agents" if bottleneck == 'agents' else "Optimize network" if bottleneck == 'network' else "Monitor performance"
            st.write(f"**Recommendation:** {recommendation}")
    
    with col3:
        st.markdown("**💰 Cost Efficiency Analysis:**")
        
        cost_per_agent = agent_analysis.get('monthly_cost', 0) / config.get('number_of_agents', 1) if config.get('number_of_agents', 1) > 0 else 0
        cost_per_mbps = agent_analysis.get('monthly_cost', 0) / agent_analysis.get('total_effective_throughput', 1) if agent_analysis.get('total_effective_throughput', 0) > 0 else 0
        
        storage_cost_impact = {
            'S3': 1.0,
            'FSx_Windows': 1.1,
            'FSx_Lustre': 1.2
        }.get(destination_storage, 1.0)
        
        with st.container():
            st.info("Cost Efficiency Metrics")
            st.write(f"**Cost per Agent:** ${cost_per_agent:,.0f}/month")
            st.write(f"**Cost per Mbps:** ${cost_per_mbps:.2f}/month")
            st.write(f"**Total Monthly:** ${agent_analysis.get('monthly_cost', 0):,.0f}")
            st.write(f"**Storage Impact:** {storage_cost_impact:.1f}x base cost")
            st.write(f"**Efficiency Rating:** {agent_config.get('optimal_configuration', {}).get('cost_efficiency', 'Unknown')}")
            st.write(f"**Storage Optimization:** {agent_config.get('optimal_configuration', {}).get('storage_optimization', 'Standard')}")
    
    # AI Optimization Recommendations using expandable sections
    ai_optimization = agent_analysis.get('ai_optimization', {})
    optimization_recommendations = ai_optimization.get('recommendations', [])
    
    if optimization_recommendations:
        st.markdown("**🤖 AI Agent Optimization Recommendations:**")
        
        for i, recommendation in enumerate(optimization_recommendations, 1):
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            timeframe = "Immediate" if i <= 2 else "Short-term" if i <= 4 else "Long-term"
            
            # Determine if recommendation is storage-specific
            storage_specific = any(storage in recommendation.lower() for storage in ['s3', 'fsx', 'lustre', 'windows'])
            
            with st.expander(f"Optimization {i}: {recommendation}", expanded=(i <= 2)):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if complexity == "Low":
                        st.success(f"**Complexity:** {complexity}")
                    elif complexity == "Medium":
                        st.warning(f"**Complexity:** {complexity}")
                    else:
                        st.error(f"**Complexity:** {complexity}")
                
                with col2:
                    st.write(f"**Timeframe:** {timeframe}")
                
                with col3:
                    expected_benefit = ai_optimization.get('optimization_potential_percent', 0) // i
                    st.write(f"**Expected Benefit:** {expected_benefit}% improvement")
                
                with col4:
                    relevance = "High" if storage_specific else "General"
                    st.write(f"**Storage Relevance:** {relevance}")
                
                with col5:
                    destination_text = "Specific to " + destination_storage if storage_specific else "All destinations"
                    st.write(f"**Destination:** {destination_text}")
                    
                    # Enhanced DYNAMIC warning about bandwidth vs throughput
    network_perf = analysis.get('network_performance', {})
    agent_analysis = analysis.get('agent_analysis', {})

    network_bw = network_perf.get('effective_bandwidth_mbps', 0)
    actual_throughput = agent_analysis.get('total_effective_throughput', 0)

    if network_bw > actual_throughput:
        # Get ACTUAL user configuration values
        nic_speed = config.get('nic_speed', 1000)
        nic_type = config.get('nic_type', 'gigabit_fiber')
        os_name = config.get('operating_system', 'Unknown').replace('_', ' ').title()
        server_type = config.get('server_type', 'physical')
        num_agents = config.get('number_of_agents', 1)
        tool_name = analysis.get('primary_tool', 'DMS').upper()
        
        # Calculate ACTUAL efficiency losses
        nic_efficiency = get_nic_efficiency(nic_type)
        os_efficiency = analysis.get('onprem_performance', {}).get('os_impact', {}).get('network_efficiency', 0.90)
        
        st.warning(f"""
        ⚠️ **Why Your {nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} Shows {network_bw:,.0f} Mbps But Agents Only Achieve {actual_throughput:,.0f} Mbps**

        🔍 **Your Actual Configuration Performance Stack:**
        1. **Your Network:** {nic_speed:,.0f} Mbps {nic_type.replace('_', ' ')} connection
        2. **Your NIC Hardware:** -{(1-nic_efficiency)*100:.0f}% efficiency loss ({nic_type.replace('_', ' ')})
        3. **Your OS ({os_name}):** -{(1-os_efficiency)*100:.0f}% network stack overhead
        4. **Your Platform ({server_type.title()}):** {'-8%' if server_type == 'vmware' else 'No virtualization overhead'}
        5. **Protocol Overhead:** -{15 if config.get('environment') != 'production' else 18}% (TCP/IP, encryption, compression)
        6. **Your Migration Setup:** {num_agents}x {tool_name} agents = {actual_throughput:,.0f} Mbps final throughput

        💡 **For Your Migration:** Plan timelines using **{actual_throughput:,.0f} Mbps actual throughput**, not the {nic_speed:,.0f} Mbps network capacity.
        
        📊 **See the Network Intelligence tab for your complete bandwidth waterfall analysis.**
        """)
    
    
    
    # Agent Configuration Details with Storage Impact
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Current Agent Configuration:**")
        
        agent_config = agent_analysis.get('agent_configuration', {})
        destination_storage = agent_analysis.get('destination_storage', 'S3')
        
        with st.container():
            st.success("Current Setup Details")
            st.write(f"**Agent Type:** {agent_analysis.get('primary_tool', 'Unknown').upper()}")
            # FIXED: Proper null checking for agent_size
            agent_size = agent_analysis.get('agent_size', 'Unknown')
            if agent_size and agent_size != 'Unknown':
                st.write(f"**Agent Size:** {agent_size.title()}")
            else:
                st.write(f"**Agent Size:** Unknown")
            st.write(f"**Number of Agents:** {agent_analysis.get('number_of_agents', 1)}")
            st.write(f"**Destination Storage:** {destination_storage}")
            st.write(f"**Per-Agent vCPU:** {agent_config.get('per_agent_spec', {}).get('vcpu', 'N/A')}")
            st.write(f"**Per-Agent Memory:** {agent_config.get('per_agent_spec', {}).get('memory_gb', 'N/A')} GB")
            st.write(f"**Per-Agent Throughput:** {agent_config.get('max_throughput_mbps_per_agent', 0):,.0f} Mbps")
            st.write(f"**Storage Performance Bonus:** {agent_config.get('storage_performance_multiplier', 1.0):.1f}x")
            st.write(f"**Total Concurrent Tasks:** {agent_config.get('total_concurrent_tasks', 0)}")
    
    with col2:
        st.markdown("**🎯 Optimal Configuration Recommendation:**")
        
        if optimal_recommendations.get('optimal_configuration'):
            optimal_config = optimal_recommendations['optimal_configuration']
            with st.container():
                st.info("AI-Recommended Optimal Setup")
                st.write(f"**Optimal Agents:** {optimal_config['configuration']['number_of_agents']}")
                st.write(f"**Optimal Size:** {optimal_config['configuration']['agent_size'].title()}")
                st.write(f"**Recommended Destination:** {optimal_config['configuration']['destination_storage']}")
                st.write(f"**Total Throughput:** {optimal_config['total_throughput_mbps']:,.0f} Mbps")
                st.write(f"**Storage Performance:** {optimal_config.get('storage_performance_multiplier', 1.0):.1f}x")
                st.write(f"**Monthly Cost:** ${optimal_config['total_cost_per_hour'] * 24 * 30:,.0f}")
                st.write(f"**Overall Score:** {optimal_config['overall_score']:.1f}/100")
                
                efficiency_gain = optimal_config['overall_score'] - agent_config.get('optimal_configuration', {}).get('efficiency_score', 0)
                st.write(f"**Efficiency Gain:** {efficiency_gain:.1f} points")
        else:
            with st.container():
                st.warning("Configuration Analysis")
                st.write(f"Current configuration appears optimal for {destination_storage}")
                st.write(f"**Efficiency Score:** {agent_config.get('optimal_configuration', {}).get('efficiency_score', 0):.1f}/100")
                st.write(f"**Management Complexity:** {agent_config.get('optimal_configuration', {}).get('management_complexity', 'Unknown')}")
                st.write(f"**Storage Optimization:** {agent_config.get('optimal_configuration', {}).get('storage_optimization', 'Unknown')}")
    
    # Throughput Analysis Chart
    st.markdown("**📊 Throughput Analysis with Storage Impact:**")
    
    throughput_data = {
        'Component': ['Per-Agent Base', 'Per-Agent with Storage', 'Total Agent Capacity', 'Network Limit', 'Effective Throughput'],
        'Throughput (Mbps)': [
            agent_config.get('max_throughput_mbps_per_agent', 0),
            agent_config.get('max_throughput_mbps_per_agent', 0) * agent_config.get('storage_performance_multiplier', 1.0),
            agent_analysis.get('total_max_throughput_mbps', 0),
            analysis.get('network_performance', {}).get('effective_bandwidth_mbps', 0),
            agent_analysis.get('total_effective_throughput', 0)
        ],
        'Type': ['Base', 'Enhanced', 'Aggregate', 'Network', 'Effective']
    }
    
    fig_throughput = px.bar(
        throughput_data, 
        x='Component', 
        y='Throughput (Mbps)',
        color='Type',
        title=f"Agent vs Network Throughput Analysis ({destination_storage} Destination)",
        color_discrete_map={'Base': '#95a5a6', 'Enhanced': '#3498db', 'Aggregate': '#2ecc71', 'Network': '#e74c3c', 'Effective': '#9b59b6'}
    )
    
    st.plotly_chart(fig_throughput, use_container_width=True)
    
    # Scaling Efficiency Analysis using native components
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📈 Scaling Efficiency Analysis:**")
        
        scaling_recommendations = agent_config.get('scaling_recommendations', [])
        
        with st.container():
            st.info("Scaling Assessment")
            st.write(f"**Current Efficiency:** {agent_analysis.get('scaling_efficiency', 1.0)*100:.1f}%")
            st.write(f"**Management Overhead:** {(agent_analysis.get('management_overhead', 1.0)-1)*100:.1f}%")
            st.write(f"**Storage Overhead:** {(agent_analysis.get('storage_management_overhead', 1.0)-1)*100:.1f}%")
            st.write(f"**Coordination Complexity:** {agent_config.get('optimal_configuration', {}).get('management_complexity', 'Unknown')}")
            
            if scaling_recommendations:
                st.write("**Key Recommendations:**")
                for rec in scaling_recommendations[:3]:
                    st.write(f"• {rec}")
    
    with col2:
        st.markdown("**🔍 Bottleneck Analysis:**")
        
        bottleneck = agent_analysis.get('bottleneck', 'unknown')
        bottleneck_severity = agent_analysis.get('bottleneck_severity', 'medium')
        
        with st.container():
            if bottleneck_severity == 'high':
                st.error("Performance Constraints")
            elif bottleneck_severity == 'medium':
                st.warning("Performance Constraints")
            else:
                st.success("Performance Constraints")
                
            st.write(f"**Primary Bottleneck:** {bottleneck.title()}")
            st.write(f"**Severity:** {bottleneck_severity.title()}")
            st.write(f"**Throughput Impact:** {agent_analysis.get('throughput_impact', 0)*100:.1f}%")
            st.write(f"**Storage Impact:** {destination_storage} provides {agent_analysis.get('storage_performance_multiplier', 1.0):.1f}x multiplier")
            
            recommendation = "Scale agents" if bottleneck == 'agents' else "Optimize network" if bottleneck == 'network' else "Monitor performance"
            st.write(f"**Recommendation:** {recommendation}")
    
    with col3:
        st.markdown("**💰 Cost Efficiency Analysis:**")
        
        cost_per_agent = agent_analysis.get('monthly_cost', 0) / config.get('number_of_agents', 1) if config.get('number_of_agents', 1) > 0 else 0
        cost_per_mbps = agent_analysis.get('monthly_cost', 0) / agent_analysis.get('total_effective_throughput', 1) if agent_analysis.get('total_effective_throughput', 0) > 0 else 0
        
        storage_cost_impact = {
            'S3': 1.0,
            'FSx_Windows': 1.1,
            'FSx_Lustre': 1.2
        }.get(destination_storage, 1.0)
        
        with st.container():
            st.info("Cost Efficiency Metrics")
            st.write(f"**Cost per Agent:** ${cost_per_agent:,.0f}/month")
            st.write(f"**Cost per Mbps:** ${cost_per_mbps:.2f}/month")
            st.write(f"**Total Monthly:** ${agent_analysis.get('monthly_cost', 0):,.0f}")
            st.write(f"**Storage Impact:** {storage_cost_impact:.1f}x base cost")
            st.write(f"**Efficiency Rating:** {agent_config.get('optimal_configuration', {}).get('cost_efficiency', 'Unknown')}")
            st.write(f"**Storage Optimization:** {agent_config.get('optimal_configuration', {}).get('storage_optimization', 'Standard')}")
    
    # AI Optimization Recommendations using expandable sections
    ai_optimization = agent_analysis.get('ai_optimization', {})
    optimization_recommendations = ai_optimization.get('recommendations', [])
    
    if optimization_recommendations:
        st.markdown("**🤖 AI Agent Optimization Recommendations:**")
        
        for i, recommendation in enumerate(optimization_recommendations, 1):
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            timeframe = "Immediate" if i <= 2 else "Short-term" if i <= 4 else "Long-term"
            
            # Determine if recommendation is storage-specific
            storage_specific = any(storage in recommendation.lower() for storage in ['s3', 'fsx', 'lustre', 'windows'])
            
            with st.expander(f"Optimization {i}: {recommendation}", expanded=(i <= 2)):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if complexity == "Low":
                        st.success(f"**Complexity:** {complexity}")
                    elif complexity == "Medium":
                        st.warning(f"**Complexity:** {complexity}")
                    else:
                        st.error(f"**Complexity:** {complexity}")
                
                with col2:
                    st.write(f"**Timeframe:** {timeframe}")
                
                with col3:
                    expected_benefit = ai_optimization.get('optimization_potential_percent', 0) // i
                    st.write(f"**Expected Benefit:** {expected_benefit}% improvement")
                
                with col4:
                    relevance = "High" if storage_specific else "General"
                    st.write(f"**Storage Relevance:** {relevance}")
                
                with col5:
                    destination_text = "Specific to " + destination_storage if storage_specific else "All destinations"
                    st.write(f"**Destination:** {destination_text}")


async def main():
    """Enhanced main function with agent positioning tab and comprehensive costs"""
    render_enhanced_header()
    
    # Get enhanced configuration
    config = render_enhanced_sidebar_controls()
    
    # Initialize enhanced analyzer
    analyzer = EnhancedMigrationAnalyzer()
    
    # Run analysis
    analysis_placeholder = st.empty()
    
    with analysis_placeholder.container():
        if config['enable_ai_analysis']:
            with st.spinner("🧠 Running comprehensive AI-powered migration analysis..."):
                try:
                    analysis = await analyzer.comprehensive_ai_migration_analysis(config)
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    analysis = create_fallback_analysis_with_proper_readers_writers(config)
        else:
            with st.spinner("🔬 Running standard migration analysis..."):
                analysis = create_fallback_analysis_with_proper_readers_writers(config)
    
    analysis_placeholder.empty()
    
    # Enhanced tabs with agent positioning and comprehensive costs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🧠 AI Insights & Analysis", 
        "🤖 Agent Scaling Analysis",
        "🏢 Agent Positioning",  # NEW TAB
        "🗄️ FSx Destination Comparison",
        "🌐 Network Intelligence",
        "💰 Comprehensive Costs",  # ENHANCED TAB
        "💻 OS Performance Analysis",
        "📊 Migration Dashboard",
        "🎯 AWS Sizing & Configuration",
        "📄 Executive PDF Reports"
    ])
    
    with tab1:
        if config['enable_ai_analysis']:
            render_ai_insights_tab_enhanced(analysis, config)
        else:
            st.info("🤖 Enable AI Analysis in the sidebar for comprehensive migration insights")
    
    with tab2:
        render_agent_scaling_tab(analysis, config)
    
    with tab3:  # NEW: Agent Positioning Tab
        render_agent_positioning_tab(analysis, config)
    
    with tab4:
        render_fsx_destination_comparison_tab(analysis, config)
    
    with tab5:
        render_network_intelligence_tab(analysis, config)
    
    with tab6:  # ENHANCED: Comprehensive Cost Analysis
        render_comprehensive_cost_pricing_tab(analysis, config)
    
    with tab7:
        render_os_performance_tab(analysis, config)
    
    with tab8:
        render_migration_dashboard_tab(analysis, config)
    
    with tab9:
        render_aws_sizing_tab(analysis, config)
    
    with tab10:
        render_pdf_reports_tab(analysis, config)
    
def render_pdf_reports_tab(analysis: Dict, config: Dict):
    """Render PDF reports generation tab with FSx destination support"""
    st.subheader("📄 Executive PDF Reports")
    
    # PDF Generation Section
    st.markdown(f"""
    <div class="pdf-section">
        <h3>🎯 Generate Executive Migration Report</h3>
        <p>Create a comprehensive PDF report for stakeholders with all analysis results, recommendations, technical details, agent scaling analysis, and FSx destination comparisons.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("📊 Generate Executive PDF Report", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating comprehensive PDF report with agent scaling analysis and FSx destination comparisons..."):
                try:
                    # Try to generate PDF report
                    try:
                        pdf_generator = PDFReportGenerator()
                        pdf_bytes = pdf_generator.generate_executive_report(analysis, config)
                        
                        # Create download button
                        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        destination = config.get('destination_storage_type', 'S3')
                        filename = f"AWS_Migration_Analysis_Agent_Scaling_FSx_{destination}_{current_time}.pdf"
                        
                        st.success("✅ PDF Report Generated Successfully!")
                        
                        st.download_button(
                            label="📥 Download Executive Report PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except ImportError:
                        st.warning("📋 PDF generation requires additional libraries. Showing text report instead.")
                        # Generate text report as fallback
                        text_report = generate_text_report(analysis, config)
                        
                        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"AWS_Migration_Analysis_{current_time}.txt"
                        
                        st.download_button(
                            label="📥 Download Text Report",
                            data=text_report,
                            file_name=filename,
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
                    st.info("💡 Report generation temporarily unavailable. Analysis data is still available in other tabs.")
    
    # Report Contents Preview
    st.markdown("**📋 Report Contents:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="professional-card">
            <h4>📈 Executive Summary</h4>
            <ul>
                <li>Migration overview and key metrics</li>
                <li>AI readiness assessment</li>
                <li>Cost summary and ROI analysis</li>
                <li>Performance baseline evaluation</li>
                <li>Agent scaling configuration</li>
                <li>FSx destination storage analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="professional-card">
            <h4>⚙️ Technical Analysis</h4>
            <ul>
                <li>Current performance breakdown</li>
                <li>Network path analysis</li>
                <li>OS performance impact</li>
                <li>Identified bottlenecks and solutions</li>
                <li>Agent throughput analysis</li>
                <li>Storage destination performance impact</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="professional-card">
            <h4>🤖 Agent Scaling Analysis</h4>
            <ul>
                <li>Current agent configuration assessment</li>
                <li>Scaling efficiency analysis</li>
                <li>Optimal configuration recommendations</li>
                <li>Cost efficiency metrics</li>
                <li>Bottleneck identification</li>
                <li>Storage destination impact on agents</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="professional-card">
            <h4>🗄️ FSx Destination Comparison</h4>
            <ul>
                <li>S3 vs FSx for Windows vs FSx for Lustre</li>
                <li>Performance comparison analysis</li>
                <li>Cost impact assessment</li>
                <li>Migration time variations</li>
                <li>Complexity and risk factors</li>
                <li>Destination-specific recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="professional-card">
            <h4>☁️ AWS Recommendations</h4>
            <ul>
                <li>Deployment recommendations (RDS vs EC2)</li>
                <li>Instance sizing and configuration</li>
                <li>Reader/writer setup ({analysis.get('aws_sizing_recommendations', {}).get('reader_writer_config', {}).get('writers', 1)} writers, {analysis.get('aws_sizing_recommendations', {}).get('reader_writer_config', {}).get('readers', 0)} readers)</li>
                <li>AI complexity analysis</li>
                <li>Agent impact on AWS sizing</li>
                <li>Storage destination optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="professional-card">
            <h4>📊 Cost & Risk Analysis</h4>
            <ul>
                <li>Detailed cost breakdown with agent costs</li>
                <li>FSx destination cost comparisons</li>
                <li>Financial projections and ROI</li>
                <li>Risk assessment matrix</li>
                <li>Recommended timeline and next steps</li>
                <li>Agent scaling cost optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Report Metrics
    st.markdown("**📊 Report Metrics:**")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("📄 Report Sections", "7", delta="Comprehensive analysis")
    
    with metrics_col2:
        recommendation_count = len(analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('performance_recommendations', []))
        st.metric("💡 AI Recommendations", str(max(recommendation_count, 5)), delta="Including FSx insights")
    
    with metrics_col3:
        complexity_score = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('ai_complexity_score', 6)
        st.metric("🎯 AI Complexity", f"{complexity_score}/10", delta="Migration difficulty")
    
    with metrics_col4:
        readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 75)
        st.metric("📈 Readiness Score", f"{readiness_score}/100", delta="Migration preparedness")

def generate_text_report(analysis: Dict, config: Dict) -> str:
    """Generate a comprehensive text report as fallback when PDF generation fails"""
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
AWS DATABASE MIGRATION ANALYSIS REPORT
Generated: {current_time}

================================================================================
EXECUTIVE SUMMARY
================================================================================

Migration Overview:
- Source Database: {config['source_database_engine'].upper()} ({config['database_size_gb']:,} GB)
- Target Database: AWS {config['database_engine'].upper()}
- Migration Type: {'Homogeneous' if config['source_database_engine'] == config['database_engine'] else 'Heterogeneous'}
- Environment: {config['environment'].title()}
- Destination Storage: {config.get('destination_storage_type', 'S3')}
- Migration Agents: {config.get('number_of_agents', 1)}

Key Metrics:
- Estimated Migration Time: {analysis.get('estimated_migration_time_hours', 0):.1f} hours
- Migration Readiness Score: {analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0):.0f}/100
- Risk Level: {analysis.get('ai_overall_assessment', {}).get('risk_level', 'Unknown')}
- Monthly AWS Cost: ${analysis.get('cost_analysis', {}).get('total_monthly_cost', 0):,.0f}
- ROI Timeline: {analysis.get('cost_analysis', {}).get('roi_months', 'TBD')} months

================================================================================
AWS SIZING RECOMMENDATIONS
================================================================================

Deployment Recommendation: {analysis.get('aws_sizing_recommendations', {}).get('deployment_recommendation', {}).get('recommendation', 'RDS').upper()}

Reader/Writer Configuration:
- Writer Instances: {analysis.get('aws_sizing_recommendations', {}).get('reader_writer_config', {}).get('writers', 1)}
- Reader Instances: {analysis.get('aws_sizing_recommendations', {}).get('reader_writer_config', {}).get('readers', 0)}
- Total Instances: {analysis.get('aws_sizing_recommendations', {}).get('reader_writer_config', {}).get('total_instances', 1)}
- Read Capacity: {analysis.get('aws_sizing_recommendations', {}).get('reader_writer_config', {}).get('read_capacity_percent', 0):.1f}%

================================================================================
AGENT SCALING ANALYSIS
================================================================================

Agent Configuration:
- Agent Type: {analysis.get('agent_analysis', {}).get('primary_tool', 'Unknown').upper()}
- Number of Agents: {analysis.get('agent_analysis', {}).get('number_of_agents', 1)}
- Scaling Efficiency: {analysis.get('agent_analysis', {}).get('scaling_efficiency', 1.0)*100:.1f}%
- Total Throughput: {analysis.get('agent_analysis', {}).get('total_effective_throughput', 0):,.0f} Mbps
- Monthly Agent Cost: ${analysis.get('agent_analysis', {}).get('monthly_cost', 0):,.0f}

================================================================================
DESTINATION STORAGE ANALYSIS
================================================================================

Selected Destination: {config.get('destination_storage_type', 'S3')}
Storage Performance Multiplier: {analysis.get('agent_analysis', {}).get('storage_performance_multiplier', 1.0):.1f}x
Monthly Storage Cost: ${analysis.get('cost_analysis', {}).get('destination_storage_cost', 0):,.0f}

================================================================================
COST ANALYSIS
================================================================================

Monthly Costs:
- AWS Compute: ${analysis.get('cost_analysis', {}).get('aws_compute_cost', 0):,.0f}
- AWS Storage: ${analysis.get('cost_analysis', {}).get('aws_storage_cost', 0):,.0f}
- Agent Costs: ${analysis.get('cost_analysis', {}).get('agent_cost', 0):,.0f}
- Destination Storage: ${analysis.get('cost_analysis', {}).get('destination_storage_cost', 0):,.0f}
- Network: ${analysis.get('cost_analysis', {}).get('network_cost', 0):,.0f}
- Total Monthly: ${analysis.get('cost_analysis', {}).get('total_monthly_cost', 0):,.0f}

One-time Costs:
- Migration Cost: ${analysis.get('cost_analysis', {}).get('one_time_migration_cost', 0):,.0f}

Financial Projections:
- Annual Cost: ${analysis.get('cost_analysis', {}).get('total_monthly_cost', 0) * 12:,.0f}
- Estimated Monthly Savings: ${analysis.get('cost_analysis', {}).get('estimated_monthly_savings', 0):,.0f}

================================================================================
RECOMMENDATIONS
================================================================================

Next Steps:
"""
    
    # Add recommendations
    next_steps = analysis.get('ai_overall_assessment', {}).get('recommended_next_steps', [])
    for i, step in enumerate(next_steps, 1):
        report += f"{i}. {step}\n"
    
    report += f"""

================================================================================
CONFIGURATION DETAILS
================================================================================

Operating System: {config['operating_system']}
Server Type: {config['server_type']}
Hardware: {config['cpu_cores']} cores, {config['ram_gb']} GB RAM
Database Size: {config['database_size_gb']:,} GB
Performance Requirements: {config['performance_requirements']}
Downtime Tolerance: {config['downtime_tolerance_minutes']} minutes

================================================================================
END OF REPORT
================================================================================
"""
    
    return report

# Fix 4: Update the fallback analysis to include proper reader/writer config
# In the EnhancedMigrationAnalyzer class, update the fallback analysis section:

def create_fallback_analysis_with_proper_readers_writers(config):
    """Create fallback analysis with proper reader/writer configuration and complete AWS sizing recommendations"""
    
    from datetime import datetime
    
    # Extract actual database engine for homogeneous check
    source_engine = config['source_database_engine']
    
    if config['database_engine'].startswith('rds_'):
        target_engine = config['database_engine'].replace('rds_', '')
    elif config['database_engine'].startswith('ec2_'):
        target_engine = config.get('ec2_database_engine', 'mysql')
    else:
        target_engine = config['database_engine']
    
    # Calculate readers based on database size for better defaults
    database_size_gb = config['database_size_gb']
    readers = 0
    
    if database_size_gb > 500:
        readers = 1
    if database_size_gb > 2000:
        readers = 2
    if database_size_gb > 10000:
        readers = 3
    if config.get('performance_requirements') == 'high':
        readers += 1
    if config.get('environment') == 'production':
        readers = max(readers, 2)
    
    # Ensure minimum sensible configuration
    if database_size_gb > 1000 and readers == 0:
        readers = 1
    
    writers = 1
    total_instances = writers + readers
    
    # FIXED: Create network performance analysis - DYNAMIC based on environment
    environment = config.get('environment', 'non-production')
    destination_storage = config.get('destination_storage_type', 'S3')
    
    # Determine bandwidth based on environment (matching actual network paths)
    if environment == 'production':
        # Production: Multi-hop high-bandwidth path (no bottleneck)
        effective_bandwidth = 10000  # Full 10Gbps throughput
        total_latency = 21  # SA → SJ → AWS
        total_reliability = 0.999 * 0.9995 * 0.9999  # Product of all segments
        network_quality_score = 90
        ai_enhanced_quality_score = 95
        cost_factor = 7.0  # Higher cost for production path
        segments = [
            {
                'name': 'San Antonio Linux NAS to Jump Server',
                'effective_bandwidth_mbps': 10000,
                'effective_latency_ms': 1,
                'reliability': 0.999,
                'connection_type': 'internal_lan',
                'cost_factor': 0.0
            },
            {
                'name': 'San Antonio to San Jose (Private Line)',
                'effective_bandwidth_mbps': 10000,
                'effective_latency_ms': 12,
                'reliability': 0.9995,
                'connection_type': 'private_line',
                'cost_factor': 3.0
            },
            {
                'name': f'San Jose to AWS {destination_storage} (DX)',
                'effective_bandwidth_mbps': 10000,
                'effective_latency_ms': 8,
                'reliability': 0.9999,
                'connection_type': 'direct_connect',
                'cost_factor': 4.0
            }
        ]
    else:
        # Non-production: DX connection bottleneck at 2Gbps
        effective_bandwidth = 2000  # Bottlenecked by DX connection
        total_latency = 17  # SJ local → AWS
        total_reliability = 0.999 * 0.998  # Product of segments
        network_quality_score = 80
        ai_enhanced_quality_score = 85
        cost_factor = 2.0  # Lower cost for non-prod
        segments = [
            {
                'name': 'San Jose Linux NAS to Jump Server',
                'effective_bandwidth_mbps': 10000,
                'effective_latency_ms': 2,
                'reliability': 0.999,
                'connection_type': 'internal_lan',
                'cost_factor': 0.0
            },
            {
                'name': f'San Jose to AWS {destination_storage} (DX)',
                'effective_bandwidth_mbps': 2000,  # This is the bottleneck!
                'effective_latency_ms': 15,
                'reliability': 0.998,
                'connection_type': 'direct_connect',
                'cost_factor': 2.0
            }
        ]
    
    # Apply destination storage bonuses
    storage_bonus = 0
    if destination_storage == 'FSx_Windows':
        storage_bonus = 10
        ai_enhanced_quality_score += storage_bonus
        # Adjust latency for FSx Windows (better performance)
        total_latency *= 0.9
        for segment in segments:
            if 'AWS' in segment['name']:
                segment['effective_latency_ms'] *= 0.9
    elif destination_storage == 'FSx_Lustre':
        storage_bonus = 20
        ai_enhanced_quality_score += storage_bonus
        # Adjust latency for FSx Lustre (much better performance)
        total_latency *= 0.7
        for segment in segments:
            if 'AWS' in segment['name']:
                segment['effective_latency_ms'] *= 0.7
    
    # Ensure quality scores don't exceed 100
    ai_enhanced_quality_score = min(100, ai_enhanced_quality_score)
    
    # Create network performance analysis - NOW FULLY DYNAMIC
    network_performance = {
        'path_name': f'{environment.title()}: Network path to AWS {destination_storage}',
        'destination_storage': destination_storage,
        'network_quality_score': network_quality_score,
        'ai_enhanced_quality_score': ai_enhanced_quality_score,
        'effective_bandwidth_mbps': effective_bandwidth,  # NOW DYNAMIC!
        'total_latency_ms': total_latency,
        'total_reliability': total_reliability,
        'total_cost_factor': cost_factor,
        'storage_performance_bonus': storage_bonus,
        'ai_optimization_potential': 15,
        'environment': environment,
        'os_type': 'linux' if any(os_name in config.get('operating_system', '') for os_name in ['linux', 'ubuntu', 'rhel']) else 'windows',
        'storage_type': 'nas',
        'segments': segments,
        'ai_insights': {
            'performance_bottlenecks': [f'{environment.title()} network path analysis'] + 
                                     (['DX connection bandwidth limit'] if environment != 'production' else ['No significant bottlenecks']),
            'optimization_opportunities': ['Network path optimization available', f'{destination_storage} performance tuning'],
            'recommended_improvements': [f'{environment.title()} network best practices', f'Optimize for {destination_storage}'],
            'risk_factors': ['No significant network risks identified']
        }
    }
    
    # Generate comprehensive AWS sizing recommendations
    def get_fallback_rds_sizing():
        # Determine instance size based on database size and requirements
        if database_size_gb < 1000:
            instance_type = 'db.t3.medium'
            vcpu = 2
            memory = 4
            cost_per_hour = 0.068
        elif database_size_gb < 5000:
            instance_type = 'db.r6g.large'
            vcpu = 2
            memory = 16
            cost_per_hour = 0.48
        elif database_size_gb < 20000:
            instance_type = 'db.r6g.xlarge'
            vcpu = 4
            memory = 32
            cost_per_hour = 0.96
        else:
            instance_type = 'db.r6g.2xlarge'
            vcpu = 8
            memory = 64
            cost_per_hour = 1.92
        
        # Storage sizing
        storage_size = max(database_size_gb * 1.5, 100)
        storage_type = 'gp3' if database_size_gb < 10000 else 'io1'
        storage_cost_per_gb = 0.08 if storage_type == 'gp3' else 0.125
        
        monthly_instance_cost = cost_per_hour * 24 * 30
        monthly_storage_cost = storage_size * storage_cost_per_gb
        
        return {
            'primary_instance': instance_type,
            'instance_specs': {
                'vcpu': vcpu,
                'memory': memory,
                'cost_per_hour': cost_per_hour
            },
            'storage_type': storage_type,
            'storage_size_gb': storage_size,
            'monthly_instance_cost': monthly_instance_cost,
            'monthly_storage_cost': monthly_storage_cost,
            'total_monthly_cost': monthly_instance_cost + monthly_storage_cost,
            'multi_az': config.get('environment') == 'production',
            'backup_retention_days': 30 if config.get('environment') == 'production' else 7,
            'ai_sizing_factors': {
                'complexity_multiplier': 1.0,
                'agent_scaling_factor': 1.0,
                'ai_complexity_score': 6,
                'storage_multiplier': 1.5
            }
        }
    
    def get_fallback_ec2_sizing():
        # Determine instance size based on database size and requirements
        if database_size_gb < 1000:
            instance_type = 't3.large'
            vcpu = 2
            memory = 8
            cost_per_hour = 0.0832
        elif database_size_gb < 5000:
            instance_type = 'r6i.large'
            vcpu = 2
            memory = 16
            cost_per_hour = 0.252
        elif database_size_gb < 20000:
            instance_type = 'r6i.xlarge'
            vcpu = 4
            memory = 32
            cost_per_hour = 0.504
        else:
            instance_type = 'r6i.2xlarge'
            vcpu = 8
            memory = 64
            cost_per_hour = 1.008
        
        # Storage sizing (EC2 needs more overhead)
        storage_size = max(database_size_gb * 2.0, 100)
        storage_type = 'gp3'
        storage_cost_per_gb = 0.08
        
        monthly_instance_cost = cost_per_hour * 24 * 30
        monthly_storage_cost = storage_size * storage_cost_per_gb
        
        return {
            'primary_instance': instance_type,
            'instance_specs': {
                'vcpu': vcpu,
                'memory': memory,
                'cost_per_hour': cost_per_hour
            },
            'storage_type': storage_type,
            'storage_size_gb': storage_size,
            'monthly_instance_cost': monthly_instance_cost,
            'monthly_storage_cost': monthly_storage_cost,
            'total_monthly_cost': monthly_instance_cost + monthly_storage_cost,
            'ebs_optimized': True,
            'enhanced_networking': True,
            'ai_sizing_factors': {
                'complexity_multiplier': 1.2,
                'agent_scaling_factor': 1.0,
                'ai_complexity_score': 6,
                'storage_multiplier': 2.0
            }
        }
    
    # Determine which deployment to recommend
    rds_score = 75
    ec2_score = 65
    
    # Adjust scores based on configuration
    if config.get('environment') == 'production':
        rds_score += 10
    if database_size_gb > 20000:
        ec2_score += 15
    if config.get('performance_requirements') == 'high':
        ec2_score += 10
    
    recommended_deployment = 'rds' if rds_score > ec2_score else 'ec2'
    confidence = abs(rds_score - ec2_score) / max(rds_score, ec2_score)
    
    # Get both RDS and EC2 sizing
    rds_sizing = get_fallback_rds_sizing()
    ec2_sizing = get_fallback_ec2_sizing()
    
    # Create reader/writer configuration
    reader_writer_config = {
        'writers': writers,
        'readers': readers,
        'total_instances': total_instances,
        'write_capacity_percent': (writers / total_instances) * 100 if total_instances > 0 else 100,
        'read_capacity_percent': (readers / total_instances) * 100 if total_instances > 0 else 0,
        'recommended_read_split': min(80, (readers / total_instances) * 100) if total_instances > 0 else 0,
        'reasoning': f"AI-optimized for {database_size_gb}GB, {config.get('performance_requirements', 'standard')} performance",
        'ai_insights': {
            'complexity_impact': 6,
            'agent_scaling_impact': config.get('number_of_agents', 1),
            'scaling_factors': [
                f"Database size drives {readers} reader replicas",
                f"Performance requirement: {config.get('performance_requirements', 'standard')}",
                f"Environment: {config.get('environment', 'non-production')} scaling applied"
            ]
        }
    }
    
    # Create AI analysis
    ai_analysis = {
        'ai_complexity_score': 6,
        'confidence_level': 'medium',
        'risk_factors': [
            'Standard migration complexity',
            'Database size may require careful planning',
            'Agent coordination needed'
        ],
        'risk_percentages': {
            'migration_risk': 15,
            'performance_risk': 10,
            'complexity_risk': 20
        },
        'mitigation_strategies': [
            'Implement comprehensive testing',
            'Plan adequate migration windows',
            'Configure proper monitoring'
        ],
        'performance_recommendations': [
            'Optimize database before migration',
            'Configure proper instance sizing',
            'Implement monitoring and alerting'
        ],
        'performance_improvements': {
            'overall_optimization': '15-25%',
            'instance_upgrade': '20-30%'
        },
        'timeline_suggestions': [
            'Phase 1: Assessment and Planning (2-3 weeks)',
            'Phase 2: Environment Setup (2-4 weeks)',
            'Phase 3: Testing and Validation (1-2 weeks)',
            'Phase 4: Migration Execution (1-3 days)',
            'Phase 5: Post-Migration Optimization (1 week)'
        ],
        'resource_allocation': {
            'migration_team_size': 3,
            'aws_specialists_needed': 1,
            'database_experts_required': 1,
            'testing_resources': 'Standard',
            'infrastructure_requirements': 'Standard setup'
        },
        'cost_optimization': [
            'Consider Reserved Instances for cost savings',
            'Implement auto-scaling policies',
            'Optimize storage configuration'
        ],
        'best_practices': [
            'Implement comprehensive backup strategy',
            'Use AWS Migration Hub for tracking',
            'Establish detailed communication plan'
        ],
        'testing_strategy': [
            'Unit Testing: Validate migration components',
            'Integration Testing: End-to-end workflow',
            'Performance Testing: AWS environment validation'
        ],
        'post_migration_monitoring': [
            'Implement CloudWatch monitoring',
            'Set up automated alerts',
            'Monitor application performance'
        ],
        'detailed_assessment': {
            'overall_readiness': 'needs_preparation',
            'success_probability': 80,
            'recommended_approach': 'staged_migration'
        }
    }
    
    # Create deployment recommendation
    deployment_recommendation = {
        'recommendation': recommended_deployment,
        'confidence': confidence,
        'rds_score': rds_score,
        'ec2_score': ec2_score,
        'primary_reasons': [
            f'Recommended for {database_size_gb}GB database',
            f'Suitable for {config.get("environment", "non-production")} environment',
            f'Matches {config.get("performance_requirements", "standard")} performance requirements'
        ]
    }
    
    # Create on-premises performance analysis
    onprem_performance = {
        'performance_score': 75,
        'overall_performance': {
            'cpu_score': 70,
            'memory_score': 75,
            'storage_score': 80,
            'network_score': 85,
            'database_score': 78,
            'composite_score': 75
        },
        'os_impact': {
            'name': config.get('operating_system', 'Unknown OS'),
            'total_efficiency': 0.85,
            'base_efficiency': 0.90,
            'cpu_efficiency': 0.88,
            'memory_efficiency': 0.85,
            'io_efficiency': 0.87,
            'network_efficiency': 0.90,
            'db_optimization': 0.85,
            'licensing_cost_factor': 1.5,
            'management_complexity': 0.6,
            'ai_insights': {
                'strengths': ['Good performance characteristics', 'Stable platform'],
                'weaknesses': ['Some optimization opportunities'],
                'migration_considerations': ['Standard migration approach']
            }
        },
        'bottlenecks': ['No major bottlenecks identified'],
        'ai_insights': ['System appears well-configured for migration']
    }
    
    # Create agent analysis
    is_homogeneous = source_engine == target_engine
    primary_tool = 'datasync' if is_homogeneous else 'dms'
    num_agents = config.get('number_of_agents', 1)
    
    # Calculate agent throughput
    base_throughput_per_agent = 500 if primary_tool == 'datasync' else 400
    storage_multiplier = {
        'S3': 1.0,
        'FSx_Windows': 1.15,
        'FSx_Lustre': 1.4
    }.get(destination_storage, 1.0)
    
    scaling_efficiency = min(1.0, 1.0 - (num_agents - 1) * 0.05)  # Diminishing returns
    total_throughput = base_throughput_per_agent * num_agents * scaling_efficiency * storage_multiplier
    
    agent_analysis = {
        'primary_tool': primary_tool,
        'agent_size': config.get('datasync_agent_size', config.get('dms_agent_size', 'medium')),
        'number_of_agents': num_agents,
        'destination_storage': destination_storage,
        'total_max_throughput_mbps': total_throughput,
        'total_effective_throughput': min(total_throughput, effective_bandwidth),  # Limited by network
        'scaling_efficiency': scaling_efficiency,
        'storage_performance_multiplier': storage_multiplier,
        'management_overhead': 1.0 + (num_agents - 1) * 0.05,
        'monthly_cost': num_agents * 200,  # Rough estimate
        'bottleneck': 'network' if total_throughput > effective_bandwidth else 'agents',
        'bottleneck_severity': 'medium',
        'agent_configuration': {
            'per_agent_spec': {
                'vcpu': 2,
                'memory_gb': 4
            },
            'max_throughput_mbps_per_agent': base_throughput_per_agent,
            'storage_performance_multiplier': storage_multiplier,
            'optimal_configuration': {
                'efficiency_score': 85,
                'management_complexity': 'Low' if num_agents <= 2 else 'Medium',
                'cost_efficiency': 'Good'
            }
        },
        'ai_optimization': {
            'optimization_potential_percent': 15,
            'recommendations': [
                'Optimize agent configuration for workload',
                'Implement proper load balancing',
                'Monitor agent performance'
            ]
        }
    }
    
    # Create FSx destination comparisons
    fsx_comparisons = {}
    
    for dest_type in ['S3', 'FSx_Windows', 'FSx_Lustre']:
        dest_storage_multiplier = {
            'S3': 1.0,
            'FSx_Windows': 1.15,
            'FSx_Lustre': 1.4
        }.get(dest_type, 1.0)
        
        dest_throughput = base_throughput_per_agent * num_agents * scaling_efficiency * dest_storage_multiplier
        dest_migration_time = (database_size_gb * 8 * 1000) / (min(dest_throughput, effective_bandwidth) * 3600)
        
        dest_storage_cost = database_size_gb * {
            'S3': 0.023,
            'FSx_Windows': 0.13,
            'FSx_Lustre': 0.14
        }.get(dest_type, 0.023)
        
        fsx_comparisons[dest_type] = {
            'destination_type': dest_type,
            'estimated_migration_time_hours': dest_migration_time,
            'migration_throughput_mbps': min(dest_throughput, effective_bandwidth),
            'estimated_monthly_storage_cost': dest_storage_cost,
            'performance_rating': {
                'S3': 'Good',
                'FSx_Windows': 'Very Good',
                'FSx_Lustre': 'Excellent'
            }.get(dest_type, 'Good'),
            'cost_rating': {
                'S3': 'Excellent',
                'FSx_Windows': 'Good',
                'FSx_Lustre': 'Fair'
            }.get(dest_type, 'Good'),
            'complexity_rating': {
                'S3': 'Low',
                'FSx_Windows': 'Medium',
                'FSx_Lustre': 'High'
            }.get(dest_type, 'Low'),
            'recommendations': [
                f'{dest_type} is suitable for this workload',
                f'Consider performance vs cost trade-offs',
                f'Validate {dest_type} integration requirements'
            ],
            'network_performance': network_performance,
            'agent_configuration': {
                'number_of_agents': num_agents,
                'total_monthly_cost': num_agents * 200,
                'storage_performance_multiplier': dest_storage_multiplier
            }
        }
    
    # Create cost analysis
    selected_sizing = rds_sizing if recommended_deployment == 'rds' else ec2_sizing
    
    cost_analysis = {
        'aws_compute_cost': selected_sizing['monthly_instance_cost'],
        'aws_storage_cost': selected_sizing['monthly_storage_cost'],
        'agent_cost': agent_analysis['monthly_cost'],
        'agent_base_cost': agent_analysis['monthly_cost'],
        'destination_storage_cost': fsx_comparisons[destination_storage]['estimated_monthly_storage_cost'],
        'destination_storage_type': destination_storage,
        'network_cost': 500,
        'os_licensing_cost': 300,
        'management_cost': 200,
        'total_monthly_cost': (selected_sizing['monthly_instance_cost'] + 
                             selected_sizing['monthly_storage_cost'] + 
                             agent_analysis['monthly_cost'] + 
                             fsx_comparisons[destination_storage]['estimated_monthly_storage_cost'] + 
                             500 + 300 + 200),
        'one_time_migration_cost': database_size_gb * 0.1 + num_agents * 500,
        'agent_setup_cost': num_agents * 500,
        'agent_coordination_cost': max(0, (num_agents - 1) * 200),
        'storage_setup_cost': {'S3': 100, 'FSx_Windows': 1000, 'FSx_Lustre': 2000}.get(destination_storage, 100),
        'estimated_monthly_savings': 500,
        'roi_months': 12,
        'ai_cost_insights': {
            'ai_optimization_factor': 0.1,
            'complexity_multiplier': 1.0,
            'management_reduction': 0.05,
            'agent_efficiency_bonus': 0.1,
            'storage_efficiency_bonus': 0.1,
            'potential_additional_savings': '10-15%'
        }
    }
    
    # Create AI overall assessment
    ai_overall_assessment = {
        'migration_readiness_score': 80,
        'success_probability': 85,
        'risk_level': 'Medium',
        'readiness_factors': [
            'System appears ready for migration',
            'Standard complexity migration',
            'Proper planning required'
        ],
        'ai_confidence': 0.8,
        'agent_scaling_impact': {
            'scaling_efficiency': scaling_efficiency * 100,
            'optimal_agents': 2,
            'current_agents': num_agents,
            'efficiency_bonus': 5
        },
        'destination_storage_impact': {
            'storage_type': destination_storage,
            'performance_bonus': fsx_comparisons[destination_storage]['performance_rating'],
            'storage_performance_multiplier': storage_multiplier
        },
        'recommended_next_steps': [
            'Conduct detailed performance baseline',
            'Set up AWS environment and testing',
            'Plan comprehensive testing strategy',
            'Develop detailed migration runbook'
        ],
        'timeline_recommendation': {
            'planning_phase_weeks': 2,
            'testing_phase_weeks': 3,
            'migration_window_hours': 24,
            'total_project_weeks': 6,
            'recommended_approach': 'staged'
        }
    }
    
    # Return complete analysis
    return {
        'api_status': {
            'anthropic_connected': False,
            'aws_pricing_connected': False,
            'last_update': datetime.now()
        },
        'onprem_performance': onprem_performance,
        'network_performance': network_performance,
        'migration_type': 'homogeneous' if is_homogeneous else 'heterogeneous',
        'primary_tool': primary_tool,
        'agent_analysis': agent_analysis,
        'migration_throughput_mbps': min(total_throughput, effective_bandwidth),
        'estimated_migration_time_hours': (database_size_gb * 8 * 1000) / (min(total_throughput, effective_bandwidth) * 3600),
        'aws_sizing_recommendations': {
            'rds_recommendations': rds_sizing,
            'ec2_recommendations': ec2_sizing,
            'reader_writer_config': reader_writer_config,
            'deployment_recommendation': deployment_recommendation,
            'ai_analysis': ai_analysis,
            'pricing_data': {
                'data_source': 'fallback',
                'last_updated': datetime.now(),
                'ec2_instances': {},
                'rds_instances': {},
                'storage': {}
            }
        },
        'cost_analysis': cost_analysis,
        'fsx_comparisons': fsx_comparisons,
        'ai_overall_assessment': ai_overall_assessment
    }

def render_ai_insights_tab_enhanced(analysis: Dict, config: Dict):
    """Render enhanced AI insights and analysis tab using native Streamlit components"""
    st.subheader("🧠 AI-Powered Migration Insights & Analysis")
    
    # Guard clause to handle None or invalid analysis
    if analysis is None:
        analysis = {}
    if config is None:
        config = {}
    
    # Ensure analysis is a dictionary
    if not isinstance(analysis, dict):
        st.error("Invalid analysis data provided. Please check your data source.")
        return
    
    ai_analysis = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {})
    ai_assessment = analysis.get('ai_overall_assessment', {})
    
    # AI Analysis Overview
    st.markdown("**🤖 AI Analysis Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        complexity_score = ai_analysis.get('ai_complexity_score', 6)
        st.metric(
            "🎯 AI Complexity Score",
            f"{complexity_score:.1f}/10",
            delta=ai_analysis.get('confidence_level', 'medium').title()
        )
    
    with col2:
        readiness_score = ai_assessment.get('migration_readiness_score', 0)
        st.metric(
            "📊 Migration Readiness",
            f"{readiness_score:.0f}/100",
            delta=ai_assessment.get('risk_level', 'Unknown')
        )
    
    with col3:
        success_probability = ai_assessment.get('success_probability', 0)
        st.metric(
            "🎯 Success Probability",
            f"{success_probability:.0f}%",
            delta=f"Confidence: {ai_analysis.get('confidence_level', 'medium').title()}"
        )
    
    with col4:
        num_agents = config.get('number_of_agents', 1)
        agent_efficiency = analysis.get('agent_analysis', {}).get('scaling_efficiency', 1.0)
        st.metric(
            "🤖 Agent Efficiency",
            f"{agent_efficiency*100:.1f}%",
            delta=f"{num_agents} agents"
        )
    
    with col5:
        destination_storage = config.get('destination_storage_type', 'S3')
        storage_multiplier = analysis.get('agent_analysis', {}).get('storage_performance_multiplier', 1.0)
        st.metric(
            "🗄️ Storage Performance",
            f"{storage_multiplier:.1f}x",
            delta=destination_storage
        )
    
    # AI Risk Assessment and Factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚠️ AI-Identified Risk Factors:**")
        
        risk_factors = ai_analysis.get('risk_factors', [])
        risk_percentages = ai_analysis.get('risk_percentages', {})
        
        with st.container():
            st.warning("Risk Assessment")
            
            if risk_factors:
                st.write("**Identified Risks:**")
                for i, risk in enumerate(risk_factors[:4], 1):
                    st.write(f"{i}. {risk}")
                
                if risk_percentages:
                    st.write("**Risk Probabilities:**")
                    for risk_type, percentage in list(risk_percentages.items())[:3]:
                        risk_name = risk_type.replace('_', ' ').title()
                        st.write(f"• {risk_name}: {percentage}%")
            else:
                st.write("No significant risks identified by AI analysis")
                st.write("Migration appears to be low-risk with current configuration")
    
    with col2:
        st.markdown("**🛡️ AI-Recommended Mitigation Strategies:**")
        
        mitigation_strategies = ai_analysis.get('mitigation_strategies', [])
        
        with st.container():
            st.success("Mitigation Recommendations")
            
            if mitigation_strategies:
                for i, strategy in enumerate(mitigation_strategies[:4], 1):
                    st.write(f"{i}. {strategy}")
            else:
                st.write("• Continue with standard migration best practices")
                st.write("• Implement comprehensive testing procedures")
                st.write("• Monitor performance throughout migration")
    
    # AI Performance Recommendations
    st.markdown("**🚀 AI Performance Optimization Recommendations:**")
    
    performance_recommendations = ai_analysis.get('performance_recommendations', [])
    performance_improvements = ai_analysis.get('performance_improvements', {})
    
    if performance_recommendations:
        for i, recommendation in enumerate(performance_recommendations, 1):
            impact = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            
            with st.expander(f"Recommendation {i}: {recommendation}", expanded=(i <= 2)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if impact == "High":
                        st.success(f"**Expected Impact:** {impact}")
                    elif impact == "Medium":
                        st.warning(f"**Expected Impact:** {impact}")
                    else:
                        st.info(f"**Expected Impact:** {impact}")
                
                with col2:
                    st.write(f"**Implementation Complexity:** {complexity}")
                
                with col3:
                    # Try to find corresponding improvement percentage
                    improvement_key = recommendation.lower().replace(' ', '_')[:20]
                    improvement = None
                    for key, value in performance_improvements.items():
                        if any(word in key.lower() for word in improvement_key.split('_')[:2]):
                            improvement = value
                            break
                    
                    if improvement:
                        st.write(f"**Expected Improvement:** {improvement}")
                    else:
                        expected_improvement = "15-25%" if impact == "High" else "5-15%" if impact == "Medium" else "2-10%"
                        st.write(f"**Expected Improvement:** {expected_improvement}")
    else:
        st.info("Current configuration appears well-optimized. Continue with standard best practices.")
    
    # AI Timeline and Resource Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 AI-Enhanced Timeline Suggestions:**")
        
        timeline_suggestions = ai_analysis.get('timeline_suggestions', [])
        
        with st.container():
            st.info("Project Timeline Recommendations")
            
            if timeline_suggestions:
                for suggestion in timeline_suggestions:
                    st.write(f"• {suggestion}")
            else:
                st.write("• Phase 1: Assessment and Planning (2-3 weeks)")
                st.write("• Phase 2: Environment Setup (2-4 weeks)")
                st.write("• Phase 3: Testing and Validation (1-2 weeks)")
                st.write("• Phase 4: Migration Execution (1-3 days)")
                st.write("• Phase 5: Post-Migration Optimization (1 week)")
            
            # Add agent-specific timeline considerations
            num_agents = config.get('number_of_agents', 1)
            if num_agents > 1:
                st.write(f"• Agent Coordination Setup: +{num_agents} hours for {num_agents} agents")
            
            destination_storage = config.get('destination_storage_type', 'S3')
            if destination_storage != 'S3':
                setup_time = {"FSx_Windows": "4-8 hours", "FSx_Lustre": "8-16 hours"}.get(destination_storage, "2-4 hours")
                st.write(f"• {destination_storage} Setup: {setup_time}")
    
    with col2:
        st.markdown("**👥 AI-Recommended Resource Allocation:**")
        
        resource_allocation = ai_analysis.get('resource_allocation', {})
        
        with st.container():
            st.success("Team & Resource Requirements")
            
            if resource_allocation:
                st.write(f"**Migration Team Size:** {resource_allocation.get('migration_team_size', 3)} people")
                st.write(f"**AWS Specialists:** {resource_allocation.get('aws_specialists_needed', 1)} required")
                st.write(f"**Database Experts:** {resource_allocation.get('database_experts_required', 1)} required")
                st.write(f"**Testing Resources:** {resource_allocation.get('testing_resources', 'Standard')}")
                st.write(f"**Infrastructure:** {resource_allocation.get('infrastructure_requirements', 'Standard setup')}")
                
                # Agent-specific resources
                if resource_allocation.get('agent_management_overhead'):
                    st.write(f"**Agent Management:** {resource_allocation.get('agent_management_overhead')}")
                
                # Storage-specific resources
                if resource_allocation.get('storage_specialists', 0) > 0:
                    st.write(f"**Storage Specialists:** {resource_allocation.get('storage_specialists')} required")
            else:
                st.write("**Migration Team Size:** 3-4 people")
                st.write("**AWS Specialists:** 1 required")
                st.write("**Database Experts:** 1-2 required")
                st.write("**Testing Resources:** 2 weeks dedicated testing")
    
    # AI Best Practices and Testing Strategy
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 AI-Recommended Best Practices:**")
        
        best_practices = ai_analysis.get('best_practices', [])
        
        with st.container():
            st.success("Migration Best Practices")
            
            if best_practices:
                for practice in best_practices[:5]:
                    st.write(f"• {practice}")
            else:
                st.write("• Implement comprehensive backup strategy")
                st.write("• Use AWS Migration Hub for tracking")
                st.write("• Establish detailed communication plan")
                st.write("• Create detailed runbook procedures")
                st.write("• Implement automated testing pipelines")
    
    with col2:
        st.markdown("**🧪 AI-Enhanced Testing Strategy:**")
        
        testing_strategy = ai_analysis.get('testing_strategy', [])
        
        with st.container():
            st.info("Testing & Validation Strategy")
            
            if testing_strategy:
                for test in testing_strategy[:5]:
                    st.write(f"• {test}")
            else:
                st.write("• Unit Testing: Validate migration components")
                st.write("• Integration Testing: End-to-end workflow")
                st.write("• Performance Testing: AWS environment validation")
                st.write("• Data Integrity Testing: Consistency verification")
                st.write("• Security Testing: Access controls and encryption")
    
    # AI Post-Migration Monitoring
    st.markdown("**📊 AI Post-Migration Monitoring Recommendations:**")
    
    monitoring_recommendations = ai_analysis.get('post_migration_monitoring', [])
    
    if monitoring_recommendations:
        monitor_col1, monitor_col2, monitor_col3 = st.columns(3)
        
        # Split recommendations into columns
        rec_per_col = len(monitoring_recommendations) // 3
        
        with monitor_col1:
            st.success("**Performance Monitoring**")
            for rec in monitoring_recommendations[:rec_per_col]:
                st.write(f"• {rec}")
        
        with monitor_col2:
            st.info("**Operational Monitoring**")
            for rec in monitoring_recommendations[rec_per_col:rec_per_col*2]:
                st.write(f"• {rec}")
        
        with monitor_col3:
            st.warning("**Cost & Optimization Monitoring**")
            for rec in monitoring_recommendations[rec_per_col*2:]:
                st.write(f"• {rec}")
    else:
        st.info("Standard CloudWatch monitoring recommended for all database and application metrics")
    
    # AI Overall Assessment Summary
    st.markdown("**🎯 AI Overall Assessment Summary:**")
    
    detailed_assessment = ai_analysis.get('detailed_assessment', {})
    
    if detailed_assessment:
        assess_col1, assess_col2, assess_col3 = st.columns(3)
        
        with assess_col1:
            overall_readiness = detailed_assessment.get('overall_readiness', 'needs_preparation')
            if overall_readiness == 'ready':
                st.success("**Migration Readiness: READY**")
            elif overall_readiness == 'needs_preparation':
                st.warning("**Migration Readiness: NEEDS PREPARATION**")
            else:
                st.error("**Migration Readiness: SIGNIFICANT PREPARATION REQUIRED**")
            
            st.write(f"**Success Probability:** {detailed_assessment.get('success_probability', 75)}%")
            st.write(f"**Recommended Approach:** {detailed_assessment.get('recommended_approach', 'staged_migration').replace('_', ' ').title()}")
        
        with assess_col2:
            st.info("**Critical Success Factors**")
            critical_factors = detailed_assessment.get('critical_success_factors', [])
            for factor in critical_factors[:4]:
                st.write(f"• {factor}")
        
        with assess_col3:
            st.warning("**Key Action Items**")
            next_steps = ai_assessment.get('recommended_next_steps', [])
            for step in next_steps[:4]:
                st.write(f"• {step}")
    
    # Raw AI Response (for debugging/transparency)
    with st.expander("🔍 Raw AI Analysis Response", expanded=False):
        raw_response = ai_analysis.get('raw_ai_response', 'No raw AI response available')
        st.text_area("AI Analysis Details", raw_response, height=200, help="Complete AI analysis response for technical review")



def render_network_intelligence_tab(analysis: Dict, config: Dict):
    """Render network intelligence analysis tab with AI insights using native components"""
    st.subheader("🌐 Network Intelligence & Path Optimization")
    
    network_perf = analysis.get('network_performance', {})
    
    # Network Overview Dashboard
    st.markdown("**📊 Network Performance Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🎯 Network Quality",
            f"{network_perf.get('network_quality_score', 0):.1f}/100",
            delta=f"AI Enhanced: {network_perf.get('ai_enhanced_quality_score', 0):.1f}"
        )
    
    # FIND THIS SECTION in render_network_intelligence_tab() and REPLACE the bandwidth metric:
    with col2:
        st.metric(
            "🌐 Network Capacity",  # Changed from "Effective Bandwidth"
            f"{network_perf.get('effective_bandwidth_mbps', 0):,.0f} Mbps",
            delta="Raw network limit (not migration speed)"  # Added context
        )
    
    with col3:
        st.metric(
            "🕐 Total Latency",
            f"{network_perf.get('total_latency_ms', 0):.1f} ms",
            delta=f"Reliability: {network_perf.get('total_reliability', 0)*100:.2f}%"
        )
    
    with col4:
        st.metric(
            "🗄️ Destination Storage",
            network_perf.get('destination_storage', 'S3'),
            delta=f"Bonus: +{network_perf.get('storage_performance_bonus', 0)}%"
        )
    
    with col5:
        ai_optimization = network_perf.get('ai_optimization_potential', 0)
        st.metric(
            "🤖 AI Optimization",
            f"{ai_optimization:.1f}%",
            delta="Improvement potential"
        )
    
    # ADD THIS AFTER THE EXISTING 5-COLUMN METRICS SECTION in render_network_intelligence_tab()

    # Add bandwidth waterfall analysis
    st.markdown("---")  # Add separator
    render_bandwidth_waterfall_analysis(analysis, config)

    st.markdown("---")  # Add separator  
    render_performance_impact_table(analysis, config)
    
    
    # Network Path Visualization
    st.markdown("**🗺️ Network Path Visualization:**")
    
    if network_perf.get('segments'):
        # Create network path diagram
        try:
            network_diagram = create_network_path_diagram(network_perf)
            st.plotly_chart(network_diagram, use_container_width=True)
        except Exception as e:
            st.warning(f"Network diagram could not be rendered: {str(e)}")
            
            # Fallback: Show path as table
            segments_data = []
            for i, segment in enumerate(network_perf.get('segments', []), 1):
                segments_data.append({
                    'Hop': i,
                    'Segment': segment['name'],
                    'Type': segment['connection_type'].replace('_', ' ').title(),
                    'Bandwidth (Mbps)': f"{segment.get('effective_bandwidth_mbps', 0):,.0f}",
                    'Latency (ms)': f"{segment.get('effective_latency_ms', 0):.1f}",
                    'Reliability': f"{segment['reliability']*100:.3f}%",
                    'Cost Factor': f"{segment['cost_factor']:.1f}x"
                })
            
            df_segments = pd.DataFrame(segments_data)
            st.dataframe(df_segments, use_container_width=True)
    else:
        st.info("Network path data not available in current analysis")
    
    # Detailed Network Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Network Performance Analysis:**")
        
        with st.container():
            st.info("Performance Metrics")
            st.write(f"**Path Name:** {network_perf.get('path_name', 'Unknown')}")
            st.write(f"**Environment:** {network_perf.get('environment', 'Unknown').title()}")
            st.write(f"**OS Type:** {network_perf.get('os_type', 'Unknown').title()}")
            st.write(f"**Storage Type:** {network_perf.get('storage_type', 'Unknown').title()}")
            st.write(f"**Destination:** {network_perf.get('destination_storage', 'S3')}")
            st.write(f"**Network Quality Score:** {network_perf.get('network_quality_score', 0):.1f}/100")
            st.write(f"**AI Enhanced Score:** {network_perf.get('ai_enhanced_quality_score', 0):.1f}/100")
            st.write(f"**Cost Factor:** {network_perf.get('total_cost_factor', 0):.1f}x")
    
    with col2:
        st.markdown("**🤖 AI Network Insights:**")
        
        ai_insights = network_perf.get('ai_insights', {})
        
        with st.container():
            st.success("AI Analysis & Recommendations")
            
            st.write("**Performance Bottlenecks:**")
            bottlenecks = ai_insights.get('performance_bottlenecks', ['No bottlenecks identified'])
            for bottleneck in bottlenecks[:3]:
                st.write(f"• {bottleneck}")
            
            st.write("**Optimization Opportunities:**")
            opportunities = ai_insights.get('optimization_opportunities', ['Standard optimization'])
            for opportunity in opportunities[:3]:
                st.write(f"• {opportunity}")
            
            st.write("**Risk Factors:**")
            risks = ai_insights.get('risk_factors', ['No significant risks'])
            for risk in risks[:2]:
                st.write(f"• {risk}")
    
    # Network Optimization Recommendations
    st.markdown("**💡 Network Optimization Recommendations:**")
    
    recommendations = ai_insights.get('recommended_improvements', [])
    if recommendations:
        for i, recommendation in enumerate(recommendations, 1):
            impact = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
            complexity = "Low" if i <= 2 else "Medium" if i <= 4 else "High"
            priority = "Immediate" if i <= 2 else "Short-term" if i <= 4 else "Long-term"
            
            with st.expander(f"Recommendation {i}: {recommendation}", expanded=(i <= 2)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if impact == "High":
                        st.success(f"**Expected Impact:** {impact}")
                    elif impact == "Medium":
                        st.warning(f"**Expected Impact:** {impact}")
                    else:
                        st.info(f"**Expected Impact:** {impact}")
                
                with col2:
                    st.write(f"**Implementation Complexity:** {complexity}")
                
                with col3:
                    st.write(f"**Priority:** {priority}")
    else:
        st.info("Network appears optimally configured for current requirements")


# Fix for the PyArrow serialization error in render_cost_pricing_tab function

def render_agent_positioning_tab(analysis: Dict, config: Dict):
    """Render agent positioning and data center architecture tab"""
    st.subheader("🏢 Migration Agent Positioning & Data Center Architecture")
    
    # Determine agent positioning based on OS and database type
    os_type = config.get('operating_system', '')
    source_engine = config.get('source_database_engine', '')
    environment = config.get('environment', 'non-production')
    
    # Agent positioning logic
    if 'linux' in os_type.lower() or any(linux_os in os_type for linux_os in ['ubuntu', 'rhel']):
        primary_storage_location = "Linux NAS Server"
        storage_protocol = "NFS/CIFS"
        agent_placement = "Near Linux NAS"
        backup_path = "Database → Linux NAS → Migration Agent → AWS S3"
    else:  # Windows
        primary_storage_location = "Windows Shared Drive (SMB)"
        storage_protocol = "SMB/CIFS"
        agent_placement = "Near Windows File Server"
        backup_path = "Database → Windows SMB Share → Migration Agent → AWS S3"
    
    # Migration agent details
    is_homogeneous = config.get('source_database_engine') == config.get('ec2_database_engine', config.get('database_engine', '').replace('rds_', ''))
    agent_type = "DataSync" if is_homogeneous else "DMS"
    num_agents = config.get('number_of_agents', 1)
    
    # Data Center Overview
    st.markdown("**🏗️ Data Center Architecture Overview:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🖥️ Database Server",
            f"{source_engine.upper()}",
            delta=f"OS: {os_type.replace('_', ' ').title()}"
        )
    
    with col2:
        st.metric(
            "🗄️ Storage Location",
            primary_storage_location,
            delta=f"Protocol: {storage_protocol}"
        )
    
    with col3:
        st.metric(
            "🤖 Agent Placement",
            agent_placement,
            delta=f"{num_agents}x {agent_type} agents"
        )
    
    with col4:
        destination = config.get('destination_storage_type', 'S3')
        st.metric(
            "☁️ AWS Destination",
            f"AWS {destination}",
            delta=f"Environment: {environment.title()}"
        )
    
    # Visual Architecture Diagram
    st.markdown("**🗺️ Migration Architecture Flow:**")
    
    # Create architecture flow diagram
    fig = go.Figure()
    
    # Define positions for architecture components
    components = [
        {"name": f"{source_engine.upper()}\nDatabase Server", "x": 1, "y": 3, "color": "#e74c3c"},
        {"name": f"{primary_storage_location}", "x": 2, "y": 3, "color": "#f39c12"},
        {"name": f"{num_agents}x {agent_type}\nMigration Agents", "x": 3, "y": 3, "color": "#3498db"},
        {"name": "Data Center\nNetwork", "x": 4, "y": 3, "color": "#9b59b6"},
        {"name": f"AWS {destination}\nDestination", "x": 5, "y": 3, "color": "#27ae60"}
    ]
    
    # Add components
    for i, comp in enumerate(components):
        fig.add_trace(go.Scatter(
            x=[comp["x"]],
            y=[comp["y"]],
            mode='markers+text',
            marker=dict(size=60, color=comp["color"]),
            text=[comp["name"]],
            textposition='middle center',
            textfont=dict(color='white', size=10),
            name=comp["name"],
            showlegend=False
        ))
        
        # Add arrows between components
        if i < len(components) - 1:
            fig.add_annotation(
                x=comp["x"] + 0.4,
                y=comp["y"],
                ax=comp["x"] + 0.1,
                ay=comp["y"],
                xref="x", yref="y",
                axref="x", ayref="y",
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="#2c3e50"
            )
    
    fig.update_layout(
        title="Migration Data Flow Architecture",
        xaxis=dict(range=[0.5, 5.5], showgrid=False, showticklabels=False),
        yaxis=dict(range=[2.5, 3.5], showgrid=False, showticklabels=False),
        height=300,
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Agent Positioning Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📍 Agent Positioning Details:**")
        
        with st.container():
            st.success("Migration Agent Configuration")
            st.write(f"**Agent Type:** {agent_type}")
            st.write(f"**Number of Agents:** {num_agents}")
            st.write(f"**Placement Strategy:** {agent_placement}")
            st.write(f"**Storage Access:** {storage_protocol}")
            st.write(f"**Network Path:** {environment.title()} network")
            
            if source_engine == "sqlserver":
                st.write(f"**SQL Server Specific:** Custom backup to {primary_storage_location}")
                st.write(f"**Backup Method:** SQL Server native backup + {agent_type}")
            
            st.write(f"**Data Flow:** {backup_path}")
    
    with col2:
        st.markdown("**🔧 Technical Implementation:**")
        
        with st.container():
            st.info("Implementation Requirements")
            
            if 'linux' in os_type.lower():
                st.write("**Linux NAS Configuration:**")
                st.write("• NFS exports configured for agent access")
                st.write("• CIFS/SMB shares for Windows compatibility")
                st.write("• Proper permissions and security")
                st.write("• Network connectivity to agents")
            else:
                st.write("**Windows SMB Share Configuration:**")
                st.write("• SMB shares accessible to migration agents")
                st.write("• Proper Active Directory permissions")
                st.write("• File system permissions for agent service account")
                st.write("• Network connectivity and firewall rules")
            
            st.write(f"**Agent Requirements:**")
            st.write(f"• {num_agents}x {agent_type} agent(s) deployment")
            st.write(f"• Network access to {primary_storage_location}")
            st.write(f"• AWS connectivity for {destination} uploads")
    
    # Network Connectivity Analysis
    st.markdown("**🌐 Network Connectivity Analysis:**")
    
    network_perf = analysis.get('network_performance', {})
    
    # Create network path table
    connectivity_data = []
    
    # Database to Storage
    connectivity_data.append({
        "Connection": f"{source_engine.upper()} → {primary_storage_location}",
        "Protocol": storage_protocol,
        "Bandwidth": "Local LAN (1-10 Gbps)",
        "Latency": "< 1ms",
        "Reliability": "99.9%",
        "Security": "Internal network"
    })
    
    # Storage to Agents
    connectivity_data.append({
        "Connection": f"{primary_storage_location} → {agent_type} Agents",
        "Protocol": f"{storage_protocol} access",
        "Bandwidth": f"Shared with {num_agents} agents",
        "Latency": "< 2ms",
        "Reliability": "99.9%",
        "Security": "Service account authentication"
    })
    
    # Agents to AWS
    aws_bandwidth = network_perf.get('effective_bandwidth_mbps', 2000)
    connectivity_data.append({
        "Connection": f"{agent_type} Agents → AWS {config.get('destination_storage_type', 'S3')}",
        "Protocol": "HTTPS/TLS",
        "Bandwidth": f"{aws_bandwidth:,.0f} Mbps",
        "Latency": f"{network_perf.get('total_latency_ms', 15):.1f}ms",
        "Reliability": f"{network_perf.get('total_reliability', 0.998)*100:.1f}%",
        "Security": "AWS IAM + TLS encryption"
    })
    
    df_connectivity = pd.DataFrame(connectivity_data)
    st.dataframe(df_connectivity, use_container_width=True)
    
    # Agent Placement Recommendations
    st.markdown("**💡 Agent Placement Recommendations:**")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.success("✅ **Optimal Placement**")
        st.write(f"• Place agents close to {primary_storage_location}")
        st.write("• Minimize network hops to storage")
        st.write("• Ensure high-bandwidth connectivity")
        st.write("• Consider storage I/O capacity")
        
    with rec_col2:
        st.warning("⚠️ **Considerations**")
        st.write("• Agent compute resources requirements")
        st.write("• Network bandwidth sharing")
        st.write("• Storage concurrent access limits")
        st.write(f"• {num_agents} agents coordination overhead")
        
    with rec_col3:
        st.info("📋 **Implementation Steps**")
        st.write("1. Prepare storage access permissions")
        st.write(f"2. Deploy {num_agents}x {agent_type} agents")
        st.write("3. Configure AWS connectivity")
        st.write("4. Test end-to-end data flow")
        st.write("5. Monitor performance and optimize")
    
    # Special SQL Server Considerations
    if source_engine == "sqlserver":
        st.markdown("**🪟 SQL Server Specific Considerations:**")
        
        with st.expander("SQL Server Migration Architecture", expanded=True):
            sql_col1, sql_col2 = st.columns(2)
            
            with sql_col1:
                st.warning("**SQL Server Backup Strategy**")
                st.write("• Full database backup to shared storage")
                st.write("• Transaction log backups for point-in-time recovery")
                st.write("• Backup compression to reduce transfer time")
                st.write("• Backup verification before migration")
                st.write("• Consider Always On Availability Groups")
                
            with sql_col2:
                st.info("**Migration Process Flow**")
                st.write("1. SQL Server backup to SMB share")
                st.write("2. DMS agents access backup files")
                st.write("3. Schema conversion (if heterogeneous)")
                st.write("4. Data transfer to AWS")
                st.write("5. Target SQL Server on EC2 restore")
                st.write("6. Application cutover and testing")


# Fixed render_os_performance_tab function
def render_os_performance_tab(analysis: Dict, config: Dict):
    """Render OS performance analysis tab using native Streamlit components"""
    st.subheader("💻 Operating System Performance Analysis")
    
    os_impact = analysis.get('onprem_performance', {}).get('os_impact', {})
    
    # OS Overview
    st.markdown("**🖥️ Operating System Overview:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💻 Current OS",
            os_impact.get('name', 'Unknown'),
            delta=f"Platform: {config.get('server_type', 'Unknown').title()}"
        )
    
    with col2:
        st.metric(
            "⚡ Total Efficiency",
            f"{os_impact.get('total_efficiency', 0)*100:.1f}%",
            delta=f"Base: {os_impact.get('base_efficiency', 0)*100:.1f}%"
        )
    
    with col3:
        st.metric(
            "🗄️ DB Optimization",
            f"{os_impact.get('db_optimization', 0)*100:.1f}%",
            delta=f"Engine: {os_impact.get('actual_database_engine', 'Unknown').upper()}"
        )
    
    with col4:
        st.metric(
            "💰 Licensing Factor",
            f"{os_impact.get('licensing_cost_factor', 1.0):.1f}x",
            delta=f"Complexity: {os_impact.get('management_complexity', 0)*100:.0f}%"
        )
    
    with col5:
        virt_overhead = os_impact.get('virtualization_overhead', 0)
        st.metric(
            "☁️ Virtualization",
            f"{virt_overhead*100:.1f}%" if config.get('server_type') == 'vmware' else "N/A",
            delta="Overhead" if config.get('server_type') == 'vmware' else "Physical"
        )
    
    # OS Performance Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 OS Performance Metrics:**")
        
        # Create radar chart for OS performance
        performance_metrics = {
            'CPU Efficiency': os_impact.get('cpu_efficiency', 0) * 100,
            'Memory Efficiency': os_impact.get('memory_efficiency', 0) * 100,
            'I/O Efficiency': os_impact.get('io_efficiency', 0) * 100,
            'Network Efficiency': os_impact.get('network_efficiency', 0) * 100,
            'DB Optimization': os_impact.get('db_optimization', 0) * 100
        }
        
        fig_radar = go.Figure()
        
        categories = list(performance_metrics.keys())
        values = list(performance_metrics.values())
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name='OS Performance',
            line_color='#3498db'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="OS Performance Profile"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.markdown("**🤖 AI OS Insights:**")
        
        ai_insights = os_impact.get('ai_insights', {})
        
        with st.container():
            st.success(f"AI Analysis of {os_impact.get('name', 'Current OS')}")
            
            st.write("**Strengths:**")
            for strength in ai_insights.get('strengths', ['General purpose OS'])[:3]:
                st.write(f"• {strength}")
            
            st.write("**Weaknesses:**")
            for weakness in ai_insights.get('weaknesses', ['No significant issues'])[:3]:
                st.write(f"• {weakness}")
            
            st.write("**Migration Considerations:**")
            for consideration in ai_insights.get('migration_considerations', ['Standard migration'])[:3]:
                st.write(f"• {consideration}")
    
    # Database Engine Optimization
    st.markdown("**🗄️ Database Engine Optimization Analysis:**")
    
    database_optimizations = os_impact.get('database_optimizations', {})
    
    if database_optimizations:
        opt_data = []
        # Get the actual database engine being used
        current_engine = os_impact.get('actual_database_engine', 'mysql')
        
        for engine, optimization in database_optimizations.items():
            performance_rating = ('Excellent' if optimization > 0.95 else 
                                'Very Good' if optimization > 0.90 else 
                                'Good' if optimization > 0.85 else 'Fair')
            
            opt_data.append({
                'Database Engine': engine.upper(),
                'Optimization Score': f"{optimization*100:.1f}%",
                'Performance Rating': performance_rating,
                'Current Selection': '✅' if engine == current_engine else ''
            })
        
        df_opt = pd.DataFrame(opt_data)
        st.dataframe(df_opt, use_container_width=True)
    
    # OS Comparison Analysis
    st.markdown("**⚖️ OS Comparison Analysis:**")
    
    # Create comparison with other OS options
    os_manager = OSPerformanceManager()
    comparison_data = []
    
    current_os = config.get('operating_system')
    current_platform = config.get('server_type')
    current_db_engine = config.get('database_engine')
    ec2_db_engine = config.get('ec2_database_engine')
    
    for os_name, os_config in os_manager.operating_systems.items():
        # FIXED: Pass the ec2_database_engine parameter
        os_perf = os_manager.calculate_os_performance_impact(
            os_name, 
            current_platform, 
            current_db_engine,
            ec2_db_engine
        )
        
        comparison_data.append({
            'Operating System': os_config['name'],
            'Total Efficiency': f"{os_perf['total_efficiency']*100:.1f}%",
            'CPU Efficiency': f"{os_perf['cpu_efficiency']*100:.1f}%",
            'Memory Efficiency': f"{os_perf['memory_efficiency']*100:.1f}%",
            'I/O Efficiency': f"{os_perf['io_efficiency']*100:.1f}%",
            'Licensing Cost': f"{os_perf['licensing_cost_factor']:.1f}x",
            'Current Choice': '✅' if os_name == current_os else ''
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Performance Impact Analysis using native components
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🎯 **Performance Impact Assessment**")
        st.write(f"**Base Efficiency:** {os_impact.get('base_efficiency', 0)*100:.1f}%")
        st.write(f"**Database Optimization:** {os_impact.get('db_optimization', 0)*100:.1f}%")
        st.write(f"**Platform Optimization:** {os_impact.get('platform_optimization', 1.0)*100:.1f}%")
        improvement = ((os_impact.get('total_efficiency', 0) - 0.8) / 0.2 * 100)
        st.write(f"**Overall Impact:** {improvement:.1f}% above baseline")
    
    with col2:
        st.warning("💰 **Cost Impact Analysis**")
        st.write(f"**Licensing Cost Factor:** {os_impact.get('licensing_cost_factor', 1.0):.1f}x")
        monthly_licensing = os_impact.get('licensing_cost_factor', 1.0) * 150
        st.write(f"**Monthly Licensing Est.:** ${monthly_licensing:.0f}")
        st.write(f"**Management Complexity:** {os_impact.get('management_complexity', 0)*100:.0f}%")
        st.write(f"**Security Overhead:** {os_impact.get('security_overhead', 0)*100:.1f}%")
    
    with col3:
        suitability = ("Excellent" if os_impact.get('total_efficiency', 0) > 0.9 else 
                      "Good" if os_impact.get('total_efficiency', 0) > 0.8 else "Fair")
        
        migration_complexity = ("Low" if 'windows' in current_os and 'windows' in current_os else "Medium")
        
        recommendation = ("Keep current OS" if os_impact.get('total_efficiency', 0) > 0.85 else 
                         "Consider optimization")
        
        st.success("🔧 **Migration Recommendations**")
        st.write(f"**Current OS Suitability:** {suitability}")
        st.write(f"**Migration Complexity:** {migration_complexity}")
        st.write(f"**Recommended Action:** {recommendation}")


def render_migration_dashboard_tab(analysis: Dict, config: Dict):
    """Render comprehensive migration dashboard with key metrics and visualizations"""
    st.subheader("📊 Enhanced Migration Performance Dashboard")
    
    # Executive Summary Dashboard
    st.markdown("**🎯 Executive Migration Summary:**")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        readiness_score = analysis.get('ai_overall_assessment', {}).get('migration_readiness_score', 0)
        st.metric(
            "🎯 Readiness Score",
            f"{readiness_score:.0f}/100",
            delta=analysis.get('ai_overall_assessment', {}).get('risk_level', 'Unknown')
        )
    
    with col2:
        migration_time = analysis.get('estimated_migration_time_hours', 0)
        st.metric(
            "⏱️ Migration Time",
            f"{migration_time:.1f} hours",
            delta=f"Window: {config.get('downtime_tolerance_minutes', 60)} min"
        )
    
    with col3:
        throughput = analysis.get('migration_throughput_mbps', 0)
        st.metric(
            "🚀 Throughput",
            f"{throughput:,.0f} Mbps",
            delta=f"Agents: {config.get('number_of_agents', 1)}"
        )
    
    with col4:
        monthly_cost = analysis.get('cost_analysis', {}).get('total_monthly_cost', 0)
        st.metric(
            "💰 Monthly Cost",
            f"${monthly_cost:,.0f}",
            delta=f"ROI: {analysis.get('cost_analysis', {}).get('roi_months', 'TBD')} months"
        )
    
    with col5:
        destination = config.get('destination_storage_type', 'S3')
        agent_efficiency = analysis.get('agent_analysis', {}).get('scaling_efficiency', 1.0)
        st.metric(
            "🗄️ Destination",
            destination,
            delta=f"Efficiency: {agent_efficiency*100:.1f}%"
        )
    
    with col6:
        complexity = analysis.get('aws_sizing_recommendations', {}).get('ai_analysis', {}).get('ai_complexity_score', 6)
        confidence = analysis.get('ai_overall_assessment', {}).get('ai_confidence', 0.5)
        st.metric(
            "🤖 AI Confidence",
            f"{confidence*100:.1f}%",
            delta=f"Complexity: {complexity:.1f}/10"
        )
    
    # Performance Overview Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Performance Analysis:**")
        
        # Current vs Target Performance
        onprem_perf = analysis.get('onprem_performance', {}).get('overall_performance', {})
        
        performance_comparison = {
            'Metric': ['CPU', 'Memory', 'Storage', 'Network', 'Database'],
            'Current Score': [
                onprem_perf.get('cpu_score', 0),
                onprem_perf.get('memory_score', 0),
                onprem_perf.get('storage_score', 0),
                onprem_perf.get('network_score', 0),
                onprem_perf.get('database_score', 0)
            ],
            'Target (AWS)': [85, 90, 95, 90, 88]  # Estimated AWS performance
        }
        
        fig_perf = px.bar(
            performance_comparison,
            x='Metric',
            y=['Current Score', 'Target (AWS)'],
            title="Current vs AWS Target Performance",
            barmode='group'
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with col2:
        st.markdown("**🔄 Migration Timeline:**")
        
        # Timeline breakdown
        timeline = analysis.get('ai_overall_assessment', {}).get('timeline_recommendation', {})
        
        timeline_data = {
            'Phase': ['Planning', 'Testing', 'Migration', 'Validation'],
            'Duration (Weeks)': [
                timeline.get('planning_phase_weeks', 2),
                timeline.get('testing_phase_weeks', 3),
                timeline.get('migration_window_hours', 24) / (7 * 24),  # Convert to weeks
                1  # Validation week
            ]
        }
        
        fig_timeline = px.bar(
            timeline_data,
            x='Phase',
            y='Duration (Weeks)',
            title="Project Timeline Breakdown",
            color='Duration (Weeks)',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Agent and Network Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🤖 Agent Performance Analysis:**")
        
        agent_analysis = analysis.get('agent_analysis', {})
        
        # Agent throughput breakdown
        agent_data = {
            'Component': ['Per Agent', 'Total Agents', 'Network Limit', 'Effective'],
            'Throughput (Mbps)': [
                agent_analysis.get('total_max_throughput_mbps', 0) / config.get('number_of_agents', 1),
                agent_analysis.get('total_max_throughput_mbps', 0),
                analysis.get('network_performance', {}).get('effective_bandwidth_mbps', 0),
                agent_analysis.get('total_effective_throughput', 0)
            ]
        }
        
        fig_agent = px.bar(
            agent_data,
            x='Component',
            y='Throughput (Mbps)',
            title=f"Agent Throughput Analysis ({config.get('number_of_agents', 1)} agents)",
            color='Throughput (Mbps)',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_agent, use_container_width=True)
    
    with col2:
        st.markdown("**🌐 Network Performance Analysis:**")
        
        network_perf = analysis.get('network_performance', {})
        
        # Network metrics
        network_data = {
            'Metric': ['Quality Score', 'AI Enhanced', 'Bandwidth Util', 'Reliability'],
            'Score/Percentage': [
                network_perf.get('network_quality_score', 0),
                network_perf.get('ai_enhanced_quality_score', 0),
                min(100, network_perf.get('effective_bandwidth_mbps', 0) / 100),
                network_perf.get('total_reliability', 0) * 100
            ]
        }
        
        fig_network = px.bar(
            network_data,
            x='Metric',
            y='Score/Percentage',
            title="Network Performance Metrics",
            color='Score/Percentage',
            color_continuous_scale='greens'
        )
        st.plotly_chart(fig_network, use_container_width=True)
    
    # Cost and ROI Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Cost Analysis Dashboard:**")
        
        cost_analysis = analysis.get('cost_analysis', {})
        
        # Cost breakdown
        cost_categories = {
            'Compute': cost_analysis.get('aws_compute_cost', 0),
            'Storage': cost_analysis.get('aws_storage_cost', 0) + cost_analysis.get('destination_storage_cost', 0),
            'Agents': cost_analysis.get('agent_cost', 0),
            'Network': cost_analysis.get('network_cost', 0),
            'Other': cost_analysis.get('os_licensing_cost', 0) + cost_analysis.get('management_cost', 0)
        }
        
        fig_cost = px.pie(
            values=list(cost_categories.values()),
            names=list(cost_categories.keys()),
            title="Monthly Cost Distribution"
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col2:
        st.markdown("**📈 ROI Projection:**")
        
        # ROI over time
        monthly_savings = cost_analysis.get('estimated_monthly_savings', 0)
        one_time_cost = cost_analysis.get('one_time_migration_cost', 0)
        
        months = list(range(0, 37, 6))  # 0 to 36 months
        cumulative_savings = []
        
        for month in months:
            if month == 0:
                cumulative_savings.append(-one_time_cost)
            else:
                cumulative_savings.append((monthly_savings * month) - one_time_cost)
        
        roi_data = {
            'Months': months,
            'Cumulative Savings ($)': cumulative_savings
        }
        
        fig_roi = px.line(
            roi_data,
            x='Months',
            y='Cumulative Savings ($)',
            title="ROI Projection Over Time",
            markers=True
        )
        fig_roi.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        st.plotly_chart(fig_roi, use_container_width=True)
    
    # Risk and Readiness Assessment
    st.markdown("**⚠️ Risk and Readiness Assessment:**")
    
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        ai_assessment = analysis.get('ai_overall_assessment', {})
        
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🎯 Migration Readiness</h4>
            <p><strong>Readiness Score:</strong> {ai_assessment.get('migration_readiness_score', 0):.0f}/100</p>
            <p><strong>Success Probability:</strong> {ai_assessment.get('success_probability', 0):.0f}%</p>
            <p><strong>Risk Level:</strong> {ai_assessment.get('risk_level', 'Unknown')}</p>
            <p><strong>Recommended Approach:</strong> {ai_assessment.get('timeline_recommendation', {}).get('recommended_approach', 'Unknown').replace('_', ' ').title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col2:
        agent_impact = ai_assessment.get('agent_scaling_impact', {})
        
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🤖 Agent Scaling Assessment</h4>
            <p><strong>Scaling Efficiency:</strong> {agent_impact.get('scaling_efficiency', 0):.1f}%</p>
            <p><strong>Optimal Agents:</strong> {agent_impact.get('optimal_agents', config.get('number_of_agents', 1))}</p>
            <p><strong>Current Agents:</strong> {agent_impact.get('current_agents', config.get('number_of_agents', 1))}</p>
            <p><strong>Efficiency Bonus:</strong> {agent_impact.get('efficiency_bonus', 0):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col3:
        storage_impact = ai_assessment.get('destination_storage_impact', {})
        
        st.markdown(f"""
        <div class="detailed-analysis-section">
            <h4>🗄️ Storage Destination Impact</h4>
            <p><strong>Storage Type:</strong> {storage_impact.get('storage_type', 'Unknown')}</p>
            <p><strong>Performance Bonus:</strong> +{storage_impact.get('performance_bonus', 0)}%</p>
            <p><strong>Performance Multiplier:</strong> {storage_impact.get('storage_performance_multiplier', 1.0):.1f}x</p>
            <p><strong>Optimization Level:</strong> {"High" if storage_impact.get('performance_multiplier', 1.0) > 1.2 else "Medium" if storage_impact.get('performance_multiplier', 1.0) > 1.0 else "Standard"}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Recommendations Summary
    st.markdown("**💡 Key Recommendations Summary:**")
    
    recommendations = ai_assessment.get('recommended_next_steps', [])
    if recommendations:
        for i, rec in enumerate(recommendations[:4], 1):
            priority = "🔴 High" if i <= 2 else "🟡 Medium"
            st.markdown(f"**{priority} Priority:** {rec}")
    
    # Migration Health Score
    st.markdown("**🏥 Migration Health Score:**")
    
    health_factors = {
        'Performance Readiness': min(100, analysis.get('onprem_performance', {}).get('performance_score', 0)),
        'Network Quality': network_perf.get('ai_enhanced_quality_score', 0),
        'Agent Optimization': agent_analysis.get('scaling_efficiency', 1.0) * 100,
        'Cost Efficiency': min(100, (monthly_savings / monthly_cost * 100)) if monthly_cost > 0 else 0,
        'Risk Mitigation': readiness_score
    }
    
    fig_health = px.bar(
        x=list(health_factors.keys()),
        y=list(health_factors.values()),
        title="Migration Health Score Breakdown",
        color=list(health_factors.values()),
        color_continuous_scale='RdYlGn',
        labels={'x': 'Factor', 'y': 'Score'}
    )
    
    # Add benchmark line at 80%
    fig_health.add_hline(y=80, line_dash="dash", line_color="blue", annotation_text="Target: 80%")
    
    st.plotly_chart(fig_health, use_container_width=True)

def render_aws_sizing_tab(analysis: Dict, config: Dict):
    """Render AWS sizing and configuration recommendations tab using native components"""
    st.subheader("🎯 AWS Sizing & Configuration Recommendations")
    
    aws_sizing = analysis.get('aws_sizing_recommendations', {})
    deployment_rec = aws_sizing.get('deployment_recommendation', {})
    
    # Deployment Recommendation Overview
    st.markdown("**☁️ Deployment Recommendation:**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        recommendation = deployment_rec.get('recommendation', 'unknown').upper()
        confidence = deployment_rec.get('confidence', 0)
        st.metric(
            "🎯 Recommended Deployment",
            recommendation,
            delta=f"Confidence: {confidence*100:.1f}%"
        )
    
    with col2:
        rds_score = deployment_rec.get('rds_score', 0)
        ec2_score = deployment_rec.get('ec2_score', 0)
        st.metric(
            "📊 RDS Score",
            f"{rds_score:.0f}",
            delta=f"EC2: {ec2_score:.0f}"
        )
    
    with col3:
        ai_analysis = aws_sizing.get('ai_analysis', {})
        complexity = ai_analysis.get('ai_complexity_score', 6)
        st.metric(
            "🤖 AI Complexity",
            f"{complexity:.1f}/10",
            delta=ai_analysis.get('confidence_level', 'medium').title()
        )
    
    with col4:
        if recommendation == 'RDS':
            monthly_cost = aws_sizing.get('rds_recommendations', {}).get('total_monthly_cost', 0)
        else:
            monthly_cost = aws_sizing.get('ec2_recommendations', {}).get('total_monthly_cost', 0)
        st.metric(
            "💰 Monthly Cost",
            f"${monthly_cost:,.0f}",
            delta="Compute + Storage"
        )
    
    with col5:
        reader_writer = aws_sizing.get('reader_writer_config', {})
        total_instances = reader_writer.get('total_instances', 1)
        writers = reader_writer.get('writers', 1)
        readers = reader_writer.get('readers', 0)
        st.metric(
            "🖥️ Total Instances",
            f"{total_instances}",
            delta=f"Writers: {writers}, Readers: {readers}"
        )
    
    # Detailed Sizing Recommendations using native components
    col1, col2 = st.columns(2)
    
    with col1:
        if recommendation == 'RDS':
            st.markdown("**🗄️ RDS Recommended Configuration:**")
            
            rds_rec = aws_sizing.get('rds_recommendations', {})
            
            with st.container():
                st.success("Amazon RDS Managed Service")
                st.write(f"**Instance Type:** {rds_rec.get('primary_instance', 'N/A')}")
                st.write(f"**vCPU:** {rds_rec.get('instance_specs', {}).get('vcpu', 'N/A')}")
                st.write(f"**Memory:** {rds_rec.get('instance_specs', {}).get('memory', 'N/A')} GB")
                st.write(f"**Storage:** {rds_rec.get('storage_size_gb', 0):,.0f} GB")
                st.write(f"**Storage Type:** {rds_rec.get('storage_type', 'gp3').upper()}")
                st.write(f"**Multi-AZ:** {'Yes' if rds_rec.get('multi_az', False) else 'No'}")
                st.write(f"**Backup Retention:** {rds_rec.get('backup_retention_days', 7)} days")
                st.write(f"**Monthly Instance Cost:** ${rds_rec.get('monthly_instance_cost', 0):,.0f}")
                st.write(f"**Monthly Storage Cost:** ${rds_rec.get('monthly_storage_cost', 0):,.0f}")
                st.write(f"**Total Monthly Cost:** ${rds_rec.get('total_monthly_cost', 0):,.0f}")
        else:
            st.markdown("**🖥️ EC2 Recommended Configuration:**")
            
            ec2_rec = aws_sizing.get('ec2_recommendations', {})
            
            with st.container():
                st.info("Amazon EC2 Self-Managed")
                st.write(f"**Instance Type:** {ec2_rec.get('primary_instance', 'N/A')}")
                st.write(f"**vCPU:** {ec2_rec.get('instance_specs', {}).get('vcpu', 'N/A')}")
                st.write(f"**Memory:** {ec2_rec.get('instance_specs', {}).get('memory', 'N/A')} GB")
                st.write(f"**Storage:** {ec2_rec.get('storage_size_gb', 0):,.0f} GB")
                st.write(f"**Storage Type:** {ec2_rec.get('storage_type', 'gp3').upper()}")
                st.write(f"**EBS Optimized:** {'Yes' if ec2_rec.get('ebs_optimized', False) else 'No'}")
                st.write(f"**Enhanced Networking:** {'Yes' if ec2_rec.get('enhanced_networking', False) else 'No'}")
                st.write(f"**Monthly Instance Cost:** ${ec2_rec.get('monthly_instance_cost', 0):,.0f}")
                st.write(f"**Monthly Storage Cost:** ${ec2_rec.get('monthly_storage_cost', 0):,.0f}")
                st.write(f"**Total Monthly Cost:** ${ec2_rec.get('total_monthly_cost', 0):,.0f}")
    
    with col2:
        st.markdown("**🎯 AI Sizing Factors:**")
        
        if recommendation == 'RDS':
            sizing_factors = aws_sizing.get('rds_recommendations', {}).get('ai_sizing_factors', {})
        else:
            sizing_factors = aws_sizing.get('ec2_recommendations', {}).get('ai_sizing_factors', {})
        
        with st.container():
            st.warning("AI-Enhanced Sizing Analysis")
            st.write(f"**Complexity Multiplier:** {sizing_factors.get('complexity_multiplier', 1.0):.2f}x")
            st.write(f"**Agent Scaling Factor:** {sizing_factors.get('agent_scaling_factor', 1.0):.2f}x")
            st.write(f"**AI Complexity Score:** {sizing_factors.get('ai_complexity_score', 6):.1f}/10")
            st.write(f"**Storage Multiplier:** {sizing_factors.get('storage_multiplier', 1.5):.1f}x")
            st.write(f"**Database Size:** {config.get('database_size_gb', 0):,} GB")
            st.write(f"**Performance Requirement:** {config.get('performance_requirements', 'standard').title()}")
            st.write(f"**Environment:** {config.get('environment', 'unknown').title()}")
            st.write(f"**Number of Agents:** {config.get('number_of_agents', 1)}")
    
    # RDS vs EC2 Comparison
    st.markdown("**⚖️ RDS vs EC2 Deployment Comparison:**")
    
    comparison_col1, comparison_col2 = st.columns(2)
    
    with comparison_col1:
        # Create comparison chart
        comparison_data = {
            'Criteria': ['Management', 'Cost', 'Performance', 'Scalability', 'Control', 'Total Score'],
            'RDS Score': [90, 80, 85, 90, 60, deployment_rec.get('rds_score', 0)],
            'EC2 Score': [60, 85, 90, 80, 95, deployment_rec.get('ec2_score', 0)]
        }
        
        fig_comparison = px.bar(
            comparison_data,
            x='Criteria',
            y=['RDS Score', 'EC2 Score'],
            title="RDS vs EC2 Scoring Breakdown",
            barmode='group'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with comparison_col2:
        st.markdown("**📊 Deployment Decision Factors:**")
        
        primary_reasons = deployment_rec.get('primary_reasons', [])
        
        with st.container():
            st.success(f"Why {recommendation}?")
            for reason in primary_reasons[:4]:
                st.write(f"• {reason}")
            
            st.write(f"**Confidence Level:** {confidence*100:.1f}%")
            decision_strength = ("Strong" if abs(rds_score - ec2_score) > 20 else 
                               "Moderate" if abs(rds_score - ec2_score) > 10 else "Weak")
            st.write(f"**Decision Strength:** {decision_strength}")
    
    # ENHANCED: Reader/Writer Instance Sizing Details
        with st.expander("🔄 Writer/Reader Instance Sizing Details", expanded=True):
            reader_writer = aws_sizing.get('reader_writer_config', {})
            
            # Get the base instance recommendation
            if recommendation == 'RDS':
                base_instance = aws_sizing.get('rds_recommendations', {}).get('primary_instance', 'db.r6g.large')
                base_specs = aws_sizing.get('rds_recommendations', {}).get('instance_specs', {})
                base_cost = aws_sizing.get('rds_recommendations', {}).get('monthly_instance_cost', 0)
            else:
                base_instance = aws_sizing.get('ec2_recommendations', {}).get('primary_instance', 'r6i.large')
                base_specs = aws_sizing.get('ec2_recommendations', {}).get('instance_specs', {})
                base_cost = aws_sizing.get('ec2_recommendations', {}).get('monthly_instance_cost', 0)
            
            writers = reader_writer.get('writers', 1)
            readers = reader_writer.get('readers', 0)
            total_instances = writers + readers
            
            # Calculate per-instance cost
            per_instance_cost = base_cost / total_instances if total_instances > 0 else base_cost
            
            config_col1, config_col2, config_col3 = st.columns(3)
            
            with config_col1:
                st.success("✍️ **Writer Instance Details**")
                st.write(f"**Number of Writers:** {writers}")
                st.write(f"**Instance Type:** {base_instance}")
                st.write(f"**vCPU per Writer:** {base_specs.get('vcpu', 'N/A')}")
                st.write(f"**Memory per Writer:** {base_specs.get('memory', 'N/A')} GB")
                st.write(f"**Cost per Writer:** ${per_instance_cost:,.0f}/month")
                st.write(f"**Total Writer Cost:** ${per_instance_cost * writers:,.0f}/month")
                st.write(f"**Write Capacity:** {reader_writer.get('write_capacity_percent', 100):.1f}%")
                st.write(f"**Role:** Primary database operations")
                
            with config_col2:
                st.info("📖 **Reader Instance Details**")
                if readers > 0:
                    # For readers, we might use the same instance type or a smaller one
                    reader_instance = base_instance  # Could be optimized to use smaller instances
                    reader_cost = per_instance_cost * 0.8  # Readers typically cost slightly less
                    
                    st.write(f"**Number of Readers:** {readers}")
                    st.write(f"**Instance Type:** {reader_instance}")
                    st.write(f"**vCPU per Reader:** {base_specs.get('vcpu', 'N/A')}")
                    st.write(f"**Memory per Reader:** {base_specs.get('memory', 'N/A')} GB")
                    st.write(f"**Cost per Reader:** ${reader_cost:,.0f}/month")
                    st.write(f"**Total Reader Cost:** ${reader_cost * readers:,.0f}/month")
                    st.write(f"**Read Capacity:** {reader_writer.get('read_capacity_percent', 0):.1f}%")
                    st.write(f"**Role:** Read-only query processing")
                else:
                    st.write("**No reader instances configured**")
                    st.write("**Reason:** Database size and workload")
                    st.write("**Alternative:** Single writer handles all operations")
                    st.write("**Scaling:** Can add readers later as needed")
                    st.write("**Cost Savings:** ${:,.0f}/month".format(per_instance_cost * 2))
                    st.write("**Performance Impact:** Minimal for current workload")
            
            with config_col3:
                st.warning("📊 **Total Configuration Summary**")
                total_writer_cost = per_instance_cost * writers
                total_reader_cost = per_instance_cost * 0.8 * readers if readers > 0 else 0
                total_config_cost = total_writer_cost + total_reader_cost
                
                st.write(f"**Total Instances:** {total_instances}")
                st.write(f"**Total vCPU:** {base_specs.get('vcpu', 0) * total_instances}")
                st.write(f"**Total Memory:** {base_specs.get('memory', 0) * total_instances} GB")
                st.write(f"**Total Monthly Cost:** ${total_config_cost:,.0f}")
                st.write(f"**Cost per GB/Month:** ${total_config_cost / config.get('database_size_gb', 1):.2f}")
                st.write(f"**Recommended Read Split:** {reader_writer.get('recommended_read_split', 0):.0f}%")
                
                # Show scaling recommendations
                database_size = config.get('database_size_gb', 0)
                if database_size > 5000 and readers == 0:
                    st.warning("💡 Consider adding 1-2 read replicas")
                elif database_size > 20000 and readers < 3:
                    st.info("💡 Consider additional read replicas")
        
        # Instance Scaling Recommendations
        with st.expander("📈 Instance Scaling Recommendations", expanded=False):
            scaling_col1, scaling_col2 = st.columns(2)
            
            with scaling_col1:
                st.markdown("**🔮 Future Scaling Scenarios:**")
                
                database_size = config.get('database_size_gb', 0)
                performance_req = config.get('performance_requirements', 'standard')
                
                # Calculate scaling scenarios
                scenarios = []
                
                # Current scenario
                scenarios.append({
                    'Scenario': 'Current',
                    'Database Size': f"{database_size:,} GB",
                    'Writers': writers,
                    'Readers': readers,
                    'Monthly Cost': f"${total_config_cost:,.0f}" if 'total_config_cost' in locals() else "TBD"
                })
                
                # 2x growth scenario
                future_readers = min(readers + 1, 3) if database_size * 2 > 5000 else readers
                scenarios.append({
                    'Scenario': '2x Growth',
                    'Database Size': f"{database_size * 2:,} GB",
                    'Writers': writers,
                    'Readers': future_readers,
                    'Monthly Cost': f"${(total_config_cost * (writers + future_readers) / total_instances if total_instances > 0 else total_config_cost):,.0f}" if 'total_config_cost' in locals() else "TBD"
                })
                
                # High performance scenario
                hp_readers = max(readers + 1, 2) if performance_req == 'standard' else readers + 1
                scenarios.append({
                    'Scenario': 'High Performance',
                    'Database Size': f"{database_size:,} GB",
                    'Writers': writers,
                    'Readers': hp_readers,
                    'Monthly Cost': f"${(total_config_cost * (writers + hp_readers) / total_instances if total_instances > 0 else total_config_cost):,.0f}" if 'total_config_cost' in locals() else "TBD"
                })
                
                df_scenarios = pd.DataFrame(scenarios)
                st.dataframe(df_scenarios, use_container_width=True)
            
            with scaling_col2:
                st.markdown("**⚡ Performance Optimization Tips:**")
                
                st.write("**Read Scaling:**")
                st.write("• Add read replicas to distribute query load")
                st.write("• Use connection pooling for efficient connections")
                st.write("• Implement read/write splitting in application")
                
                st.write("**Write Scaling:**")
                st.write("• Optimize database queries and indexes")
                st.write("• Consider write partitioning for large datasets")
                st.write("• Use write-through caching strategies")
                
                st.write("**Cost Optimization:**")
                st.write("• Use Reserved Instances for 20-30% savings")
                st.write("• Monitor and right-size based on utilization")
                st.write("• Consider Aurora Serverless for variable workloads")

        # AI Configuration Insights (Enhanced)
        with st.expander("🤖 AI Configuration Insights & Reasoning", expanded=False):
            ai_insights_rw = reader_writer.get('ai_insights', {})
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.success("🧠 **AI Reasoning Process**")
                st.write(f"**Database Size Analysis:** {config.get('database_size_gb', 0):,} GB")
                st.write(f"**Performance Requirement:** {config.get('performance_requirements', 'standard').title()}")
                st.write(f"**Environment Type:** {config.get('environment', 'unknown').title()}")
                st.write(f"**Agent Impact:** {config.get('number_of_agents', 1)} agents considered")
                
                reasoning = reader_writer.get('reasoning', 'Standard configuration applied')
                st.write(f"**AI Decision Logic:** {reasoning}")
                
                st.write("**Scaling Factors Applied:**")
                for factor in ai_insights_rw.get('scaling_factors', ['Standard scaling applied'])[:3]:
                    st.write(f"• {factor}")
            
            with insight_col2:
                st.info("📈 **Optimization Potential**")
                st.write(f"**Complexity Impact:** {ai_insights_rw.get('complexity_impact', 0):.0f}/10")
                st.write(f"**Agent Scaling Impact:** {ai_insights_rw.get('agent_scaling_impact', 1)} agents")
                st.write(f"**Optimization Potential:** {ai_insights_rw.get('optimization_potential', '5-10%')}")
                
                # Performance predictions
                expected_improvement = 15 + (readers * 10)  # Rough calculation
                st.write(f"**Expected Read Performance:** +{expected_improvement}%")
                st.write(f"**Expected Write Performance:** Consistent")
                st.write(f"**Availability Improvement:** {'+99.5%' if readers > 0 else 'Standard'}")
    
    # ENHANCED: Comprehensive Cost Analysis
def render_comprehensive_cost_pricing_tab(analysis: Dict, config: Dict):
    """Render comprehensive cost analysis including ALL AWS services"""
    st.subheader("💰 Comprehensive AWS Migration Cost Analysis")
    
    # Base cost analysis
    cost_analysis = analysis.get('cost_analysis', {})
    
    # Enhanced cost breakdown with ALL AWS services
    st.markdown("**💸 Complete AWS Service Cost Breakdown:**")
    
    # Primary AWS Services Costs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        compute_cost = cost_analysis.get('aws_compute_cost', 0)
        st.metric(
            "🖥️ Compute (RDS/EC2)",
            f"${compute_cost:,.0f}/mo",
            delta="Primary database hosting"
        )
    
    with col2:
        storage_cost = cost_analysis.get('aws_storage_cost', 0) + cost_analysis.get('destination_storage_cost', 0)
        st.metric(
            "💾 Storage (EBS/S3/FSx)",
            f"${storage_cost:,.0f}/mo",
            delta="Database + destination storage"
        )
    
    with col3:
        # NEW: DataSync/DMS service costs
        migration_service_cost = cost_analysis.get('agent_cost', 0)
        st.metric(
            "🔄 Migration Services",
            f"${migration_service_cost:,.0f}/mo",
            delta=f"DataSync/DMS agents"
        )
    
    with col4:
        # NEW: Direct Connect costs
        dx_cost = cost_analysis.get('network_cost', 0)
        st.metric(
            "🌐 Direct Connect (DX)",
            f"${dx_cost:,.0f}/mo",
            delta="Dedicated network connection"
        )
    
    with col5:
        # Total monthly cost
        total_monthly = cost_analysis.get('total_monthly_cost', 0)
        st.metric(
            "💰 Total Monthly",
            f"${total_monthly:,.0f}",
            delta=f"Annual: ${total_monthly * 12:,.0f}"
        )
    
    # Detailed AWS Service Breakdown
    st.markdown("**📊 Detailed AWS Service Cost Analysis:**")
    
    # Create comprehensive cost breakdown
    service_costs = []
    
    # 1. Compute Services
    if config.get('is_sql_server') or config.get('database_engine', '').startswith('ec2_'):
        # EC2 costs for SQL Server or self-managed
        service_costs.append({
            "Service Category": "Compute",
            "AWS Service": "Amazon EC2",
            "Description": f"SQL Server on {config.get('database_engine', 'EC2')}",
            "Monthly Cost": f"${cost_analysis.get('aws_compute_cost', 0):,.0f}",
            "Usage": "Database hosting (self-managed)",
            "Optimization": "Consider Reserved Instances for 30% savings"
        })
    else:
        # RDS costs
        service_costs.append({
            "Service Category": "Compute",
            "AWS Service": "Amazon RDS",
            "Description": f"Managed {config.get('database_engine', 'MySQL').replace('rds_', '')}",
            "Monthly Cost": f"${cost_analysis.get('aws_compute_cost', 0):,.0f}",
            "Usage": "Managed database service",
            "Optimization": "Consider Reserved Instances for 30% savings"
        })
    
    # 2. Storage Services
    # EBS Storage
    ebs_cost = cost_analysis.get('aws_storage_cost', 0)
    if ebs_cost > 0:
        service_costs.append({
            "Service Category": "Storage",
            "AWS Service": "Amazon EBS",
            "Description": f"Database storage (GP3/IO2)",
            "Monthly Cost": f"${ebs_cost:,.0f}",
            "Usage": f"Database storage ({config.get('database_size_gb', 0):,} GB)",
            "Optimization": "Right-size based on IOPS requirements"
        })
    
    # Destination Storage (S3/FSx)
    dest_storage_cost = cost_analysis.get('destination_storage_cost', 0)
    destination_type = config.get('destination_storage_type', 'S3')
    service_costs.append({
        "Service Category": "Storage",
        "AWS Service": f"Amazon {destination_type}",
        "Description": f"Migration destination storage",
        "Monthly Cost": f"${dest_storage_cost:,.0f}",
        "Usage": f"Backup/archive storage ({destination_type})",
        "Optimization": "Use lifecycle policies for cost optimization" if destination_type == "S3" else f"Right-size {destination_type} for workload"
    })
    
    # 3. Migration Services
    agent_cost = cost_analysis.get('agent_cost', 0)
    num_agents = config.get('number_of_agents', 1)
    is_homogeneous = config.get('source_database_engine') == config.get('ec2_database_engine', 'mysql')
    migration_service = "DataSync" if is_homogeneous else "DMS"
    
    service_costs.append({
        "Service Category": "Migration",
        "AWS Service": f"AWS {migration_service}",
        "Description": f"{num_agents}x {migration_service} agents",
        "Monthly Cost": f"${agent_cost:,.0f}",
        "Usage": f"Data migration and sync ({num_agents} agents)",
        "Optimization": "Optimize agent count based on throughput needs"
    })
    
    # 4. Network Services
    dx_cost = cost_analysis.get('network_cost', 0)
    environment = config.get('environment', 'non-production')
    
    service_costs.append({
        "Service Category": "Networking",
        "AWS Service": "AWS Direct Connect",
        "Description": f"{environment.title()} DX connection",
        "Monthly Cost": f"${dx_cost:,.0f}",
        "Usage": f"Dedicated network connectivity ({environment})",
        "Optimization": "Consider DX Gateway for multiple VPCs"
    })
    
    # 5. Additional AWS Services
    
    # VPC and Security
    service_costs.append({
        "Service Category": "Networking",
        "AWS Service": "Amazon VPC",
        "Description": "Virtual Private Cloud setup",
        "Monthly Cost": "$50",
        "Usage": "Network isolation and security",
        "Optimization": "Included in base networking costs"
    })
    
    # CloudWatch Monitoring
    monitoring_cost = 100 + (num_agents * 20)  # Base + per agent
    service_costs.append({
        "Service Category": "Management",
        "AWS Service": "Amazon CloudWatch",
        "Description": "Monitoring and alerting",
        "Monthly Cost": f"${monitoring_cost:,.0f}",
        "Usage": "Database and migration monitoring",
        "Optimization": "Optimize log retention and metrics"
    })
    
    # AWS Backup (if applicable)
    if not config.get('database_engine', '').startswith('rds_'):
        backup_cost = cost_analysis.get('aws_storage_cost', 0) * 0.2  # 20% of storage for backups
        service_costs.append({
            "Service Category": "Backup",
            "AWS Service": "AWS Backup",
            "Description": "Automated backup service",
            "Monthly Cost": f"${backup_cost:,.0f}",
            "Usage": "Database backup and recovery",
            "Optimization": "Configure retention policies"
        })
    
    # IAM and Security Services
    service_costs.append({
        "Service Category": "Security",
        "AWS Service": "AWS IAM + KMS",
        "Description": "Identity and encryption management",
        "Monthly Cost": "$25",
        "Usage": "Access control and encryption keys",
        "Optimization": "Included in security baseline"
    })
    
    # SQL Server Specific Costs
    if config.get('is_sql_server'):
        # SQL Server licensing
        sql_licensing_cost = cost_analysis.get('os_licensing_cost', 0)
        service_costs.append({
            "Service Category": "Licensing",
            "AWS Service": "SQL Server License",
            "Description": "BYOL or License Included",
            "Monthly Cost": f"${sql_licensing_cost:,.0f}",
            "Usage": "SQL Server database engine licensing",
            "Optimization": "Consider BYOL for long-term savings"
        })
        
        # Windows Server licensing
        windows_licensing_cost = 200  # Estimated Windows Server cost
        service_costs.append({
            "Service Category": "Licensing",
            "AWS Service": "Windows Server License",
            "Description": "Windows OS licensing on EC2",
            "Monthly Cost": f"${windows_licensing_cost:,.0f}",
            "Usage": "Windows Server OS for SQL Server",
            "Optimization": "Include in EC2 pricing or BYOL"
        })
    
    # Create comprehensive service table
    df_services = pd.DataFrame(service_costs)
    st.dataframe(df_services, use_container_width=True)
    
    # Cost by Category Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Cost by Service Category:**")
        
        # Calculate category totals
        category_costs = {}
        for service in service_costs:
            category = service["Service Category"]
            cost_str = service["Monthly Cost"].replace("$", "").replace(",", "")
            try:
                cost = float(cost_str)
                category_costs[category] = category_costs.get(category, 0) + cost
            except:
                continue
        
        if category_costs:
            fig_category = px.pie(
                values=list(category_costs.values()),
                names=list(category_costs.keys()),
                title="Monthly Cost by Service Category"
            )
            st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        st.markdown("**💡 Cost Optimization Recommendations:**")
        
        with st.container():
            st.success("Cost Optimization Strategies")
            
            # Generate optimization recommendations based on configuration
            optimizations = []
            
            if not config.get('database_engine', '').startswith('rds_'):
                optimizations.append("• Consider Reserved Instances for 30-50% compute savings")
            
            if config.get('destination_storage_type') == 'S3':
                optimizations.append("• Implement S3 Intelligent Tiering for storage optimization")
            
            if num_agents > 3:
                optimizations.append(f"• Optimize {num_agents} agents - consider consolidation")
            
            if config.get('environment') == 'non-production':
                optimizations.append("• Use Spot Instances for non-prod workloads (60% savings)")
            
            optimizations.append("• Implement auto-scaling policies")
            optimizations.append("• Regular cost reviews and rightsizing")
            
            for opt in optimizations:
                st.write(opt)
    
    # One-time Migration Costs
    st.markdown("**🔄 One-time Migration and Setup Costs:**")
    
    onetime_col1, onetime_col2, onetime_col3 = st.columns(3)
    
    with onetime_col1:
        st.info("**Migration Setup Costs**")
        setup_costs = {
            "Professional Services": "$15,000",
            "Agent Setup": f"${cost_analysis.get('agent_setup_cost', 0):,.0f}",
            "Network Configuration": "$5,000",
            "Testing & Validation": "$8,000",
            "Training": "$3,000"
        }
        
        for item, cost in setup_costs.items():
            st.write(f"**{item}:** {cost}")
    
    with onetime_col2:
        st.warning("**Data Transfer Costs**")
        
        database_size_gb = config.get('database_size_gb', 0)
        data_transfer_cost = database_size_gb * 0.02  # $0.02 per GB estimate
        
        st.write(f"**Initial Data Transfer:** ${data_transfer_cost:,.0f}")
        st.write(f"**Database Size:** {database_size_gb:,} GB")
        st.write(f"**Transfer Rate:** $0.02/GB")
        st.write(f"**Ongoing Sync:** Included in DX")
        st.write(f"**Backup Transfer:** ${data_transfer_cost * 0.1:,.0f}/month")
    
    with onetime_col3:
        st.error("**Risk Mitigation Costs**")
        
        risk_costs = {
            "Rollback Preparation": "$5,000",
            "Extended Support": "$10,000",
            "Additional Testing": "$5,000",
            "Contingency (10%)": f"${cost_analysis.get('one_time_migration_cost', 0) * 0.1:,.0f}"
        }
        
        for item, cost in risk_costs.items():
            st.write(f"**{item}:** {cost}")
    
    # Total Cost Summary
    st.markdown("**📊 Total Cost of Ownership (TCO) Analysis:**")
    
    # Calculate 3-year TCO
    monthly_total = sum(category_costs.values()) if category_costs else total_monthly
    one_time_total = cost_analysis.get('one_time_migration_cost', 0) + 51000  # Professional services + setup
    
    tco_data = {
        "Timeline": ["Month 1", "Year 1", "Year 2", "Year 3", "3-Year Total"],
        "Monthly Costs": [f"${monthly_total:,.0f}", f"${monthly_total * 12:,.0f}", 
                         f"${monthly_total * 12:,.0f}", f"${monthly_total * 12:,.0f}",
                         f"${monthly_total * 36:,.0f}"],
        "One-time Costs": [f"${one_time_total:,.0f}", "$0", "$0", "$0", f"${one_time_total:,.0f}"],
        "Total": [f"${monthly_total + one_time_total:,.0f}", 
                 f"${monthly_total * 12:,.0f}",
                 f"${monthly_total * 12:,.0f}", 
                 f"${monthly_total * 12:,.0f}",
                 f"${monthly_total * 36 + one_time_total:,.0f}"]
    }
    
    df_tco = pd.DataFrame(tco_data)
    st.dataframe(df_tco, use_container_width=True)
    
    
    
    
    
    # Professional footer with FSx capabilities
    st.markdown("""
    <div class="enterprise-footer">
        <h4>🚀 AWS Enterprise Database Migration Analyzer AI v3.0</h4>
        <p>Powered by Anthropic Claude AI • Real-time AWS Integration • Professional Migration Analysis • Advanced Agent Scaling • FSx Destination Analysis</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            🔬 Advanced Network Intelligence • 🎯 AI-Driven Recommendations • 📊 Executive Reporting • 🤖 Multi-Agent Optimization • 🗄️ S3/FSx Comparisons
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


