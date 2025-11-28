"""
🔧 Registry Integration Example - Complete System Demonstration

This example demonstrates the full Enhanced Registry system integration,
showing how all 4 registry components work together to provide a complete
modular architecture for the gaming AI bot application.

Features demonstrated:
• Dynamic page loading with dependency injection
• Service discovery and lifecycle management  
• UI component registration and theming
• AI engine management with performance monitoring
• Complete end-to-end modular architecture
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import all registry components
from streamlit_app.registry import (
    EnhancedPageRegistry, 
    PageInfo,
    NavigationContext,
    ServicesRegistry,
    ServiceInfo,
    ComponentsRegistry,
    ComponentInfo,
    AIEnginesRegistry,
    EngineInfo,
    EngineType,
    EngineStatus,
    EngineCapability
)

app_log = logging.getLogger(__name__)


class RegistryIntegrationDemo:
    """
    Comprehensive demonstration of the Enhanced Registry System.
    
    This class shows how all registry components work together to provide:
    - Dynamic page management
    - Service dependency injection
    - Component lifecycle management
    - AI engine orchestration
    """
    
    def __init__(self):
        """Initialize the registry integration demo."""
        self.page_registry = None
        self.services_registry = None
        self.components_registry = None
        self.ai_engines_registry = None
        
        # Demo state
        self.demo_initialized = False
        self.demo_stats = {
            'pages_loaded': 0,
            'services_active': 0,
            'components_registered': 0,
            'engines_ready': 0,
            'demo_started': None
        }
        
        self.initialize_demo()
    
    def initialize_demo(self) -> None:
        """Initialize all registry components for the demo."""
        try:
            st.header("🔧 Registry System Integration Demo")
            st.write("Initializing comprehensive registry system...")
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize registries step by step
            with st.expander("📋 Registry Initialization Progress", expanded=True):
                
                # Step 1: Services Registry
                status_text.text("Step 1/4: Initializing Services Registry...")
                progress_bar.progress(25)
                
                self.services_registry = ServicesRegistry()
                services_count = len(self.services_registry.get_all_services())
                st.success(f"✅ Services Registry initialized with {services_count} services")
                self.demo_stats['services_active'] = services_count
                
                # Step 2: Components Registry  
                status_text.text("Step 2/4: Initializing Components Registry...")
                progress_bar.progress(50)
                
                self.components_registry = ComponentsRegistry()
                components_count = len(self.components_registry.get_all_components())
                st.success(f"✅ Components Registry initialized with {components_count} components")
                self.demo_stats['components_registered'] = components_count
                
                # Step 3: AI Engines Registry
                status_text.text("Step 3/4: Initializing AI Engines Registry...")
                progress_bar.progress(75)
                
                self.ai_engines_registry = AIEnginesRegistry()
                engines_count = len(self.ai_engines_registry.get_all_engines())
                st.success(f"✅ AI Engines Registry initialized with {engines_count} engines")
                self.demo_stats['engines_ready'] = engines_count
                
                # Step 4: Page Registry (depends on other registries)
                status_text.text("Step 4/4: Initializing Page Registry with dependencies...")
                progress_bar.progress(100)
                
                self.page_registry = EnhancedPageRegistry(
                    services_registry=self.services_registry,
                    components_registry=self.components_registry,
                    ai_engines_registry=self.ai_engines_registry
                )
                pages_count = len(self.page_registry.get_all_pages())
                st.success(f"✅ Page Registry initialized with {pages_count} pages")
                self.demo_stats['pages_loaded'] = pages_count
            
            status_text.text("✅ All registries initialized successfully!")
            self.demo_initialized = True
            self.demo_stats['demo_started'] = datetime.now()
            
            # Show demo navigation
            self.show_demo_navigation()
            
        except Exception as e:
            st.error(f"❌ Error initializing registry demo: {e}")
            app_log.error(f"Registry demo initialization error: {e}")
    
    def show_demo_navigation(self) -> None:
        """Show demo navigation and features."""
        if not self.demo_initialized:
            return
        
        st.divider()
        st.header("🎮 Registry Demo Features")
        
        # Create tabs for different demo sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 System Overview",
            "📄 Page Management", 
            "⚙️ Service Management",
            "🎨 Component Management",
            "🤖 AI Engine Management"
        ])
        
        with tab1:
            self.show_system_overview()
        
        with tab2:
            self.show_page_management_demo()
        
        with tab3:
            self.show_service_management_demo()
        
        with tab4:
            self.show_component_management_demo()
        
        with tab5:
            self.show_ai_engine_management_demo()
    
    def show_system_overview(self) -> None:
        """Show comprehensive system overview."""
        st.subheader("📊 Registry System Overview")
        
        # System statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pages Loaded", self.demo_stats['pages_loaded'])
        
        with col2:
            st.metric("Services Active", self.demo_stats['services_active'])
        
        with col3:
            st.metric("Components Registered", self.demo_stats['components_registered'])
        
        with col4:
            st.metric("AI Engines Ready", self.demo_stats['engines_ready'])
        
        # System architecture diagram
        st.subheader("🏗️ System Architecture")
        with st.expander("Registry System Architecture", expanded=True):
            st.code("""
Registry System Architecture:

┌─────────────────────────────────────────────────┐
│                 Application Layer                │
├─────────────────────────────────────────────────┤
│              Enhanced Page Registry             │
│  • Dynamic page loading & navigation            │
│  • Dependency injection coordination            │
│  • Performance monitoring                       │
├─────────────────────────────────────────────────┤
│    Services Registry  │  Components Registry    │
│  • Service discovery  │  • UI component mgmt    │
│  • Lifecycle mgmt     │  • Theme management     │
│  • Health monitoring  │  • Performance tracking │
├─────────────────────────────────────────────────┤
│               AI Engines Registry               │
│  • Model lifecycle management                   │
│  • Performance optimization                     │
│  • A/B testing capabilities                     │
└─────────────────────────────────────────────────┘
            """, language="text")
        
        # System health status
        st.subheader("💊 System Health")
        self.show_system_health()
    
    def show_system_health(self) -> None:
        """Display system health information."""
        try:
            health_data = []
            
            # Check each registry health
            registries = [
                ("Page Registry", self.page_registry, "pages"),
                ("Services Registry", self.services_registry, "services"),
                ("Components Registry", self.components_registry, "components"),
                ("AI Engines Registry", self.ai_engines_registry, "engines")
            ]
            
            for name, registry, item_type in registries:
                if registry:
                    try:
                        # Get basic health info
                        if hasattr(registry, 'get_health_status'):
                            health_status = registry.get_health_status()
                        else:
                            health_status = "✅ Healthy"
                        
                        health_data.append({
                            "Registry": name,
                            "Status": health_status,
                            "Items": len(getattr(registry, f'get_all_{item_type}', lambda: {})()),
                            "Last Check": datetime.now().strftime("%H:%M:%S")
                        })
                    except Exception as e:
                        health_data.append({
                            "Registry": name,
                            "Status": f"❌ Error: {str(e)[:50]}...",
                            "Items": "N/A",
                            "Last Check": datetime.now().strftime("%H:%M:%S")
                        })
            
            if health_data:
                st.dataframe(health_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying system health: {e}")
    
    def show_page_management_demo(self) -> None:
        """Demonstrate page management features."""
        st.subheader("📄 Page Management Demo")
        
        if not self.page_registry:
            st.warning("Page Registry not initialized")
            return
        
        # Show all available pages
        all_pages = self.page_registry.get_all_pages()
        st.write(f"**Total Pages Available:** {len(all_pages)}")
        
        # Page selection and navigation
        if all_pages:
            page_names = list(all_pages.keys())
            selected_page = st.selectbox("Select a page to view details:", page_names)
            
            if selected_page:
                page_info = all_pages[selected_page]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Page Information:**")
                    st.write(f"- **Title:** {page_info.title}")
                    st.write(f"- **Description:** {page_info.description}")
                    st.write(f"- **Module:** {page_info.module_path}")
                    st.write(f"- **Icon:** {page_info.icon}")
                    st.write(f"- **Category:** {page_info.category}")
                    st.write(f"- **Dependencies:** {', '.join(page_info.dependencies)}")
                
                with col2:
                    st.write("**Page Capabilities:**")
                    for capability in page_info.capabilities:
                        st.write(f"- ✅ {capability}")
                
                # Navigation context demo
                if st.button(f"🚀 Navigate to {page_info.title}"):
                    context = NavigationContext(
                        source_page="demo",
                        target_page=selected_page,
                        user_id="demo_user",
                        session_data={"demo_mode": True}
                    )
                    
                    success = self.page_registry.navigate_to_page(selected_page, context)
                    if success:
                        st.success(f"✅ Successfully navigated to {page_info.title}")
                    else:
                        st.error(f"❌ Failed to navigate to {page_info.title}")
    
    def show_service_management_demo(self) -> None:
        """Demonstrate service management features."""
        st.subheader("⚙️ Service Management Demo")
        
        if not self.services_registry:
            st.warning("Services Registry not initialized")
            return
        
        # Show all services
        all_services = self.services_registry.get_all_services()
        st.write(f"**Total Services Available:** {len(all_services)}")
        
        # Service status overview
        service_status = self.services_registry.get_service_status()
        
        if service_status:
            st.subheader("📊 Service Status Overview")
            status_data = []
            
            for service_name, status in service_status.items():
                status_data.append({
                    "Service": status.get('title', service_name),
                    "Status": status.get('status', 'unknown'),
                    "Health": status.get('health', 'unknown'),
                    "Start Time": status.get('start_time', 'N/A'),
                    "Requests": status.get('request_count', 0),
                    "Errors": status.get('error_count', 0)
                })
            
            st.dataframe(status_data, use_container_width=True)
        
        # Service interaction demo
        if all_services:
            service_names = list(all_services.keys())
            selected_service = st.selectbox("Select a service for interaction:", service_names)
            
            if selected_service and st.button(f"🔄 Test Service: {selected_service}"):
                try:
                    service_instance = self.services_registry.get_service(selected_service)
                    if service_instance:
                        st.success(f"✅ Successfully retrieved service: {selected_service}")
                        
                        # Show service info
                        service_info = all_services[selected_service]
                        st.json({
                            "name": service_info.name,
                            "title": service_info.title,
                            "version": service_info.version,
                            "status": service_info.status.value if hasattr(service_info.status, 'value') else str(service_info.status)
                        })
                    else:
                        st.error(f"❌ Failed to retrieve service: {selected_service}")
                except Exception as e:
                    st.error(f"❌ Service interaction error: {e}")
    
    def show_component_management_demo(self) -> None:
        """Demonstrate component management features."""
        st.subheader("🎨 Component Management Demo")
        
        if not self.components_registry:
            st.warning("Components Registry not initialized")
            return
        
        # Show all components
        all_components = self.components_registry.get_all_components()
        st.write(f"**Total Components Available:** {len(all_components)}")
        
        # Component categories
        categories = set()
        for comp in all_components.values():
            categories.add(comp.category)
        
        st.write(f"**Component Categories:** {', '.join(sorted(categories))}")
        
        # Component filtering
        selected_category = st.selectbox("Filter by category:", ["All"] + sorted(list(categories)))
        
        filtered_components = all_components
        if selected_category != "All":
            filtered_components = {
                name: comp for name, comp in all_components.items()
                if comp.category == selected_category
            }
        
        # Display components
        if filtered_components:
            st.subheader(f"📋 Components ({len(filtered_components)})")
            
            for comp_name, comp_info in filtered_components.items():
                with st.expander(f"{comp_info.icon} {comp_info.title}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Name:** {comp_info.name}")
                        st.write(f"**Description:** {comp_info.description}")
                        st.write(f"**Category:** {comp_info.category}")
                        st.write(f"**Version:** {comp_info.version}")
                    
                    with col2:
                        st.write(f"**Module Path:** {comp_info.module_path}")
                        st.write(f"**Dependencies:** {', '.join(comp_info.dependencies)}")
                        st.write(f"**Props:** {', '.join(comp_info.props.keys()) if comp_info.props else 'None'}")
                    
                    # Component usage demo
                    if st.button(f"🧪 Test Component: {comp_info.title}", key=f"test_{comp_name}"):
                        try:
                            component = self.components_registry.get_component(comp_name)
                            if component:
                                st.success(f"✅ Component {comp_info.title} loaded successfully")
                            else:
                                st.error(f"❌ Failed to load component {comp_info.title}")
                        except Exception as e:
                            st.error(f"❌ Component test error: {e}")
    
    def show_ai_engine_management_demo(self) -> None:
        """Demonstrate AI engine management features."""
        st.subheader("🤖 AI Engine Management Demo")
        
        if not self.ai_engines_registry:
            st.warning("AI Engines Registry not initialized")
            return
        
        # Show all engines
        all_engines = self.ai_engines_registry.get_all_engines()
        st.write(f"**Total AI Engines Available:** {len(all_engines)}")
        
        # Engine status overview
        engine_status = self.ai_engines_registry.get_engine_status()
        
        if engine_status:
            st.subheader("📊 AI Engine Status")
            status_data = []
            
            for engine_name, status in engine_status.items():
                status_data.append({
                    "Engine": status.get('title', engine_name),
                    "Type": status.get('engine_type', 'unknown'),
                    "Status": status.get('status', 'unknown'),
                    "Health": status.get('health', 'unknown'),
                    "Inferences": status.get('inference_count', 0),
                    "Avg Time (ms)": round(status.get('avg_inference_time', 0) * 1000, 2),
                    "Errors": status.get('error_count', 0)
                })
            
            st.dataframe(status_data, use_container_width=True)
        
        # Engine interaction demo
        if all_engines:
            st.subheader("🎯 Engine Interaction Demo")
            
            engine_names = list(all_engines.keys())
            selected_engine = st.selectbox("Select an AI engine to test:", engine_names)
            
            if selected_engine:
                engine_info = all_engines[selected_engine]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Engine Details:**")
                    st.write(f"- **Type:** {engine_info.engine_type.value}")
                    st.write(f"- **Status:** {engine_info.status.value}")
                    st.write(f"- **Version:** {engine_info.version}")
                    st.write(f"- **Supported Models:** {', '.join(engine_info.supported_models)}")
                
                with col2:
                    st.write("**Capabilities:**")
                    for capability in engine_info.capabilities:
                        st.write(f"- ✅ {capability.value.replace('_', ' ').title()}")
                
                # Engine testing
                if st.button(f"🚀 Load Engine: {engine_info.title}"):
                    success = self.ai_engines_registry.load_engine(selected_engine)
                    if success:
                        st.success(f"✅ Engine {engine_info.title} loaded successfully")
                    else:
                        st.error(f"❌ Failed to load engine {engine_info.title}")
                
                # Prediction demo (with dummy data)
                if st.button(f"🎯 Test Prediction: {engine_info.title}"):
                    dummy_data = {"test_data": [1, 2, 3, 4, 5]}
                    
                    with st.spinner(f"Running prediction with {engine_info.title}..."):
                        result = self.ai_engines_registry.predict(selected_engine, dummy_data)
                        
                        if result:
                            st.success("✅ Prediction completed successfully")
                            st.json(result)
                        else:
                            st.error("❌ Prediction failed")


def main():
    """Main demo function."""
    st.set_page_config(
        page_title="Registry Integration Demo",
        page_icon="🔧",
        layout="wide"
    )
    
    # Initialize and run demo
    demo = RegistryIntegrationDemo()


if __name__ == "__main__":
    main()