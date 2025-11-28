"""
Table components for the lottery prediction system.

This module provides comprehensive table components for displaying
data in tabular format with sorting, filtering, pagination, and
export functionality extracted from the legacy application patterns.

Enhanced Components:
- TableComponents: Complete table library for all data display needs
- DataTable: Legacy table system (backward compatibility)
- TableBuilder: Utility functions for table creation and management
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, date
import logging
import io
import json

logger = logging.getLogger(__name__)


class TableComponents:
    """
    Comprehensive table component library for lottery prediction system.
    
    This class provides a complete set of reusable table components
    extracted from the legacy application UI patterns. All components maintain
    consistent styling, sorting, filtering, and export capabilities.
    
    Key Features:
    - Data Tables: Interactive tables with sorting and filtering
    - Sortable Grids: Advanced grid displays with multi-column sorting
    - Filtered Views: Dynamic filtering with search capabilities  
    - Export Functionality: CSV, JSON, Excel export options
    - Pagination: Large dataset pagination with performance optimization
    - Custom Formatters: Column-specific formatting and styling
    - Selection Support: Row selection and bulk operations
    - Summary Displays: Aggregated data summaries and statistics
    
    Table Categories:
    1. Basic Data Tables (display, sorting, filtering)
    2. Advanced Data Grids (pagination, selection, export)
    3. Summary Tables (aggregated data, statistics)
    4. Comparison Tables (side-by-side data comparison)
    5. Interactive Tables (clickable rows, actions)
    6. Export Systems (CSV, JSON, Excel export)
    7. Filtered Views (dynamic search and filtering)
    8. Performance Tables (large dataset optimization)
    """
    
    @staticmethod
    def render_data_table(data: pd.DataFrame,
                         title: str = "Data Table",
                         columns: Optional[List[str]] = None,
                         sortable: bool = True,
                         filterable: bool = True,
                         paginated: bool = True,
                         page_size: int = 20,
                         show_index: bool = False,
                         custom_formatters: Optional[Dict[str, Callable]] = None,
                         exportable: bool = True) -> Dict[str, Any]:
        """
        Render comprehensive data table with advanced features.
        
        Args:
            data: DataFrame to display
            title: Table title
            columns: Columns to display (None for all)
            sortable: Enable column sorting
            filterable: Enable data filtering
            paginated: Enable pagination
            page_size: Rows per page
            show_index: Show row indices
            custom_formatters: Custom column formatters
            exportable: Enable export functionality
            
        Returns:
            Table interaction results and state
        """
        try:
            if data.empty:
                st.info(f"📊 {title}: No data available")
                return {'empty': True}
            
            st.subheader(f"📊 {title}")
            
            # Table configuration
            table_state = {
                'total_rows': len(data),
                'displayed_rows': 0,
                'filtered_data': data.copy(),
                'selected_rows': []
            }
            
            # Column selection
            if columns is not None:
                available_columns = [col for col in columns if col in data.columns]
                display_data = data[available_columns].copy()
            else:
                display_data = data.copy()
                available_columns = list(data.columns)
            
            # Filtering controls
            if filterable:
                display_data = TableComponents._apply_table_filters(
                    display_data, available_columns, f"{title}_filter"
                )
            
            # Sorting controls
            if sortable:
                display_data = TableComponents._apply_table_sorting(
                    display_data, available_columns, f"{title}_sort"
                )
            
            # Apply custom formatters
            if custom_formatters:
                display_data = TableComponents._apply_custom_formatters(
                    display_data, custom_formatters
                )
            
            # Pagination
            if paginated and len(display_data) > page_size:
                display_data = TableComponents._apply_pagination(
                    display_data, page_size, f"{title}_page"
                )
            
            # Update table state
            table_state['displayed_rows'] = len(display_data)
            table_state['filtered_data'] = display_data
            
            # Display table
            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=not show_index
            )
            
            # Table summary
            TableComponents._render_table_summary(table_state, title)
            
            # Export functionality
            if exportable:
                TableComponents._render_export_controls(
                    display_data, f"{title}_export"
                )
            
            return table_state
        
        except Exception as e:
            logger.error(f"Error rendering data table: {e}")
            st.error("Error displaying data table")
            return {'error': True}
    
    @staticmethod
    def render_comparison_table(datasets: Dict[str, pd.DataFrame],
                              comparison_columns: List[str],
                              title: str = "Data Comparison") -> Dict[str, Any]:
        """
        Render side-by-side comparison table for multiple datasets.
        
        Args:
            datasets: Dictionary of dataset name -> DataFrame
            comparison_columns: Columns to compare across datasets
            title: Table title
            
        Returns:
            Comparison results and statistics
        """
        try:
            st.subheader(f"⚔️ {title}")
            
            if not datasets:
                st.info("No datasets provided for comparison")
                return {'empty': True}
            
            comparison_data = []
            
            # Build comparison matrix
            for dataset_name, df in datasets.items():
                dataset_stats = {'Dataset': dataset_name}
                
                for col in comparison_columns:
                    if col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            dataset_stats[f"{col}_mean"] = df[col].mean()
                            dataset_stats[f"{col}_std"] = df[col].std()
                            dataset_stats[f"{col}_count"] = df[col].count()
                        else:
                            dataset_stats[f"{col}_unique"] = df[col].nunique()
                            dataset_stats[f"{col}_count"] = df[col].count()
                
                dataset_stats['Total_Rows'] = len(df)
                comparison_data.append(dataset_stats)
            
            # Display comparison table
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Comparison insights
            TableComponents._render_comparison_insights(comparison_df, datasets)
            
            return {'comparison_data': comparison_df, 'datasets': datasets}
        
        except Exception as e:
            logger.error(f"Error rendering comparison table: {e}")
            st.error("Error displaying comparison table")
            return {'error': True}
    
    @staticmethod
    def render_summary_table(data: pd.DataFrame,
                           group_by: str = None,
                           aggregations: Dict[str, List[str]] = None,
                           title: str = "Summary Table") -> Dict[str, Any]:
        """
        Render summary table with aggregated statistics.
        
        Args:
            data: Source DataFrame
            group_by: Column to group by
            aggregations: Dictionary of column -> aggregation functions
            title: Table title
            
        Returns:
            Summary data and statistics
        """
        try:
            st.subheader(f"📋 {title}")
            
            if data.empty:
                st.info("No data available for summary")
                return {'empty': True}
            
            # Default aggregations if none provided
            if aggregations is None:
                aggregations = {}
                for col in data.select_dtypes(include=[np.number]).columns:
                    aggregations[col] = ['count', 'mean', 'std', 'min', 'max']
            
            # Create summary
            if group_by and group_by in data.columns:
                # Grouped summary
                summary_data = TableComponents._create_grouped_summary(
                    data, group_by, aggregations
                )
            else:
                # Overall summary
                summary_data = TableComponents._create_overall_summary(
                    data, aggregations
                )
            
            # Display summary table
            st.dataframe(summary_data, use_container_width=True)
            
            # Summary insights
            TableComponents._render_summary_insights(summary_data, data)
            
            return {'summary_data': summary_data, 'source_data': data}
        
        except Exception as e:
            logger.error(f"Error rendering summary table: {e}")
            st.error("Error displaying summary table")
            return {'error': True}
    
    @staticmethod
    def render_interactive_table(data: pd.DataFrame,
                               action_column: str = "Actions",
                               row_actions: List[Dict] = None,
                               selectable: bool = True,
                               title: str = "Interactive Table") -> Dict[str, Any]:
        """
        Render interactive table with row actions and selection.
        
        Args:
            data: DataFrame to display
            action_column: Name for action column
            row_actions: List of action configurations
            selectable: Enable row selection
            title: Table title
            
        Returns:
            Table state with selections and actions
        """
        try:
            st.subheader(f"🎮 {title}")
            
            if data.empty:
                st.info("No data available for interaction")
                return {'empty': True}
            
            interaction_state = {
                'selected_rows': [],
                'performed_actions': [],
                'data': data.copy()
            }
            
            # Add selection column if enabled
            if selectable:
                data_with_selection = data.copy()
                data_with_selection.insert(0, 'Select', False)
            else:
                data_with_selection = data.copy()
            
            # Display table with selections
            if selectable:
                # Use experimental data editor for selection
                edited_df = st.data_editor(
                    data_with_selection,
                    column_config={
                        "Select": st.column_config.CheckboxColumn(
                            "Select",
                            help="Select rows for bulk actions",
                            default=False,
                        )
                    },
                    disabled=[col for col in data.columns],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Track selected rows
                selected_indices = edited_df.index[edited_df['Select']].tolist()
                interaction_state['selected_rows'] = selected_indices
                
                # Bulk actions
                if selected_indices and row_actions:
                    st.markdown("### 🎯 Bulk Actions")
                    TableComponents._render_bulk_actions(
                        row_actions, selected_indices, interaction_state
                    )
            else:
                st.dataframe(data, use_container_width=True)
            
            # Individual row actions
            if row_actions:
                TableComponents._render_row_actions(
                    data, row_actions, interaction_state
                )
            
            return interaction_state
        
        except Exception as e:
            logger.error(f"Error rendering interactive table: {e}")
            st.error("Error displaying interactive table")
            return {'error': True}
    
    @staticmethod
    def render_performance_table(data: pd.DataFrame,
                               max_rows: int = 1000,
                               virtual_scrolling: bool = True,
                               title: str = "Performance Table") -> Dict[str, Any]:
        """
        Render optimized table for large datasets.
        
        Args:
            data: Large DataFrame to display
            max_rows: Maximum rows to display at once
            virtual_scrolling: Enable virtual scrolling
            title: Table title
            
        Returns:
            Performance metrics and table state
        """
        try:
            st.subheader(f"⚡ {title}")
            
            performance_metrics = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'memory_usage': data.memory_usage(deep=True).sum(),
                'display_rows': 0
            }
            
            # Performance info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{performance_metrics['total_rows']:,}")
            with col2:
                st.metric("Total Columns", f"{performance_metrics['total_columns']:,}")
            with col3:
                memory_mb = performance_metrics['memory_usage'] / (1024 * 1024)
                st.metric("Memory Usage", f"{memory_mb:.2f} MB")
            
            # Data sampling and display options
            display_options = st.columns(4)
            
            with display_options[0]:
                sample_size = st.selectbox(
                    "Sample Size",
                    [100, 500, 1000, 5000, "All"],
                    index=2
                )
            
            with display_options[1]:
                sort_column = st.selectbox(
                    "Sort By",
                    ["None"] + list(data.columns),
                    index=0
                )
            
            with display_options[2]:
                sort_order = st.selectbox(
                    "Sort Order",
                    ["Ascending", "Descending"],
                    index=1
                )
            
            with display_options[3]:
                show_sample_only = st.checkbox(
                    "Show Sample Only",
                    value=True,
                    help="Show only a sample for better performance"
                )
            
            # Prepare display data
            display_data = data.copy()
            
            # Apply sorting
            if sort_column != "None" and sort_column in data.columns:
                ascending = sort_order == "Ascending"
                display_data = display_data.sort_values(sort_column, ascending=ascending)
            
            # Apply sampling
            if show_sample_only and sample_size != "All":
                display_data = display_data.head(int(sample_size))
            
            performance_metrics['display_rows'] = len(display_data)
            
            # Display table
            st.dataframe(display_data, use_container_width=True)
            
            # Performance summary
            st.caption(f"Displaying {performance_metrics['display_rows']:,} of {performance_metrics['total_rows']:,} rows")
            
            return performance_metrics
        
        except Exception as e:
            logger.error(f"Error rendering performance table: {e}")
            st.error("Error displaying performance table")
            return {'error': True}
    
    # Utility methods for table functionality
    @staticmethod
    def _apply_table_filters(data: pd.DataFrame, columns: List[str], key_prefix: str) -> pd.DataFrame:
        """Apply filtering controls to table data."""
        try:
            with st.expander("🔍 Filters"):
                filtered_data = data.copy()
                
                # Text search
                search_term = st.text_input(
                    "Search in all columns",
                    key=f"{key_prefix}_search",
                    help="Search across all text columns"
                )
                
                if search_term:
                    search_mask = pd.Series([False] * len(data))
                    for col in data.select_dtypes(include=['object']).columns:
                        search_mask |= data[col].astype(str).str.contains(
                            search_term, case=False, na=False
                        )
                    filtered_data = filtered_data[search_mask]
                
                # Column-specific filters
                for col in columns[:3]:  # Limit to first 3 columns for UI space
                    if col in data.columns:
                        if data[col].dtype in ['int64', 'float64']:
                            # Numeric filter
                            min_val = float(data[col].min())
                            max_val = float(data[col].max())
                            
                            filter_range = st.slider(
                                f"{col} Range",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"{key_prefix}_{col}_range"
                            )
                            
                            filtered_data = filtered_data[
                                (filtered_data[col] >= filter_range[0]) &
                                (filtered_data[col] <= filter_range[1])
                            ]
                        
                        elif data[col].nunique() <= 20:  # Categorical filter
                            unique_values = sorted(data[col].unique())
                            selected_values = st.multiselect(
                                f"{col} Values",
                                unique_values,
                                default=unique_values,
                                key=f"{key_prefix}_{col}_values"
                            )
                            
                            if selected_values:
                                filtered_data = filtered_data[
                                    filtered_data[col].isin(selected_values)
                                ]
                
                return filtered_data
        
        except Exception as e:
            logger.error(f"Error applying table filters: {e}")
            return data
    
    @staticmethod
    def _apply_table_sorting(data: pd.DataFrame, columns: List[str], key_prefix: str) -> pd.DataFrame:
        """Apply sorting controls to table data."""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                sort_column = st.selectbox(
                    "Sort By",
                    ["None"] + columns,
                    key=f"{key_prefix}_sort_col"
                )
            
            with col2:
                sort_ascending = st.selectbox(
                    "Sort Order",
                    ["Ascending", "Descending"],
                    key=f"{key_prefix}_sort_order"
                ) == "Ascending"
            
            if sort_column and sort_column != "None":
                return data.sort_values(sort_column, ascending=sort_ascending)
            
            return data
        
        except Exception as e:
            logger.error(f"Error applying table sorting: {e}")
            return data
    
    @staticmethod
    def _apply_pagination(data: pd.DataFrame, page_size: int, key_prefix: str) -> pd.DataFrame:
        """Apply pagination to table data."""
        try:
            total_pages = max(1, (len(data) + page_size - 1) // page_size)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                current_page = st.number_input(
                    f"Page (1-{total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    key=f"{key_prefix}_page_num"
                )
            
            start_idx = (current_page - 1) * page_size
            end_idx = min(start_idx + page_size, len(data))
            
            st.caption(f"Showing rows {start_idx + 1}-{end_idx} of {len(data)}")
            
            return data.iloc[start_idx:end_idx]
        
        except Exception as e:
            logger.error(f"Error applying pagination: {e}")
            return data.head(page_size)
    
    @staticmethod
    def _apply_custom_formatters(data: pd.DataFrame, formatters: Dict[str, Callable]) -> pd.DataFrame:
        """Apply custom formatters to table columns."""
        try:
            formatted_data = data.copy()
            
            for column, formatter in formatters.items():
                if column in data.columns:
                    formatted_data[column] = data[column].apply(formatter)
            
            return formatted_data
        
        except Exception as e:
            logger.error(f"Error applying custom formatters: {e}")
            return data
    
    @staticmethod
    def _render_table_summary(table_state: Dict, title: str) -> None:
        """Render table summary statistics."""
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", f"{table_state['total_rows']:,}")
            with col2:
                st.metric("Displayed Rows", f"{table_state['displayed_rows']:,}")
            with col3:
                filter_ratio = table_state['displayed_rows'] / table_state['total_rows'] * 100
                st.metric("Filter Ratio", f"{filter_ratio:.1f}%")
        
        except Exception as e:
            logger.error(f"Error rendering table summary: {e}")
    
    @staticmethod
    def _render_export_controls(data: pd.DataFrame, key_prefix: str) -> None:
        """Render export functionality controls."""
        try:
            st.markdown("### 💾 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Export CSV", key=f"{key_prefix}_csv"):
                    csv_data = data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"table_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Export Excel", key=f"{key_prefix}_excel"):
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        data.to_excel(writer, sheet_name='Data', index=False)
                    
                    st.download_button(
                        label="Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"table_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col3:
                if st.button("📋 Export JSON", key=f"{key_prefix}_json"):
                    json_data = data.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"table_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        except Exception as e:
            logger.error(f"Error rendering export controls: {e}")
    
    @staticmethod
    def _create_grouped_summary(data: pd.DataFrame, group_by: str, aggregations: Dict) -> pd.DataFrame:
        """Create grouped summary statistics."""
        try:
            grouped = data.groupby(group_by)
            summary_data = []
            
            for name, group in grouped:
                summary_row = {group_by: name}
                
                for column, agg_funcs in aggregations.items():
                    if column in group.columns:
                        for func in agg_funcs:
                            try:
                                if func == 'count':
                                    summary_row[f"{column}_{func}"] = group[column].count()
                                elif func == 'mean':
                                    summary_row[f"{column}_{func}"] = group[column].mean()
                                elif func == 'std':
                                    summary_row[f"{column}_{func}"] = group[column].std()
                                elif func == 'min':
                                    summary_row[f"{column}_{func}"] = group[column].min()
                                elif func == 'max':
                                    summary_row[f"{column}_{func}"] = group[column].max()
                                elif func == 'sum':
                                    summary_row[f"{column}_{func}"] = group[column].sum()
                            except Exception:
                                summary_row[f"{column}_{func}"] = None
                
                summary_data.append(summary_row)
            
            return pd.DataFrame(summary_data)
        
        except Exception as e:
            logger.error(f"Error creating grouped summary: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _create_overall_summary(data: pd.DataFrame, aggregations: Dict) -> pd.DataFrame:
        """Create overall summary statistics."""
        try:
            summary_data = {}
            
            for column, agg_funcs in aggregations.items():
                if column in data.columns:
                    for func in agg_funcs:
                        try:
                            if func == 'count':
                                summary_data[f"{column}_{func}"] = data[column].count()
                            elif func == 'mean':
                                summary_data[f"{column}_{func}"] = data[column].mean()
                            elif func == 'std':
                                summary_data[f"{column}_{func}"] = data[column].std()
                            elif func == 'min':
                                summary_data[f"{column}_{func}"] = data[column].min()
                            elif func == 'max':
                                summary_data[f"{column}_{func}"] = data[column].max()
                            elif func == 'sum':
                                summary_data[f"{column}_{func}"] = data[column].sum()
                        except Exception:
                            summary_data[f"{column}_{func}"] = None
            
            return pd.DataFrame([summary_data])
        
        except Exception as e:
            logger.error(f"Error creating overall summary: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _render_summary_insights(summary_data: pd.DataFrame, source_data: pd.DataFrame) -> None:
        """Render insights from summary data."""
        try:
            st.markdown("#### 💡 Summary Insights")
            
            insights = []
            
            # Find highest values
            numeric_columns = summary_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if '_mean' in col:
                    base_col = col.replace('_mean', '')
                    mean_val = summary_data[col].iloc[0] if not summary_data.empty else 0
                    insights.append(f"Average {base_col}: {mean_val:.2f}")
            
            for insight in insights[:3]:  # Show top 3 insights
                st.info(f"ℹ️ {insight}")
        
        except Exception as e:
            logger.error(f"Error rendering summary insights: {e}")
    
    @staticmethod
    def _render_comparison_insights(comparison_df: pd.DataFrame, datasets: Dict) -> None:
        """Render insights from comparison data."""
        try:
            st.markdown("#### ⚔️ Comparison Insights")
            
            insights = []
            
            # Find dataset with most rows
            if 'Total_Rows' in comparison_df.columns:
                max_rows_idx = comparison_df['Total_Rows'].idxmax()
                max_dataset = comparison_df.loc[max_rows_idx, 'Dataset']
                max_rows = comparison_df.loc[max_rows_idx, 'Total_Rows']
                insights.append(f"Largest dataset: {max_dataset} ({max_rows:,} rows)")
            
            # Find datasets with significant differences
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'Total_Rows' and comparison_df[col].std() > 0:
                    cv = comparison_df[col].std() / comparison_df[col].mean()
                    if cv > 0.5:  # High coefficient of variation
                        insights.append(f"High variation in {col} across datasets")
            
            for insight in insights[:3]:  # Show top 3 insights
                st.info(f"ℹ️ {insight}")
        
        except Exception as e:
            logger.error(f"Error rendering comparison insights: {e}")
    
    @staticmethod
    def _render_bulk_actions(row_actions: List[Dict], selected_indices: List[int], interaction_state: Dict) -> None:
        """Render bulk action controls."""
        try:
            cols = st.columns(len(row_actions))
            
            for i, action in enumerate(row_actions):
                with cols[i]:
                    if st.button(
                        f"{action.get('icon', '🎯')} {action['label']} ({len(selected_indices)})",
                        key=f"bulk_{action['key']}",
                        help=f"Apply {action['label']} to selected rows"
                    ):
                        interaction_state['performed_actions'].append({
                            'action': action['key'],
                            'type': 'bulk',
                            'rows': selected_indices,
                            'timestamp': datetime.now()
                        })
                        
                        if 'callback' in action:
                            action['callback'](selected_indices)
                        
                        st.success(f"Applied {action['label']} to {len(selected_indices)} rows")
        
        except Exception as e:
            logger.error(f"Error rendering bulk actions: {e}")
    
    @staticmethod
    def _render_row_actions(data: pd.DataFrame, row_actions: List[Dict], interaction_state: Dict) -> None:
        """Render individual row action controls."""
        try:
            with st.expander("🎮 Row Actions"):
                st.markdown("Select a row to perform individual actions:")
                
                row_index = st.selectbox(
                    "Select Row",
                    range(len(data)),
                    format_func=lambda x: f"Row {x+1}: {data.iloc[x].to_dict()}"
                )
                
                if row_index is not None:
                    cols = st.columns(len(row_actions))
                    
                    for i, action in enumerate(row_actions):
                        with cols[i]:
                            if st.button(
                                f"{action.get('icon', '🎯')} {action['label']}",
                                key=f"row_{action['key']}_{row_index}",
                                help=f"Apply {action['label']} to row {row_index+1}"
                            ):
                                interaction_state['performed_actions'].append({
                                    'action': action['key'],
                                    'type': 'individual',
                                    'row': row_index,
                                    'timestamp': datetime.now()
                                })
                                
                                if 'callback' in action:
                                    action['callback'](row_index, data.iloc[row_index])
                                
                                st.success(f"Applied {action['label']} to row {row_index+1}")
        
        except Exception as e:
            logger.error(f"Error rendering row actions: {e}")


class DataTable:
    """Generic data table component with advanced features."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data table.
        
        Args:
            config: Table configuration
        """
        self.config = config or {}
        self.page_size = self.config.get('page_size', 20)
        self.sortable = self.config.get('sortable', True)
        self.filterable = self.config.get('filterable', True)
    
    def render(self, data: pd.DataFrame, 
               title: str = "Data Table",
               columns: Optional[List[str]] = None,
               show_index: bool = False,
               custom_formatters: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """
        Render data table.
        
        Args:
            data: DataFrame to display
            title: Table title
            columns: Columns to display (None for all)
            show_index: Whether to show row index
            custom_formatters: Custom column formatters
            
        Returns:
            Table interaction results
        """
        try:
            if data.empty:
                st.warning("⚠️ No data to display")
                return {}
            
            st.subheader(title)
            
            # Column selection
            if columns is None:
                columns = data.columns.tolist()
            
            display_data = data[columns].copy()
            
            # Apply custom formatters
            if custom_formatters:
                display_data = self._apply_formatters(display_data, custom_formatters)
            
            # Filtering
            if self.filterable:
                display_data = self._render_filters(display_data)
            
            # Sorting
            if self.sortable:
                display_data = self._render_sorting(display_data)
            
            # Pagination
            display_data, pagination_info = self._render_pagination(display_data)
            
            # Display table
            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=not show_index
            )
            
            # Table info
            self._render_table_info(pagination_info)
            
            # Action buttons
            return self._render_table_actions(data, display_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to render data table: {e}")
            st.error(f"Failed to display data table: {e}")
            return {}
    
    def _apply_formatters(self, data: pd.DataFrame, 
                         formatters: Dict[str, Callable]) -> pd.DataFrame:
        """Apply custom formatters to columns."""
        formatted_data = data.copy()
        
        for column, formatter in formatters.items():
            if column in formatted_data.columns:
                try:
                    formatted_data[column] = formatted_data[column].apply(formatter)
                except Exception as e:
                    logger.warning(f"⚠️ Formatter failed for column {column}: {e}")
        
        return formatted_data
    
    def _render_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Render column filters."""
        filtered_data = data.copy()
        
        with st.expander("🔍 Filters"):
            filter_cols = st.columns(min(len(data.columns), 3))
            
            for i, column in enumerate(data.columns):
                with filter_cols[i % len(filter_cols)]:
                    if data[column].dtype == 'object':
                        # Text filter
                        filter_value = st.text_input(
                            f"Filter {column}:",
                            key=f"filter_{column}"
                        )
                        if filter_value:
                            filtered_data = filtered_data[
                                filtered_data[column].astype(str).str.contains(
                                    filter_value, case=False, na=False
                                )
                            ]
                    
                    elif pd.api.types.is_numeric_dtype(data[column]):
                        # Numeric range filter
                        min_val = float(data[column].min())
                        max_val = float(data[column].max())
                        
                        if min_val != max_val:
                            range_values = st.slider(
                                f"Range {column}:",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"range_{column}"
                            )
                            filtered_data = filtered_data[
                                (filtered_data[column] >= range_values[0]) &
                                (filtered_data[column] <= range_values[1])
                            ]
        
        return filtered_data
    
    def _render_sorting(self, data: pd.DataFrame) -> pd.DataFrame:
        """Render sorting controls."""
        if data.empty:
            return data
        
        col1, col2 = st.columns(2)
        
        with col1:
            sort_column = st.selectbox(
                "Sort by:",
                options=['None'] + data.columns.tolist(),
                key="sort_column"
            )
        
        with col2:
            sort_order = st.selectbox(
                "Sort order:",
                options=['Ascending', 'Descending'],
                key="sort_order"
            )
        
        if sort_column and sort_column != 'None':
            ascending = sort_order == 'Ascending'
            data = data.sort_values(by=sort_column, ascending=ascending)
        
        return data
    
    def _render_pagination(self, data: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Render pagination controls."""
        total_rows = len(data)
        
        if total_rows <= self.page_size:
            return data, {
                'total_rows': total_rows,
                'current_page': 1,
                'total_pages': 1,
                'showing_from': 1,
                'showing_to': total_rows
            }
        
        total_pages = (total_rows - 1) // self.page_size + 1
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            current_page = st.number_input(
                f"Page (1-{total_pages}):",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="table_page"
            )
        
        start_idx = (current_page - 1) * self.page_size
        end_idx = min(start_idx + self.page_size, total_rows)
        
        paginated_data = data.iloc[start_idx:end_idx]
        
        pagination_info = {
            'total_rows': total_rows,
            'current_page': current_page,
            'total_pages': total_pages,
            'showing_from': start_idx + 1,
            'showing_to': end_idx
        }
        
        return paginated_data, pagination_info
    
    def _render_table_info(self, pagination_info: Dict[str, Any]) -> None:
        """Render table information."""
        st.markdown(
            f"Showing {pagination_info['showing_from']}-{pagination_info['showing_to']} "
            f"of {pagination_info['total_rows']} rows "
            f"(Page {pagination_info['current_page']} of {pagination_info['total_pages']})"
        )
    
    def _render_table_actions(self, original_data: pd.DataFrame, 
                            displayed_data: pd.DataFrame) -> Dict[str, Any]:
        """Render table action buttons."""
        actions = {}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📥 Export CSV", key="export_csv"):
                csv = displayed_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                actions['export_csv'] = True
        
        with col2:
            if st.button("📊 Show Stats", key="show_stats"):
                actions['show_stats'] = True
                self._show_table_statistics(displayed_data)
        
        with col3:
            if st.button("🔄 Refresh", key="refresh_table"):
                actions['refresh'] = True
        
        with col4:
            if st.button("⚙️ Configure", key="config_table"):
                actions['configure'] = True
        
        return actions
    
    def _show_table_statistics(self, data: pd.DataFrame) -> None:
        """Show table statistics."""
        with st.expander("📊 Table Statistics", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Stats:**")
                st.write(f"Rows: {len(data)}")
                st.write(f"Columns: {len(data.columns)}")
                st.write(f"Memory Usage: {data.memory_usage().sum() / 1024:.1f} KB")
            
            with col2:
                st.markdown("**Data Types:**")
                for dtype, count in data.dtypes.value_counts().items():
                    st.write(f"{dtype}: {count} columns")
            
            # Numeric columns summary
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.markdown("**Numeric Summary:**")
                st.dataframe(data[numeric_cols].describe(), use_container_width=True)
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class HistoryTable:
    """Table component for displaying prediction history."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize history table."""
        self.config = config or {}
    
    def render(self, history_data: List[Dict[str, Any]], 
               title: str = "Prediction History") -> Dict[str, Any]:
        """
        Render history table.
        
        Args:
            history_data: List of prediction history records
            title: Table title
            
        Returns:
            Table interaction results
        """
        try:
            if not history_data:
                st.warning("⚠️ No history data available")
                return {}
            
            st.subheader(title)
            
            # Convert to DataFrame
            df = self._prepare_history_dataframe(history_data)
            
            # Custom formatters
            formatters = {
                'numbers': self._format_numbers,
                'confidence': lambda x: f"{x:.1%}" if pd.notna(x) else "N/A",
                'date': lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, datetime) else str(x)
            }
            
            # Use DataTable component
            data_table = DataTable(self.config)
            actions = data_table.render(
                df,
                title="",
                custom_formatters=formatters
            )
            
            # History-specific actions
            return self._render_history_actions(df, actions)
            
        except Exception as e:
            logger.error(f"❌ Failed to render history table: {e}")
            st.error(f"Failed to display history table: {e}")
            return {}
    
    def _prepare_history_dataframe(self, history_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare history data for display."""
        records = []
        
        for i, record in enumerate(history_data):
            records.append({
                'ID': i + 1,
                'date': record.get('generated_at', datetime.now()),
                'numbers': record.get('numbers', []),
                'confidence': record.get('confidence', 0.0),
                'strategy': record.get('strategy', 'unknown'),
                'model': record.get('model_name', 'unknown'),
                'result': record.get('result', 'pending')
            })
        
        return pd.DataFrame(records)
    
    def _format_numbers(self, numbers: List[int]) -> str:
        """Format numbers for display."""
        if isinstance(numbers, list):
            return ", ".join(map(str, sorted(numbers)))
        return str(numbers)
    
    def _render_history_actions(self, df: pd.DataFrame, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Render history-specific actions."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Replay Prediction", key="replay_prediction"):
                actions['replay'] = True
        
        with col2:
            if st.button("📈 Analyze Performance", key="analyze_performance"):
                actions['analyze'] = True
                self._show_performance_analysis(df)
        
        with col3:
            if st.button("🗑️ Clear History", key="clear_history"):
                actions['clear'] = True
        
        return actions
    
    def _show_performance_analysis(self, df: pd.DataFrame) -> None:
        """Show performance analysis."""
        with st.expander("📈 Performance Analysis", expanded=True):
            if 'confidence' in df.columns:
                avg_confidence = df['confidence'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
            
            if 'strategy' in df.columns:
                strategy_counts = df['strategy'].value_counts()
                st.markdown("**Strategy Usage:**")
                for strategy, count in strategy_counts.items():
                    st.write(f"- {strategy}: {count}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class MetricsTable:
    """Table component for displaying model metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize metrics table."""
        self.config = config or {}
    
    def render(self, metrics_data: Dict[str, Dict[str, float]], 
               title: str = "Model Metrics") -> Dict[str, Any]:
        """
        Render metrics table.
        
        Args:
            metrics_data: Dictionary of model metrics
            title: Table title
            
        Returns:
            Table interaction results
        """
        try:
            if not metrics_data:
                st.warning("⚠️ No metrics data available")
                return {}
            
            st.subheader(title)
            
            # Convert to DataFrame
            df = self._prepare_metrics_dataframe(metrics_data)
            
            # Custom formatters for metrics
            formatters = {}
            for col in df.columns:
                if col != 'Model':
                    if 'accuracy' in col.lower() or 'confidence' in col.lower():
                        formatters[col] = lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                    elif 'time' in col.lower():
                        formatters[col] = lambda x: f"{x:.2f}s" if pd.notna(x) else "N/A"
                    else:
                        formatters[col] = lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            
            # Display table
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Metrics analysis
            return self._render_metrics_analysis(df)
            
        except Exception as e:
            logger.error(f"❌ Failed to render metrics table: {e}")
            st.error(f"Failed to display metrics table: {e}")
            return {}
    
    def _prepare_metrics_dataframe(self, metrics_data: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Prepare metrics data for display."""
        records = []
        
        for model_name, metrics in metrics_data.items():
            record = {'Model': model_name}
            record.update(metrics)
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _render_metrics_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Render metrics analysis."""
        actions = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Compare Models", key="compare_models"):
                actions['compare'] = True
                self._show_model_comparison(df)
        
        with col2:
            if st.button("🏆 Best Performer", key="best_performer"):
                actions['best_performer'] = True
                self._show_best_performer(df)
        
        with col3:
            if st.button("📈 Trends", key="metrics_trends"):
                actions['trends'] = True
        
        return actions
    
    def _show_model_comparison(self, df: pd.DataFrame) -> None:
        """Show model comparison."""
        with st.expander("📊 Model Comparison", expanded=True):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                comparison_metric = st.selectbox(
                    "Compare by:",
                    options=numeric_cols.tolist(),
                    key="comparison_metric"
                )
                
                if comparison_metric:
                    sorted_df = df.sort_values(by=comparison_metric, ascending=False)
                    st.dataframe(
                        sorted_df[['Model', comparison_metric]], 
                        use_container_width=True,
                        hide_index=True
                    )
    
    def _show_best_performer(self, df: pd.DataFrame) -> None:
        """Show best performing model."""
        with st.expander("🏆 Best Performer", expanded=True):
            # Find best performer by accuracy (or first numeric column)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                best_metric = 'accuracy' if 'accuracy' in numeric_cols else numeric_cols[0]
                best_model_idx = df[best_metric].idxmax()
                best_model = df.iloc[best_model_idx]
                
                st.success(f"🏆 **Best Model:** {best_model['Model']}")
                st.write(f"**{best_metric.title()}:** {best_model[best_metric]:.3f}")
                
                # Show all metrics for best model
                for col in numeric_cols:
                    if col != best_metric:
                        st.write(f"**{col.title()}:** {best_model[col]:.3f}")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class ComparisonTable:
    """Table component for side-by-side comparisons."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize comparison table."""
        self.config = config or {}
    
    def render(self, comparison_data: Dict[str, List[Any]], 
               title: str = "Comparison Table") -> Dict[str, Any]:
        """
        Render comparison table.
        
        Args:
            comparison_data: Dictionary with comparison categories and values
            title: Table title
            
        Returns:
            Table interaction results
        """
        try:
            if not comparison_data:
                st.warning("⚠️ No comparison data available")
                return {}
            
            st.subheader(title)
            
            # Convert to DataFrame
            df = pd.DataFrame(comparison_data)
            
            # Style the comparison table
            styled_df = self._style_comparison_table(df)
            
            # Display table
            st.dataframe(styled_df, use_container_width=True)
            
            # Comparison analysis
            return self._render_comparison_analysis(df)
            
        except Exception as e:
            logger.error(f"❌ Failed to render comparison table: {e}")
            st.error(f"Failed to display comparison table: {e}")
            return {}
    
    def _style_comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply styling to comparison table."""
        # This is a simplified styling - in a real implementation,
        # you might use pandas styling features
        return df
    
    def _render_comparison_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Render comparison analysis."""
        actions = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Highlight Differences", key="highlight_diffs"):
                actions['highlight'] = True
                self._highlight_differences(df)
        
        with col2:
            if st.button("📋 Summary", key="comparison_summary"):
                actions['summary'] = True
                self._show_comparison_summary(df)
        
        return actions
    
    def _highlight_differences(self, df: pd.DataFrame) -> None:
        """Highlight differences in comparison."""
        with st.expander("📊 Differences Highlighted", expanded=True):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if len(df[col].unique()) > 1:
                    max_val = df[col].max()
                    min_val = df[col].min()
                    
                    max_idx = df[col].idxmax()
                    min_idx = df[col].idxmin()
                    
                    st.write(f"**{col}:**")
                    st.write(f"  🔹 Highest: {max_val} (Row {max_idx})")
                    st.write(f"  🔹 Lowest: {min_val} (Row {min_idx})")
    
    def _show_comparison_summary(self, df: pd.DataFrame) -> None:
        """Show comparison summary."""
        with st.expander("📋 Comparison Summary", expanded=True):
            st.write(f"**Items Compared:** {len(df)}")
            st.write(f"**Categories:** {len(df.columns)}")
            
            # Identify best in each category
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                st.markdown("**Best in Category:**")
                for col in numeric_cols:
                    best_idx = df[col].idxmax()
                    st.write(f"- {col}: Row {best_idx} ({df.iloc[best_idx][col]})")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


# Utility functions for tables
def create_sample_table_data() -> pd.DataFrame:
    """Create sample data for testing tables."""
    np.random.seed(42)
    
    data = {
        'Model': ['Mathematical', 'Ensemble', 'Temporal', 'Optimizer'],
        'Accuracy': np.random.uniform(0.6, 0.9, 4),
        'Confidence': np.random.uniform(0.5, 0.8, 4),
        'Training Time': np.random.uniform(30, 300, 4),
        'Predictions': np.random.randint(100, 1000, 4),
        'Status': ['Active', 'Training', 'Active', 'Inactive']
    }
    
    return pd.DataFrame(data)


def export_table_data(df: pd.DataFrame, format: str = 'csv') -> str:
    """
    Export table data to various formats.
    
    Args:
        df: DataFrame to export
        format: Export format ('csv', 'json', 'excel')
        
    Returns:
        Exported data as string
    """
    try:
        if format == 'csv':
            return df.to_csv(index=False)
        elif format == 'json':
            return df.to_json(orient='records', indent=2)
        elif format == 'excel':
            # This would require additional handling for binary data
            return df.to_csv(index=False)  # Fallback to CSV
        else:
            return df.to_csv(index=False)
            
    except Exception as e:
        logger.error(f"❌ Failed to export table data: {e}")
        return ""


def filter_dataframe(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filters to DataFrame.
    
    Args:
        df: DataFrame to filter
        filters: Dictionary of column filters
        
    Returns:
        Filtered DataFrame
    """
    try:
        filtered_df = df.copy()
        
        for column, filter_value in filters.items():
            if column in filtered_df.columns and filter_value is not None:
                if isinstance(filter_value, str):
                    # Text filter
                    filtered_df = filtered_df[
                        filtered_df[column].astype(str).str.contains(
                            filter_value, case=False, na=False
                        )
                    ]
                elif isinstance(filter_value, (int, float)):
                    # Exact match for numeric
                    filtered_df = filtered_df[filtered_df[column] == filter_value]
                elif isinstance(filter_value, tuple) and len(filter_value) == 2:
                    # Range filter
                    min_val, max_val = filter_value
                    filtered_df = filtered_df[
                        (filtered_df[column] >= min_val) & 
                        (filtered_df[column] <= max_val)
                    ]
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"❌ Failed to filter DataFrame: {e}")
        return df


# Backward compatibility classes - maintain original interfaces
class DataTable:
    """
    Legacy DataTable class for backward compatibility.
    
    This class maintains the original simple interface while delegating
    to the enhanced TableComponents class for actual functionality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize data table (legacy constructor)."""
        self.config = config or {}
        self.page_size = self.config.get('page_size', 20)
        self.sortable = self.config.get('sortable', True)
        self.filterable = self.config.get('filterable', True)
    
    def render(self, data: pd.DataFrame, 
               title: str = "Data Table",
               columns: Optional[List[str]] = None,
               show_index: bool = False,
               custom_formatters: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """Render data table (legacy method)."""
        return TableComponents.render_data_table(
            data=data,
            title=title,
            columns=columns,
            sortable=self.sortable,
            filterable=self.filterable,
            paginated=True,
            page_size=self.page_size,
            show_index=show_index,
            custom_formatters=custom_formatters,
            exportable=True
        )
    
    def set_page_size(self, page_size: int) -> None:
        """Set page size (legacy method)."""
        self.page_size = page_size
    
    def enable_filtering(self, enabled: bool = True) -> None:
        """Enable/disable filtering (legacy method)."""
        self.filterable = enabled
    
    def enable_sorting(self, enabled: bool = True) -> None:
        """Enable/disable sorting (legacy method)."""
        self.sortable = enabled


class TableBuilder:
    """
    Legacy TableBuilder class for backward compatibility.
    
    This class provides helper methods that maintain the original interface.
    """
    
    @staticmethod
    def create_simple_table(data: pd.DataFrame, title: str = "Simple Table") -> Dict[str, Any]:
        """Create simple table (legacy method)."""
        return TableComponents.render_data_table(
            data, title, sortable=False, filterable=False, paginated=False
        )
    
    @staticmethod
    def create_sortable_table(data: pd.DataFrame, title: str = "Sortable Table") -> Dict[str, Any]:
        """Create sortable table (legacy method)."""
        return TableComponents.render_data_table(
            data, title, sortable=True, filterable=False, paginated=False
        )
    
    @staticmethod
    def create_filtered_table(data: pd.DataFrame, title: str = "Filtered Table") -> Dict[str, Any]:
        """Create filtered table (legacy method)."""
        return TableComponents.render_data_table(
            data, title, sortable=True, filterable=True, paginated=True
        )
    
    @staticmethod
    def create_comparison_table(datasets: Dict[str, pd.DataFrame], 
                              comparison_columns: List[str],
                              title: str = "Comparison") -> Dict[str, Any]:
        """Create comparison table (legacy method)."""
        return TableComponents.render_comparison_table(datasets, comparison_columns, title)
    
    @staticmethod
    def create_summary_table(data: pd.DataFrame, 
                           group_by: str = None,
                           title: str = "Summary") -> Dict[str, Any]:
        """Create summary table (legacy method)."""
        return TableComponents.render_summary_table(data, group_by, None, title)


class TableExporter:
    """
    Legacy TableExporter class for backward compatibility.
    
    This class provides export methods that maintain the original interface.
    """
    
    @staticmethod
    def export_to_csv(data: pd.DataFrame, filename: str = None) -> str:
        """Export to CSV (legacy method)."""
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        csv_data = data.to_csv(index=False)
        return csv_data
    
    @staticmethod
    def export_to_json(data: pd.DataFrame, filename: str = None) -> str:
        """Export to JSON (legacy method)."""
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        json_data = data.to_json(orient='records', indent=2)
        return json_data
    
    @staticmethod
    def export_to_excel(data: pd.DataFrame, filename: str = None) -> bytes:
        """Export to Excel (legacy method)."""
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            data.to_excel(writer, sheet_name='Data', index=False)
        
        return excel_buffer.getvalue()


# Export classes for easy importing
__all__ = [
    'TableComponents',
    'DataTable', 
    'TableBuilder', 
    'TableExporter'
]