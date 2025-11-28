"""
Export service module for the lottery prediction system.

This module provides comprehensive data export capabilities including 
CSV, JSON, Excel, and PDF exports with report generation features.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging
from pathlib import Path
from io import BytesIO, StringIO
import zipfile
import base64

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExportFormat:
    """Export format constants."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "xlsx"
    PDF = "pdf"
    ZIP = "zip"
    
    @classmethod
    def get_all_formats(cls) -> List[str]:
        """Get all available export formats."""
        formats = [cls.CSV, cls.JSON, cls.EXCEL, cls.ZIP]
        if PDF_AVAILABLE:
            formats.append(cls.PDF)
        return formats


class ExportService:
    """
    Comprehensive data export service.
    
    This service handles exporting predictions, historical data, statistics,
    and reports in multiple formats including CSV, JSON, Excel, and PDF.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize export service."""
        self.config = config or {}
        self.export_dir = Path(self.config.get('export_dir', 'exports'))
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export settings
        self.max_file_size_mb = self.config.get('max_file_size_mb', 100)
        self.include_metadata = self.config.get('include_metadata', True)
        self.timestamp_files = self.config.get('timestamp_files', True)
        
        logger.info(f"✅ Export service initialized with directory: {self.export_dir}")
    
    def export_predictions(self, predictions: List[Dict[str, Any]], 
                          format_type: str = ExportFormat.CSV,
                          filename: Optional[str] = None,
                          include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export prediction data.
        
        Args:
            predictions: List of prediction dictionaries
            format_type: Export format (csv, json, excel, pdf)
            filename: Custom filename (optional)
            include_metadata: Include metadata in export
            
        Returns:
            Export result with file path and metadata
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.timestamp_files else ""
                filename = f"predictions_{timestamp}.{format_type}"
            
            # Prepare data
            export_data = self._prepare_prediction_data(predictions, include_metadata)
            
            # Export based on format
            if format_type == ExportFormat.CSV:
                result = self._export_to_csv(export_data, filename)
            elif format_type == ExportFormat.JSON:
                result = self._export_to_json(export_data, filename)
            elif format_type == ExportFormat.EXCEL:
                result = self._export_to_excel(export_data, filename, "Predictions")
            elif format_type == ExportFormat.PDF:
                result = self._export_predictions_to_pdf(predictions, filename)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"✅ Predictions exported to {result['file_path']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to export predictions: {e}")
            raise
    
    def export_historical_data(self, data: pd.DataFrame,
                              format_type: str = ExportFormat.CSV,
                              filename: Optional[str] = None,
                              date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Export historical lottery data.
        
        Args:
            data: Historical data DataFrame
            format_type: Export format
            filename: Custom filename (optional)
            date_range: Optional date range filter
            
        Returns:
            Export result with file path and metadata
        """
        try:
            # Filter by date range if provided
            if date_range and 'draw_date' in data.columns:
                start_date, end_date = date_range
                data = data[
                    (pd.to_datetime(data['draw_date']) >= start_date) &
                    (pd.to_datetime(data['draw_date']) <= end_date)
                ]
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.timestamp_files else ""
                filename = f"historical_data_{timestamp}.{format_type}"
            
            # Export based on format
            if format_type == ExportFormat.CSV:
                result = self._export_dataframe_to_csv(data, filename)
            elif format_type == ExportFormat.JSON:
                result = self._export_dataframe_to_json(data, filename)
            elif format_type == ExportFormat.EXCEL:
                result = self._export_dataframe_to_excel(data, filename, "Historical Data")
            elif format_type == ExportFormat.PDF:
                result = self._export_dataframe_to_pdf(data, filename, "Historical Lottery Data")
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"✅ Historical data exported to {result['file_path']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to export historical data: {e}")
            raise
    
    def export_statistics(self, statistics: Dict[str, Any],
                         format_type: str = ExportFormat.JSON,
                         filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Export statistical analysis data.
        
        Args:
            statistics: Statistics dictionary
            format_type: Export format
            filename: Custom filename (optional)
            
        Returns:
            Export result with file path and metadata
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.timestamp_files else ""
                filename = f"statistics_{timestamp}.{format_type}"
            
            # Prepare statistics data
            export_data = self._prepare_statistics_data(statistics)
            
            # Export based on format
            if format_type == ExportFormat.JSON:
                result = self._export_to_json(export_data, filename)
            elif format_type == ExportFormat.CSV:
                # Convert statistics to DataFrame for CSV export
                df = self._statistics_to_dataframe(statistics)
                result = self._export_dataframe_to_csv(df, filename)
            elif format_type == ExportFormat.EXCEL:
                result = self._export_statistics_to_excel(statistics, filename)
            elif format_type == ExportFormat.PDF:
                result = self._export_statistics_to_pdf(statistics, filename)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"✅ Statistics exported to {result['file_path']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to export statistics: {e}")
            raise
    
    def export_model_report(self, model_data: Dict[str, Any],
                           performance_data: Dict[str, Any],
                           filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Export comprehensive model performance report.
        
        Args:
            model_data: Model configuration and metadata
            performance_data: Model performance metrics
            filename: Custom filename (optional)
            
        Returns:
            Export result with file path and metadata
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.timestamp_files else ""
                filename = f"model_report_{timestamp}.pdf"
            
            if PDF_AVAILABLE:
                result = self._export_model_report_to_pdf(model_data, performance_data, filename)
            else:
                # Fallback to JSON if PDF not available
                filename = filename.replace('.pdf', '.json')
                combined_data = {
                    'model_data': model_data,
                    'performance_data': performance_data,
                    'report_generated': datetime.now().isoformat()
                }
                result = self._export_to_json(combined_data, filename)
            
            logger.info(f"✅ Model report exported to {result['file_path']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to export model report: {e}")
            raise
    
    def create_export_package(self, export_items: List[Dict[str, Any]],
                             package_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a ZIP package containing multiple exports.
        
        Args:
            export_items: List of export items with type and data
            package_name: Custom package name (optional)
            
        Returns:
            Export result with package path and metadata
        """
        try:
            # Generate package name if not provided
            if not package_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.timestamp_files else ""
                package_name = f"lottery_data_package_{timestamp}.zip"
            
            package_path = self.export_dir / package_name
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for item in export_items:
                    item_type = item.get('type', 'data')
                    item_data = item.get('data')
                    item_filename = item.get('filename', f"{item_type}.json")
                    
                    # Export individual item to memory
                    if item_type == 'predictions':
                        temp_data = self._prepare_prediction_data(item_data, True)
                        content = json.dumps(temp_data, indent=2, default=str)
                    elif item_type == 'historical_data':
                        content = item_data.to_csv(index=False) if isinstance(item_data, pd.DataFrame) else str(item_data)
                    elif item_type == 'statistics':
                        temp_data = self._prepare_statistics_data(item_data)
                        content = json.dumps(temp_data, indent=2, default=str)
                    else:
                        content = json.dumps(item_data, indent=2, default=str)
                    
                    # Add to ZIP
                    zip_file.writestr(item_filename, content)
                
                # Add package manifest
                manifest = {
                    'package_name': package_name,
                    'created_at': datetime.now().isoformat(),
                    'items': [
                        {
                            'type': item.get('type'),
                            'filename': item.get('filename'),
                            'description': item.get('description', '')
                        }
                        for item in export_items
                    ],
                    'total_items': len(export_items)
                }
                zip_file.writestr('manifest.json', json.dumps(manifest, indent=2))
            
            # Get file size
            file_size = package_path.stat().st_size
            
            result = {
                'success': True,
                'file_path': str(package_path),
                'filename': package_name,
                'format': 'zip',
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'items_count': len(export_items),
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Export package created: {package_path}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to create export package: {e}")
            raise
    
    def _prepare_prediction_data(self, predictions: List[Dict[str, Any]], 
                                include_metadata: bool) -> Dict[str, Any]:
        """Prepare prediction data for export."""
        try:
            export_data = {
                'predictions': predictions,
                'total_predictions': len(predictions),
                'exported_at': datetime.now().isoformat()
            }
            
            if include_metadata:
                export_data['metadata'] = {
                    'format_version': '1.0',
                    'source': 'lottery_prediction_system',
                    'export_settings': {
                        'include_metadata': include_metadata,
                        'timestamp_files': self.timestamp_files
                    }
                }
                
                # Add summary statistics
                if predictions:
                    confidences = [p.get('confidence', 0) for p in predictions if 'confidence' in p]
                    if confidences:
                        export_data['metadata']['statistics'] = {
                            'avg_confidence': np.mean(confidences),
                            'min_confidence': np.min(confidences),
                            'max_confidence': np.max(confidences),
                            'confidence_std': np.std(confidences)
                        }
            
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare prediction data: {e}")
            raise
    
    def _prepare_statistics_data(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare statistics data for export."""
        try:
            export_data = {
                'statistics': statistics,
                'exported_at': datetime.now().isoformat(),
                'metadata': {
                    'format_version': '1.0',
                    'source': 'lottery_prediction_system'
                }
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare statistics data: {e}")
            raise
    
    def _export_to_csv(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export data to CSV format."""
        try:
            file_path = self.export_dir / filename
            
            # Convert predictions to DataFrame
            if 'predictions' in data:
                records = []
                for i, pred in enumerate(data['predictions']):
                    record = {
                        'prediction_id': i + 1,
                        'numbers': json.dumps(pred.get('numbers', [])),
                        'confidence': pred.get('confidence', 0),
                        'strategy': pred.get('strategy', ''),
                        'generated_at': pred.get('generated_at', ''),
                        'engine': pred.get('engine', '')
                    }
                    
                    # Add any additional fields
                    for key, value in pred.items():
                        if key not in record and not isinstance(value, (dict, list)):
                            record[key] = value
                    
                    records.append(record)
                
                df = pd.DataFrame(records)
                df.to_csv(file_path, index=False)
            else:
                # Generic data export
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    f.write(json.dumps(data, indent=2, default=str))
            
            return self._create_export_result(file_path, 'csv')
            
        except Exception as e:
            logger.error(f"❌ CSV export failed: {e}")
            raise
    
    def _export_to_json(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export data to JSON format."""
        try:
            file_path = self.export_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            
            return self._create_export_result(file_path, 'json')
            
        except Exception as e:
            logger.error(f"❌ JSON export failed: {e}")
            raise
    
    def _export_to_excel(self, data: Dict[str, Any], filename: str, sheet_name: str) -> Dict[str, Any]:
        """Export data to Excel format."""
        try:
            file_path = self.export_dir / filename
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Export predictions
                if 'predictions' in data:
                    records = []
                    for i, pred in enumerate(data['predictions']):
                        record = {
                            'Prediction ID': i + 1,
                            'Numbers': ', '.join(map(str, pred.get('numbers', []))),
                            'Confidence': pred.get('confidence', 0),
                            'Strategy': pred.get('strategy', ''),
                            'Generated At': pred.get('generated_at', ''),
                            'Engine': pred.get('engine', '')
                        }
                        records.append(record)
                    
                    df = pd.DataFrame(records)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Export metadata if available
                if 'metadata' in data:
                    metadata_df = pd.DataFrame([data['metadata']])
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            return self._create_export_result(file_path, 'excel')
            
        except Exception as e:
            logger.error(f"❌ Excel export failed: {e}")
            raise
    
    def _export_dataframe_to_csv(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Export DataFrame to CSV."""
        try:
            file_path = self.export_dir / filename
            df.to_csv(file_path, index=False)
            return self._create_export_result(file_path, 'csv')
            
        except Exception as e:
            logger.error(f"❌ DataFrame CSV export failed: {e}")
            raise
    
    def _export_dataframe_to_json(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Export DataFrame to JSON."""
        try:
            file_path = self.export_dir / filename
            
            export_data = {
                'data': df.to_dict('records'),
                'metadata': {
                    'total_records': len(df),
                    'columns': list(df.columns),
                    'exported_at': datetime.now().isoformat()
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
            
            return self._create_export_result(file_path, 'json')
            
        except Exception as e:
            logger.error(f"❌ DataFrame JSON export failed: {e}")
            raise
    
    def _export_dataframe_to_excel(self, df: pd.DataFrame, filename: str, sheet_name: str) -> Dict[str, Any]:
        """Export DataFrame to Excel."""
        try:
            file_path = self.export_dir / filename
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Records', 'Columns', 'Date Range Start', 'Date Range End'],
                    'Value': [
                        len(df),
                        len(df.columns),
                        df['draw_date'].min() if 'draw_date' in df.columns else 'N/A',
                        df['draw_date'].max() if 'draw_date' in df.columns else 'N/A'
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            return self._create_export_result(file_path, 'excel')
            
        except Exception as e:
            logger.error(f"❌ DataFrame Excel export failed: {e}")
            raise
    
    def _export_predictions_to_pdf(self, predictions: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
        """Export predictions to PDF format."""
        if not PDF_AVAILABLE:
            raise ImportError("PDF export requires reportlab package")
        
        try:
            file_path = self.export_dir / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(file_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph("Lottery Predictions Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Summary
            summary_text = f"""
            <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Total Predictions:</b> {len(predictions)}<br/>
            <b>Export Format:</b> PDF Report
            """
            summary = Paragraph(summary_text, styles['Normal'])
            story.append(summary)
            story.append(Spacer(1, 20))
            
            # Predictions table
            if predictions:
                # Table headers
                table_data = [['ID', 'Numbers', 'Confidence', 'Strategy', 'Engine']]
                
                # Add prediction rows
                for i, pred in enumerate(predictions[:50]):  # Limit to first 50 for PDF
                    row = [
                        str(i + 1),
                        ', '.join(map(str, pred.get('numbers', []))),
                        f"{pred.get('confidence', 0):.2%}",
                        pred.get('strategy', ''),
                        pred.get('engine', '')
                    ]
                    table_data.append(row)
                
                # Create table
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
                
                # Add note if truncated
                if len(predictions) > 50:
                    note = Paragraph(f"<i>Note: Showing first 50 of {len(predictions)} predictions</i>", styles['Normal'])
                    story.append(Spacer(1, 12))
                    story.append(note)
            
            # Build PDF
            doc.build(story)
            
            return self._create_export_result(file_path, 'pdf')
            
        except Exception as e:
            logger.error(f"❌ PDF export failed: {e}")
            raise
    
    def _export_statistics_to_excel(self, statistics: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export statistics to Excel with multiple sheets."""
        try:
            file_path = self.export_dir / filename
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Main statistics sheet
                stats_data = []
                for key, value in statistics.items():
                    if not isinstance(value, (dict, list)):
                        stats_data.append({'Metric': key, 'Value': value})
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                # Frequency analysis if available
                if 'frequency_analysis' in statistics:
                    freq_data = statistics['frequency_analysis']
                    if isinstance(freq_data, dict):
                        freq_records = [{'Number': k, 'Frequency': v} for k, v in freq_data.items()]
                        freq_df = pd.DataFrame(freq_records)
                        freq_df.to_excel(writer, sheet_name='Frequency Analysis', index=False)
                
                # Hot/Cold numbers if available
                if 'hot_numbers' in statistics or 'cold_numbers' in statistics:
                    hot_cold_data = []
                    if 'hot_numbers' in statistics:
                        for num in statistics['hot_numbers'][:10]:
                            hot_cold_data.append({'Number': num, 'Type': 'Hot'})
                    if 'cold_numbers' in statistics:
                        for num in statistics['cold_numbers'][:10]:
                            hot_cold_data.append({'Number': num, 'Type': 'Cold'})
                    
                    if hot_cold_data:
                        hot_cold_df = pd.DataFrame(hot_cold_data)
                        hot_cold_df.to_excel(writer, sheet_name='Hot Cold Numbers', index=False)
            
            return self._create_export_result(file_path, 'excel')
            
        except Exception as e:
            logger.error(f"❌ Statistics Excel export failed: {e}")
            raise
    
    def _statistics_to_dataframe(self, statistics: Dict[str, Any]) -> pd.DataFrame:
        """Convert statistics dictionary to DataFrame."""
        try:
            records = []
            
            def flatten_dict(d, prefix=''):
                for key, value in d.items():
                    new_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        flatten_dict(value, new_key)
                    elif isinstance(value, list):
                        records.append({'Metric': new_key, 'Value': ', '.join(map(str, value))})
                    else:
                        records.append({'Metric': new_key, 'Value': value})
            
            flatten_dict(statistics)
            return pd.DataFrame(records)
            
        except Exception as e:
            logger.error(f"❌ Statistics to DataFrame conversion failed: {e}")
            raise
    
    def _create_export_result(self, file_path: Path, format_type: str) -> Dict[str, Any]:
        """Create standardized export result."""
        try:
            file_size = file_path.stat().st_size
            
            return {
                'success': True,
                'file_path': str(file_path),
                'filename': file_path.name,
                'format': format_type,
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create export result: {e}")
            raise
    
    def get_export_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get history of recent exports.
        
        Args:
            limit: Maximum number of exports to return
            
        Returns:
            List of export metadata
        """
        try:
            exports = []
            
            # Get all files in export directory
            for file_path in self.export_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    exports.append({
                        'filename': file_path.name,
                        'file_path': str(file_path),
                        'size_bytes': stat.st_size,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'format': file_path.suffix.lstrip('.')
                    })
            
            # Sort by creation time (newest first) and limit
            exports.sort(key=lambda x: x['created_at'], reverse=True)
            return exports[:limit]
            
        except Exception as e:
            logger.error(f"❌ Failed to get export history: {e}")
            return []
    
    def cleanup_old_exports(self, days_old: int = 30) -> Dict[str, Any]:
        """
        Clean up old export files.
        
        Args:
            days_old: Delete files older than this many days
            
        Returns:
            Cleanup summary
        """
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days_old)
            deleted_files = []
            total_size_freed = 0
            
            for file_path in self.export_dir.iterdir():
                if file_path.is_file():
                    file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_date < cutoff_date:
                        file_size = file_path.stat().st_size
                        total_size_freed += file_size
                        deleted_files.append({
                            'filename': file_path.name,
                            'size_bytes': file_size,
                            'created_at': file_date.isoformat()
                        })
                        file_path.unlink()
            
            summary = {
                'deleted_count': len(deleted_files),
                'total_size_freed_mb': round(total_size_freed / (1024 * 1024), 2),
                'cutoff_date': cutoff_date.isoformat(),
                'deleted_files': deleted_files
            }
            
            logger.info(f"✅ Cleaned up {len(deleted_files)} old exports, freed {summary['total_size_freed_mb']}MB")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Export cleanup failed: {e}")
            raise
    
    @staticmethod
    def health_check() -> bool:
        """Check export service health."""
        return True


# Alias for backward compatibility
ExportManager = ExportService