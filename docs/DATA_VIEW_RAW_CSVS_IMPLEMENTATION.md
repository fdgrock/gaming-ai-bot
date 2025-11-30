# Data View (Raw CSVs) Feature - Implementation Summary

## Overview

Added a new **"Data View (Raw CSVs)"** section to the Data Management tab in the application. This section allows users to browse, view, and explore raw CSV files for the selected game with a visually appealing interface.

## Location

**File:** `streamlit_app/pages/data_training.py`
**Section:** Data Management tab (after Data Extraction, before Advanced Features)
**Lines:** ~619-710

## Features

### 1. **CSV File Selection**
- Dropdown list showing all available CSV files for the selected game
- Dynamically populates based on game selection
- File list pulled from the game's data directory

### 2. **File Information Dashboard**
Displays key metrics for the selected CSV file:
- **Total Rows:** Number of records in the file
- **Total Columns:** Number of fields/variables
- **File Size:** Size in MB
- **Modified Date:** Last modification timestamp

### 3. **Data Preview Table**
- Full-width, interactive table display
- Shows all rows and columns from the CSV
- Fixed height (400px) with scrolling for large files
- Maintains proper formatting and data types

### 4. **Data Insights Expander**
An expandable section with detailed data exploration tools:

#### Data Types Panel
- Lists all columns with their data types
- Helps identify numeric vs. text fields
- Useful for understanding data structure

#### Missing Values Panel
- Shows count of missing values per column
- Displays percentage of missing data
- Helps identify data quality issues

#### Summary Statistics
- Displays descriptive statistics (mean, std, min, max, etc.)
- Only available for numeric columns
- Automatically calculated by pandas

### 5. **Download Button**
- Allows users to download the viewed CSV file
- Maintains original filename
- Useful for external analysis

## User Experience Flow

```
1. Select Game (from Data Management header)
   ↓
2. Select CSV File (from dropdown)
   ↓
3. View File Metrics (4 cards showing stats)
   ↓
4. Browse Data (interactive table)
   ↓
5. [Optional] Expand Data Insights section
   ↓
6. [Optional] Download CSV file
```

## Code Structure

### Main Section Code:
```python
# Section Header
st.markdown("### 📖 Data View (Raw CSVs)")

# File Selection
csv_files = _get_csv_files(selected_game)
selected_file_name = st.selectbox("📄 Select CSV File", file_options)

# File Info Metrics (4 columns)
st.metric("📊 Total Rows", ...)
st.metric("📋 Total Columns", ...)
st.metric("💾 File Size", ...)
st.metric("🕐 Modified", ...)

# Data Preview
st.dataframe(csv_data, use_container_width=True, height=400)

# Data Insights Expander
with st.expander("📊 Data Insights"):
    # Data Types
    # Missing Values
    # Summary Statistics
```

## Helper Function Used

**`_get_csv_files(game: str) -> List[Path]`**
- Location: Lines 52-57 (already existing)
- Returns: Sorted list of CSV files in game directory
- Pattern: `training_data_*.csv`

## UI/UX Features

### Visual Design:
- 📖 Icon indicates browsing functionality
- 📄 Icon for file selection
- 📊 Icons for metrics and insights
- ⬇️ Icon for download button
- Clean layout with proper spacing via `st.divider()`

### Responsive Design:
- Full-width containers (`use_container_width=True`)
- Adaptive column layouts (4 metrics, 2 insight panels)
- Scrollable table for large datasets
- Expandable sections for advanced exploration

### Error Handling:
- Displays message if no CSV files exist
- Graceful error handling for file read errors
- User-friendly error messages

## Dependencies

All dependencies are already imported at the top of `data_training.py`:
- `streamlit` (st)
- `pandas` (pd)
- `datetime`
- `Path` from pathlib

No new imports required.

## Integration Points

1. **Game Selection:** Uses the same `selected_game` variable from Data Management header
2. **File Discovery:** Leverages existing `_get_csv_files()` helper function
3. **Error Logging:** Uses existing `app_log()` function for error tracking
4. **Session State:** Uses Streamlit's session state for file selection persistence

## Features Breakdown

### Feature 1: CSV File Dropdown
```python
file_options = [f.name for f in csv_files]
selected_file_name = st.selectbox(
    "📄 Select CSV File",
    file_options,
    key="csv_viewer_selector"
)
```
- Automatically updated when game changes
- Session state preserves selection during reruns

### Feature 2: File Metrics
```python
col1, col2, col3, col4 = st.columns(4)
st.metric("📊 Total Rows", f"{len(csv_data):,}")
st.metric("📋 Total Columns", csv_data.shape[1])
st.metric("💾 File Size", f"{file_size_mb:.2f} MB")
st.metric("🕐 Modified", last_modified.strftime("%Y-%m-%d"))
```
- Calculated on-the-fly from file data
- Formatted for readability (comma separators, 2 decimal places)

### Feature 3: Interactive Data Table
```python
st.dataframe(
    csv_data,
    use_container_width=True,
    height=400,
    hide_index=False
)
```
- Full-featured Streamlit dataframe with sorting/filtering
- Fixed height prevents excessive page length
- Index column visible for row identification

### Feature 4: Data Insights
```python
with st.expander("📊 Data Insights", expanded=False):
    # Data Types Panel
    dtype_info = pd.DataFrame({
        'Column': csv_data.columns,
        'Type': [str(dtype) for dtype in csv_data.dtypes]
    })
    
    # Missing Values Panel
    missing_info = pd.DataFrame({
        'Column': csv_data.columns,
        'Missing': [csv_data[col].isnull().sum() for col in csv_data.columns],
        'Percentage': [f"{percentage:.1f}%" for col in csv_data.columns]
    })
    
    # Summary Statistics
    st.dataframe(csv_data.describe())
```
- Collapsed by default (clean UI)
- Provides data quality information
- Numeric summary statistics

### Feature 5: Download Button
```python
csv_string = csv_data.to_csv(index=False)
st.download_button(
    label="⬇️ Download CSV",
    data=csv_string,
    file_name=selected_file_name,
    mime="text/csv"
)
```
- Full-width button for easy access
- Downloads currently viewed file
- Maintains original filename

## Testing Considerations

1. **Empty Game Directory:** Section displays informative message
2. **Single File:** Dropdown still works with one option
3. **Large Files:** Dataframe height limits prevent performance issues
4. **Special Characters:** Filenames with special characters handled gracefully
5. **Data Types:** Mixed data types correctly identified and displayed

## Performance

- CSV loading on-demand (only when selected)
- Pandas operations efficient for typical lottery data sizes
- Metrics calculated in-memory (no file I/O after initial read)
- Expander collapsed by default to reduce initial render time

## Future Enhancements

Possible improvements for future versions:
1. Search/filter functionality within the data table
2. Column sorting and selection
3. Data export to different formats (Excel, JSON)
4. Statistical visualizations (histograms, scatter plots)
5. Comparison tool for multiple CSV files
6. Data quality indicators (completeness score)
7. Data validation rules display

## Summary

The new "Data View (Raw CSVs)" section provides an intuitive interface for users to explore their lottery data files. It combines a user-friendly file browser with comprehensive data exploration tools, all within the familiar Streamlit environment. The implementation is clean, efficient, and follows the existing code patterns in the application.
