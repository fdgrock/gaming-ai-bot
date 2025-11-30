# Data View (Raw CSVs) - Quick Start Guide

## Feature Location

**Tab:** Data Training → Data Management Section
**Position:** After "Data Extraction", before "Advanced Feature Generation"

## How to Use

### Step 1: Navigate to Data Management
1. Open the application
2. Go to "Data Training" page (if not already there)
3. Scroll to the "📊 Data Management" section

### Step 2: Select a Game
- Use the "Select Game" dropdown at the top of Data Management
- Choose either "Lotto 6/49" or "Lotto Max"

### Step 3: View CSV Files
After the Data Extraction section, you'll see the new section:
```
### 📖 Data View (Raw CSVs)
Browse and view raw CSV files for the selected game
```

### Step 4: Select a File
- Click the "📄 Select CSV File" dropdown
- Choose a CSV file from the list
- The file will automatically load

### Step 5: Explore the Data

#### Option A: Quick Overview
- View the 4 metric cards showing:
  - Total number of rows (records)
  - Number of columns
  - File size in MB
  - Last modification date

#### Option B: Browse Full Data
- Scroll through the interactive data table
- Click column headers to sort
- Use the search feature to find specific values

#### Option C: Detailed Analysis
- Click the "📊 Data Insights" expander to reveal:
  - Data Types table (shows column types)
  - Missing Values table (data quality check)
  - Summary Statistics (mean, min, max, etc.)

#### Option D: Export Data
- Click the "⬇️ Download CSV" button
- File downloads with original filename
- Useful for external analysis or backup

## What Each Section Shows

### File Metrics (4 Cards)
| Card | Shows | Example |
|------|-------|---------|
| 📊 Total Rows | Number of records | 1,234 |
| 📋 Total Columns | Number of fields | 6 |
| 💾 File Size | Size in MB | 0.45 MB |
| 🕐 Modified | Last update date | 2025-11-23 |

### Data Preview Table
- **Full table display** with all columns visible
- **Sortable columns** - click header to sort
- **Searchable** - use Streamlit's built-in search
- **Interactive rows** - click to expand details
- **Fixed height** - scrollable for large datasets

### Data Insights Panel

#### Data Types Sub-section
Lists each column with its type:
- `int64` = Integer numbers
- `float64` = Decimal numbers
- `object` = Text or mixed
- `datetime64` = Dates/times

#### Missing Values Sub-section
Shows data quality issues:
- Count of empty cells per column
- Percentage of missing data
- Helps identify incomplete records

#### Summary Statistics Sub-section
Statistical overview (numeric columns only):
- Count (non-null values)
- Mean (average value)
- Std (standard deviation)
- Min/Max (range)
- 25th, 50th, 75th percentiles (quartiles)

## Features

✅ **Dynamic File List** - Updates based on selected game
✅ **No CSV Files Message** - Helpful info if directory is empty
✅ **File Information** - Quick stats dashboard
✅ **Interactive Table** - Sortable, searchable, scrollable
✅ **Data Exploration** - Detailed analysis tools
✅ **Download** - Export viewed file
✅ **Error Handling** - Graceful error messages
✅ **Responsive Design** - Works on any screen size

## Common Tasks

### Task 1: Check How Many Records Are in a File
1. Select the CSV file
2. Look at the first metric: "📊 Total Rows"
3. This shows the number of lottery draws

### Task 2: Find Files with Missing Data
1. Select a CSV file
2. Click the "📊 Data Insights" expander
3. Look at the "Missing Values" table
4. Any value > 0% means some data is missing

### Task 3: Check Data Types
1. Select a CSV file
2. Click the "📊 Data Insights" expander
3. View the "Data Types" table
4. Verify columns have expected types

### Task 4: Find the Largest/Smallest Value in a Column
1. Select a CSV file
2. Click the "📊 Data Insights" expander
3. View the "Summary Statistics" section
4. Look for Min/Max rows

### Task 5: Download a File for External Analysis
1. Select the CSV file
2. Scroll to the bottom
3. Click the "⬇️ Download CSV" button
4. File downloads automatically

### Task 6: Compare Two CSV Files
1. Select first CSV file
2. View the metrics and data
3. Change selection in dropdown to second CSV file
4. Compare side-by-side

## Tips & Tricks

💡 **Tip 1: Column Sorting**
Click any column header to sort ascending/descending

💡 **Tip 2: Search Data**
Use the search box that appears in the Streamlit dataframe

💡 **Tip 3: Expandable Insights**
Keep "Data Insights" collapsed for cleaner UI, expand when needed

💡 **Tip 4: File Size Check**
If a file is very large (>50MB), it may take time to load

💡 **Tip 5: Data Quality**
Check "Missing Values" percentage to assess data quality

💡 **Tip 6: Game Context**
Remember to select the correct game before viewing CSV files

## Troubleshooting

### Problem: "No CSV files found"
**Solution:** 
1. Use the Data Extraction section above to scrape lottery data
2. Or manually add CSV files to the game's data directory
3. Files should follow pattern: `training_data_*.csv`

### Problem: Table loads slowly
**Solution:**
1. Files with 100,000+ rows may take a moment to load
2. The 400px table height prevents display issues
3. Data Insights are calculated on-demand, not immediately

### Problem: Can't see all columns
**Solution:**
1. Scroll horizontally in the data table
2. Use the column selector if available in Streamlit
3. Or download the CSV to view in Excel/spreadsheet

### Problem: Missing values show as empty
**Solution:**
1. This is normal - empty cells represent missing data
2. Check the "Missing Values" table for quantitative info
3. Some lottery systems use blank or 0 for missing data

### Problem: Download doesn't work
**Solution:**
1. Check browser download settings
2. Ensure CSV file is valid and not corrupted
3. Try a different browser if issue persists

## Data Structure Example

Typical lottery CSV file contains columns like:
- `Draw_Date` - Date of the draw (YYYY-MM-DD)
- `Ball_1`, `Ball_2`, etc. - Main numbers
- `Bonus` - Bonus/additional number
- `Jackpot` - Prize amount

Example of what you might see:
```
Draw_Date,Ball_1,Ball_2,Ball_3,Ball_4,Ball_5,Ball_6,Bonus
2025-01-01,7,14,21,28,35,42,49
2025-01-02,5,12,19,26,33,40,48
2025-01-03,11,22,33,44,55,66,47
```

## Summary

The "Data View (Raw CSVs)" section provides an easy way to browse, explore, and understand your lottery data files without needing external tools. It's perfect for:
- ✓ Quick data quality checks
- ✓ Understanding data structure
- ✓ Verifying file contents
- ✓ Downloading data for analysis
- ✓ Learning about your datasets

Enjoy exploring your data! 📊
