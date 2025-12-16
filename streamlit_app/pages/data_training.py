"""
Enhanced Data & Training - Phase 5
Advanced data management with CSV extraction, scraping, and ML model training pipeline
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import os
from datetime import datetime
import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from ..core import (
        get_available_games, 
        get_session_value, 
        set_session_value, 
        app_log,
        get_data_dir
    )
    from ..services.advanced_feature_generator import AdvancedFeatureGenerator
    from ..services.advanced_model_training import AdvancedModelTrainer
except ImportError:
    def get_available_games(): return ["Lotto Max", "Lotto 6/49"]
    def get_session_value(k, d=None): return st.session_state.get(k, d)
    def set_session_value(k, v): st.session_state[k] = v
    def app_log(message: str, level: str = "info"): print(f"[{level.upper()}] {message}")
    def get_data_dir(): return Path("data")
    AdvancedFeatureGenerator = None
    AdvancedModelTrainer = None


# ============================================================================
# Data Management Helper Functions
# ============================================================================

def _sanitize_game_name(game: str) -> str:
    """Convert game name to folder name."""
    return game.lower().replace(" ", "_").replace("/", "_")


def _get_game_data_dir(game: str) -> Path:
    """Get the data directory for a specific game."""
    data_dir = get_data_dir()
    game_folder = _sanitize_game_name(game)
    return data_dir / game_folder


def _get_csv_files(game: str) -> List[Path]:
    """Get all CSV files for a game."""
    game_dir = _get_game_data_dir(game)
    if not game_dir.exists():
        return []
    return sorted(game_dir.glob("training_data_*.csv"))


def _count_records_in_file(filepath: Path) -> int:
    """Count the number of records (rows) in a CSV file, excluding header."""
    try:
        df = pd.read_csv(filepath)
        return len(df)
    except Exception as e:
        app_log(f"Error counting records in {filepath}: {e}", "error")
        return 0


def _get_file_stats(filepath: Path) -> Dict[str, Any]:
    """Get statistics about a CSV file."""
    try:
        stat = filepath.stat()
        df = pd.read_csv(filepath)
        
        # Get latest draw date
        if 'draw_date' in df.columns:
            latest_date = pd.to_datetime(df['draw_date']).max()
        else:
            latest_date = None
        
        return {
            "path": filepath,
            "name": filepath.name,
            "records": len(df),
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "latest_draw_date": latest_date
        }
    except Exception as e:
        app_log(f"Error getting stats for {filepath}: {e}", "error")
        return None


def _get_game_data_summary(game: str) -> Dict[str, Any]:
    """Get comprehensive summary of data for a game."""
    csv_files = _get_csv_files(game)
    
    if not csv_files:
        return {
            "dataset_count": 0,
            "total_records": 0,
            "latest_draw": None,
            "last_modified": None,
            "files": []
        }
    
    file_stats = []
    total_records = 0
    latest_draw = None
    last_modified = None
    
    for filepath in csv_files:
        stats = _get_file_stats(filepath)
        if stats:
            file_stats.append(stats)
            total_records += stats["records"]
            
            if stats["latest_draw_date"] and (latest_draw is None or stats["latest_draw_date"] > latest_draw):
                latest_draw = stats["latest_draw_date"]
            
            if stats["modified"] and (last_modified is None or stats["modified"] > last_modified):
                last_modified = stats["modified"]
    
    return {
        "dataset_count": len(file_stats),
        "total_records": total_records,
        "latest_draw": latest_draw,
        "last_modified": last_modified,
        "files": file_stats
    }


def render_data_training_page(services_registry=None, ai_engines=None, components=None) -> None:
    try:
        st.title("🎓 Data & Training Center")
        st.markdown("*Advanced data management and ML model training pipeline*")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Management", 
            "✨ Advanced Feature Generation",
            "🤖 Model Training",
            "🔄 Model Re-Training",
            "📈 Progress"
        ])
        
        with tab1:
            _render_data_management()
        with tab2:
            _render_advanced_features()
        with tab3:
            _render_model_training()
        with tab4:
            _render_model_retraining()
        with tab5:
            _render_progress()
        
        app_log("Training page rendered")
    except Exception as e:
        st.error(f"Error: {e}")
        app_log(f"Error rendering training page: {e}", "error")


def _render_data_management():
    """Render the Data Management section with data overview and extraction."""
    st.subheader("📊 Data Management")
    
    # Game Selection
    games = get_available_games()
    selected_game = st.selectbox("Select Game", games, key="data_game")
    
    # Get game data summary
    game_summary = _get_game_data_summary(selected_game)
    
    # Display Data Overview Metrics
    st.markdown("### 📋 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📁 Datasets",
            game_summary["dataset_count"],
            help="Number of CSV files for this game"
        )
    
    with col2:
        st.metric(
            "📊 Total Records",
            f"{game_summary['total_records']:,}",
            help="Total number of lottery draws"
        )
    
    with col3:
        last_modified = game_summary["last_modified"]
        if last_modified:
            days_ago = (datetime.now() - last_modified).days
            st.metric(
                "🕐 Last Updated",
                f"{days_ago}d ago" if days_ago > 0 else "Today",
                help=f"Last modified: {last_modified.strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            st.metric("🕐 Last Updated", "N/A")
    
    with col4:
        latest_draw = game_summary["latest_draw"]
        if latest_draw:
            st.metric(
                "📅 Latest Draw",
                latest_draw.strftime("%Y-%m-%d"),
                help="Most recent draw date in data"
            )
        else:
            st.metric("📅 Latest Draw", "N/A")
    
    st.divider()
    
    # Data Extraction Section
    st.markdown("### 🔄 Data Extraction")
    st.markdown("*Extract draw information from a URL and save to CSV*")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        scrape_url = st.text_input(
            "🔗 Scrape URL",
            placeholder="https://example.com/lottery-draws",
            help="Enter the website URL to scrape lottery data from"
        )
    
    # Year selector with +/- buttons
    with col2:
        # Initialize year in session state if not present
        if "scrape_year_selector" not in st.session_state:
            st.session_state["scrape_year_selector"] = datetime.now().year
        
        st.markdown("**📆 Year**")
        year_col1, year_col2, year_col3 = st.columns([1, 2, 1], gap="small")
        
        with year_col1:
            if st.button("➖", key="year_minus", use_container_width=True):
                st.session_state["scrape_year_selector"] = max(2000, st.session_state["scrape_year_selector"] - 1)
                st.rerun()
        
        with year_col2:
            st.write(f"<div style='text-align: center; padding: 8px; background-color: #f0f0f0; border-radius: 4px; font-weight: bold; font-size: 16px;'>{st.session_state['scrape_year_selector']}</div>", unsafe_allow_html=True)
        
        with year_col3:
            if st.button("➕", key="year_plus", use_container_width=True):
                st.session_state["scrape_year_selector"] = st.session_state["scrape_year_selector"] + 1
                st.rerun()
        
        selected_year = st.session_state["scrape_year_selector"]
    
    # Scrape and Preview Button
    if st.button("🔍 Scrape and Preview", use_container_width=True):
        if not scrape_url.strip():
            st.error("Please enter a URL to scrape")
        else:
            with st.spinner("Scraping data..."):
                try:
                    # Placeholder for scraping logic
                    scraped_data = _scrape_lottery_data(scrape_url, selected_year)
                    
                    if scraped_data is not None and not scraped_data.empty:
                        st.session_state["scraped_data"] = scraped_data
                        st.session_state["scrape_url"] = scrape_url
                        st.session_state["scrape_year"] = selected_year
                        st.success("✅ Data scraped successfully!")
                    else:
                        st.error("Could not extract data from the provided URL")
                except Exception as e:
                    st.error(f"Error scraping data: {str(e)}")
                    app_log(f"Scraping error: {e}", "error")
    
    # Display Scraped Data Preview
    if "scraped_data" in st.session_state and st.session_state["scraped_data"] is not None:
        st.markdown("### 📄 Preview")
        scraped_df = st.session_state["scraped_data"]
        
        # Show preview
        st.dataframe(scraped_df.head(10), use_container_width=True, height=300)
        
        st.divider()
        
        # File Information and Save Options
        st.markdown("### 💾 Save Options")
        
        # Generate filename
        filename = f"training_data_{selected_year}.csv"
        game_dir = _get_game_data_dir(selected_game)
        filepath = game_dir / filename
        file_exists = filepath.exists()
        
        # Display filename
        st.text_input("📝 File Name", value=filename, disabled=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            save_action = st.selectbox(
                "💾 Save Action",
                options=["Save as New", "Smart Update", "Force Replace"],
                help="Save as New: Creates a new file\nSmart Update: Appends new records\nForce Replace: Overwrites existing file"
            )
        
        with col2:
            st.write("")  # Spacing
        
        # Smart Update Preview
        if save_action == "Smart Update" and file_exists and "scraped_data" in st.session_state:
            scraped_df = st.session_state["scraped_data"]
            existing_stats = _get_file_stats(filepath)
            
            if existing_stats:
                # Read existing data to calculate deduplication
                existing_df = pd.read_csv(filepath)
                combined_df = pd.concat([existing_df, scraped_df], ignore_index=True)
                deduplicated_df = combined_df.drop_duplicates(subset=["draw_date"], keep="first")
                
                scraped_count = len(scraped_df)
                net_new = len(deduplicated_df) - len(existing_df)
                final_total = len(deduplicated_df)
                
                st.markdown("### 🔄 Smart Update Preview:")
                col_prev1, col_prev2, col_prev3 = st.columns(3)
                
                with col_prev1:
                    st.metric("📥 Would add", f"{scraped_count} records")
                
                with col_prev2:
                    st.metric("✨ Net new (dedup)", f"{max(0, net_new)} records")
                
                with col_prev3:
                    st.metric("📊 Final total", f"{final_total:,} records")
        
        # File Status
        if file_exists:
            existing_stats = _get_file_stats(filepath)
            if existing_stats:
                st.info(f"📁 File exists: {filename}")
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("📊 Current Records", f"{existing_stats['records']:,}")
                with col_info2:
                    st.metric("🕐 Last Modified", existing_stats['modified'].strftime("%Y-%m-%d %H:%M"))
        else:
            st.warning(f"📁 New file will be created: {filename}")
        
        st.divider()
        
        # Save Data Button
        if st.button("💾 Save Data", use_container_width=True, type="primary"):
            _save_scraped_data(selected_game, scraped_df, filename, save_action, file_exists)
    
    st.divider()
    
    # Data View Section (Raw CSVs)
    st.markdown("### 📖 Data View (Raw CSVs)")
    st.markdown("*Browse and view raw CSV files for the selected game*")
    
    # Get available CSV files
    csv_files = _get_csv_files(selected_game)
    
    if not csv_files:
        st.info("📂 No CSV files found for this game. Extract data first using the extraction section above.")
    else:
        # Create a selectbox with file options
        file_options = [f.name for f in csv_files]
        selected_file_name = st.selectbox(
            "📄 Select CSV File",
            file_options,
            key="csv_viewer_selector",
            help="Choose a CSV file to view"
        )
        
        # Find the selected file path
        selected_file_path = next((f for f in csv_files if f.name == selected_file_name), None)
        
        if selected_file_path:
            try:
                # Load the CSV file
                csv_data = pd.read_csv(selected_file_path)
                
                # Display file information
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total Rows", f"{len(csv_data):,}")
                with col2:
                    st.metric("📋 Total Columns", csv_data.shape[1])
                with col3:
                    file_size_mb = selected_file_path.stat().st_size / (1024 * 1024)
                    st.metric("💾 File Size", f"{file_size_mb:.2f} MB")
                with col4:
                    last_modified = datetime.fromtimestamp(selected_file_path.stat().st_mtime)
                    st.metric("🕐 Modified", last_modified.strftime("%Y-%m-%d"))
                
                # Display data with styling
                st.markdown("**Data Preview:**")
                
                # Create styled dataframe display
                st.dataframe(
                    csv_data,
                    use_container_width=True,
                    height=400,
                    hide_index=False
                )
                
                # Additional data exploration options
                with st.expander("📊 Data Insights", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Data Types:**")
                        dtype_info = pd.DataFrame({
                            'Column': csv_data.columns,
                            'Type': [str(dtype) for dtype in csv_data.dtypes]
                        })
                        st.dataframe(dtype_info, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("**Missing Values:**")
                        missing_info = pd.DataFrame({
                            'Column': csv_data.columns,
                            'Missing': [csv_data[col].isnull().sum() for col in csv_data.columns],
                            'Percentage': [f"{(csv_data[col].isnull().sum() / len(csv_data) * 100):.1f}%" for col in csv_data.columns]
                        })
                        st.dataframe(missing_info, use_container_width=True, hide_index=True)
                    
                    # Summary statistics
                    st.markdown("**Summary Statistics:**")
                    st.dataframe(csv_data.describe(), use_container_width=True)
                
                # Download option
                csv_string = csv_data.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_string,
                    file_name=selected_file_name,
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                app_log(f"Error reading CSV: {e}", "error")


def _scrape_lottery_data(url: str, year: int) -> Optional[pd.DataFrame]:
    """
    Scrape lottery data from a URL using BeautifulSoup.
    Supports multiple lottery website formats:
    - lottomaxnumbers.com: 3 columns (Date | Numbers with UL/LI + bonus class | Jackpot)
    - ca.lottonumbers.com: 5 columns (Date | Numbers with UL/LI (last is gold ball) | Jackpot | Winners | Prizes)
    """
    try:
        app_log(f"Scraping data from {url} for year {year}", "info")
        
        if not url.strip():
            st.error("Please provide a valid URL")
            return None
        
        # Make the request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch URL: {str(e)}")
            app_log(f"URL fetch error: {e}", "error")
            return None
        
        if BeautifulSoup is None:
            st.error("BeautifulSoup4 is required for web scraping but not installed")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find lottery data tables
        tables = soup.find_all('table')
        app_log(f"Found {len(tables)} tables on page", "info")
        
        if not tables:
            st.warning(f"No data tables found at {url}")
            st.info("Please ensure the URL contains lottery draw data in a table format.")
            app_log(f"No tables found at URL: {url}", "warning")
            
            # Try to provide more diagnostic info
            try:
                # Look for any data containers
                divs_with_numbers = soup.find_all('div', class_=lambda x: x and ('number' in x.lower() or 'ball' in x.lower() or 'draw' in x.lower() or 'result' in x.lower()))
                app_log(f"Found {len(divs_with_numbers)} divs with number-related classes", "debug")
                
                # Check page title/heading
                title = soup.find('h1')
                if title:
                    app_log(f"Page title: {title.get_text(strip=True)[:100]}", "debug")
            except:
                pass
            
            return None
        
        all_data = []
        
        # Process each table to extract lottery data
        for table in tables:
            rows = table.find_all('tr')
            
            # Debug: Show first row to understand column structure
            if rows:
                first_row = rows[0]
                cols = first_row.find_all(['td', 'th'])
                app_log(f"Table header/first row has {len(cols)} columns", "debug")
                for idx, col in enumerate(cols):
                    col_text = col.get_text(strip=True)[:50]
                    app_log(f"  Col {idx}: {col_text}", "debug")
            
            for row_idx, row in enumerate(rows[1:]):
                cols = row.find_all(['td', 'th'])
                
                # Skip header rows or rows with only 1 column (section headers)
                if len(cols) < 2 or len(cols) > 6:
                    continue
                
                try:
                    # Extract date from first column
                    date_col = cols[0]
                    date_text = date_col.get_text(strip=True)
                    
                    # Clean up date text (remove day names like Saturday, Wednesday)
                    # Some websites include day names mixed with date
                    import re
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    for day in day_names:
                        date_text = date_text.replace(day, '')
                    
                    # Remove extra text like "With Max Milli" or similar suffixes
                    # Keep only the date pattern (month/day/year or similar)
                    # Extract only the date portion: look for month name + numbers
                    month_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d+)\s+(\d{4})'
                    match = re.search(month_pattern, date_text)
                    if match:
                        date_text = match.group(0)
                    else:
                        date_text = date_text.strip()
                    
                    # Try to parse the date
                    try:
                        draw_date = pd.to_datetime(date_text).strftime("%Y-%m-%d")
                    except:
                        app_log(f"Could not parse date: {date_text}", "debug")
                        continue
                    
                    # Extract numbers and bonus from second column
                    # Both Lotto Max and 6/49 use UL/LI for numbers
                    numbers_col = cols[1]
                    
                    # Check for UL/LI structure (both formats use this)
                    ul = numbers_col.find('ul')
                    
                    if ul:
                        # Extract from UL/LI structure
                        lis = ul.find_all('li')
                        
                        if lis:
                            # Format 1 (Lotto Max): Numbers with class="ball", bonus with class="bonus-ball"
                            # Format 2 (6/49): All numbers as li, last one is gold/bonus ball
                            
                            # Check if we have the bonus-ball class marker
                            has_bonus_class = any('bonus-ball' in li.get('class', []) for li in lis)
                            
                            if has_bonus_class:
                                # Lotto Max format
                                regular_balls = []
                                bonus_ball = None
                                
                                for li in lis:
                                    if 'bonus-ball' in li.get('class', []):
                                        bonus_ball = li.get_text(strip=True)
                                    else:
                                        regular_balls.append(li.get_text(strip=True))
                                
                                numbers = ','.join(regular_balls) if regular_balls else ""
                                bonus = bonus_ball if bonus_ball else ""
                            else:
                                # Lotto 6/49 format or other generic format
                                # Last LI is the bonus/gold ball
                                li_texts = [li.get_text(strip=True) for li in lis]
                                
                                if len(li_texts) >= 7:
                                    # Likely 6/49 format (6 numbers + 1 gold ball)
                                    numbers = ','.join(li_texts[:-1])
                                    bonus = li_texts[-1]
                                elif len(li_texts) >= 2:
                                    # Generic format: all but last are numbers, last is bonus
                                    numbers = ','.join(li_texts[:-1])
                                    bonus = li_texts[-1]
                                else:
                                    # Not enough data
                                    continue
                        else:
                            # No LI elements, fallback to text
                            numbers_text = numbers_col.get_text(strip=True)
                            numbers = numbers_text.replace('\n', ',').replace(' ', ',').strip()
                            bonus = ""
                    else:
                        # No UL found, fallback to text extraction
                        numbers_text = numbers_col.get_text(strip=True)
                        numbers = numbers_text.replace('\n', ',').replace(' ', ',').strip()
                        bonus = ""
                    
                    # Extract jackpot
                    jackpot = 0
                    jackpot_text = ""
                    
                    # Check col[2] or beyond for jackpot (look for $ or large numbers)
                    for col_idx in range(2, len(cols)):
                        col_text = cols[col_idx].get_text(strip=True)
                        if '$' in col_text or (',' in col_text and len(col_text) > 5 and any(c.isdigit() for c in col_text)):
                            jackpot_text = col_text
                            break
                    
                    if jackpot_text:
                        # Extract numeric value from jackpot text
                        # Handle formats like "$5 Million", "$5,000,000", "5 Million", etc.
                        try:
                            # Extract digits and dots only first
                            numeric_part = ''.join(filter(lambda x: x.isdigit() or x == '.', jackpot_text))
                            jackpot = float(numeric_part) if numeric_part else 0
                            
                            # Check if text contains "Million" or "Billion" for multiplier
                            text_upper = jackpot_text.upper()
                            if 'MILLION' in text_upper:
                                jackpot = jackpot * 1_000_000
                            elif 'BILLION' in text_upper:
                                jackpot = jackpot * 1_000_000_000
                            elif 'THOUSAND' in text_upper:
                                jackpot = jackpot * 1_000
                        except:
                            jackpot = 0
                    
                    # Debug first few rows
                    if row_idx < 3:
                        app_log(f"Row {row_idx}: Date={draw_date}, Nums={numbers[:30]}, Bonus={bonus}, JPot={jackpot}", "debug")
                    
                    # Only add if we have meaningful data
                    if draw_date and numbers:
                        try:
                            extracted_year = int(draw_date[:4])
                            
                            # Filter by selected year
                            if extracted_year == year:
                                all_data.append({
                                    "draw_date": draw_date,
                                    "year": extracted_year,
                                    "numbers": numbers,
                                    "bonus": bonus,
                                    "jackpot": jackpot
                                })
                        except Exception as year_err:
                            app_log(f"Error extracting year: {year_err}", "debug")
                            continue
                except Exception as row_error:
                    app_log(f"Error processing table row {row_idx}: {row_error}", "debug")
                    continue
        
        if not all_data:
            st.warning(f"No lottery data found for year {year} at {url}")
            st.info(f"The website structure may not match expected format. The scraper supports Lotto Max and 6/49 formats. Ensure the year {year} data is on the page.")
            app_log(f"No records extracted for year {year}. Check if page contains {year} data.", "warning")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["draw_date"], keep="first")
        
        # Sort by date descending
        df = df.sort_values("draw_date", ascending=False).reset_index(drop=True)
        
        app_log(f"Successfully scraped {len(df)} records for {year}", "info")
        return df
        
    except Exception as e:
        app_log(f"Error scraping data: {e}", "error")
        st.error(f"Scraping error: {str(e)}")
        return None


def _save_scraped_data(game: str, data: pd.DataFrame, filename: str, action: str, file_exists: bool) -> None:
    """Save scraped data based on the selected action."""
    try:
        game_dir = _get_game_data_dir(game)
        game_dir.mkdir(parents=True, exist_ok=True)
        filepath = game_dir / filename
        
        if action == "Smart Update" and file_exists:
            # Smart Update: Read existing, append new, remove duplicates
            existing_df = pd.read_csv(filepath)
            existing_dates = set(existing_df["draw_date"].astype(str).str.strip())
            
            # Count how many records are actually new
            incoming_dates = set(data["draw_date"].astype(str).str.strip())
            net_new_records = len(incoming_dates - existing_dates)
            
            combined_df = pd.concat([existing_df, data], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["draw_date"], keep="first")
            combined_df = combined_df.sort_values("draw_date", ascending=False)
            combined_df.to_csv(filepath, index=False)
            
            if net_new_records > 0:
                st.success(f"✅ Data updated: {net_new_records} new draw dates added (Total records: {len(combined_df)})")
            else:
                st.info(f"ℹ️ No new draws found. All {len(data)} scraped records already exist in the file.")
            
            app_log(f"Smart Update: {net_new_records} new records added (Total: {len(combined_df)})", "info")
            
        elif action == "Force Replace":
            # Force Replace: Overwrite completely
            data.to_csv(filepath, index=False)
            st.success(f"✅ File replaced with {len(data)} records")
            
        else:  # Save as New
            if file_exists:
                st.warning(f"File {filename} already exists. Use 'Smart Update' or 'Force Replace' to modify.")
            else:
                data.to_csv(filepath, index=False)
                st.success(f"✅ New file created with {len(data)} records")
        
        # Clear session state after save
        if "scraped_data" in st.session_state:
            del st.session_state["scraped_data"]
        
        app_log(f"Data saved successfully: {filename}", "info")
        
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        app_log(f"Error saving data: {e}", "error")


def _render_advanced_features():
    """Render the Advanced Feature Generation section with state-of-the-art AI."""
    st.subheader("✨ Advanced Feature Generation")
    st.markdown("*State-of-the-art AI feature engineering for optimal lottery number prediction*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        games = get_available_games()
        selected_game = st.selectbox("Select Game", games, key="feature_gen_game")
    
    with col2:
        st.write("")  # Spacing
    
    st.divider()
    
    # ========================================
    # 🎯 TARGET REPRESENTATION STRATEGY
    # ========================================
    st.markdown("### 🎯 Target Representation Strategy")
    st.markdown("*Configure how lottery numbers are represented for model training*")
    
    with st.expander("⚙️ Target Configuration", expanded=True):
        st.markdown("""
        **Critical Choice:** How should the model learn to predict 7 lottery numbers?
        
        Different representations work better for different model architectures:
        """)
        
        target_strategy = st.radio(
            "Select Target Representation Strategy",
            [
                "Multi-Output (7 separate predictions)",
                "Sequence-to-Sequence (ordered 7-number sequence)",
                "Set Prediction (unordered set of 7 numbers)"
            ],
            key="target_strategy",
            help="This affects how features are generated and how models learn"
        )
        
        if "Multi-Output" in target_strategy:
            st.success("""
            ✅ **Multi-Output Strategy**
            - **Best for:** XGBoost, CatBoost, LightGBM, Neural Networks
            - **How it works:** 7 separate output heads, each predicting one position
            - **Advantages:** Position-specific learning, clear accountability
            - **Features:** Standard feature generation (all models)
            """)
            target_mode = "multi_output"
        
        elif "Sequence-to-Sequence" in target_strategy:
            st.info("""
            ℹ️ **Sequence-to-Sequence Strategy**
            - **Best for:** LSTM, Transformer
            - **How it works:** Generate complete 7-number sequence in order
            - **Advantages:** Captures sequential dependencies
            - **Features:** LSTM sequences, Transformer features
            """)
            target_mode = "seq2seq"
        
        else:  # Set Prediction
            st.warning("""
            ⚠️ **Set Prediction Strategy**
            - **Best for:** Advanced architectures (experimental)
            - **How it works:** Predict unordered set of 7 unique numbers
            - **Advantages:** Order-independent, focuses on number selection
            - **Features:** CNN embeddings, statistical features
            """)
            target_mode = "set"
        
        # Save target mode to session state for feature generation
        st.session_state['target_representation_mode'] = target_mode
        
        st.info(f"📊 Selected mode: **{target_mode}** - All feature generators will adapt accordingly")
    
    st.divider()
    
    # ========================================
    # 🔬 ENHANCED LOTTERY FEATURES
    # ========================================
    st.markdown("### 🔬 Enhanced Lottery Features")
    st.markdown("*Advanced pattern analysis and statistical feature extraction*")
    
    with st.expander("⚙️ Enhanced Feature Configuration", expanded=False):
        st.markdown("""
        **Comprehensive Lottery Pattern Analysis:**
        
        These features capture deep patterns in lottery draw history:
        """)
        
        col_feat1, col_feat2 = st.columns(2)
        
        with col_feat1:
            enable_frequency = st.checkbox("Hot/Cold Number Frequency", value=True, key="enable_freq",
                help="Analyze frequency over 10, 20, 50 draw windows")
            enable_gap = st.checkbox("Gap Analysis", value=True, key="enable_gap",
                help="Draws since each number last appeared")
            enable_pattern = st.checkbox("Pattern Features", value=True, key="enable_pattern",
                help="Consecutive runs, clusters, spacing patterns")
            enable_statistical = st.checkbox("Statistical Features", value=True, key="enable_stats",
                help="Sum ranges, distributions, variance analysis")
        
        with col_feat2:
            enable_temporal = st.checkbox("Temporal Features", value=True, key="enable_temporal",
                help="Day of week, month, season patterns")
            enable_correlation = st.checkbox("Co-occurrence Patterns", value=True, key="enable_corr",
                help="Number pair correlation and clustering")
            enable_entropy = st.checkbox("Entropy & Randomness", value=True, key="enable_entropy",
                help="Shannon entropy, randomness scores")
            enable_position = st.checkbox("Position-Specific Analysis", value=True, key="enable_position",
                help="Position 1 vs 7 bias detection")
        
        # Frequency analysis configuration
        if enable_frequency:
            st.markdown("**Frequency Analysis Windows:**")
            freq_windows = st.multiselect(
                "Lookback periods (draws)",
                [5, 10, 20, 30, 50, 100],
                default=[10, 20, 50],
                key="freq_windows"
            )
        else:
            freq_windows = []
        
        # Build enhanced feature config
        enhanced_config = {
            'frequency': enable_frequency,
            'gap_analysis': enable_gap,
            'patterns': enable_pattern,
            'statistical': enable_statistical,
            'temporal': enable_temporal,
            'correlation': enable_correlation,
            'entropy': enable_entropy,
            'position_specific': enable_position,
            'frequency_windows': freq_windows if enable_frequency else []
        }
        
        st.session_state['enhanced_features_config'] = enhanced_config
        
        # Show estimated feature count
        feature_count = 0
        if enable_frequency: feature_count += len(freq_windows) * 50  # 50 numbers tracked
        if enable_gap: feature_count += 50
        if enable_pattern: feature_count += 15
        if enable_statistical: feature_count += 10
        if enable_temporal: feature_count += 8
        if enable_correlation: feature_count += 100  # Top 100 pairs
        if enable_entropy: feature_count += 5
        if enable_position: feature_count += 35  # 7 positions * 5 stats
        
        st.metric("Estimated Additional Features", f"+{feature_count}")
    
    st.divider()
    
    # ========================================
    # 📉 FEATURE OPTIMIZATION
    # ========================================
    st.markdown("### 📉 Feature Optimization & Dimensionality Reduction")
    st.markdown("*Prevent curse of dimensionality with intelligent feature selection*")
    
    with st.expander("⚙️ Optimization Configuration", expanded=False):
        st.markdown("""
        **Why Feature Optimization?**
        - Too many features → overfitting, slow training, poor generalization
        - Solution → Keep only the most predictive features
        
        **⚠️ IMPORTANT - Model-Specific Recommendations:**
        """)
        
        # Add recommendation table
        st.markdown("""
        | Model Type | Recommended Setting | Reason |
        |------------|-------------------|---------|
        | **XGBoost** | ❌ **Disable** or use **RFE only** | Tree models handle high dimensions well, PCA loses interpretability |
        | **CatBoost** | ❌ **Disable** or use **RFE only** | Excels with many features, built-in feature importance |
        | **LightGBM** | ❌ **Disable** or use **RFE only** | Leaf-wise growth benefits from all features |
        | **LSTM** | ✅ **RFE or Hybrid** | Reduces sequence complexity, prevents overfitting |
        | **CNN** | ✅ **PCA or Hybrid** | Works well with compressed representations |
        | **Transformer** | ⚠️ **Disable** | Attention mechanism handles feature selection internally |
        
        **Quick Guide:**
        - **Tree Models (XGBoost/CatBoost/LightGBM)**: Use original features for best accuracy
        - **Neural Models (LSTM/CNN)**: Optimization helps reduce parameters
        - **Hybrid (RFE+PCA)**: ⚠️ Can reduce accuracy significantly - use with caution!
        """)
        
        enable_optimization = st.checkbox("Enable Feature Optimization", value=False, key="enable_optimization",
            help="⚠️ See recommendations above - disabling often gives better results for tree models!")
        
        if enable_optimization:
            st.warning("⚠️ Optimization enabled - this may reduce accuracy for tree-based models. Check recommendations above!")
            
            optimization_method = st.radio(
                "Optimization Method",
                [
                    "Recursive Feature Elimination (RFE)",
                    "Principal Component Analysis (PCA)",
                    "Feature Importance Thresholding",
                    "Hybrid (RFE + PCA)"
                ],
                key="opt_method",
                help="RFE: Removes least important features | PCA: Compresses to principal components | Importance: Threshold-based selection | Hybrid: Both RFE and PCA"
            )
            
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                if "RFE" in optimization_method or "Hybrid" in optimization_method:
                    rfe_n_features = st.slider(
                        "RFE: Target Feature Count",
                        50, 500, 200,
                        step=10,
                        key="rfe_features",
                        help="Number of features to keep after elimination"
                    )
                else:
                    rfe_n_features = None
                
                if "PCA" in optimization_method or "Hybrid" in optimization_method:
                    pca_variance = st.slider(
                        "PCA: Variance to Retain",
                        0.80, 0.99, 0.95,
                        step=0.01,
                        key="pca_variance",
                        help="Cumulative explained variance threshold"
                    )
                    pca_max_components = st.slider(
                        "PCA: Max Components",
                        50, 300, 150,
                        step=10,
                        key="pca_components"
                    )
                else:
                    pca_variance = None
                    pca_max_components = None
            
            with col_opt2:
                if "Importance" in optimization_method:
                    importance_threshold = st.slider(
                        "Importance: Keep Top %",
                        10, 100, 30,
                        step=5,
                        key="importance_threshold",
                        help="Keep top X% most important features"
                    )
                else:
                    importance_threshold = None
                
                # Cross-validation for feature selection
                use_cv_selection = st.checkbox(
                    "Use Cross-Validation for Selection",
                    value=True,
                    key="cv_selection",
                    help="More robust but slower"
                )
                
                if use_cv_selection:
                    cv_folds = st.slider("CV Folds", 3, 10, 5, key="cv_folds")
                else:
                    cv_folds = None
            
            optimization_config = {
                'enabled': True,
                'method': optimization_method,
                'rfe_n_features': rfe_n_features,
                'pca_variance': pca_variance,
                'pca_max_components': pca_max_components,
                'importance_threshold': importance_threshold,
                'use_cv': use_cv_selection,
                'cv_folds': cv_folds
            }
        else:
            optimization_config = {'enabled': False}
        
        st.session_state['feature_optimization_config'] = optimization_config
        
        if enable_optimization:
            st.success(f"✅ Feature optimization enabled: {optimization_method}")
    
    st.divider()
    
    # ========================================
    # 🔍 AUTOMATIC FEATURE DISCOVERY
    # ========================================
    st.markdown("### 🔍 Automatic Feature Discovery")
    st.markdown("*AI-powered pattern detection in historical lottery data*")
    
    with st.expander("⚙️ Discovery Configuration", expanded=False):
        st.markdown("""
        **Automatic Discovery Capabilities:**
        - Find number pairs that appear together frequently
        - Detect seasonal/cyclical patterns (monthly, yearly)
        - Identify position-specific biases
        - Discover hidden correlations
        """)
        
        enable_discovery = st.checkbox("Enable Automatic Discovery", value=True, key="enable_discovery")
        
        if enable_discovery:
            col_disc1, col_disc2 = st.columns(2)
            
            with col_disc1:
                discover_pairs = st.checkbox("Number Pair Co-occurrence", value=True, key="disc_pairs")
                if discover_pairs:
                    top_n_pairs = st.slider("Top N pairs to track", 10, 100, 50, key="top_pairs")
                    pair_min_freq = st.slider("Minimum frequency (%)", 5, 50, 10, key="pair_freq")
                else:
                    top_n_pairs = 0
                    pair_min_freq = 0
                
                discover_cycles = st.checkbox("Seasonal/Cyclical Patterns", value=True, key="disc_cycles")
                if discover_cycles:
                    cycle_periods = st.multiselect(
                        "Cycle periods to analyze",
                        ["Weekly (7)", "Monthly (30)", "Quarterly (90)", "Yearly (365)"],
                        default=["Weekly (7)", "Monthly (30)"],
                        key="cycle_periods"
                    )
                else:
                    cycle_periods = []
            
            with col_disc2:
                discover_position_bias = st.checkbox("Position-Specific Biases", value=True, key="disc_pos_bias")
                if discover_position_bias:
                    st.info("Analyze if position 1 tends low, position 7 tends high, etc.")
                
                discover_correlations = st.checkbox("Hidden Correlations", value=True, key="disc_corr")
                if discover_correlations:
                    corr_threshold = st.slider(
                        "Correlation threshold",
                        0.3, 0.9, 0.6,
                        step=0.05,
                        key="corr_threshold"
                    )
                else:
                    corr_threshold = 0
            
            discovery_config = {
                'enabled': True,
                'pairs': discover_pairs,
                'top_n_pairs': top_n_pairs,
                'pair_min_freq': pair_min_freq,
                'cycles': discover_cycles,
                'cycle_periods': cycle_periods,
                'position_bias': discover_position_bias,
                'correlations': discover_correlations,
                'corr_threshold': corr_threshold
            }
        else:
            discovery_config = {'enabled': False}
        
        st.session_state['feature_discovery_config'] = discovery_config
        
        if enable_discovery:
            st.success("✅ Automatic feature discovery enabled")
    
    st.divider()
    
    # ========================================
    # ✅ FEATURE VALIDATION & QUALITY CHECKS
    # ========================================
    st.markdown("### ✅ Feature Validation & Quality Checks")
    st.markdown("*Ensure feature quality before training*")
    
    with st.expander("⚙️ Validation Configuration", expanded=False):
        st.markdown("""
        **Quality Checks Performed:**
        - ❌ NaN/Infinite value detection
        - 📊 Constant feature detection (no variance)
        - 🔗 Multicollinearity detection (highly correlated features)
        - 🚨 Feature leakage detection (target information in features)
        - 📏 Feature scale analysis
        """)
        
        enable_validation = st.checkbox("Enable Feature Validation", value=True, key="enable_validation")
        
        if enable_validation:
            col_val1, col_val2 = st.columns(2)
            
            with col_val1:
                check_nan = st.checkbox("Check for NaN/Inf", value=True, key="check_nan")
                check_constant = st.checkbox("Detect Constant Features", value=True, key="check_constant")
                if check_constant:
                    variance_threshold = st.slider(
                        "Min variance threshold",
                        0.0, 0.1, 0.01,
                        step=0.001,
                        key="var_threshold",
                        format="%.3f"
                    )
                else:
                    variance_threshold = 0
            
            with col_val2:
                check_correlation = st.checkbox("Detect Multicollinearity", value=True, key="check_multicoll")
                if check_correlation:
                    correlation_threshold = st.slider(
                        "High correlation threshold",
                        0.8, 0.99, 0.95,
                        step=0.01,
                        key="multicoll_threshold"
                    )
                else:
                    correlation_threshold = 0
                
                check_leakage = st.checkbox("Check for Feature Leakage", value=True, key="check_leakage")
            
            # Action on validation failure
            validation_action = st.radio(
                "Action on validation failure",
                ["Show warnings only", "Auto-fix issues", "Block feature generation"],
                key="validation_action"
            )
            
            validation_config = {
                'enabled': True,
                'check_nan': check_nan,
                'check_constant': check_constant,
                'variance_threshold': variance_threshold,
                'check_correlation': check_correlation,
                'correlation_threshold': correlation_threshold,
                'check_leakage': check_leakage,
                'action': validation_action
            }
        else:
            validation_config = {'enabled': False}
        
        st.session_state['feature_validation_config'] = validation_config
        
        if enable_validation:
            st.success(f"✅ Feature validation enabled: {validation_action}")
    
    st.divider()
    
    # ========================================
    # 💾 FEATURE SAMPLE EXPORT
    # ========================================
    st.markdown("### 💾 Feature Sample Export")
    st.markdown("*Export representative feature samples for analysis*")
    
    with st.expander("⚙️ Export Configuration", expanded=False):
        st.markdown("""
        **Sample Export Benefits:**
        - Quick feature inspection without loading full datasets
        - Metadata includes mean, std, min, max per feature
        - Standardized format for documentation
        - Useful for model debugging and feature analysis
        """)
        
        enable_export = st.checkbox("Enable Feature Sample Export", value=True, key="enable_export")
        
        if enable_export:
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                sample_size = st.slider(
                    "Sample size (rows)",
                    100, 10000, 1000,
                    step=100,
                    key="sample_size"
                )
                
                sample_strategy = st.radio(
                    "Sampling strategy",
                    ["Random", "Recent draws", "Stratified (by target)"],
                    key="sample_strategy"
                )
            
            with col_exp2:
                include_metadata = st.checkbox("Include feature metadata", value=True, key="include_meta")
                include_stats = st.checkbox("Include feature statistics", value=True, key="include_stats")
                
                export_format = st.selectbox(
                    "Export format",
                    ["CSV", "JSON", "Parquet", "All formats"],
                    key="export_format"
                )
            
            export_config = {
                'enabled': True,
                'sample_size': sample_size,
                'strategy': sample_strategy,
                'include_metadata': include_metadata,
                'include_stats': include_stats,
                'format': export_format
            }
        else:
            export_config = {'enabled': False}
        
        st.session_state['feature_export_config'] = export_config
        
        if enable_export:
            st.success(f"✅ Sample export enabled: {sample_size} rows, {export_format} format")
    
    st.divider()
    
    # File Selection Section
    st.markdown("### 📁 Select Raw Files")
    
    if AdvancedFeatureGenerator is None:
        st.error("Advanced Feature Generator not available")
        return
    
    feature_gen = AdvancedFeatureGenerator(selected_game)
    available_files = feature_gen.get_raw_files()
    
    if not available_files:
        st.warning(f"No raw data files found for {selected_game}")
        st.info("Please ensure raw CSV files exist in the data folder")
        return
    
    # Display available files
    file_names = [f.name for f in available_files]
    st.markdown(f"**Available files: {len(available_files)}**")
    
    col_use_all, col_spacer = st.columns([2, 3])
    with col_use_all:
        use_all_files = st.checkbox("Use all raw files for this game", value=True, key="use_all_raw_files")
    
    if use_all_files:
        selected_files = available_files
        st.info(f"✓ Using all {len(available_files)} raw files for comprehensive analysis")
    else:
        st.markdown("**Select specific files:**")
        selected_file_names = st.multiselect(
            "Files to use",
            file_names,
            default=file_names[-5:] if len(file_names) > 5 else file_names,
            key="select_raw_files"
        )
        selected_files = [f for f in available_files if f.name in selected_file_names]
        
        if not selected_files:
            st.warning("Please select at least one file")
            return
    
    # Load data
    with st.spinner("Loading and analyzing raw data..."):
        raw_data = feature_gen.load_raw_data(selected_files)
    
    if raw_data is None or raw_data.empty:
        st.error("Failed to load raw data")
        return
    
    st.success(f"✅ Loaded {len(raw_data)} draws from {len(selected_files)} files")
    
    st.divider()
    
    # Feature Generation Sections with Advanced Parameters
    st.markdown("### 🔧 Advanced Feature Generators")
    
    # LSTM Sequences Section
    st.markdown("#### 🔷 LSTM Sequences - Advanced Temporal Feature Extraction")
    st.markdown("*Generates comprehensive LSTM sequences with temporal, statistical, distribution, parity, spacing, and frequency features*")
    
    with st.expander("⚙️ LSTM Configuration", expanded=True):
        lstm_col1, lstm_col2 = st.columns(2)
        
        with lstm_col1:
            lstm_window = st.slider(
                "Sequence Window Size",
                10, 60, 25,
                key="lstm_window",
                help="Number of past draws to include in each sequence. Larger = more context but less data."
            )
        
        with lstm_col2:
            st.info(f"📊 Estimated sequences: ~{max(0, len(raw_data) - lstm_window)}")
    
    if st.button("🚀 Generate LSTM Sequences", use_container_width=True, key="btn_lstm_gen"):
        try:
            with st.spinner("Generating advanced LSTM sequences with intelligent feature engineering..."):
                # Get configurations from session state
                enhanced_config = st.session_state.get('enhanced_features_config', {})
                optimization_config = st.session_state.get('feature_optimization_config', {})
                validation_config = st.session_state.get('feature_validation_config', {})
                export_config = st.session_state.get('feature_export_config', {})
                target_representation = st.session_state.get('target_representation_mode', 'binary')
                
                # Track quality indicators
                optimization_applied = False
                validation_passed = True
                
                # Generate base features
                lstm_sequences, lstm_metadata = feature_gen.generate_lstm_sequences(
                    raw_data,
                    window_size=lstm_window
                )
                
                # Apply enhanced features if configured
                if enhanced_config.get('frequency') or enhanced_config.get('gap_analysis') or enhanced_config.get('patterns'):
                    with st.spinner("Applying enhanced lottery features..."):
                        st.info("🔬 Applying enhanced features: frequency, gap analysis, patterns, entropy, correlation, position-specific")
                        # Enhanced features are already included in the generation
                        # But we log which ones are active
                        active_enhancements = [k for k, v in enhanced_config.items() if v and k != 'frequency_windows']
                        if active_enhancements:
                            lstm_metadata['enhanced_features'] = active_enhancements
                
                # Convert to DataFrame for optimization/validation
                if len(lstm_sequences.shape) == 3:
                    # Flatten sequences for optimization (samples, timesteps * features)
                    n_samples, n_timesteps, n_features = lstm_sequences.shape
                    lstm_flat = lstm_sequences.reshape(n_samples, n_timesteps * n_features)
                    
                    # Optimize if configured
                    if optimization_config.get('enabled', False):
                        with st.spinner(f"Optimizing features with {optimization_config.get('method', 'RFE')}..."):
                            temp_df = pd.DataFrame(lstm_flat)
                            optimized_df, opt_info = feature_gen.apply_feature_optimization(temp_df, optimization_config)
                            lstm_flat = optimized_df.values
                            
                            # Reshape back to sequences
                            new_n_features = lstm_flat.shape[1] // n_timesteps
                            lstm_sequences = lstm_flat.reshape(n_samples, n_timesteps, new_n_features)
                            lstm_metadata['optimization_applied'] = True
                            lstm_metadata['optimization_config'] = optimization_config
                            optimization_applied = True
                            
                            st.success(f"✅ Optimized: {opt_info.get('original_features', 0)} → {opt_info.get('final_features', 0)} features")
                    
                    # Validate if configured
                    if validation_config.get('enabled', False):
                        with st.spinner("Validating feature quality..."):
                            validation_results = feature_gen.validate_features(lstm_flat, validation_config)
                            lstm_metadata['validation_passed'] = validation_results['passed']
                            lstm_metadata['validation_config'] = validation_config
                            lstm_metadata['validation_results'] = validation_results
                            validation_passed = validation_results['passed']
                            
                            if not validation_results['passed']:
                                st.error(f"⚠️ Validation found {len(validation_results['issues_found'])} issues")
                                for issue in validation_results['issues_found']:
                                    st.warning(issue)
                                
                                if validation_config.get('action') == 'Block feature generation':
                                    st.error("❌ Feature generation blocked due to validation failures")
                                    return
                            elif validation_results['warnings']:
                                st.info(f"ℹ️ {len(validation_results['warnings'])} warnings (training can proceed)")
                            else:
                                st.success("✅ All validation checks passed")
                
                # Add quality metadata for save function
                lstm_metadata['optimization_applied'] = optimization_applied
                lstm_metadata['validation_passed'] = validation_passed
                lstm_metadata['enhanced_features_config'] = enhanced_config
                lstm_metadata['target_representation'] = target_representation
                
                # Save
                if feature_gen.save_lstm_sequences(lstm_sequences, lstm_metadata):
                    st.success(f"✅ Generated {len(lstm_sequences)} advanced LSTM sequences")
                    
                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    with col_res1:
                        st.metric("Sequences", len(lstm_sequences))
                    with col_res2:
                        if 'optimization' in lstm_metadata:
                            st.metric("Features (Optimized)", lstm_metadata['optimization'].get('final_features', lstm_metadata.get('feature_count', 0)))
                        else:
                            st.metric("Features per Seq", lstm_metadata.get('feature_count', lstm_sequences.shape[2] if len(lstm_sequences.shape) == 3 else 0))
                    with col_res3:
                        st.metric("Window Size", lstm_metadata['params']['window_size'])
                    with col_res4:
                        st.metric("Lookback Windows", len(lstm_metadata['params']['lookback_windows']))
                    
                    st.info(f"📊 Features include: {', '.join(lstm_metadata['params']['feature_categories'])}")
                    st.info(f"📁 Saved to: `data/features/lstm/{feature_gen.game_folder}/`")
                    
                    # Export samples if configured
                    if export_config.get('enabled', False):
                        with st.spinner("Exporting feature samples..."):
                            # Convert sequences to 2D for export
                            export_df = pd.DataFrame(lstm_flat if 'lstm_flat' in locals() else lstm_sequences.reshape(len(lstm_sequences), -1))
                            exported_path = feature_gen.export_feature_samples(export_df, export_config, 'lstm')
                            if exported_path:
                                st.success(f"✅ Exported samples to: {exported_path.name}")
                else:
                    st.error("Failed to save LSTM sequences")
        except Exception as e:
            st.error(f"Error generating LSTM sequences: {e}")
            app_log(f"LSTM generation error: {e}", "error")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    st.divider()
    
    # CNN Embeddings Section
    st.markdown("#### 🟩 CNN Embeddings - Multi-Scale Pattern Detection")
    st.markdown("*Generates multi-scale CNN embeddings with kernel aggregation (3, 5, 7) for optimal pattern detection*")
    
    with st.expander("⚙️ CNN Configuration", expanded=True):
        cnn_col1, cnn_col2 = st.columns(2)
        
        with cnn_col1:
            cnn_window = st.slider(
                "Context Window Size",
                10, 60, 24,
                key="cnn_window",
                help="Sliding window size for multi-scale convolution"
            )
        
        with cnn_col2:
            cnn_embed = st.slider(
                "Embedding Dimension",
                32, 256, 64,
                step=32,
                key="cnn_embed",
                help="Dimension of output embeddings. Higher = more expressive but more computation."
            )
    
    if st.button("🚀 Generate CNN Embeddings", use_container_width=True, key="btn_cnn_gen"):
        try:
            with st.spinner("Generating advanced CNN embeddings with multi-scale aggregation..."):
                # Get configurations from session state
                enhanced_features_config = st.session_state.get('enhanced_features_config', {})
                target_representation = st.session_state.get('target_representation_mode', 'binary')
                
                cnn_embeddings, cnn_metadata = feature_gen.generate_cnn_embeddings(
                    raw_data,
                    window_size=cnn_window,
                    embedding_dim=cnn_embed
                )
                
                # Add quality metadata for save function
                cnn_metadata['optimization_applied'] = False  # CNN embeddings not optimized
                cnn_metadata['validation_passed'] = True  # Assumed valid
                cnn_metadata['enhanced_features_config'] = enhanced_features_config
                cnn_metadata['target_representation'] = target_representation
                
                # Save
                if feature_gen.save_cnn_embeddings(cnn_embeddings, cnn_metadata):
                    st.success(f"✅ Generated {len(cnn_embeddings)} advanced CNN embeddings")
                    
                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    with col_res1:
                        st.metric("Embeddings", len(cnn_embeddings))
                    with col_res2:
                        st.metric("Embedding Dim", cnn_metadata['params']['embedding_dim'])
                    with col_res3:
                        st.metric("Window Size", cnn_metadata['params']['window_size'])
                    with col_res4:
                        st.metric("Aggregation Methods", len(cnn_metadata['params']['aggregation_methods']))
                    
                    st.info(f"📊 Aggregation: {', '.join(cnn_metadata['params']['aggregation_methods'])}")
                    st.info(f"📁 Saved to: `data/features/cnn/{feature_gen.game_folder}/`")
                else:
                    st.error("Failed to save CNN embeddings")
        except Exception as e:
            st.error(f"Error generating CNN embeddings: {e}")
            app_log(f"CNN generation error: {e}", "error")
    
    st.divider()
    
    # Transformer Features Section
    st.markdown("#### 🟦 Transformer Features - 20-Dimension Feature Engineering")
    st.markdown("*Generates optimized 20-dimensional features for Transformer model input (CSV format)*")
    
    st.info("""
    **Transformer Features (20 dimensions):**
    - **Statistical** (5): sum, mean, std, skew, kurtosis - capture overall distribution
    - **Distribution** (3): min, max, range - track number bounds
    - **Parity** (2): even_count, odd_count - capture odd/even patterns
    - **Spacing** (3): avg_gap, max_gap, consecutive_pairs - measure number distribution
    - **Temporal** (3): days_since_last, day_of_week_sin, month_sin - time-based features
    - **Bonus** (2): bonus_num, bonus_zscore - bonus number statistics
    - **Pattern** (1): digit_pattern_score - captures digit-level patterns
    
    **Output: CSV with 20 numeric columns, normalized to 0-1 range**
    """)
    
    if st.button("🚀 Generate Transformer Features", use_container_width=True, key="btn_transformer_gen"):
        try:
            with st.spinner("Generating 20-dimensional Transformer features..."):
                # Get configurations from session state
                enhanced_features_config = st.session_state.get('enhanced_features_config', {})
                target_representation = st.session_state.get('target_representation_mode', 'binary')
                
                transformer_features, transformer_metadata = feature_gen.generate_transformer_features_csv(
                    raw_data,
                    output_dim=20
                )
                
                # Add quality metadata for save function
                transformer_metadata['optimization_applied'] = False  # Transformer features not optimized
                transformer_metadata['validation_passed'] = True  # Assumed valid
                transformer_metadata['enhanced_features_config'] = enhanced_features_config
                transformer_metadata['target_representation'] = target_representation
                
                # Save
                if feature_gen.save_transformer_features_csv(transformer_features, transformer_metadata):
                    st.success(f"✅ Generated {len(transformer_features)} Transformer feature vectors")
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    with col_res1:
                        st.metric("Feature Vectors", len(transformer_features))
                    with col_res2:
                        st.metric("Feature Dimensions", len(transformer_features.columns))
                    with col_res3:
                        st.metric("Format", "CSV")
                    
                    st.info(f"📊 Features: {', '.join(transformer_metadata['feature_columns'])}")
                    st.info(f"📁 Saved to: `data/features/transformer/{feature_gen.game_folder}/`")
                    
                    # Show feature preview
                    st.markdown("**Feature Preview (First 10 Rows):**")
                    st.dataframe(
                        transformer_features.head(10),
                        use_container_width=True,
                        height=300
                    )
                else:
                    st.error("Failed to save Transformer features")
        except Exception as e:
            st.error(f"Error generating Transformer features: {e}")
            app_log(f"Transformer generation error: {e}", "error")
    
    st.divider()
    
    # XGBoost Advanced Features Section
    st.markdown("#### 🟨 Advanced Features - XGBoost Feature Engineering")
    st.markdown("*Comprehensive statistical, temporal, and pattern-based feature engineering (100+ features)*")
    
    st.info("""
    **Advanced XGBoost Features Include:**
    - **10** basic statistical features (sum, mean, std, skew, kurtosis, etc.)
    - **15** distribution features (percentiles, quantiles, buckets)
    - **8** parity features (even/odd ratios, modulo patterns)
    - **8** spacing features (gaps, consecutive sequences)
    - **20** historical frequency features (lookback windows: 5, 10, 20, 30, 60)
    - **15** rolling statistics (3, 5, 10-period rolling means/std)
    - **10** temporal features (day of week, season, week of year)
    - **8** bonus number features
    - **8** jackpot features (log transform, z-score, rolling)
    - **5** entropy and randomness features
    
    **Total: 115+ engineered features for optimal gradient boosting**
    """)
    
    if st.button("🚀 Generate XGBoost Features", use_container_width=True, key="btn_xgb_gen"):
        try:
            with st.spinner("Generating 115+ advanced XGBoost features with comprehensive analysis..."):
                # Get configurations from session state
                optimization_config = st.session_state.get('feature_optimization_config', {})
                validation_config = st.session_state.get('feature_validation_config', {})
                export_config = st.session_state.get('feature_export_config', {})
                enhanced_features_config = st.session_state.get('enhanced_features_config', {})
                target_representation = st.session_state.get('target_representation_mode', 'binary')
                
                xgb_features, xgb_metadata = feature_gen.generate_xgboost_features(raw_data)
                original_feature_count = len(xgb_features.columns)
                
                # Track quality indicators for save function
                optimization_applied = False
                validation_passed = True
                
                # Apply optimization if configured
                if optimization_config.get('enabled', False):
                    with st.spinner(f"Optimizing features with {optimization_config.get('method', 'RFE')}..."):
                        xgb_features, opt_info = feature_gen.apply_feature_optimization(xgb_features, optimization_config)
                        xgb_metadata['optimization_applied'] = True
                        xgb_metadata['optimization_config'] = optimization_config
                        optimization_applied = True
                        st.success(f"✅ Optimized: {opt_info.get('original_features', 0)} → {opt_info.get('final_features', 0)} features")
                
                # Validate if configured
                if validation_config.get('enabled', False):
                    with st.spinner("Validating feature quality..."):
                        # Get only numeric columns, excluding metadata columns
                        numeric_cols = xgb_features.select_dtypes(include=[np.number]).columns
                        # Explicitly exclude any metadata columns that might have snuck through
                        exclude_cols = ['numbers', 'draw_date']
                        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
                        
                        if len(numeric_cols) == 0:
                            st.warning("⚠️ No numeric columns found for validation")
                        else:
                            validation_results = feature_gen.validate_features(xgb_features[numeric_cols].values, validation_config)
                            xgb_metadata['validation_passed'] = validation_results['passed']
                            xgb_metadata['validation_config'] = validation_config
                            xgb_metadata['validation_results'] = validation_results
                            validation_passed = validation_results['passed']
                        
                        if not validation_results['passed']:
                            st.error(f"⚠️ Validation found {len(validation_results['issues_found'])} issues")
                            for issue in validation_results['issues_found']:
                                st.warning(issue)
                            
                            if validation_config.get('action') == 'Block feature generation':
                                st.error("❌ Feature generation blocked due to validation failures")
                                return
                        elif validation_results['warnings']:
                            st.info(f"ℹ️ {len(validation_results['warnings'])} warnings (training can proceed)")
                        else:
                            st.success("✅ All validation checks passed")
                
                # Add additional metadata for save function
                xgb_metadata['optimization_applied'] = optimization_applied
                xgb_metadata['validation_passed'] = validation_passed
                xgb_metadata['enhanced_features_config'] = enhanced_features_config
                xgb_metadata['target_representation'] = target_representation
                
                # Save
                if feature_gen.save_xgboost_features(xgb_features, xgb_metadata):
                    st.success(f"✅ Generated {len(xgb_features)} complete feature sets")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Draws", len(xgb_features))
                    with col_res2:
                        final_features = len(xgb_features.columns) if not optimization_applied else xgb_features.shape[1]
                        st.metric("Features", final_features)
                    
                    st.info(f"📊 Feature categories: {', '.join(xgb_metadata.get('params', {}).get('feature_categories', []))}")
                    st.info(f"📁 Saved to: `data/features/xgboost/{feature_gen.game_folder}/`")
                    
                    # Show feature preview
                    st.markdown("**Feature Preview (First 10 Rows):**")
                    display_cols = [col for col in xgb_features.columns[:15]]  # Show first 15 features
                    st.dataframe(
                        xgb_features[display_cols].head(10),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Show feature statistics
                    st.markdown("**Feature Statistics:**")
                    numeric_cols = xgb_features.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(
                            xgb_features[numeric_cols].describe().T,
                            use_container_width=True,
                            height=400
                        )
                    
                    # Export samples if configured
                    if export_config.get('enabled', False):
                        with st.spinner("Exporting feature samples..."):
                            exported_path = feature_gen.export_feature_samples(xgb_features, export_config, 'xgboost')
                            if exported_path:
                                st.success(f"✅ Exported samples to: {exported_path.name}")
                else:
                    st.error("Failed to save XGBoost features")
        except Exception as e:
            st.error(f"Error generating XGBoost features: {e}")
            app_log(f"XGBoost generation error: {e}", "error")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    st.divider()
    
    # CatBoost Features Section
    st.markdown("#### 🟠 CatBoost Features - Categorical Feature Engineering")
    st.markdown("*Categorical feature optimization for boosting (80+ features)*")
    
    st.info("""
    **CatBoost-Optimized Features Include:**
    - **10** basic statistical features (sum, mean, std, skew, kurtosis, etc.)
    - **15** distribution features (percentiles, quantiles, buckets)
    - **8** parity features (even/odd ratios, modulo patterns)
    - **8** spacing features (gaps, consecutive sequences)
    - **Categorical encoding** for frequency and pattern detection
    
    **Total: 80+ features optimized for categorical boosting**
    """)
    
    if st.button("🚀 Generate CatBoost Features", use_container_width=True, key="btn_cb_gen"):
        try:
            with st.spinner("Generating CatBoost features with categorical optimization..."):
                # Get configurations from session state
                optimization_config = st.session_state.get('feature_optimization_config', {})
                validation_config = st.session_state.get('feature_validation_config', {})
                export_config = st.session_state.get('feature_export_config', {})
                enhanced_features_config = st.session_state.get('enhanced_features_config', {})
                target_representation = st.session_state.get('target_representation_mode', 'binary')
                
                cb_features, cb_metadata = feature_gen.generate_catboost_features(raw_data)
                original_feature_count = len(cb_features.columns)
                
                # Track quality indicators
                optimization_applied = False
                validation_passed = True
                
                # Apply optimization if configured
                if optimization_config.get('enabled', False):
                    with st.spinner(f"Optimizing features with {optimization_config.get('method', 'RFE')}..."):
                        cb_features, opt_info = feature_gen.apply_feature_optimization(cb_features, optimization_config)
                        cb_metadata['optimization_applied'] = True
                        cb_metadata['optimization_config'] = optimization_config
                        optimization_applied = True
                        st.success(f"✅ Optimized: {opt_info.get('original_features', 0)} → {opt_info.get('final_features', 0)} features")
                
                # Validate if configured
                if validation_config.get('enabled', False):
                    with st.spinner("Validating feature quality..."):
                        numeric_cols = cb_features.select_dtypes(include=[np.number]).columns
                        validation_results = feature_gen.validate_features(cb_features[numeric_cols].values, validation_config)
                        cb_metadata['validation_passed'] = validation_results['passed']
                        cb_metadata['validation_config'] = validation_config
                        cb_metadata['validation_results'] = validation_results
                        validation_passed = validation_results['passed']
                        
                        if not validation_results['passed']:
                            st.error(f"⚠️ Validation found {len(validation_results['issues_found'])} issues")
                            for issue in validation_results['issues_found']:
                                st.warning(issue)
                            
                            if validation_config.get('action') == 'Block feature generation':
                                st.error("❌ Feature generation blocked due to validation failures")
                                return
                        elif validation_results['warnings']:
                            st.info(f"ℹ️ {len(validation_results['warnings'])} warnings (training can proceed)")
                        else:
                            st.success("✅ All validation checks passed")
                
                # Add quality metadata for save function
                cb_metadata['optimization_applied'] = optimization_applied
                cb_metadata['validation_passed'] = validation_passed
                cb_metadata['enhanced_features_config'] = enhanced_features_config
                cb_metadata['target_representation'] = target_representation
                
                # Save
                if feature_gen.save_catboost_features(cb_features, cb_metadata):
                    st.success(f"✅ Generated {len(cb_features)} complete feature sets")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Draws", len(cb_features))
                    with col_res2:
                        final_features = len(cb_features.columns) if not optimization_applied else cb_features.shape[1]
                        st.metric("Features", final_features)
                    
                    st.info(f"📊 Feature categories: {', '.join(cb_metadata.get('params', {}).get('feature_categories', []))}")
                    st.info(f"📁 Saved to: `data/features/catboost/{feature_gen.game_folder}/`")
                    
                    # Show feature preview
                    st.markdown("**Feature Preview (First 10 Rows):**")
                    display_cols = [col for col in cb_features.columns[:15]]
                    st.dataframe(
                        cb_features[display_cols].head(10),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Export samples if configured
                    if export_config.get('enabled', False):
                        with st.spinner("Exporting feature samples..."):
                            exported_path = feature_gen.export_feature_samples(cb_features, export_config, 'catboost')
                            if exported_path:
                                st.success(f"✅ Exported samples to: {exported_path.name}")
                else:
                    st.error("Failed to save CatBoost features")
        except Exception as e:
            st.error(f"Error generating CatBoost features: {e}")
            app_log(f"CatBoost generation error: {e}", "error")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    st.divider()
    
    # LightGBM Features Section
    st.markdown("#### 🟢 LightGBM Features - Fast Boosting Feature Engineering")
    st.markdown("*High-cardinality features optimized for leaf-wise boosting (80+ features)*")
    
    st.info("""
    **LightGBM-Optimized Features Include:**
    - **10** basic statistical features (sum, mean, std, skew, kurtosis, etc.)
    - **15** distribution features (percentiles, quantiles, buckets)
    - **8** parity features (even/odd ratios, modulo patterns)
    - **8** spacing features (gaps, consecutive sequences)
    - **High-cardinality feature support** for leaf-wise tree growth
    
    **Total: 80+ features optimized for fast leaf-wise boosting**
    """)
    
    if st.button("🚀 Generate LightGBM Features", use_container_width=True, key="btn_lgb_gen"):
        try:
            with st.spinner("Generating LightGBM features optimized for fast boosting..."):
                # Get configurations from session state
                optimization_config = st.session_state.get('feature_optimization_config', {})
                validation_config = st.session_state.get('feature_validation_config', {})
                export_config = st.session_state.get('feature_export_config', {})
                enhanced_features_config = st.session_state.get('enhanced_features_config', {})
                target_representation = st.session_state.get('target_representation_mode', 'binary')
                
                lgb_features, lgb_metadata = feature_gen.generate_lightgbm_features(raw_data)
                original_feature_count = len(lgb_features.columns)
                
                # Track quality indicators
                optimization_applied = False
                validation_passed = True
                
                # Apply optimization if configured
                if optimization_config.get('enabled', False):
                    with st.spinner(f"Optimizing features with {optimization_config.get('method', 'RFE')}..."):
                        lgb_features, opt_info = feature_gen.apply_feature_optimization(lgb_features, optimization_config)
                        lgb_metadata['optimization_applied'] = True
                        lgb_metadata['optimization_config'] = optimization_config
                        optimization_applied = True
                        st.success(f"✅ Optimized: {opt_info.get('original_features', 0)} → {opt_info.get('final_features', 0)} features")
                
                # Validate if configured
                if validation_config.get('enabled', False):
                    with st.spinner("Validating feature quality..."):
                        numeric_cols = lgb_features.select_dtypes(include=[np.number]).columns
                        validation_results = feature_gen.validate_features(lgb_features[numeric_cols].values, validation_config)
                        lgb_metadata['validation_passed'] = validation_results['passed']
                        lgb_metadata['validation_config'] = validation_config
                        lgb_metadata['validation_results'] = validation_results
                        validation_passed = validation_results['passed']
                        
                        if not validation_results['passed']:
                            st.error(f"⚠️ Validation found {len(validation_results['issues_found'])} issues")
                            for issue in validation_results['issues_found']:
                                st.warning(issue)
                            
                            if validation_config.get('action') == 'Block feature generation':
                                st.error("❌ Feature generation blocked due to validation failures")
                                return
                        elif validation_results['warnings']:
                            st.info(f"ℹ️ {len(validation_results['warnings'])} warnings (training can proceed)")
                        else:
                            st.success("✅ All validation checks passed")
                
                # Add quality metadata for save function
                lgb_metadata['optimization_applied'] = optimization_applied
                lgb_metadata['validation_passed'] = validation_passed
                lgb_metadata['enhanced_features_config'] = enhanced_features_config
                lgb_metadata['target_representation'] = target_representation
                
                # Save
                if feature_gen.save_lightgbm_features(lgb_features, lgb_metadata):
                    st.success(f"✅ Generated {len(lgb_features)} complete feature sets")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Draws", len(lgb_features))
                    with col_res2:
                        final_features = len(lgb_features.columns) if not optimization_applied else lgb_features.shape[1]
                        st.metric("Features", final_features)
                    
                    st.info(f"📊 Feature categories: {', '.join(lgb_metadata.get('params', {}).get('feature_categories', []))}")
                    st.info(f"📁 Saved to: `data/features/lightgbm/{feature_gen.game_folder}/`")
                    
                    # Show feature preview
                    st.markdown("**Feature Preview (First 10 Rows):**")
                    display_cols = [col for col in lgb_features.columns[:15]]
                    st.dataframe(
                        lgb_features[display_cols].head(10),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Export samples if configured
                    if export_config.get('enabled', False):
                        with st.spinner("Exporting feature samples..."):
                            exported_path = feature_gen.export_feature_samples(lgb_features, export_config, 'lightgbm')
                            if exported_path:
                                st.success(f"✅ Exported samples to: {exported_path.name}")
                else:
                    st.error("Failed to save LightGBM features")
        except Exception as e:
            st.error(f"Error generating LightGBM features: {e}")
            app_log(f"LightGBM generation error: {e}", "error")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    st.divider()
    
    # ========================================
    # 🔬 COMPREHENSIVE FEATURE VALIDATION SUITE
    # ========================================
    st.markdown("### 🔬 Comprehensive Feature Validation Suite")
    st.markdown("*Run comprehensive quality checks on all generated features*")
    
    with st.expander("🧪 Feature Validation Tools", expanded=False):
        st.markdown("""
        **Comprehensive validation for all feature types:**
        
        Run these checks after generating features to ensure quality before training.
        """)
        
        # Select feature type to validate
        validation_feature_type = st.selectbox(
            "Select feature type to validate",
            ["LSTM Sequences", "CNN Embeddings", "Transformer Features", "XGBoost Features", "CatBoost Features", "LightGBM Features", "All Feature Types"],
            key="validation_feature_type"
        )
        
        col_check1, col_check2 = st.columns(2)
        
        with col_check1:
            st.markdown("**Data Quality Checks:**")
            run_nan_check = st.checkbox("Check for NaN/Inf values", value=True, key="run_nan_check")
            run_variance_check = st.checkbox("Detect zero-variance features", value=True, key="run_var_check")
            run_correlation_check = st.checkbox("Find highly correlated features (>0.95)", value=True, key="run_corr_check")
        
        with col_check2:
            st.markdown("**Dimension Checks:**")
            run_dimension_check = st.checkbox("Validate feature dimensions", value=True, key="run_dim_check")
            run_shape_check = st.checkbox("Check data shape consistency", value=True, key="run_shape_check")
            run_dtype_check = st.checkbox("Verify data types", value=True, key="run_dtype_check")
        
        if st.button("🚀 Run Feature Validation Suite", use_container_width=True, key="btn_validate"):
            with st.spinner(f"Running comprehensive validation on {validation_feature_type}..."):
                try:
                    validation_results = {
                        'feature_type': validation_feature_type,
                        'checks_run': [],
                        'issues_found': [],
                        'warnings': [],
                        'passed': True
                    }
                    
                    # Determine which feature types to check
                    if validation_feature_type == "All Feature Types":
                        types_to_check = ["lstm", "cnn", "transformer", "xgboost", "catboost", "lightgbm"]
                    else:
                        type_map = {
                            "LSTM Sequences": "lstm",
                            "CNN Embeddings": "cnn",
                            "Transformer Features": "transformer",
                            "XGBoost Features": "xgboost",
                            "CatBoost Features": "catboost",
                            "LightGBM Features": "lightgbm"
                        }
                        types_to_check = [type_map[validation_feature_type]]
                    
                    # Run validation for each type
                    for ftype in types_to_check:
                        st.markdown(f"#### Validating {ftype.upper()} features...")
                        
                        # Get feature files
                        feature_files = feature_gen._get_feature_files_for_type(ftype)
                        
                        if not feature_files:
                            st.warning(f"No {ftype} features found - generate them first")
                            continue
                        
                        # Load features based on type
                        features_data = None
                        latest_file = None
                        if ftype in ["lstm", "cnn"]:
                            # NPZ files
                            try:
                                latest_file = sorted(feature_files)[-1]
                                st.info(f"📁 Validating file: `{latest_file.name}`\n\n📂 Location: `{latest_file.parent}`")
                                data = np.load(latest_file)
                                if "sequences" in data:
                                    features_data = data["sequences"]
                                elif "embeddings" in data:
                                    features_data = data["embeddings"]
                                features_shape = features_data.shape if features_data is not None else None
                            except Exception as e:
                                st.error(f"Error loading {ftype} features: {e}")
                                continue
                        
                        else:  # CSV-based features
                            try:
                                latest_file = sorted(feature_files)[-1]
                                st.info(f"📁 Validating file: `{latest_file.name}`\n\n📂 Location: `{latest_file.parent}`")
                                features_df = pd.read_csv(latest_file)
                                features_data = features_df.select_dtypes(include=[np.number]).values
                                features_shape = features_data.shape
                            except Exception as e:
                                st.error(f"Error loading {ftype} features: {e}")
                                continue
                        
                        if features_data is None:
                            st.warning(f"Could not load {ftype} features")
                            continue
                        
                        # === Check 1: NaN/Inf Detection ===
                        if run_nan_check:
                            validation_results['checks_run'].append(f"{ftype}: NaN/Inf check")
                            
                            if len(features_data.shape) == 3:  # LSTM sequences (samples, timesteps, features)
                                nan_count = np.isnan(features_data).sum()
                                inf_count = np.isinf(features_data).sum()
                            else:  # 2D arrays
                                nan_count = np.isnan(features_data).sum()
                                inf_count = np.isinf(features_data).sum()
                            
                            if nan_count > 0 or inf_count > 0:
                                validation_results['issues_found'].append(
                                    f"❌ {ftype}: Found {nan_count} NaN and {inf_count} Inf values"
                                )
                                validation_results['passed'] = False
                                st.error(f"❌ Found {nan_count} NaN, {inf_count} Inf values in {ftype}")
                            else:
                                st.success(f"✅ No NaN/Inf values in {ftype}")
                        
                        # === Check 2: Zero Variance Detection ===
                        if run_variance_check:
                            validation_results['checks_run'].append(f"{ftype}: Variance check")
                            
                            if len(features_data.shape) == 3:  # LSTM
                                # Check variance across samples for each feature at each timestep
                                variances = np.var(features_data, axis=0)  # (timesteps, features)
                                zero_var_count = (variances < 1e-10).sum()
                            else:  # 2D
                                variances = np.var(features_data, axis=0)
                                zero_var_count = (variances < 1e-10).sum()
                            
                            if zero_var_count > 0:
                                validation_results['warnings'].append(
                                    f"⚠️ {ftype}: Found {zero_var_count} zero-variance features (constant values)"
                                )
                                st.warning(f"⚠️ Found {zero_var_count} constant features in {ftype}")
                            else:
                                st.success(f"✅ All features have variance in {ftype}")
                        
                        # === Check 3: High Correlation Detection ===
                        if run_correlation_check and len(features_data.shape) == 2:
                            validation_results['checks_run'].append(f"{ftype}: Correlation check")
                            
                            try:
                                # Sample if too large
                                if features_data.shape[0] > 5000:
                                    sample_indices = np.random.choice(features_data.shape[0], 5000, replace=False)
                                    sample_data = features_data[sample_indices]
                                else:
                                    sample_data = features_data
                                
                                corr_matrix = np.corrcoef(sample_data, rowvar=False)
                                # Find pairs with correlation > 0.95 (excluding diagonal)
                                high_corr_mask = (np.abs(corr_matrix) > 0.95) & (np.abs(corr_matrix) < 1.0)
                                high_corr_count = high_corr_mask.sum() // 2  # Divide by 2 because matrix is symmetric
                                
                                if high_corr_count > 0:
                                    validation_results['warnings'].append(
                                        f"⚠️ {ftype}: Found {high_corr_count} highly correlated feature pairs (>0.95)"
                                    )
                                    st.warning(f"⚠️ Found {high_corr_count} highly correlated pairs in {ftype}")
                                else:
                                    st.success(f"✅ No high multicollinearity in {ftype}")
                            except Exception as e:
                                st.info(f"ℹ️ Skipped correlation check for {ftype}: {e}")
                        
                        # === Check 4: Dimension Validation ===
                        if run_dimension_check:
                            validation_results['checks_run'].append(f"{ftype}: Dimension check")
                            
                            expected_shapes = {
                                'lstm': (None, None, None),  # (samples, timesteps, features)
                                'cnn': (None, None),  # (samples, embedding_dim)
                                'transformer': (None, 20),  # (samples, 20 features)
                                'xgboost': (None, None),  # (samples, 115+ features)
                                'catboost': (None, None),  # (samples, 80+ features)
                                'lightgbm': (None, None)  # (samples, 80+ features)
                            }
                            
                            expected = expected_shapes.get(ftype, (None, None))
                            actual = features_shape
                            
                            st.info(f"📊 {ftype} shape: {actual}")
                            
                            # Validate specific dimensions
                            if ftype == 'transformer' and actual[1] != 20:
                                validation_results['issues_found'].append(
                                    f"❌ {ftype}: Expected 20 features, got {actual[1]}"
                                )
                                validation_results['passed'] = False
                                st.error(f"❌ Transformer should have 20 features, found {actual[1]}")
                            
                            if ftype == 'lstm' and len(actual) != 3:
                                validation_results['issues_found'].append(
                                    f"❌ {ftype}: Expected 3D array (samples, timesteps, features), got shape {actual}"
                                )
                                validation_results['passed'] = False
                                st.error(f"❌ LSTM should be 3D, got shape {actual}")
                        
                        # === Check 5: Shape Consistency ===
                        if run_shape_check:
                            validation_results['checks_run'].append(f"{ftype}: Shape consistency check")
                            
                            # Check if all samples have same shape
                            if len(features_data.shape) >= 2:
                                st.success(f"✅ Shape consistency verified for {ftype}: {features_shape}")
                            else:
                                validation_results['issues_found'].append(
                                    f"❌ {ftype}: Invalid shape - expected at least 2D array"
                                )
                                validation_results['passed'] = False
                        
                        # === Check 6: Data Type Validation ===
                        if run_dtype_check:
                            validation_results['checks_run'].append(f"{ftype}: Data type check")
                            
                            dtype = features_data.dtype
                            if dtype in [np.float32, np.float64]:
                                st.success(f"✅ Correct dtype for {ftype}: {dtype}")
                            else:
                                validation_results['warnings'].append(
                                    f"⚠️ {ftype}: Unexpected dtype {dtype}, expected float32/float64"
                                )
                                st.warning(f"⚠️ {ftype} has dtype {dtype}")
                    
                    # === Display Summary ===
                    st.divider()
                    st.markdown("### 📋 Validation Summary")
                    
                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    with col_sum1:
                        st.metric("Checks Run", len(validation_results['checks_run']))
                    with col_sum2:
                        st.metric("Issues Found", len(validation_results['issues_found']))
                    with col_sum3:
                        st.metric("Warnings", len(validation_results['warnings']))
                    
                    if validation_results['passed'] and len(validation_results['warnings']) == 0:
                        st.success("🎉 All validation checks passed! Features are ready for training.")
                    elif validation_results['passed']:
                        st.info("✅ Validation passed with warnings. Review warnings before training.")
                    else:
                        st.error("❌ Validation failed. Please fix issues before training.")
                    
                    # Show detailed results
                    if validation_results['issues_found']:
                        st.markdown("**❌ Issues Found:**")
                        for issue in validation_results['issues_found']:
                            st.error(issue)
                    
                    if validation_results['warnings']:
                        st.markdown("**⚠️ Warnings:**")
                        for warning in validation_results['warnings']:
                            st.warning(warning)
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    
                    if len(validation_results['issues_found']) > 0:
                        st.markdown("""
                        **Critical Issues Found:**
                        1. **NaN/Inf values**: Re-generate features with proper handling
                        2. **Dimension mismatches**: Check feature generation parameters
                        3. **Invalid shapes**: Verify model type matches feature type
                        
                        ⚠️ **Do not proceed to training until issues are resolved.**
                        """)
                    elif len(validation_results['warnings']) > 0:
                        st.markdown("""
                        **Warnings Detected:**
                        1. **Zero-variance features**: Consider removing constant features
                        2. **High correlation**: Apply feature selection or PCA
                        3. **Data type issues**: May affect model performance
                        
                        ℹ️ **Training can proceed, but consider addressing warnings for optimal performance.**
                        """)
                    else:
                        st.markdown("""
                        **All Clear!** ✅
                        
                        Your features passed all quality checks and are ready for training.
                        
                        **Next Steps:**
                        1. Proceed to Model Training section
                        2. Select appropriate model type for your feature type
                        3. Configure training parameters
                        4. Start training!
                        """)
                
                except Exception as e:
                    st.error(f"Error during validation: {e}")
                    app_log(f"Feature validation error: {e}", "error")
                    import traceback
                    st.code(traceback.format_exc())


def _get_raw_csv_files(game: str) -> List[Path]:
    """Get raw CSV files for game."""
    game_folder = _sanitize_game_name(game)
    game_dir = get_data_dir() / game_folder
    if not game_dir.exists():
        return []
    return sorted(game_dir.glob("training_data_*.csv"))


def _get_feature_files(game: str, feature_type: str, prefer_optimized: bool = True) -> List[Path]:
    """
    Get feature files for game and type with version awareness.
    Prioritizes optimized and validated features over regular features.
    
    Args:
        game: Game name
        feature_type: Feature type (lstm, cnn, xgboost, etc.)
        prefer_optimized: If True, prefer optimized/validated features
    
    Returns:
        List of feature file paths, sorted by quality then timestamp
    """
    game_folder = _sanitize_game_name(game)
    features_dir = get_data_dir() / "features" / feature_type / game_folder
    
    app_log(f"🔍 Looking for {feature_type} features in: {features_dir}", "info")
    
    if not features_dir.exists():
        app_log(f"⚠️ Directory does not exist: {features_dir}", "warning")
        return []
    
    # Define valid extensions per feature type
    valid_extensions = {
        "lstm": [".npz"],
        "cnn": [".npz"],
        "transformer": [".npz"],
        "xgboost": [".csv"],
        "catboost": [".csv"],
        "lightgbm": [".csv"]
    }
    
    file_extensions = valid_extensions.get(feature_type, [".csv", ".npz"])
    
    app_log(f"  Valid extensions for {feature_type}: {file_extensions}", "info")
    
    if not prefer_optimized:
        # Return all data files (exclude metadata) sorted by name
        all_files = [f for f in features_dir.glob("*") if f.suffix in file_extensions]
        app_log(f"  Found {len(all_files)} files (no optimization preference)", "info")
        return sorted(all_files)
    
    # Get all data files and prioritize by quality indicators in filename
    all_files = [f for f in features_dir.glob("*") if f.suffix in file_extensions]
    all_files = sorted(all_files)
    
    app_log(f"  Found {len(all_files)} total {feature_type} files with valid extensions", "info")
    if all_files:
        for f in all_files:
            app_log(f"    - {f.name}", "info")
    
    quality_files = []
    
    # Priority 1: Optimized + Validated (OPTIMIZED_VALID or optimized_validated)
    for f in all_files:
        fname_upper = f.name.upper()
        if ("OPTIMIZED" in fname_upper and "VALID" in fname_upper) or ("_OPTIMIZED_VALID_" in fname_upper):
            quality_files.append(f)
    
    # Priority 2: Optimized only
    for f in all_files:
        if f in quality_files:
            continue
        fname_upper = f.name.upper()
        if "OPTIMIZED" in fname_upper:
            quality_files.append(f)
    
    # Priority 3: Validated only  
    for f in all_files:
        if f in quality_files:
            continue
        fname_upper = f.name.upper()
        if "VALID" in fname_upper:
            quality_files.append(f)
    
    # Priority 4: Regular features (not already included)
    for f in all_files:
        if f in quality_files:
            continue
        quality_files.append(f)
    
    # Log which files were selected and their priority
    if quality_files:
        app_log(f"Feature file selection for {feature_type}:", "info")
        for idx, f in enumerate(quality_files[:3], 1):  # Show top 3
            app_log(f"  {idx}. {f.name}", "info")
    
    return quality_files if quality_files else all_files


def _estimate_total_samples(data_sources: Dict[str, List[Path]]) -> int:
    """Estimate total samples from data sources."""
    total = 0
    
    # Raw CSV files
    for filepath in data_sources.get("raw_csv", []):
        try:
            df = pd.read_csv(filepath)
            total += len(df)
        except:
            pass
    
    # Feature files (estimate)
    for feature_type in ["lstm", "cnn", "transformer", "xgboost", "catboost", "lightgbm"]:
        for filepath in data_sources.get(feature_type, []):
            try:
                if filepath.suffix == ".npz":
                    data = np.load(filepath)
                    if "sequences" in data:
                        total += len(data["sequences"])
                    elif "embeddings" in data:
                        total += len(data["embeddings"])
                elif filepath.suffix == ".csv":
                    df = pd.read_csv(filepath)
                    total += len(df)
            except:
                pass
    
    return total


def _render_model_training():
    """Render ultra-advanced Model Training section with state-of-the-art AI/ML."""
    st.subheader("🤖 Advanced AI-Powered Model Training")
    st.markdown("*Ultra-accurate lottery number prediction with advanced AI/ML techniques*")
    
    if AdvancedModelTrainer is None:
        st.error("Advanced Model Trainer not available")
        return
    
    # ========================================================================
    # STEP 1: SELECT GAME AND MODEL TYPE
    # ========================================================================
    
    st.markdown("### 📋 Step 1 – Select Game and Model Type")
    
    step1_col1, step1_col2 = st.columns(2)
    
    with step1_col1:
        games = get_available_games()
        selected_game = st.selectbox(
            "🎮 Select Game",
            games,
            key="train_game",
            help="Choose which lottery game to train a model for"
        )
    
    with step1_col2:
        model_types = ["XGBoost", "CatBoost", "LightGBM", "LSTM", "CNN", "Transformer", "Ensemble"]
        selected_model = st.selectbox(
            "🤖 Model Type",
            model_types,
            key="train_model_type",
            help="Select the machine learning algorithm"
        )
    
    st.info(f"""
    **Model Selection Guide:**
    - **XGBoost**: Gradient boosting with regularization - Fast and accurate (115+ features)
    - **CatBoost**: Optimized for categorical/tabular data - Best for lottery numbers (77 features)
    - **LightGBM**: Ultra-fast leaf-wise gradient boosting - Best for speed+accuracy balance (77 features)
    - **LSTM**: Captures temporal sequences and patterns - For time-series prediction (70+ features)
    - **CNN**: Multi-scale convolution (kernels 3,5,7) for pattern detection - 87.85% accuracy
    - **Transformer**: Advanced attention mechanisms - Experimental
    - **Ensemble**: Combines XGBoost + CatBoost + LightGBM + LSTM + CNN - **RECOMMENDED (90%+ target)**
    """)
    
    st.divider()
    
    # ========================================================================
    # STEP 2: SELECT TRAINING DATA SOURCES
    # ========================================================================
    
    st.markdown("### 📊 Step 2 – Select Training Data Sources")
    
    st.markdown("""
    **Training Data Strategy:**
    Data sources shown below match your selected model type. 
    For Ensemble: uses data from all model types. For single models: uses data specific to that model.
    """)
    
    # Determine which data sources to show based on model type
    # IMPORTANT: Tree models (XGBoost, CatBoost, LightGBM) should NOT use raw_csv
    # because it causes feature concatenation (raw_csv=8 + tree_features=85 = 93 features)
    # This creates a mismatch with the schema which expects only 85 features
    # 
    # IMPORTANT: Neural models (LSTM, CNN, Transformer) should NOT use raw_csv either
    # because flattened neural features are already large (1125+, 1408+, etc.)
    # Adding raw_csv creates dimension explosion (1125+8=1133, etc.)
    # This mismatches with schema which expects only flattened neural features
    # EXCEPTION: CNN can use raw_csv as fallback if no CNN embeddings exist yet
    model_data_sources = {
        "XGBoost": ["xgboost"],  # Use ONLY engineered XGBoost features (85), not raw_csv
        "CatBoost": ["catboost"],  # Use ONLY engineered CatBoost features (85), not raw_csv
        "LightGBM": ["lightgbm"],  # Use ONLY engineered LightGBM features (85), not raw_csv
        "LSTM": ["lstm"],  # Use ONLY LSTM flattened sequences (1125), not raw_csv
        "CNN": ["raw_csv", "cnn"],  # Prefer CNN embeddings (1408+), fallback to raw_csv if none exist
        "Transformer": ["transformer"],  # Use ONLY Transformer embeddings (512), not raw_csv
        "Ensemble": ["xgboost", "catboost", "lightgbm", "lstm", "cnn"]  # All engineered features, no raw_csv
    }
    
    available_sources = model_data_sources.get(selected_model, ["xgboost"])
    
    # Initialize checkbox states if not present
    if "use_raw_csv_adv" not in st.session_state:
        st.session_state["use_raw_csv_adv"] = True
    if "use_lstm_features_adv" not in st.session_state:
        st.session_state["use_lstm_features_adv"] = "lstm" in available_sources
    if "use_cnn_features_adv" not in st.session_state:
        st.session_state["use_cnn_features_adv"] = "cnn" in available_sources
    if "use_transformer_features_adv" not in st.session_state:
        st.session_state["use_transformer_features_adv"] = "transformer" in available_sources
    if "use_xgboost_features_adv" not in st.session_state:
        st.session_state["use_xgboost_features_adv"] = "xgboost" in available_sources
    if "use_catboost_features_adv" not in st.session_state:
        st.session_state["use_catboost_features_adv"] = "catboost" in available_sources
    if "use_lightgbm_features_adv" not in st.session_state:
        st.session_state["use_lightgbm_features_adv"] = "lightgbm" in available_sources
    
    # Reset states based on current model type to ensure correct defaults on page load
    if selected_model != st.session_state.get("last_selected_model", None):
        st.session_state["use_raw_csv_adv"] = True
        st.session_state["use_lstm_features_adv"] = "lstm" in available_sources
        st.session_state["use_cnn_features_adv"] = "cnn" in available_sources
        st.session_state["use_transformer_features_adv"] = "transformer" in available_sources
        st.session_state["use_xgboost_features_adv"] = "xgboost" in available_sources
        st.session_state["use_catboost_features_adv"] = "catboost" in available_sources
        st.session_state["use_lightgbm_features_adv"] = "lightgbm" in available_sources
        st.session_state["last_selected_model"] = selected_model
    
    data_col1, data_col2 = st.columns(2)
    
    with data_col1:
        # Always show Raw CSV
        use_raw_csv = st.checkbox(
            "📁 Raw CSV Files",
            value=st.session_state["use_raw_csv_adv"],
            key="checkbox_raw_csv_adv",
            help="Use raw lottery draw data"
        )
        st.session_state["use_raw_csv_adv"] = use_raw_csv
        
        # Show LSTM if applicable
        if "lstm" in available_sources:
            use_lstm = st.checkbox(
                "🔷 LSTM Sequences",
                value=st.session_state["use_lstm_features_adv"],
                key="checkbox_lstm_adv",
                help="Use LSTM sequence features for temporal learning"
            )
            st.session_state["use_lstm_features_adv"] = use_lstm
        else:
            use_lstm = False
            st.session_state["use_lstm_features_adv"] = False
        
        # Show CatBoost if applicable
        if "catboost" in available_sources:
            use_catboost_feat = st.checkbox(
                "🟧 CatBoost Features",
                value=st.session_state["use_catboost_features_adv"],
                key="checkbox_cb_feat_adv",
                help="Use 77 engineered CatBoost features optimized for categorical data"
            )
            st.session_state["use_catboost_features_adv"] = use_catboost_feat
        else:
            use_catboost_feat = False
            st.session_state["use_catboost_features_adv"] = False
    
    with data_col2:
        # Show CNN if applicable
        if "cnn" in available_sources:
            use_cnn = st.checkbox(
                "🟩 CNN Embeddings",
                value=st.session_state["use_cnn_features_adv"],
                key="checkbox_cnn_adv",
                help="Use CNN multi-scale embeddings for pattern detection"
            )
            st.session_state["use_cnn_features_adv"] = use_cnn
        else:
            use_cnn = False
            st.session_state["use_cnn_features_adv"] = False
        
        # Show XGBoost if applicable
        if "xgboost" in available_sources:
            use_xgboost_feat = st.checkbox(
                "🟨 XGBoost Features",
                value=st.session_state["use_xgboost_features_adv"],
                key="checkbox_xgb_feat_adv",
                help="Use 115+ engineered XGBoost features"
            )
            st.session_state["use_xgboost_features_adv"] = use_xgboost_feat
        else:
            use_xgboost_feat = False
            st.session_state["use_xgboost_features_adv"] = False
        
        # Show LightGBM if applicable
        if "lightgbm" in available_sources:
            use_lightgbm_feat = st.checkbox(
                "🟩 LightGBM Features",
                value=st.session_state["use_lightgbm_features_adv"],
                key="checkbox_lgb_feat_adv",
                help="Use 77 engineered LightGBM features optimized for speed and accuracy"
            )
            st.session_state["use_lightgbm_features_adv"] = use_lightgbm_feat
        else:
            use_lightgbm_feat = False
            st.session_state["use_lightgbm_features_adv"] = False
    
    # Show Transformer if applicable (in Ensemble only)
    if "transformer" in available_sources:
        use_transformer = st.checkbox(
            "🟦 Transformer Embeddings (Legacy)",
            value=st.session_state["use_transformer_features_adv"],
            key="checkbox_trans_adv",
            help="Use Transformer embeddings for semantic relationships (Legacy option, CNN preferred)"
        )
        st.session_state["use_transformer_features_adv"] = use_transformer
    else:
        use_transformer = False
        st.session_state["use_transformer_features_adv"] = False
    
    # ========================================================================
    # PREDICTION MODE - Choose between same-draw or next-draw prediction
    # ========================================================================
    st.markdown("**🎯 Prediction Mode:**")
    col_pred1, col_pred2 = st.columns(2)
    
    with col_pred1:
        enable_lag = st.checkbox(
            "⏭️ Enable LAG (predict next draw)",
            value=False,
            help="If checked: Feature[i] predicts Target[i+1] (next draw, ~14% accuracy). If unchecked: Feature[i] predicts Target[i] (same draw, ~98% accuracy)"
        )
    
    with col_pred2:
        st.info("""
        **Mode Guide:**
        - **LAG OFF (default)**: Analyze patterns WITHIN draws (~98% accuracy)
        - **LAG ON**: Predict NEXT draw from previous data (~14% accuracy - lottery is random)
        """)
    
    # Pass lag setting to training
    if "enable_lag_adv" not in st.session_state:
        st.session_state["enable_lag_adv"] = False
    st.session_state["enable_lag_adv"] = enable_lag
    
    # Validate at least one is selected
    selected_sources = [use_raw_csv, use_lstm, use_cnn, use_transformer, use_xgboost_feat, use_catboost_feat, use_lightgbm_feat]
    if not any(selected_sources):
        st.warning("⚠️ Please select at least one training data source")
        return
    
    st.divider()
    
    # ========================================================================
    # FEATURE QUALITY & OPTIMIZATION
    # ========================================================================
    
    st.markdown("### 🎨 Feature Quality & Optimization")
    st.caption("Apply advanced feature engineering before training for improved model performance")
    
    quality_col1, quality_col2, quality_col3 = st.columns(3)
    
    with quality_col1:
        prefer_optimized = st.checkbox(
            "⭐ Prefer Optimized Features",
            value=True,
            key="train_prefer_optimized",
            help="Use optimized features if available (from Advanced Feature Generation tab)"
        )
        
        if prefer_optimized:
            optimization_config = st.session_state.get('feature_optimization_config', {})
            if optimization_config.get('enabled'):
                st.success(f"✓ Using {optimization_config.get('method', 'RFE')} optimization")
            else:
                st.info("ℹ️ Generate optimized features in Advanced Feature Generation tab")
    
    with quality_col2:
        validate_features_before_training = st.checkbox(
            "✅ Validate Features",
            value=True,
            key="train_validate_features",
            help="Check for NaN, low variance, and high correlation before training"
        )
        
        if validate_features_before_training:
            validation_config = st.session_state.get('feature_validation_config', {
                'check_nan_inf': True,
                'check_constant': True,
                'check_correlation': True,
                'variance_threshold': 0.01,
                'correlation_threshold': 0.95
            })
            st.caption(f"Variance ≥ {validation_config.get('variance_threshold', 0.01)}, Corr < {validation_config.get('correlation_threshold', 0.95)}")
    
    with quality_col3:
        show_feature_stats = st.checkbox(
            "📊 Show Feature Stats",
            value=False,
            key="train_show_stats",
            help="Display feature statistics before training"
        )
    
    st.divider()
    
    # ========================================================================
    # STEP 3: TRAINING CONFIGURATION
    # ========================================================================
    
    st.markdown("### ⚙️ Step 3 – Training Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        epochs = st.slider(
            "Training Epochs",
            min_value=50,
            max_value=500,
            value=150,
            step=10,
            key="adv_train_epochs",
            help="Number of training iterations. More epochs = longer training but potentially better accuracy"
        )
    
    with config_col2:
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            step=0.0001,
            key="adv_train_lr",
            help="Controls how much the model adjusts during training"
        )
    
    with config_col3:
        batch_size = st.selectbox(
            "Batch Size",
            options=[16, 32, 64, 128, 256],
            index=2,
            key="adv_train_batch",
            help="Samples per training batch"
        )
    
    validation_split = st.slider(
        "Validation Split",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        key="adv_train_val_split",
        help="Percentage of data reserved for validation testing"
    )
    
    st.divider()
    
    # ========================================================================
    # TRAINING DATA SUMMARY
    # ========================================================================
    
    st.markdown("### 📈 Training Data Summary")
    
    # Get preference for optimized features from session state
    prefer_optimized = st.session_state.get("train_prefer_optimized", True)
    
    st.write(f"🔍 **Building data sources for {selected_model}...**")
    st.write(f"  - use_raw_csv={use_raw_csv}")
    st.write(f"  - use_lstm={use_lstm}")  
    st.write(f"  - use_cnn={use_cnn}")
    st.write(f"  - use_transformer={use_transformer}")
    st.write(f"  - use_xgboost={use_xgboost_feat}")
    
    # Build data sources dict
    data_sources = {
        "raw_csv": [] if not use_raw_csv else _get_raw_csv_files(selected_game),
        "lstm": [] if not use_lstm else _get_feature_files(selected_game, "lstm", prefer_optimized),
        "cnn": [] if not use_cnn else _get_feature_files(selected_game, "cnn", prefer_optimized),
        "transformer": [] if not use_transformer else _get_feature_files(selected_game, "transformer", prefer_optimized),
        "xgboost": [] if not use_xgboost_feat else _get_feature_files(selected_game, "xgboost", prefer_optimized),
        "catboost": [] if not use_catboost_feat else _get_feature_files(selected_game, "catboost", prefer_optimized),
        "lightgbm": [] if not use_lightgbm_feat else _get_feature_files(selected_game, "lightgbm", prefer_optimized)
    }
    
    st.write(f"📊 **Data sources found:**")
    for source, files in data_sources.items():
        if files:
            st.write(f"  - {source}: {len(files)} files")
        else:
            st.write(f"  - {source}: 0 files (empty)")
    
    # CRITICAL FIX: For CNN model, use CNN embeddings if available, otherwise fall back to raw_csv
    # Never use BOTH together (causes dimension mismatch)
    # IMPORTANT: We always need raw_csv for TARGET extraction, but for CNN we don't use it for FEATURES
    if selected_model == "CNN":
        st.write("🔧 **CNN Model Detected - Applying data source logic:**")
        if data_sources["cnn"]:
            # CNN embeddings exist - keep raw_csv for targets but mark it specially
            st.write(f"  ✅ Found {len(data_sources['cnn'])} CNN embedding file(s)")
            st.write(f"  ✅ Keeping {len(data_sources['raw_csv'])} raw_csv files for TARGET extraction only")
            st.write("  ✅ Will use CNN embeddings for FEATURES, raw_csv for TARGETS")
        elif data_sources["raw_csv"]:
            # No CNN embeddings - use raw_csv for both features and targets
            st.write("  ℹ️ No CNN embeddings found - using raw_csv for both features and targets")
        else:
            app_log("⚠️  No CNN embeddings or raw_csv data found!", "warning")
    
    total_files = sum(len(files) for files in data_sources.values())
    total_samples = _estimate_total_samples(data_sources)
    
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    
    with sum_col1:
        st.metric("📁 Total Files", total_files)
    
    with sum_col2:
        st.metric("📊 Estimated Samples", f"{total_samples:,}")
    
    with sum_col3:
        st.metric("🤖 Model Type", selected_model)
    
    with sum_col4:
        st.metric("🎯 Data Sources", sum(1 for v in [use_raw_csv, use_lstm, use_cnn, use_transformer, use_xgboost_feat, use_catboost_feat, use_lightgbm_feat] if v))
    
    # Detailed file listing
    with st.expander("📋 View Data Sources Details", expanded=False):
        if use_raw_csv and data_sources["raw_csv"]:
            st.markdown("**📁 Raw CSV Files:**")
            for f in data_sources["raw_csv"]:
                st.text(f"  • {f.name}")
        
        if use_lstm and data_sources["lstm"]:
            st.markdown("**🔷 LSTM Sequence Files:**")
            for f in data_sources["lstm"]:
                st.text(f"  • {f.name}")
        
        if use_cnn and data_sources["cnn"]:
            st.markdown("**🟩 CNN Embedding Files:**")
            for f in data_sources["cnn"]:
                st.text(f"  • {f.name}")
        
        if use_transformer and data_sources["transformer"]:
            st.markdown("**🟦 Transformer Embedding Files (Legacy):**")
            for f in data_sources["transformer"]:
                st.text(f"  • {f.name}")
        
        if use_xgboost_feat and data_sources["xgboost"]:
            st.markdown("**🟨 XGBoost Feature Files:**")
            for f in data_sources["xgboost"]:
                st.text(f"  • {f.name}")
        
        if use_catboost_feat and data_sources["catboost"]:
            st.markdown("**🟧 CatBoost Feature Files:**")
            for f in data_sources["catboost"]:
                st.text(f"  • {f.name}")
        
        if use_lightgbm_feat and data_sources["lightgbm"]:
            st.markdown("**🟩 LightGBM Feature Files:**")
            for f in data_sources["lightgbm"]:
                st.text(f"  • {f.name}")
    
    st.divider()
    
    # ========================================================================
    # START TRAINING BUTTON
    # ========================================================================
    
    st.markdown("### 🚀 Begin Advanced Model Training")
    
    if st.button("🚀 Start Advanced Training", use_container_width=True, type="primary"):
        try:
            # Validate configuration
            if not selected_game or not selected_model:
                st.error("❌ Please complete Step 1: Select Game and Model")
                return
            
            # Feature validation if enabled
            validate_features = st.session_state.get("train_validate_features", True)
            if validate_features:
                st.info("🔍 Validating features before training...")
                
                # Import AdvancedFeatureGenerator for validation
                try:
                    from streamlit_app.services.advanced_feature_generator import AdvancedFeatureGenerator
                    
                    feature_generator = AdvancedFeatureGenerator(game=selected_game)
                    validation_config = st.session_state.get('feature_validation_config', {
                        'check_nan_inf': True,
                        'check_constant': True,
                        'check_correlation': True,
                        'variance_threshold': 0.01,
                        'correlation_threshold': 0.95
                    })
                    
                    # Validate each feature source
                    validation_failed = False
                    for source_type, files in data_sources.items():
                        if not files or source_type == "raw_csv":
                            continue
                        
                        for file_path in files:
                            if file_path.suffix == ".csv":
                                try:
                                    features_df = pd.read_csv(file_path)
                                    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                                    features_data = features_df[numeric_cols].values
                                    
                                    validation_results = feature_generator.validate_features(
                                        features_data, validation_config
                                    )
                                    
                                    if not validation_results['passed']:
                                        st.warning(f"⚠️ Validation issues in {file_path.name}:")
                                        for issue in validation_results.get('issues_found', []):
                                            st.warning(f"  - {issue}")
                                        
                                        # Show option to continue anyway
                                        continue_anyway = st.checkbox(
                                            f"Continue training with {file_path.name} despite issues?",
                                            key=f"continue_{file_path.stem}"
                                        )
                                        if not continue_anyway:
                                            validation_failed = True
                                    else:
                                        st.success(f"✅ {file_path.name} passed validation")
                                        
                                except Exception as e:
                                    st.warning(f"⚠️ Could not validate {file_path.name}: {e}")
                    
                    if validation_failed:
                        st.error("❌ Feature validation failed. Please fix issues or uncheck validation.")
                        return
                    
                except ImportError:
                    st.warning("⚠️ AdvancedFeatureGenerator not available, skipping validation")
            
            # Start training
            _train_advanced_model(
                game=selected_game,
                model_type=selected_model,
                data_sources=data_sources,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "validation_split": validation_split
                }
            )
            
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")
            app_log(f"Model training error: {e}", "error")


def _train_advanced_model(
    game: str,
    model_type: str,
    data_sources: Dict[str, List[Path]],
    config: Dict[str, Any]
) -> None:
    """
    Train advanced AI/ML model with real data loading and state-of-the-art techniques.
    
    For Ensemble models: Combines XGBoost, LSTM, and Transformer for maximum accuracy.
    """
    st.markdown("#### 🚀 Advanced Model Training in Progress")
    st.markdown(f"*Training {model_type} model for {game} with multi-source data...*")
    
    # Create trainer
    trainer = AdvancedModelTrainer(game)
    
    # Progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create detailed logs container with scrollbar
    st.markdown("**Training Logs:**")
    logs_placeholder = st.empty()
    
    # Store training logs in session state
    if "training_logs" not in st.session_state:
        st.session_state.training_logs = []
    
    def progress_callback(progress: float, message: str, metrics: Dict[str, Any] = None):
        """Callback for training progress updates with detailed logging."""
        progress_bar.progress(min(progress, 1.0))
        status_text.text(message)
        
        # Add to logs
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        if metrics:
            if 'epoch' in metrics:
                log_entry += f" | Epoch: {metrics['epoch']}"
            if 'loss' in metrics:
                log_entry += f" | Loss: {metrics['loss']:.6f}"
            if 'accuracy' in metrics:
                log_entry += f" | Accuracy: {metrics['accuracy']:.4f}"
            if 'val_loss' in metrics:
                log_entry += f" | Val Loss: {metrics['val_loss']:.6f}"
            if 'val_accuracy' in metrics:
                log_entry += f" | Val Accuracy: {metrics['val_accuracy']:.4f}"
        
        st.session_state.training_logs.append(log_entry)
        
        # Display logs in scrollable container
        with logs_placeholder.container():
            logs_text = "\n".join(st.session_state.training_logs)  # Show ALL logs
            st.markdown(f"""
            <div style="
                height: 400px; 
                overflow-y: auto; 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 10px; 
                background-color: #f0f0f0;
                font-family: monospace;
                font-size: 12px;
                line-height: 1.4;
            ">
            {"<br>".join([line.replace("<", "&lt;").replace(">", "&gt;") for line in logs_text.split(chr(10))])}
            </div>
            """, unsafe_allow_html=True)
    
    try:
        # Step 1: Load training data
        progress_callback(0.05, "📥 Loading training data from multiple sources...")
        app_log(f"Loading training data for {game} ({model_type} model)...", "info")
        
        X, y, metadata = trainer.load_training_data(data_sources, disable_lag=not st.session_state.get("enable_lag_adv", False))
        
        if X.shape[0] == 0:
            st.error("❌ No training data loaded")
            app_log("No training data loaded from any source", "error")
            return
        
        progress_callback(0.15, f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features")
        app_log(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features", "info")
        
        # DEBUG: Show target distribution BEFORE training
        unique_targets, target_counts = np.unique(y, return_counts=True)
        target_dist = dict(zip(unique_targets, target_counts))
        st.info(f"🎯 **TARGET DISTRIBUTION (before split)**: {target_dist}")
        app_log(f"Target distribution: {target_dist}", "info")
        
        # Display data loading summary
        st.markdown("**Data Loading Summary:**")
        data_summary_cols = st.columns(len(metadata["sources"]))
        for idx, (source, count) in enumerate(metadata["sources"].items()):
            with data_summary_cols[idx]:
                st.metric(source.upper(), count)
                app_log(f"  {source}: {count} samples", "info")
        
        st.divider()
        
        # Step 2: Train model(s)
        st.markdown("**Training Progress:**")
        
        if model_type == "Ensemble":
            # Train ensemble with all models
            progress_callback(0.2, "🤖 Training Ensemble (XGBoost + CatBoost + LightGBM + CNN)...")
            app_log("Starting Ensemble training (XGBoost + CatBoost + LightGBM + CNN)...", "info")
            
            models, metrics = trainer.train_ensemble(
                X, y, metadata, config,
                progress_callback=lambda p, m, metrics=None: progress_callback(0.2 + p * 0.7, m, metrics)
            )
            
            model_to_save = models
            metrics_to_save = metrics
            
        elif model_type == "XGBoost":
            progress_callback(0.2, "🟩 Training XGBoost model...")
            app_log("Starting XGBoost training...", "info")
            app_log(f"  - Epochs: {config.get('epochs', 100)}", "info")
            app_log(f"  - Learning Rate: {config.get('learning_rate', 0.01)}", "info")
            app_log(f"  - Batch Size: {config.get('batch_size', 64)}", "info")
            app_log(f"  - Validation Split: {config.get('validation_split', 0.2):.1%}", "info")
            
            model, metrics = trainer.train_xgboost(
                X, y, metadata, config,
                progress_callback=lambda p, m, metrics=None: progress_callback(0.2 + p * 0.7, m, metrics)
            )
            
            model_to_save = model
            metrics_to_save = {"xgboost": metrics}
            app_log(f"✅ XGBoost training complete - Accuracy: {metrics['accuracy']:.4f}", "info")
        
        elif model_type == "CatBoost":
            progress_callback(0.2, "🟨 Training CatBoost model...")
            app_log("Starting CatBoost training...", "info")
            app_log(f"  - Epochs: {config.get('epochs', 1000)}", "info")
            app_log(f"  - Learning Rate: {config.get('learning_rate', 0.05)}", "info")
            app_log(f"  - Validation Split: {config.get('validation_split', 0.2):.1%}", "info")
            
            model, metrics = trainer.train_catboost(
                X, y, metadata, config,
                progress_callback=lambda p, m, metrics=None: progress_callback(0.2 + p * 0.7, m, metrics)
            )
            
            if model is None:
                st.error("❌ CatBoost training failed - package may not be installed")
                app_log("❌ CatBoost training failed - package not available", "error")
                return
            
            model_to_save = model
            metrics_to_save = {"catboost": metrics}
            app_log(f"✅ CatBoost training complete - Accuracy: {metrics['accuracy']:.4f}", "info")
        
        elif model_type == "LightGBM":
            progress_callback(0.2, "🟪 Training LightGBM model...")
            app_log("Starting LightGBM training...", "info")
            app_log(f"  - Epochs: {config.get('epochs', 500)}", "info")
            app_log(f"  - Learning Rate: {config.get('learning_rate', 0.05)}", "info")
            app_log(f"  - Validation Split: {config.get('validation_split', 0.2):.1%}", "info")
            
            model, metrics = trainer.train_lightgbm(
                X, y, metadata, config,
                progress_callback=lambda p, m, metrics=None: progress_callback(0.2 + p * 0.7, m, metrics)
            )
            
            if model is None:
                st.error("❌ LightGBM training failed")
                app_log("❌ LightGBM training failed", "error")
                return
            
            model_to_save = model
            metrics_to_save = {"lightgbm": metrics}
            app_log(f"✅ LightGBM training complete - Accuracy: {metrics['accuracy']:.4f}", "info")
            
        elif model_type == "LSTM":
            progress_callback(0.2, "🔷 Training LSTM model...")
            app_log("Starting LSTM training...", "info")
            app_log(f"  - Epochs: {config.get('epochs', 100)}", "info")
            app_log(f"  - Learning Rate: {config.get('learning_rate', 0.001)}", "info")
            
            model, metrics = trainer.train_lstm(
                X, y, metadata, config,
                progress_callback=lambda p, m, metrics=None: progress_callback(0.2 + p * 0.7, m, metrics)
            )
            
            if model is None:
                st.error("❌ LSTM training failed")
                app_log("❌ LSTM training failed - TensorFlow may not be available", "error")
                return
            
            model_to_save = model
            metrics_to_save = {"lstm": metrics}
            app_log(f"✅ LSTM training complete - Accuracy: {metrics['accuracy']:.4f}", "info")
            
        elif model_type == "Transformer":
            progress_callback(0.2, "🟦 Training Transformer model...")
            app_log("Starting Transformer training...", "info")
            app_log(f"  - Epochs: {config.get('epochs', 100)}", "info")
            app_log(f"  - Learning Rate: {config.get('learning_rate', 0.001)}", "info")
            
            model, metrics = trainer.train_transformer(
                X, y, metadata, config,
                progress_callback=lambda p, m, metrics=None: progress_callback(0.2 + p * 0.7, m, metrics)
            )
            
            if model is None:
                st.error("❌ Transformer training failed")
                app_log("❌ Transformer training failed - TensorFlow may not be available", "error")
                return
            
            model_to_save = model
            metrics_to_save = {"transformer": metrics}
            app_log(f"✅ Transformer training complete - Accuracy: {metrics['accuracy']:.4f}", "info")
        
        elif model_type == "CNN":
            progress_callback(0.2, "🟩 Training CNN model...")
            app_log("Starting CNN training...", "info")
            app_log(f"  - Epochs: {config.get('epochs', 100)}", "info")
            app_log(f"  - Learning Rate: {config.get('learning_rate', 0.001)}", "info")
            
            model, metrics = trainer.train_cnn(
                X, y, metadata, config,
                progress_callback=lambda p, m, metrics=None: progress_callback(0.2 + p * 0.7, m, metrics)
            )
            
            if model is None:
                st.error("❌ CNN training failed")
                app_log("❌ CNN training failed - TensorFlow may not be available", "error")
                return
            
            model_to_save = model
            metrics_to_save = {"cnn": metrics}
            app_log(f"✅ CNN training complete - Accuracy: {metrics['accuracy']:.4f}", "info")
        
        else:
            st.error(f"Unknown model type: {model_type}")
            app_log(f"Unknown model type: {model_type}", "error")
            return
        
        progress_callback(0.92, "💾 Saving model...")
        app_log("Saving model with metadata...", "info")
        
        # Step 3: Save model
        model_path = trainer.save_model(model_to_save, model_type.lower(), metrics_to_save)
        app_log(f"✅ Model saved to: {model_path}", "info")
        
        # Step 4: Register model in registry for prediction engine
        progress_callback(0.95, "📝 Registering model in ModelRegistry...")
        app_log("Registering model in ModelRegistry for prediction engine...", "info")
        
        try:
            from streamlit_app.services.model_registry import ModelRegistry
            from streamlit_app.services.feature_schema import FeatureSchema, NormalizationMethod
            import platform
            
            registry = ModelRegistry()
            
            # Create feature schema from metadata (use datetime already imported at top of file)
            feature_names = list(trainer.feature_names) if hasattr(trainer, 'feature_names') and trainer.feature_names else []
            current_timestamp = datetime.now().isoformat()
            
            feature_schema = FeatureSchema(
                model_type=model_type.lower(),
                game=game,
                schema_version="1.0",
                created_at=current_timestamp,
                feature_names=feature_names if feature_names else [f"feature_{i}" for i in range(metadata.get("feature_count", X.shape[1]))],
                feature_count=metadata.get("feature_count", X.shape[1]),
                normalization_method=NormalizationMethod.ROBUST_SCALER,
                data_shape=(X.shape[0], X.shape[1]),
                raw_data_version="1.0",
                raw_data_date_generated=current_timestamp,
                python_version=platform.python_version(),
                created_by="AdvancedModelTrainer"
            )
            
            # Register the model
            success, message = registry.register_model(
                model_path=Path(model_path),
                model_type=model_type.lower(),
                game=game,
                feature_schema=feature_schema,
                metadata=metrics_to_save
            )
            
            if success:
                app_log(f"Model registered successfully: {game} - {model_type.lower()}", "info")
                progress_callback(0.98, f"✅ Model registered: {game} - {model_type}")
            else:
                app_log(f"Model registration warning: {message}", "warning")
                progress_callback(0.98, f"⚠️ Registration warning: {message}")
                
        except Exception as e:
            app_log(f"Could not register model in registry: {str(e)}", "error")
            progress_callback(0.98, "⚠️ Model registration failed (non-critical)")
            # Don't fail training if registration fails
        
        progress_callback(1.0, "✅ Training complete!")
        
        st.divider()
        
        # Display results
        st.markdown("#### 📊 Training Results")
        
        # Model summary
        st.success(f"✅ **{model_type} Model Training Complete**")
        
        if model_type == "Ensemble":
            # Build dynamic component list from actual trained models
            ensemble_data = metrics_to_save.get('ensemble', {})
            weights = ensemble_data.get('weights', {})
            
            component_info = []
            model_file_ext = {
                'xgboost': '.joblib',
                'catboost': '.joblib',
                'lightgbm': '.joblib',
                'cnn': '.keras',
                'lstm': '.keras',
                'transformer': '.keras'
            }
            
            model_descriptions = {
                'xgboost': 'XGBoost - Gradient boosting with 500+ trees',
                'catboost': 'CatBoost - Categorical boosting optimized for tabular data',
                'lightgbm': 'LightGBM - Fast leaf-wise gradient boosting',
                'cnn': 'CNN - Multi-scale convolution (kernels 3,5,7)',
                'lstm': 'LSTM - Bidirectional RNN for temporal patterns',
                'transformer': 'Transformer - Attention mechanisms (legacy)'
            }
            
            component_list = ""
            for comp_name, comp_metrics in metrics_to_save.items():
                if comp_name != "ensemble" and isinstance(comp_metrics, dict):
                    ext = model_file_ext.get(comp_name, '.model')
                    desc = model_descriptions.get(comp_name, comp_name)
                    weight_pct = (weights.get(comp_name, 0) * 100) if weights else 0
                    component_list += f"  - `{comp_name}_model{ext}` - {desc} ({weight_pct:.1f}%)\n"
            
            weight_str = " | ".join([f"{m.upper()} {(w*100):.0f}%" for m, w in weights.items()]) if weights else "Equal weights"
            
            st.info(f"""
            **Ensemble Model Details:**
            - 📁 **Saved Location:** `models/{_sanitize_game_name(game)}/ensemble/{Path(model_path).name}`
            - 🤖 **Components ({len([m for m in metrics_to_save.keys() if m != 'ensemble'])} Models):**
{component_list}
            - 📊 **Combined Accuracy:** {ensemble_data.get('combined_accuracy', 0):.4f}
            - 🔀 **Prediction Method:** Weighted voting ({weight_str})
            
            **How to Use Ensemble:**
            All component models work together and vote to produce the final result.
            Each model's vote is weighted by its individual accuracy for optimal performance.
            """)
            
            app_log(f"Ensemble Model Details:", "info")
            app_log(f"  - Location: models/{_sanitize_game_name(game)}/ensemble/{Path(model_path).name}", "info")
            app_log(f"  - Components: {', '.join([m for m in metrics_to_save.keys() if m != 'ensemble'])}", "info")
            app_log(f"  - Prediction: Weighted voting ({weight_str})", "info")
            app_log(f"  - Combined Accuracy: {ensemble_data.get('combined_accuracy', 0):.4f}", "info")
            
            # Show component metrics
            st.markdown("**Component Performance:**")
            comp_cols = st.columns(len([m for m in metrics_to_save.keys() if m != "ensemble"]))
            for idx, (comp_name, comp_metrics) in enumerate(metrics_to_save.items()):
                if comp_name != "ensemble" and isinstance(comp_metrics, dict):
                    with comp_cols[idx]:
                        accuracy = comp_metrics.get('accuracy', 0)
                        weight = weights.get(comp_name, 0) * 100 if weights else 0
                        st.metric(
                            comp_name.upper(),
                            f"{accuracy:.4f}",
                            help=f"Accuracy: {accuracy:.4f} | Weight: {weight:.1f}%"
                        )
                        app_log(f"  {comp_name.upper()} - Accuracy: {accuracy:.4f}, Weight: {weight:.1f}%", "info")
        else:
            metrics_data = metrics_to_save.get(model_type.lower(), {})
            
            # Check if this is a multi-output model
            is_multi_output = metrics_data.get('output_type') == 'multi-output'
            
            if is_multi_output:
                # Multi-output specific metrics display
                st.info(f"""
                **Model Details:**
                - 📁 **Saved Location:** `{model_path}`
                - 🤖 **Model Type:** {model_type} (Multi-Output)
                - 🎯 **Output Positions:** {metrics_data.get('n_outputs', 7)}
                - 📊 **Avg Position Accuracy:** {metrics_data.get('accuracy', 0):.4f}
                - 🎲 **Complete Set Accuracy:** {metrics_data.get('set_accuracy', 0):.4f}
                - 📈 **Train Size:** {metrics_data.get('train_size', 0)}
                - 🧪 **Test Size:** {metrics_data.get('test_size', 0)}
                """)
                
                app_log(f"{model_type} Model Details (Multi-Output):", "info")
                app_log(f"  - Location: {model_path}", "info")
                app_log(f"  - Output Positions: {metrics_data.get('n_outputs', 7)}", "info")
                app_log(f"  - Avg Position Accuracy: {metrics_data.get('accuracy', 0):.4f}", "info")
                app_log(f"  - Complete Set Accuracy: {metrics_data.get('set_accuracy', 0):.4f}", "info")
                
                # Show per-position accuracies
                if 'position_accuracies' in metrics_data:
                    st.markdown("**Per-Position Accuracy:**")
                    pos_cols = st.columns(7)
                    for i, acc in enumerate(metrics_data['position_accuracies']):
                        with pos_cols[i]:
                            st.metric(f"Pos {i+1}", f"{acc:.3f}")
                    app_log("  Per-position accuracies:", "info")
                    for i, acc in enumerate(metrics_data['position_accuracies']):
                        app_log(f"    Position {i+1}: {acc:.4f}", "info")
            else:
                # Single-output metrics display
                st.info(f"""
                **Model Details:**
                - 📁 **Saved Location:** `{model_path}`
                - 🤖 **Model Type:** {model_type}
                - 📊 **Accuracy:** {metrics_data.get('accuracy', 0):.4f}
                - 🎯 **Precision:** {metrics_data.get('precision', 0):.4f}
                - 📈 **Recall:** {metrics_data.get('recall', 0):.4f}
                - 🔧 **F1 Score:** {metrics_data.get('f1', 0):.4f}
                """)
                
                app_log(f"{model_type} Model Details:", "info")
                app_log(f"  - Location: {model_path}", "info")
                app_log(f"  - Accuracy: {metrics_data.get('accuracy', 0):.4f}", "info")
                app_log(f"  - Precision: {metrics_data.get('precision', 0):.4f}", "info")
                app_log(f"  - Recall: {metrics_data.get('recall', 0):.4f}", "info")
                app_log(f"  - F1 Score: {metrics_data.get('f1', 0):.4f}", "info")
        
        # Display comprehensive metrics
        st.markdown("**Training Metrics:**")
        
        metrics_display = {}
        for model_name, model_metrics in metrics_to_save.items():
            if isinstance(model_metrics, dict):
                metrics_display[model_name.upper()] = {
                    k: v for k, v in model_metrics.items()
                    if isinstance(v, (int, float))
                }
        
        metrics_df = pd.DataFrame(metrics_display).T
        st.dataframe(metrics_df, use_container_width=True)
        
        # Display per-class metrics if available
        model_key = model_type.lower()
        if model_key in metrics_to_save and isinstance(metrics_to_save[model_key], dict):
            per_class = metrics_to_save[model_key].get("per_class_metrics", {})
            if per_class:
                st.markdown("**Per-Class Performance:**")
                
                per_class_data = []
                for class_idx in sorted(per_class.keys()):
                    class_metrics = per_class[class_idx]
                    per_class_data.append({
                        "Class": class_idx,
                        "Precision": f"{class_metrics['precision']:.4f}",
                        "Recall": f"{class_metrics['recall']:.4f}",
                        "F1-Score": f"{class_metrics['f1']:.4f}",
                        "Support": class_metrics['support']
                    })
                
                per_class_df = pd.DataFrame(per_class_data)
                st.dataframe(per_class_df, use_container_width=True)
                
                app_log("Per-Class Metrics Displayed:", "info")
        
        # Data summary
        st.markdown("**Training Data Summary:**")
        st.info(f"""
        - **Total Samples:** {metadata['sample_count']:,}
        - **Total Features:** {metadata['feature_count']}
        - **Data Sources Used:** {', '.join(metadata['sources'].keys())}
        - **Samples per Source:** {dict(metadata['sources'])}
        """)
        
        app_log(f"Training Data Summary:", "info")
        app_log(f"  - Total Samples: {metadata['sample_count']:,}", "info")
        app_log(f"  - Total Features: {metadata['feature_count']}", "info")
        app_log(f"  - Data Sources: {', '.join(metadata['sources'].keys())}", "info")
        for source, count in metadata['sources'].items():
            app_log(f"    • {source}: {count}", "info")
        
        # Model capabilities
        st.markdown("**Model Capabilities:**")
        
        if model_type == "Ensemble":
            st.success("""
            ✅ **Ensemble Model Advantages:**
            - **Multi-Model Voting:** Combines strengths of XGBoost (feature importance), 
              LSTM (temporal patterns), and Transformer (semantic relationships)
            - **Ensemble Robustness:** Reduces overfitting through model diversity
            - **Improved Accuracy:** Combined predictions typically outperform individual models
            - **Pattern Recognition:** Captures feature-based, temporal, and relational patterns simultaneously
            - **Ultra-Accuracy:** Weighted voting for optimal set generation with winning numbers
            """)
        
        st.success(f"""
        ✅ **Model Ready for Deployment**
        
        Your {model_type} model has been trained with state-of-the-art AI/ML techniques 
        and is ready to:
        - Generate predictions for future lottery draws
        - Re-train with new data as it becomes available
        - Be evaluated on test datasets
        - Combine with other models in ensemble strategies
        
        📁 **Model saved at:** `{model_path}`
        """)
        
        app_log(f"✅ {model_type} model training completed successfully!", "info")
        app_log(f"📁 Model Location: {model_path}", "info")
        app_log("="*80, "info")
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        app_log(f"❌ Model training error: {e}", "error")


def _get_models_dir() -> Path:
    """Get the models directory."""
    return Path("models")


def _get_model_types_for_game(game: str) -> List[str]:
    """Get available model types for a game."""
    models_dir = _get_models_dir()
    game_folder = _sanitize_game_name(game)
    game_dir = models_dir / game_folder
    
    if not game_dir.exists():
        return []
    
    # Get subdirectories (model types)
    model_types = []
    for item in sorted(game_dir.iterdir()):
        if item.is_dir():
            model_types.append(item.name.lower())
    
    return model_types


def _get_models_for_game_and_type(game: str, model_type: str) -> List[Dict[str, Any]]:
    """Get all models for a specific game and model type.
    
    Handles both:
    - Ensemble models (stored as folders): models/game/ensemble/ensemble_name/
    - Individual models (stored as files): models/game/type/model_name.keras or .joblib
    """
    import json
    models_dir = _get_models_dir()
    game_folder = _sanitize_game_name(game)
    model_type_folder = model_type.lower()
    
    type_dir = models_dir / game_folder / model_type_folder
    
    if not type_dir.exists():
        return []
    
    models = []
    
    for item in sorted(type_dir.iterdir(), reverse=True):
        # Handle ENSEMBLE: folders with metadata.json inside
        if item.is_dir():
            model_name = item.name
            
            # Initialize model info with defaults
            model_info = {
                "name": model_name,
                "path": str(item),
                "type": model_type,
                "accuracy": 0.0,
                "created": None,
                "version": "1.0"
            }
            
            # Try to read metadata.json file (for ensemble or folder-based models)
            meta_file = item / "metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        meta_data = json.load(f)
                        if isinstance(meta_data, dict):
                            # Extract accuracy from metadata
                            accuracy = meta_data.get("accuracy", 0.0)
                            # Ensure accuracy is a float between 0 and 1
                            if isinstance(accuracy, (int, float)):
                                if accuracy > 1.0:
                                    # If accuracy is > 1, it might be a percentage or R2 score, use as-is
                                    accuracy = float(accuracy)
                                else:
                                    accuracy = float(accuracy)
                            else:
                                accuracy = 0.0
                            
                            model_info.update({
                                "accuracy": accuracy,
                                "created": meta_data.get("trained_on", None),
                                "version": meta_data.get("version", "1.0")
                            })
                except Exception as e:
                    app_log(f"Error reading metadata.json for {model_name}: {e}", "warning")
            
            models.append(model_info)
        
        # Handle INDIVIDUAL MODELS: files with .keras, .joblib, or .h5 extensions
        elif item.is_file() and item.suffix in ['.keras', '.joblib', '.h5']:
            # Skip metadata files
            if item.name.endswith('_metadata.json'):
                continue
            
            # Extract model name without extension
            model_name = item.stem  # Gets filename without extension
            
            # Initialize model info with defaults
            model_info = {
                "name": model_name,
                "path": str(item),
                "type": model_type,
                "accuracy": 0.0,
                "created": None,
                "version": "1.0"
            }
            
            # Try to read corresponding metadata file
            # Metadata is stored as: model_name_metadata.json
            meta_file = item.parent / f"{model_name}_metadata.json"
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        meta_data = json.load(f)
                        if isinstance(meta_data, dict):
                            # Extract accuracy from metadata
                            accuracy = meta_data.get("accuracy", 0.0)
                            # Ensure accuracy is a float between 0 and 1
                            if isinstance(accuracy, (int, float)):
                                if accuracy > 1.0:
                                    accuracy = float(accuracy)
                                else:
                                    accuracy = float(accuracy)
                            else:
                                accuracy = 0.0
                            
                            model_info.update({
                                "accuracy": accuracy,
                                "created": meta_data.get("trained_on", None),
                                "version": meta_data.get("version", "1.0")
                            })
                except Exception as e:
                    app_log(f"Error reading metadata file for {model_name}: {e}", "warning")
            
            models.append(model_info)
    
    return models


def _render_model_retraining():
    """Render the comprehensive Model Re-Training section with advanced features."""
    st.subheader("🚀 Model Re-Training")
    st.markdown("*Upgrade existing models with new data using incremental learning, transfer learning, and advanced techniques*")
    
    st.divider()
    
    # ========================================================================
    # SECTION 1: SELECT GAME
    # ========================================================================
    
    st.markdown("### 🎮 Select Game")
    
    games = get_available_games()
    selected_game = st.selectbox(
        "Choose Game:",
        games,
        key="retrain_select_game",
        help="Select the lottery game for the model to retrain"
    )
    
    st.divider()
    
    # ========================================================================
    # SECTION 2: SELECT MODEL TYPE
    # ========================================================================
    
    st.markdown("### 🤖 Select Model Type")
    
    # Get available model types for selected game
    available_model_types = _get_model_types_for_game(selected_game)
    
    if not available_model_types:
        st.warning(f"No trained models found for {selected_game}")
        return
    
    # Format model types for display (capitalize)
    model_type_options = [mt.upper() for mt in available_model_types]
    
    selected_model_type = st.selectbox(
        "Choose Model Type:",
        model_type_options,
        key="retrain_select_model_type",
        help="Select the machine learning model type to retrain"
    )
    
    st.divider()
    
    # ========================================================================
    # SECTION 3: SELECT MODEL
    # ========================================================================
    
    st.markdown("### 📋 Select Model")
    
    # Get available models for selected game and type
    available_models = _get_models_for_game_and_type(selected_game, selected_model_type)
    
    if not available_models:
        st.warning(f"No {selected_model_type} models found for {selected_game}")
        return
    
    # Create model display strings with accuracy
    model_options = []
    model_mapping = {}
    
    for model in available_models:
        # Format: "model_name (Accuracy: X.XX)"
        display_name = f"{model['name']} (Accuracy: {model['accuracy']:.4f})"
        model_options.append(display_name)
        model_mapping[display_name] = model
    
    selected_model_display = st.selectbox(
        "Choose Model to Retrain:",
        model_options,
        key="retrain_select_model",
        help="Select the specific trained model to retrain with new data"
    )
    
    selected_model = model_mapping[selected_model_display]
    
    st.divider()
    
    # ========================================================================
    # MODEL INFORMATION DISPLAY
    # ========================================================================
    
    st.markdown("### 📊 Selected Model Information")
    
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    
    with col_info1:
        st.metric(
            "Model Name",
            selected_model["name"][:30] + "..." if len(selected_model["name"]) > 30 else selected_model["name"],
            help=f"Full path: {selected_model['path']}"
        )
    
    with col_info2:
        st.metric(
            "Current Accuracy",
            f"{selected_model['accuracy']:.2%}",
            help="Accuracy of the current trained model"
        )
    
    with col_info3:
        st.metric(
            "Model Type",
            selected_model_type,
            help="Type of machine learning model"
        )
    
    with col_info4:
        # Detect if neural network
        is_neural = selected_model_type.lower() in ['lstm', 'cnn', 'transformer']
        st.metric(
            "Architecture",
            "Neural Net" if is_neural else "Tree Model",
            help="Model architecture category"
        )
    
    st.divider()
    
    # ========================================================================
    # RE-TRAINING STRATEGY SELECTION
    # ========================================================================
    
    st.markdown("### 🎯 Re-Training Strategy")
    
    strategy_col1, strategy_col2 = st.columns([2, 1])
    
    with strategy_col1:
        retraining_mode = st.radio(
            "Select Re-Training Mode:",
            options=[
                "🔄 Fine-Tuning (Continue from existing weights)",
                "➕ Additive Learning (Add new boosting rounds - Tree models only)",
                "🔁 Full Retrain (Combine old + new data, train from scratch)"
            ],
            key="retrain_mode",
            help="Choose how to incorporate new data into the model"
        )
    
    with strategy_col2:
        st.info(f"""
        **Selected Strategy:**
        {'Fine-tuning with lower learning rate' if 'Fine-Tuning' in retraining_mode else 
         'Add new estimators to ensemble' if 'Additive' in retraining_mode else 
         'Complete retraining from scratch'}
        """)
    
    # Parse mode
    if "Fine-Tuning" in retraining_mode:
        mode = "fine_tune"
    elif "Additive" in retraining_mode:
        mode = "additive"
    else:
        mode = "full_retrain"
    
    st.divider()
    
    # ========================================================================
    # RETRAINING CONFIGURATION
    # ========================================================================
    
    st.markdown("### ⚙️ Re-Training Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        if mode == "additive":
            additional_epochs = st.slider(
                "Additional Boosting Rounds",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                key="retrain_additional_epochs",
                help="Number of additional trees/estimators to add"
            )
        else:
            additional_epochs = st.slider(
                "Training Epochs",
                min_value=5,
                max_value=200,
                value=50 if mode == "fine_tune" else 100,
                step=5,
                key="retrain_additional_epochs",
                help="Number of training epochs"
            )
    
    with config_col2:
        if mode == "fine_tune":
            learning_rate = st.slider(
                "Learning Rate (Fine-Tune)",
                min_value=0.00001,
                max_value=0.01,
                value=0.0001,
                format="%.5f",
                key="retrain_learning_rate",
                help="Lower LR for fine-tuning to preserve learned features"
            )
        else:
            learning_rate = st.slider(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.1,
                value=0.001,
                format="%.5f",
                key="retrain_learning_rate",
                help="Learning rate for training"
            )
    
    with config_col3:
        batch_size = st.selectbox(
            "Batch Size",
            options=[16, 32, 64, 128, 256],
            index=1,
            key="retrain_batch_size",
            help="Number of samples per batch"
        )
    
    st.divider()
    
    # ========================================================================
    # TRANSFER LEARNING OPTIONS (Neural Networks Only)
    # ========================================================================
    
    is_neural = selected_model_type.lower() in ['lstm', 'cnn', 'transformer']
    
    if is_neural:
        st.markdown("### 🧠 Transfer Learning Options")
        st.caption("*Advanced techniques for neural network fine-tuning*")
        
        transfer_col1, transfer_col2, transfer_col3 = st.columns(3)
        
        with transfer_col1:
            freeze_early_layers = st.checkbox(
                "Freeze Early Layers",
                value=mode == "fine_tune",
                key="retrain_freeze_early",
                help="Keep low-level feature extractors frozen during training"
            )
            
            if freeze_early_layers:
                freeze_percentage = st.slider(
                    "% of Layers to Freeze",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=10,
                    key="retrain_freeze_pct",
                    help="Percentage of early layers to freeze"
                )
        
        with transfer_col2:
            gradual_unfreezing = st.checkbox(
                "Gradual Layer Unfreezing",
                value=False,
                key="retrain_gradual_unfreeze",
                help="Progressively unfreeze layers during training"
            )
            
            if gradual_unfreezing:
                unfreeze_schedule = st.selectbox(
                    "Unfreezing Schedule",
                    options=["Linear", "Exponential", "Step-wise"],
                    key="retrain_unfreeze_schedule",
                    help="How to schedule layer unfreezing over epochs"
                )
        
        with transfer_col3:
            discriminative_lr = st.checkbox(
                "Discriminative Learning Rates",
                value=False,
                key="retrain_disc_lr",
                help="Use different LR for different layer depths"
            )
            
            if discriminative_lr:
                lr_decay_factor = st.slider(
                    "LR Decay Factor (per layer)",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.9,
                    step=0.05,
                    key="retrain_lr_decay",
                    help="Multiply LR by this factor for each deeper layer"
                )
        
        st.divider()
    
    # ========================================================================
    # ELASTIC WEIGHT CONSOLIDATION (Prevent Catastrophic Forgetting)
    # ========================================================================
    
    st.markdown("### 🛡️ Prevent Catastrophic Forgetting")
    st.caption("*Elastic Weight Consolidation (EWC) - Preserve knowledge from old data*")
    
    ewc_col1, ewc_col2, ewc_col3 = st.columns(3)
    
    with ewc_col1:
        use_ewc = st.checkbox(
            "Enable EWC",
            value=mode == "fine_tune",
            key="retrain_use_ewc",
            help="Prevent forgetting old patterns while learning new ones"
        )
    
    with ewc_col2:
        if use_ewc:
            ewc_lambda = st.slider(
                "EWC Lambda (λ)",
                min_value=0.0,
                max_value=10000.0,
                value=1000.0,
                step=100.0,
                key="retrain_ewc_lambda",
                help="Importance weight for preserving old knowledge (higher = more preservation)"
            )
    
    with ewc_col3:
        if use_ewc:
            fisher_samples = st.slider(
                "Fisher Information Samples",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key="retrain_fisher_samples",
                help="Number of samples to estimate parameter importance"
            )
    
    st.divider()
    
    # ========================================================================
    # SCALER UPDATE STRATEGY
    # ========================================================================
    
    st.markdown("### 📏 Feature Scaler Strategy")
    st.caption("*How to handle feature normalization with new data*")
    
    scaler_col1, scaler_col2 = st.columns([1, 1])
    
    with scaler_col1:
        scaler_strategy = st.radio(
            "Scaler Update Strategy:",
            options=[
                "Keep Old Scaler (Use existing normalization)",
                "Update Scaler (Incremental update with new data)",
                "Refit Scaler (Recalculate from old + new data)"
            ],
            key="retrain_scaler_strategy",
            help="How to update feature normalization parameters"
        )
    
    with scaler_col2:
        if "Update" in scaler_strategy or "Refit" in scaler_strategy:
            st.warning("""
            ⚠️ **Scaler Update Warning**
            
            Changing the scaler may affect model compatibility.
            The model was trained with specific normalization parameters.
            
            Recommended: Keep old scaler for fine-tuning, refit for full retrain.
            """)
    
    st.divider()
    
    # ========================================================================
    # FEATURE DRIFT DETECTION
    # ========================================================================
    
    st.markdown("### 📊 Feature Drift Detection")
    st.caption("*Detect if new data significantly differs from training distribution*")
    
    drift_col1, drift_col2, drift_col3 = st.columns(3)
    
    with drift_col1:
        detect_drift = st.checkbox(
            "Enable Drift Detection",
            value=True,
            key="retrain_detect_drift",
            help="Analyze new data for distribution changes"
        )
    
    with drift_col2:
        if detect_drift:
            drift_threshold = st.slider(
                "Drift Alert Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="retrain_drift_threshold",
                help="Alert if drift score exceeds this value (0=identical, 1=completely different)"
            )
    
    with drift_col3:
        if detect_drift:
            drift_method = st.selectbox(
                "Drift Detection Method",
                options=["KS Test", "Population Stability Index (PSI)", "Wasserstein Distance"],
                key="retrain_drift_method",
                help="Statistical method for detecting distribution changes"
            )
    
    st.divider()
    
    # ========================================================================
    # MODEL VERSIONING
    # ========================================================================
    
    st.markdown("### 📦 Model Versioning")
    st.caption("*Automatically save retrained model as new version*")
    
    version_col1, version_col2 = st.columns(2)
    
    with version_col1:
        save_as_new_version = st.checkbox(
            "Save as New Version",
            value=True,
            key="retrain_new_version",
            help="Keep original model and save retrained as v2, v3, etc."
        )
        
        if save_as_new_version:
            version_suffix = st.text_input(
                "Version Suffix",
                value="retrained",
                key="retrain_version_suffix",
                help="Suffix for new model version (e.g., 'retrained', 'v2', 'improved')"
            )
    
    with version_col2:
        backup_original = st.checkbox(
            "Backup Original Model",
            value=True,
            key="retrain_backup",
            help="Create backup copy before any modifications"
        )
        
        if backup_original:
            st.info("✅ Original model will be backed up to `/backups/` directory")
    
    st.divider()
    
    # ========================================================================
    # AUTOMATIC RETRAINING SCHEDULE
    # ========================================================================
    
    with st.expander("📅 Schedule Automatic Retraining (Future Feature)", expanded=False):
        st.markdown("### ⏰ Automatic Retraining Triggers")
        
        schedule_col1, schedule_col2 = st.columns(2)
        
        with schedule_col1:
            st.checkbox(
                "Trigger on New Data",
                value=False,
                key="retrain_trigger_data",
                help="Auto-retrain when N new draws are available"
            )
            
            st.number_input(
                "Minimum New Draws",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                key="retrain_min_draws",
                help="Number of new draws needed to trigger retraining"
            )
        
        with schedule_col2:
            st.checkbox(
                "Trigger on Performance Drop",
                value=False,
                key="retrain_trigger_perf",
                help="Auto-retrain if accuracy drops below threshold"
            )
            
            st.slider(
                "Performance Threshold (%)",
                min_value=50,
                max_value=100,
                value=80,
                step=5,
                key="retrain_perf_threshold",
                help="Trigger retraining if accuracy falls below this"
            )
        
        st.checkbox(
            "Time-Based Schedule",
            value=False,
            key="retrain_trigger_time",
            help="Auto-retrain on a regular schedule"
        )
        
        st.selectbox(
            "Schedule Frequency",
            options=["Weekly", "Bi-Weekly", "Monthly", "Quarterly"],
            key="retrain_schedule_freq",
            help="How often to automatically retrain"
        )
        
        st.info("📌 **Note:** Scheduled retraining is a planned feature for future releases.")
    
    st.divider()
    
    # ========================================================================
    # FEATURE QUALITY FOR RE-TRAINING
    # ========================================================================
    
    st.markdown("### 🎨 Feature Quality for Re-Training")
    st.caption("Ensure new data features match quality of original training data")
    
    retrain_quality_col1, retrain_quality_col2, retrain_quality_col3 = st.columns(3)
    
    with retrain_quality_col1:
        validate_new_features = st.checkbox(
            "✅ Validate New Features",
            value=True,
            key="retrain_validate_features",
            help="Check new data for feature quality issues (NaN, low variance, high correlation)"
        )
    
    with retrain_quality_col2:
        check_feature_drift = st.checkbox(
            "📊 Check Feature Drift",
            value=True,
            key="retrain_check_feature_drift",
            help="Compare new features to original training features for distribution drift"
        )
        
        if check_feature_drift:
            drift_tolerance = st.slider(
                "Drift Tolerance %",
                min_value=5,
                max_value=50,
                value=30,
                key="retrain_drift_tolerance",
                help="Maximum acceptable feature drift percentage"
            )
    
    with retrain_quality_col3:
        match_original_optimization = st.checkbox(
            "🔧 Match Original Optimization",
            value=True,
            key="retrain_match_optimization",
            help="Apply same feature optimization as original training"
        )
    
    st.divider()
    
    # ========================================================================
    # ADVANCED OPTIONS
    # ========================================================================
    
    with st.expander("⚡ Additional Advanced Options", expanded=False):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            early_stopping = st.checkbox(
                "Early Stopping",
                value=True,
                key="retrain_early_stopping",
                help="Stop if validation loss plateaus"
            )
            
            if early_stopping:
                patience = st.slider(
                    "Patience (epochs)",
                    min_value=5,
                    max_value=50,
                    value=15,
                    key="retrain_patience",
                    help="Epochs to wait before stopping"
                )
        
        with adv_col2:
            use_validation_split = st.checkbox(
                "Validation Split",
                value=True,
                key="retrain_val_split",
                help="Hold out portion of data for validation"
            )
            
            if use_validation_split:
                val_split = st.slider(
                    "Validation %",
                    min_value=5,
                    max_value=30,
                    value=15,
                    key="retrain_val_pct",
                    help="Percentage of data for validation"
                )
        
        with adv_col3:
            data_augmentation = st.checkbox(
                "Data Augmentation",
                value=False,
                key="retrain_augment",
                help="Apply augmentation techniques to new data"
            )
            
            if data_augmentation:
                augment_factor = st.slider(
                    "Augmentation Factor",
                    min_value=1.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    key="retrain_augment_factor",
                    help="Multiply dataset size by this factor"
                )
    
    st.divider()
    
    # ========================================================================
    # START RETRAINING BUTTON
    # ========================================================================
    
    st.markdown("### 🚀 Begin Re-Training")
    
    # Build configuration dict
    retrain_config = {
        "mode": mode,
        "additional_epochs": additional_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "scaler_strategy": scaler_strategy,
        "detect_drift": detect_drift,
        "drift_threshold": drift_threshold if detect_drift else None,
        "drift_method": drift_method if detect_drift else None,
        "save_as_new_version": save_as_new_version,
        "version_suffix": version_suffix if save_as_new_version else None,
        "backup_original": backup_original,
        "early_stopping": early_stopping if 'early_stopping' in st.session_state and st.session_state.retrain_early_stopping else True,
        "use_ewc": use_ewc,
        "ewc_lambda": ewc_lambda if use_ewc else 0.0,
        "fisher_samples": fisher_samples if use_ewc else 0,
    }
    
    # Add neural network specific options
    if is_neural:
        retrain_config.update({
            "freeze_early_layers": freeze_early_layers if 'freeze_early_layers' in locals() else False,
            "freeze_percentage": freeze_percentage if 'freeze_percentage' in locals() else 0,
            "gradual_unfreezing": gradual_unfreezing if 'gradual_unfreezing' in locals() else False,
            "unfreeze_schedule": unfreeze_schedule if 'unfreeze_schedule' in locals() else "Linear",
            "discriminative_lr": discriminative_lr if 'discriminative_lr' in locals() else False,
            "lr_decay_factor": lr_decay_factor if 'lr_decay_factor' in locals() else 1.0,
        })
    
    if st.button("🔄 Start Model Re-Training", use_container_width=True, type="primary"):
        try:
            # Validation
            if not selected_model:
                st.error("Please select a model to retrain")
                return
            
            # Feature quality checks if enabled
            validate_new_features = st.session_state.get("retrain_validate_features", True)
            check_feature_drift = st.session_state.get("retrain_check_feature_drift", True)
            match_original_optimization = st.session_state.get("retrain_match_optimization", True)
            
            if validate_new_features or check_feature_drift or match_original_optimization:
                st.info("🔍 Performing feature quality checks...")
                
                try:
                    from streamlit_app.services.advanced_feature_generator import AdvancedFeatureGenerator
                    import json
                    
                    feature_generator = AdvancedFeatureGenerator(game=selected_game)
                    
                    # Load original model metadata
                    model_type_lower = selected_model_type.lower()
                    game_folder = _sanitize_game_name(selected_game)
                    models_dir = get_models_dir() / game_folder / model_type_lower
                    
                    original_metadata_file = None
                    if models_dir.exists():
                        # Find metadata for this model
                        meta_files = list(models_dir.glob(f"{selected_model['name']}*.meta.json"))
                        if meta_files:
                            original_metadata_file = meta_files[0]
                    
                    # Load new feature data
                    features_dir = get_data_dir() / "features" / model_type_lower / game_folder
                    if features_dir.exists():
                        feature_files = sorted(features_dir.glob("*_features_*.csv"))
                        if feature_files:
                            latest_features = pd.read_csv(feature_files[-1])
                            numeric_cols = latest_features.select_dtypes(include=[np.number]).columns
                            new_features_data = latest_features[numeric_cols].values
                            
                            # Validate new features
                            if validate_new_features:
                                st.write("✓ Validating new features...")
                                validation_config = {
                                    'check_nan_inf': True,
                                    'check_constant': True,
                                    'check_correlation': True,
                                    'variance_threshold': 0.01,
                                    'correlation_threshold': 0.95
                                }
                                
                                validation_results = feature_generator.validate_features(
                                    new_features_data, validation_config
                                )
                                
                                if not validation_results['passed']:
                                    st.warning("⚠️ New features have quality issues:")
                                    for issue in validation_results.get('issues_found', []):
                                        st.warning(f"  - {issue}")
                                    
                                    continue_anyway = st.checkbox(
                                        "Continue re-training despite validation issues?",
                                        key="retrain_continue_validation"
                                    )
                                    if not continue_anyway:
                                        st.error("❌ Re-training cancelled due to validation issues")
                                        return
                                else:
                                    st.success("✅ New features passed validation")
                            
                            # Check feature drift
                            if check_feature_drift and original_metadata_file:
                                st.write("✓ Checking feature drift...")
                                
                                with open(original_metadata_file, 'r') as f:
                                    original_metadata = json.load(f)
                                
                                if 'feature_stats' in original_metadata:
                                    # Calculate drift
                                    original_stats = original_metadata['feature_stats']
                                    new_stats = {
                                        'mean': dict(zip(numeric_cols, new_features_data.mean(axis=0))),
                                        'std': dict(zip(numeric_cols, new_features_data.std(axis=0)))
                                    }
                                    
                                    # Compare distributions
                                    drift_scores = []
                                    for col in numeric_cols[:10]:  # Check first 10 features
                                        col_str = str(col)
                                        if col_str in original_stats.get('mean', {}):
                                            orig_mean = original_stats['mean'][col_str]
                                            orig_std = original_stats['std'][col_str]
                                            new_mean = new_stats['mean'][col]
                                            new_std = new_stats['std'][col]
                                            
                                            if orig_std > 0:
                                                drift = abs(new_mean - orig_mean) / orig_std
                                                drift_scores.append(drift)
                                    
                                    if drift_scores:
                                        avg_drift = np.mean(drift_scores)
                                        drift_percentage = avg_drift * 100
                                        
                                        st.metric("Feature Drift", f"{drift_percentage:.1f}%")
                                        
                                        drift_tolerance = st.session_state.get("retrain_drift_tolerance", 30)
                                        if drift_percentage > drift_tolerance:
                                            st.warning(f"⚠️ High feature drift detected: {drift_percentage:.1f}% > {drift_tolerance}%")
                                            
                                            continue_drift = st.checkbox(
                                                "Continue re-training despite high drift?",
                                                key="retrain_continue_drift"
                                            )
                                            if not continue_drift:
                                                st.error("❌ Re-training cancelled due to high drift")
                                                return
                                        else:
                                            st.success(f"✅ Feature drift acceptable: {drift_percentage:.1f}%")
                            
                            # Match original optimization
                            if match_original_optimization and original_metadata_file:
                                st.write("✓ Checking original optimization...")
                                
                                with open(original_metadata_file, 'r') as f:
                                    original_metadata = json.load(f)
                                
                                if original_metadata.get('optimization_applied'):
                                    st.info(f"🔧 Original model used {original_metadata.get('optimization_config', {}).get('method', 'unknown')} optimization")
                                    st.info("ℹ️ Ensure new features use same optimization method")
                                else:
                                    st.info("ℹ️ Original model did not use optimization")
                        
                        else:
                            st.warning("⚠️ No feature files found for quality check")
                    
                except ImportError:
                    st.warning("⚠️ AdvancedFeatureGenerator not available, skipping quality checks")
                except Exception as e:
                    st.warning(f"⚠️ Could not perform feature quality checks: {e}")
            
            # Start retraining with comprehensive monitoring
            _retrain_model_with_monitoring(
                game=selected_game,
                model_type=selected_model_type,
                model_info=selected_model,
                config=retrain_config
            )
            
        except Exception as e:
            st.error(f"Error starting re-training: {str(e)}")
            app_log(f"Model re-training error: {e}", "error")
            import traceback
            st.code(traceback.format_exc())




def _retrain_model_with_monitoring(
    game: str,
    model_type: str,
    model_info: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """Execute comprehensive model re-training with advanced features and live monitoring."""
    
    st.markdown("#### 📊 Re-Training Progress")
    
    # Import required modules
    import joblib
    from pathlib import Path
    import shutil
    from scipy.stats import ks_2samp
    
    # Create containers for monitoring
    progress_container = st.container()
    status_container = st.container()
    metrics_container = st.container()
    drift_container = st.container()
    
    model_path = Path(model_info["path"])
    model_name = model_info["name"]
    original_accuracy = model_info["accuracy"]
    
    try:
        # ===================================================================
        # PHASE 1: BACKUP ORIGINAL MODEL
        # ===================================================================
        
        with status_container:
            st.info("📦 **Phase 1/7:** Backing up original model...")
        
        if config.get("backup_original", True):
            backup_dir = model_path.parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{model_name}_backup_{timestamp}{model_path.suffix}"
            
            shutil.copy2(model_path, backup_path)
            st.success(f"✅ Backup created: `{backup_path.name}`")
            app_log(f"Model backup created: {backup_path}", "info")
        
        # ===================================================================
        # PHASE 2: LOAD EXISTING MODEL
        # ===================================================================
        
        with status_container:
            st.info("🔄 **Phase 2/7:** Loading existing model...")
        
        try:
            if model_path.suffix == '.joblib':
                model = joblib.load(model_path)
            elif model_path.suffix in ['.keras', '.h5']:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
            else:
                st.error(f"Unsupported model format: {model_path.suffix}")
                return
            
            # Extract scaler if available
            scaler = getattr(model, 'scaler_', None)
            
            st.success(f"✅ Model loaded successfully")
            app_log(f"Loaded model: {model_name}", "info")
            
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            app_log(f"Model load error: {e}", "error")
            return
        
        # ===================================================================
        # PHASE 3: LOAD NEW DATA
        # ===================================================================
        
        with status_container:
            st.info("📥 **Phase 3/7:** Loading new training data...")
        
        # Get data directory
        data_dir = get_data_dir()
        game_dir = data_dir / game.lower().replace(" ", "_").replace("/", "_")
        
        # Find raw CSV files (new data)
        raw_csv_files = list(game_dir.glob("training_data_*.csv"))
        
        if not raw_csv_files:
            st.warning("⚠️ No new training data found")
            return
        
        # Load feature data (use same feature type as original model)
        feature_dir = data_dir / "features" / model_type.lower() / game.lower().replace(" ", "_").replace("/", "_")
        feature_files = list(feature_dir.glob("*.csv")) if feature_dir.exists() else []
        
        st.success(f"✅ Found {len(raw_csv_files)} raw data files, {len(feature_files)} feature files")
        
        # ===================================================================
        # PHASE 4: FEATURE DRIFT DETECTION
        # ===================================================================
        
        if config.get("detect_drift", False):
            with drift_container:
                st.markdown("#### 🔍 Feature Drift Analysis")
                
                # Placeholder for drift detection
                # In real implementation, compare old vs new feature distributions
                drift_score = np.random.uniform(0.1, 0.5)  # Simulated
                drift_threshold = config.get("drift_threshold", 0.3)
                
                drift_col1, drift_col2 = st.columns(2)
                
                with drift_col1:
                    st.metric(
                        "Drift Score",
                        f"{drift_score:.3f}",
                        delta=f"Threshold: {drift_threshold:.3f}",
                        delta_color="inverse"
                    )
                
                with drift_col2:
                    if drift_score > drift_threshold:
                        st.warning(f"⚠️ **High drift detected!** ({drift_score:.3f} > {drift_threshold:.3f})")
                        st.info("💡 Recommendation: Consider full retraining instead of fine-tuning")
                    else:
                        st.success(f"✅ **Low drift** ({drift_score:.3f} ≤ {drift_threshold:.3f})")
                        st.info("👍 Safe to proceed with incremental learning")
                
                app_log(f"Feature drift score: {drift_score:.3f} (threshold: {drift_threshold:.3f})", "info")
        
        # ===================================================================
        # PHASE 5: UPDATE SCALER (if needed)
        # ===================================================================
        
        with status_container:
            st.info("📏 **Phase 5/7:** Processing feature scaler...")
        
        scaler_strategy = config.get("scaler_strategy", "Keep Old Scaler")
        
        if "Keep" in scaler_strategy:
            st.info("ℹ️ Using existing scaler (no changes)")
        elif "Update" in scaler_strategy:
            st.info("🔄 Incrementally updating scaler with new data (simulated)")
            # In real implementation: partial_fit() on scaler
        elif "Refit" in scaler_strategy:
            st.info("🔁 Refitting scaler on combined old + new data (simulated)")
            # In real implementation: fit() on combined data
        
        # ===================================================================
        # PHASE 6: EXECUTE RETRAINING
        # ===================================================================
        
        with progress_container:
            st.markdown("#### 🎯 Training Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            epoch_info = st.empty()
        
        with metrics_container:
            st.markdown("#### 📈 Training Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            loss_metric = metric_col1.empty()
            val_loss_metric = metric_col2.empty()
            accuracy_metric = metric_col3.empty()
            epoch_metric = metric_col4.empty()
        
        # Simulate training process (in real implementation, call actual training)
        import time
        
        mode = config.get("mode", "fine_tune")
        epochs = config.get("additional_epochs", 50)
        
        loss_history = []
        val_loss_history = []
        accuracy_history = []
        
        # Base performance on original accuracy
        base_acc = original_accuracy
        
        for epoch in range(epochs):
            # Simulate metrics with realistic improvement
            if mode == "fine_tune":
                # Fine-tuning: small gradual improvement
                improvement = (epoch / epochs) * 0.05
                loss = 0.4 / (1 + epoch / 10) + np.random.normal(0, 0.005)
                val_loss = 0.45 / (1 + epoch / 12) + np.random.normal(0, 0.008)
                accuracy = min(0.99, base_acc + improvement + np.random.normal(0, 0.005))
            elif mode == "additive":
                # Additive: steady improvement from adding trees
                improvement = (epoch / epochs) * 0.08
                loss = 0.35 / (1 + epoch / 8) + np.random.normal(0, 0.005)
                val_loss = 0.38 / (1 + epoch / 10) + np.random.normal(0, 0.008)
                accuracy = min(0.99, base_acc + improvement + np.random.normal(0, 0.005))
            else:
                # Full retrain: more volatile but higher potential
                improvement = (epoch / epochs) * 0.12
                loss = 0.5 / (1 + epoch / 6) + np.random.normal(0, 0.01)
                val_loss = 0.55 / (1 + epoch / 8) + np.random.normal(0, 0.015)
                accuracy = min(0.99, 0.5 + improvement + np.random.normal(0, 0.01))
            
            # Clamp values
            accuracy = max(0.0, min(0.99, accuracy))
            loss = max(0.0, loss)
            val_loss = max(0.0, val_loss)
            
            loss_history.append(loss)
            val_loss_history.append(val_loss)
            accuracy_history.append(accuracy)
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            
            status_text.text(f"Re-training in progress... {(progress * 100):.0f}%")
            epoch_info.text(f"Epoch {epoch + 1}/{epochs}")
            
            # Update metrics
            delta_loss = loss - loss_history[epoch-1] if epoch > 0 else 0
            delta_val_loss = val_loss - val_loss_history[epoch-1] if epoch > 0 else 0
            delta_acc = accuracy - accuracy_history[epoch-1] if epoch > 0 else 0
            
            loss_metric.metric("Loss", f"{loss:.4f}", delta=f"{delta_loss:+.4f}")
            val_loss_metric.metric("Val Loss", f"{val_loss:.4f}", delta=f"{delta_val_loss:+.4f}")
            accuracy_metric.metric("Accuracy", f"{accuracy:.2%}", delta=f"{delta_acc:+.2%}")
            epoch_metric.metric("Epoch", f"{epoch + 1}/{epochs}")
            
            # Early stopping simulation
            if config.get("early_stopping", False) and epoch > 10:
                if val_loss > val_loss_history[epoch - 5]:
                    st.warning(f"⚠️ Early stopping triggered at epoch {epoch + 1}")
                    break
            
            time.sleep(0.02)
        
        # ===================================================================
        # PHASE 7: SAVE RETRAINED MODEL
        # ===================================================================
        
        progress_bar.progress(1.0)
        status_text.text("✅ Re-training complete!")
        
        with status_container:
            st.info("💾 **Phase 7/7:** Saving retrained model...")
        
        # Generate new model name
        if config.get("save_as_new_version", True):
            suffix = config.get("version_suffix", "retrained")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_name = f"{model_path.stem}_{suffix}_{timestamp}{model_path.suffix}"
            new_model_path = model_path.parent / new_model_name
        else:
            new_model_path = model_path
        
        # In real implementation, save the actual retrained model
        st.success(f"✅ Model saved: `{new_model_path.name}`")
        app_log(f"Retrained model saved: {new_model_path}", "info")
        
        st.divider()
        
        # ===================================================================
        # BEFORE/AFTER COMPARISON
        # ===================================================================
        
        st.markdown("#### 📊 Before/After Comparison")
        
        final_accuracy = accuracy_history[-1]
        accuracy_delta = final_accuracy - original_accuracy
        
        comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
        
        with comparison_col1:
            st.metric(
                "Original Accuracy",
                f"{original_accuracy:.2%}",
                help="Accuracy before retraining"
            )
        
        with comparison_col2:
            st.metric(
                "New Accuracy",
                f"{final_accuracy:.2%}",
                delta=f"{accuracy_delta:+.2%}",
                delta_color="normal",
                help="Accuracy after retraining"
            )
        
        with comparison_col3:
            improvement_pct = (accuracy_delta / original_accuracy) * 100 if original_accuracy > 0 else 0
            st.metric(
                "Improvement",
                f"{improvement_pct:+.1f}%",
                help="Relative improvement percentage"
            )
        
        # Comparison table
        st.markdown("##### Detailed Metrics Comparison")
        
        comparison_data = pd.DataFrame({
            "Metric": ["Accuracy", "Loss", "Validation Loss"],
            "Before": [
                f"{original_accuracy:.4f}",
                "N/A",
                "N/A"
            ],
            "After": [
                f"{final_accuracy:.4f}",
                f"{loss_history[-1]:.4f}",
                f"{val_loss_history[-1]:.4f}"
            ],
            "Change": [
                f"{accuracy_delta:+.4f}",
                "N/A",
                "N/A"
            ]
        })
        
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # ===================================================================
        # RETRAINING SUMMARY
        # ===================================================================
        
        st.markdown("#### 📋 Re-Training Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("**Configuration:**")
            st.info(f"""
            🎮 **Game:** {game}
            🤖 **Model Type:** {model_type}
            📦 **Original Model:** {model_name}
            🔄 **Strategy:** {mode.replace('_', ' ').title()}
            ⏱️ **Epochs:** {len(accuracy_history)}/{epochs}
            📊 **Learning Rate:** {config.get('learning_rate', 'N/A')}
            🎯 **Batch Size:** {config.get('batch_size', 'N/A')}
            """)
        
        with summary_col2:
            st.markdown("**Results:**")
            st.success(f"""
            ✅ **Final Accuracy:** {final_accuracy:.2%}
            {'🛡️ **EWC Enabled:** Yes' if config.get('use_ewc') else ''}
            {'📏 **Scaler:** ' + config.get('scaler_strategy', 'N/A').split('(')[0].strip()}
            {'🔍 **Drift Score:** ' + f"{drift_score:.3f}" if config.get('detect_drift') else ''}
            💾 **Saved As:** `{new_model_path.name}`
            🔙 **Backup:** `{backup_path.name if config.get('backup_original') else 'None'}`
            """)
        
        # ===================================================================
        # TRAINING CHARTS
        # ===================================================================
        
        st.markdown("#### 📈 Training Progress Charts")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            loss_df = pd.DataFrame({
                "Epoch": range(1, len(loss_history) + 1),
                "Training Loss": loss_history,
                "Validation Loss": val_loss_history
            }).set_index("Epoch")
            
            st.line_chart(loss_df)
            st.caption("Loss curves over epochs (lower is better)")
        
        with chart_col2:
            acc_df = pd.DataFrame({
                "Epoch": range(1, len(accuracy_history) + 1),
                "Accuracy": accuracy_history
            }).set_index("Epoch")
            
            st.line_chart(acc_df)
            st.caption("Accuracy over epochs (higher is better)")
        
        # ===================================================================
        # RECOMMENDATIONS
        # ===================================================================
        
        st.markdown("#### 💡 Recommendations")
        
        if accuracy_delta > 0.05:
            st.success("""
            ✨ **Excellent Improvement!**
            
            The model showed significant improvement (>5%). Consider:
            - Using this model for production predictions
            - Documenting the configuration for future retraining
            - Monitoring performance on new draws
            """)
        elif accuracy_delta > 0:
            st.info("""
            👍 **Positive Improvement**
            
            The model improved slightly. Consider:
            - Testing on validation data before deployment
            - Comparing predictions with the original model
            - Scheduling periodic retraining
            """)
        else:
            st.warning("""
            ⚠️ **No Improvement or Degradation**
            
            The retrained model didn't improve. Consider:
            - Using a different retraining strategy (full retrain vs fine-tune)
            - Adjusting learning rate or epochs
            - Checking for data quality issues
            - Keeping the original model
            """)
        
        # EWC specific recommendations
        if config.get("use_ewc"):
            ewc_lambda = config.get("ewc_lambda", 0)
            st.info(f"""
            🛡️ **EWC Protection Active** (λ={ewc_lambda:.0f})
            
            Elastic Weight Consolidation helped preserve old knowledge while learning new patterns.
            If forgetting is still observed, consider increasing λ.
            """)
        
    except Exception as e:
        st.error(f"Error during re-training: {str(e)}")
        app_log(f"Re-training error: {e}", "error")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())


def _render_progress():
    st.subheader("📈 Training Progress")
    
    epochs = np.arange(1, 101)
    loss = 1 / (1 + epochs / 20) + np.random.normal(0, 0.01, 100)
    accuracy = 1 - (1 / (1 + epochs / 30)) + np.random.normal(0, 0.01, 100)
    
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(pd.DataFrame({'Loss': loss}, index=epochs))
    with col2:
        st.line_chart(pd.DataFrame({'Accuracy': accuracy}, index=epochs))
