#!/usr/bin/env python3
"""Fix prediction history display with color-coded matching accuracy."""

import re

with open('streamlit_app/pages/predictions.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Old section to replace  
old_section = '''                        # Display prediction sets
                        if sets and isinstance(sets, list):
                            st.markdown("**Predicted Numbers:**")
                            sets_cols = st.columns(len(sets))
                            for set_idx, (col, prediction_set) in enumerate(zip(sets_cols, sets)):
                                with col:
                                    # Parse prediction numbers
                                    if isinstance(prediction_set, (list, tuple)):
                                        nums = [int(n) for n in prediction_set if str(n).isdigit()]
                                    else:
                                        nums_str = str(prediction_set).strip('[]"')
                                        nums = [int(n.strip()) for n in nums_str.split(',') if n.strip().isdigit()]
                                    
                                    # Display as badges
                                    badge_html = '<div style="display: flex; gap: 5px; flex-wrap: wrap;">'
                                    for num in nums:
                                        badge_html += f'<span style="padding: 5px 10px; background: #667eea; color: white; border-radius: 5px; font-weight: bold; font-size: 14px;">{num}</span>'
                                    badge_html += '</div>'
                                    st.markdown(f"**Set {set_idx + 1}:**", unsafe_allow_html=False)
                                    st.markdown(badge_html, unsafe_allow_html=True)'''

new_section = '''                        # Display prediction sets with matching accuracy
                        if sets and isinstance(sets, list):
                            st.markdown("**Predicted Numbers & Accuracy:**")
                            
                            # Calculate accuracy if we have winning numbers
                            accuracy_data = {}
                            if draw_info:
                                winning_nums = draw_info.get('numbers', [])
                                accuracy_data = _calculate_prediction_accuracy(sets, winning_nums)
                            
                            # Display sets in rows (max 3 per row)
                            num_cols = min(len(sets), 3)
                            
                            for set_idx, prediction_set in enumerate(sets):
                                if set_idx % num_cols == 0:
                                    sets_cols = st.columns(min(num_cols, len(sets) - set_idx))
                                
                                col = sets_cols[set_idx % num_cols]
                                
                                # Parse prediction numbers
                                if isinstance(prediction_set, (list, tuple)):
                                    nums = [int(n) for n in prediction_set if str(n).isdigit()]
                                else:
                                    nums_str = str(prediction_set).strip('[]"')
                                    nums = [int(n.strip()) for n in nums_str.split(',') if n.strip().isdigit()]
                                
                                with col:
                                    # Get accuracy for this set
                                    acc = accuracy_data.get(set_idx, {})
                                    matched = acc.get('matched_numbers', [])
                                    match_count = acc.get('match_count', 0)
                                    total = acc.get('total_count', len(nums))
                                    
                                    st.markdown(f"**Set {set_idx + 1}**")
                                    
                                    # Display numbers with color coding
                                    badge_html = '<div style="display: flex; gap: 5px; flex-wrap: wrap;">'
                                    for num in nums:
                                        if num in matched:
                                            # Green for matched numbers
                                            bg_color = "#10b981"
                                            text_color = "white"
                                        else:
                                            # Light red for unmatched
                                            bg_color = "#fee2e2"
                                            text_color = "#991b1b"
                                        
                                        badge_html += f'<span style="padding: 6px 12px; background: {bg_color}; color: {text_color}; border-radius: 6px; font-weight: bold; font-size: 14px;">{num}</span>'
                                    badge_html += '</div>'
                                    st.markdown(badge_html, unsafe_allow_html=True)
                                    
                                    # Show match count
                                    if acc:
                                        st.caption(f"✓ {match_count}/{total} numbers matched")'''

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('streamlit_app/pages/predictions.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ Display section updated successfully')
else:
    print('ERROR: Pattern not found')
