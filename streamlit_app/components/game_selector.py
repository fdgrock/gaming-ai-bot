"""
Game selector components for the lottery prediction system.

This module provides components for selecting and configuring
different lottery games and their parameters.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GameSelector:
    """Component for selecting lottery games."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize game selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.games = self._load_game_definitions()
    
    def render(self, title: str = "Select Lottery Game",
               show_details: bool = True,
               compact: bool = False) -> Optional[Dict[str, Any]]:
        """
        Render game selector.
        
        Args:
            title: Selector title
            show_details: Whether to show game details
            compact: Whether to use compact view
            
        Returns:
            Selected game configuration or None
        """
        try:
            if not compact:
                st.subheader(title)
            
            # Game selection dropdown
            game_names = list(self.games.keys())
            
            if 'selected_game' not in st.session_state:
                st.session_state.selected_game = game_names[0] if game_names else None
            
            selected_game_name = st.selectbox(
                "Choose Game:",
                options=game_names,
                index=game_names.index(st.session_state.selected_game) if st.session_state.selected_game in game_names else 0,
                key="game_selector"
            )
            
            if selected_game_name:
                st.session_state.selected_game = selected_game_name
                selected_game = self.games[selected_game_name]
                
                # Show game details
                if show_details and not compact:
                    self._render_game_details(selected_game)
                
                return selected_game
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to render game selector: {e}")
            st.error(f"Failed to display game selector: {e}")
            return None
    
    def _render_game_details(self, game: Dict[str, Any]) -> None:
        """Render game details."""
        try:
            st.markdown("**Game Details:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **{game['name']}**
                - Numbers to pick: {game['numbers_to_pick']}
                - Number range: {game['min_number']} - {game['max_number']}
                - Bonus ball: {'Yes' if game.get('has_bonus', False) else 'No'}
                """)
            
            with col2:
                if 'description' in game:
                    st.markdown(f"**Description:**\n{game['description']}")
                
                if 'jackpot_odds' in game:
                    st.markdown(f"**Jackpot Odds:** 1 in {game['jackpot_odds']:,}")
            
            # Prize structure if available
            if 'prize_tiers' in game:
                st.markdown("**Prize Tiers:**")
                for tier in game['prize_tiers']:
                    st.write(f"- {tier['match']}: {tier['description']}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to render game details: {e}")
    
    def _load_game_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load game definitions."""
        return {
            "Mega Millions": {
                "name": "Mega Millions",
                "numbers_to_pick": 5,
                "min_number": 1,
                "max_number": 70,
                "has_bonus": True,
                "bonus_min": 1,
                "bonus_max": 25,
                "description": "Pick 5 numbers from 1-70 plus Mega Ball from 1-25",
                "jackpot_odds": 302575350,
                "prize_tiers": [
                    {"match": "5 + Mega Ball", "description": "Jackpot"},
                    {"match": "5", "description": "$1 Million"},
                    {"match": "4 + Mega Ball", "description": "$10,000"},
                    {"match": "4", "description": "$500"},
                    {"match": "3 + Mega Ball", "description": "$200"},
                    {"match": "3", "description": "$10"},
                    {"match": "2 + Mega Ball", "description": "$10"},
                    {"match": "1 + Mega Ball", "description": "$4"},
                    {"match": "Mega Ball", "description": "$2"}
                ]
            },
            "Powerball": {
                "name": "Powerball",
                "numbers_to_pick": 5,
                "min_number": 1,
                "max_number": 69,
                "has_bonus": True,
                "bonus_min": 1,
                "bonus_max": 26,
                "description": "Pick 5 numbers from 1-69 plus Powerball from 1-26",
                "jackpot_odds": 292201338,
                "prize_tiers": [
                    {"match": "5 + Powerball", "description": "Jackpot"},
                    {"match": "5", "description": "$1 Million"},
                    {"match": "4 + Powerball", "description": "$50,000"},
                    {"match": "4", "description": "$100"},
                    {"match": "3 + Powerball", "description": "$100"},
                    {"match": "3", "description": "$7"},
                    {"match": "2 + Powerball", "description": "$7"},
                    {"match": "1 + Powerball", "description": "$4"},
                    {"match": "Powerball", "description": "$4"}
                ]
            },
            "Lotto 6/49": {
                "name": "Lotto 6/49",
                "numbers_to_pick": 6,
                "min_number": 1,
                "max_number": 49,
                "has_bonus": False,
                "description": "Pick 6 numbers from 1-49",
                "jackpot_odds": 13983816,
                "prize_tiers": [
                    {"match": "6", "description": "Jackpot"},
                    {"match": "5", "description": "Second Prize"},
                    {"match": "4", "description": "Third Prize"},
                    {"match": "3", "description": "Fourth Prize"}
                ]
            },
            "EuroMillions": {
                "name": "EuroMillions",
                "numbers_to_pick": 5,
                "min_number": 1,
                "max_number": 50,
                "has_bonus": True,
                "bonus_count": 2,
                "bonus_min": 1,
                "bonus_max": 12,
                "description": "Pick 5 numbers from 1-50 plus 2 Lucky Stars from 1-12",
                "jackpot_odds": 139838160,
                "prize_tiers": [
                    {"match": "5 + 2 Stars", "description": "Jackpot"},
                    {"match": "5 + 1 Star", "description": "Second Prize"},
                    {"match": "5", "description": "Third Prize"},
                    {"match": "4 + 2 Stars", "description": "Fourth Prize"},
                    {"match": "4 + 1 Star", "description": "Fifth Prize"},
                    {"match": "4", "description": "Sixth Prize"},
                    {"match": "3 + 2 Stars", "description": "Seventh Prize"}
                ]
            },
            "UK Lotto": {
                "name": "UK National Lottery",
                "numbers_to_pick": 6,
                "min_number": 1,
                "max_number": 59,
                "has_bonus": True,
                "bonus_count": 1,
                "bonus_min": 1,
                "bonus_max": 59,
                "description": "Pick 6 numbers from 1-59 plus bonus ball",
                "jackpot_odds": 45057474,
                "prize_tiers": [
                    {"match": "6", "description": "Jackpot"},
                    {"match": "5 + Bonus", "description": "Second Prize"},
                    {"match": "5", "description": "Third Prize"},
                    {"match": "4", "description": "Fourth Prize"},
                    {"match": "3", "description": "Fifth Prize"},
                    {"match": "2", "description": "Free Lucky Dip"}
                ]
            }
        }
    
    def get_selected_game(self) -> Optional[Dict[str, Any]]:
        """Get currently selected game."""
        selected_name = st.session_state.get('selected_game')
        if selected_name and selected_name in self.games:
            return self.games[selected_name]
        return None
    
    def add_custom_game(self, game_config: Dict[str, Any]) -> bool:
        """
        Add a custom game configuration.
        
        Args:
            game_config: Custom game configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            required_fields = ['name', 'numbers_to_pick', 'min_number', 'max_number']
            
            for field in required_fields:
                if field not in game_config:
                    raise ValueError(f"Missing required field: {field}")
            
            self.games[game_config['name']] = game_config
            logger.info(f"✅ Added custom game: {game_config['name']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add custom game: {e}")
            return False
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class GameConfiguration:
    """Component for configuring game parameters."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize game configuration."""
        self.config = config or {}
    
    def render(self, game: Optional[Dict[str, Any]] = None,
               title: str = "Game Configuration") -> Dict[str, Any]:
        """
        Render game configuration.
        
        Args:
            game: Base game configuration
            title: Configuration title
            
        Returns:
            Updated game configuration
        """
        try:
            st.subheader(title)
            
            if not game:
                game = self._get_default_game()
            
            # Basic configuration
            st.markdown("**Basic Settings:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                numbers_to_pick = st.number_input(
                    "Numbers to Pick:",
                    min_value=1,
                    max_value=20,
                    value=game.get('numbers_to_pick', 6),
                    key="config_numbers_to_pick"
                )
                
                min_number = st.number_input(
                    "Minimum Number:",
                    min_value=0,
                    max_value=100,
                    value=game.get('min_number', 1),
                    key="config_min_number"
                )
            
            with col2:
                max_number = st.number_input(
                    "Maximum Number:",
                    min_value=min_number + numbers_to_pick,
                    max_value=200,
                    value=game.get('max_number', 49),
                    key="config_max_number"
                )
                
                has_bonus = st.checkbox(
                    "Has Bonus Ball:",
                    value=game.get('has_bonus', False),
                    key="config_has_bonus"
                )
            
            # Bonus ball configuration
            if has_bonus:
                st.markdown("**Bonus Ball Settings:**")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    bonus_count = st.number_input(
                        "Number of Bonus Balls:",
                        min_value=1,
                        max_value=5,
                        value=game.get('bonus_count', 1),
                        key="config_bonus_count"
                    )
                    
                    bonus_min = st.number_input(
                        "Bonus Min:",
                        min_value=1,
                        max_value=100,
                        value=game.get('bonus_min', 1),
                        key="config_bonus_min"
                    )
                
                with col4:
                    bonus_max = st.number_input(
                        "Bonus Max:",
                        min_value=bonus_min + bonus_count,
                        max_value=200,
                        value=game.get('bonus_max', 49),
                        key="config_bonus_max"
                    )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                draw_frequency = st.selectbox(
                    "Draw Frequency:",
                    options=["Daily", "Twice Weekly", "Weekly", "Bi-weekly", "Monthly"],
                    index=1,
                    key="config_draw_frequency"
                )
                
                cost_per_play = st.number_input(
                    "Cost per Play ($):",
                    min_value=0.50,
                    max_value=20.00,
                    value=2.00,
                    step=0.50,
                    key="config_cost_per_play"
                )
                
                description = st.text_area(
                    "Game Description:",
                    value=game.get('description', ''),
                    key="config_description"
                )
            
            # Build configuration dictionary
            config = {
                'name': game.get('name', 'Custom Game'),
                'numbers_to_pick': numbers_to_pick,
                'min_number': min_number,
                'max_number': max_number,
                'has_bonus': has_bonus,
                'draw_frequency': draw_frequency,
                'cost_per_play': cost_per_play,
                'description': description
            }
            
            if has_bonus:
                config.update({
                    'bonus_count': bonus_count,
                    'bonus_min': bonus_min,
                    'bonus_max': bonus_max
                })
            
            # Calculate and display odds
            self._display_odds_calculation(config)
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to render game configuration: {e}")
            st.error(f"Failed to display game configuration: {e}")
            return {}
    
    def _get_default_game(self) -> Dict[str, Any]:
        """Get default game configuration."""
        return {
            'name': 'Custom Game',
            'numbers_to_pick': 6,
            'min_number': 1,
            'max_number': 49,
            'has_bonus': False,
            'description': 'Custom lottery game configuration'
        }
    
    def _display_odds_calculation(self, config: Dict[str, Any]) -> None:
        """Display calculated odds for the game configuration."""
        try:
            st.markdown("**Calculated Odds:**")
            
            numbers_to_pick = config['numbers_to_pick']
            total_numbers = config['max_number'] - config['min_number'] + 1
            
            # Calculate jackpot odds (combinations)
            import math
            
            def combination(n, r):
                if r > n or r < 0:
                    return 0
                return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
            
            main_odds = combination(total_numbers, numbers_to_pick)
            
            if config.get('has_bonus', False):
                bonus_count = config.get('bonus_count', 1)
                bonus_range = config.get('bonus_max', 49) - config.get('bonus_min', 1) + 1
                bonus_odds = combination(bonus_range, bonus_count)
                total_odds = main_odds * bonus_odds
            else:
                total_odds = main_odds
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Main Numbers Odds", f"1 in {main_odds:,}")
            
            with col2:
                if config.get('has_bonus', False):
                    bonus_odds = combination(
                        config.get('bonus_max', 49) - config.get('bonus_min', 1) + 1,
                        config.get('bonus_count', 1)
                    )
                    st.metric("Bonus Ball Odds", f"1 in {bonus_odds:,}")
                else:
                    st.metric("Bonus Ball Odds", "N/A")
            
            with col3:
                st.metric("Jackpot Odds", f"1 in {total_odds:,}")
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate odds: {e}")
            st.warning("⚠️ Unable to calculate odds")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


class GameRules:
    """Component for displaying game rules and information."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize game rules component."""
        self.config = config or {}
    
    def render(self, game: Dict[str, Any], 
               title: str = "Game Rules",
               show_strategy_tips: bool = True) -> None:
        """
        Render game rules.
        
        Args:
            game: Game configuration
            title: Rules title
            show_strategy_tips: Whether to show strategy tips
        """
        try:
            st.subheader(title)
            
            # Basic rules
            self._render_basic_rules(game)
            
            # Prize structure
            if 'prize_tiers' in game:
                self._render_prize_structure(game['prize_tiers'])
            
            # Strategy tips
            if show_strategy_tips:
                self._render_strategy_tips(game)
                
        except Exception as e:
            logger.error(f"❌ Failed to render game rules: {e}")
            st.error(f"Failed to display game rules: {e}")
    
    def _render_basic_rules(self, game: Dict[str, Any]) -> None:
        """Render basic game rules."""
        st.markdown("**How to Play:**")
        
        rules_text = f"""
        1. **Pick {game['numbers_to_pick']} numbers** from {game['min_number']} to {game['max_number']}
        """
        
        if game.get('has_bonus', False):
            bonus_count = game.get('bonus_count', 1)
            bonus_min = game.get('bonus_min', 1)
            bonus_max = game.get('bonus_max', 49)
            
            if bonus_count == 1:
                rules_text += f"\n2. **Pick 1 bonus number** from {bonus_min} to {bonus_max}"
            else:
                rules_text += f"\n2. **Pick {bonus_count} bonus numbers** from {bonus_min} to {bonus_max}"
        
        rules_text += f"""
        3. **Match numbers** drawn to win prizes
        4. **Jackpot odds:** 1 in {game.get('jackpot_odds', 'Unknown'):,}
        """
        
        if 'cost_per_play' in game:
            rules_text += f"\n5. **Cost per play:** ${game['cost_per_play']:.2f}"
        
        st.markdown(rules_text)
    
    def _render_prize_structure(self, prize_tiers: List[Dict[str, str]]) -> None:
        """Render prize structure."""
        st.markdown("**Prize Structure:**")
        
        # Create a DataFrame for better display
        prize_data = []
        for tier in prize_tiers:
            prize_data.append({
                'Match': tier['match'],
                'Prize': tier['description']
            })
        
        df = pd.DataFrame(prize_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def _render_strategy_tips(self, game: Dict[str, Any]) -> None:
        """Render strategy tips."""
        with st.expander("💡 Strategy Tips"):
            tips = [
                "🎯 **Number Selection:** Consider using a mix of high and low numbers",
                "📊 **Statistical Analysis:** Look at historical frequency patterns",
                "🔄 **Consistent Play:** Stick to your chosen numbers consistently",
                "💰 **Budget Management:** Set a spending limit and stick to it",
                "📈 **Multiple Entries:** Consider using systematic entries for better coverage"
            ]
            
            if game.get('has_bonus', False):
                tips.append("⭐ **Bonus Numbers:** Don't forget to consider bonus ball patterns")
            
            # Add game-specific tips
            total_numbers = game['max_number'] - game['min_number'] + 1
            numbers_to_pick = game['numbers_to_pick']
            
            if numbers_to_pick / total_numbers < 0.15:
                tips.append("🎲 **High Odds Game:** Focus on consistent strategies rather than frequent changes")
            
            for tip in tips:
                st.markdown(tip)
            
            st.warning("⚠️ **Remember:** Lottery games are games of chance. Play responsibly and within your means.")
    
    @staticmethod
    def health_check() -> bool:
        """Check component health."""
        return True


# Utility functions for game management
def validate_game_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate game configuration.
    
    Args:
        config: Game configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        # Required fields
        required_fields = ['name', 'numbers_to_pick', 'min_number', 'max_number']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate number ranges
        if 'numbers_to_pick' in config and 'min_number' in config and 'max_number' in config:
            total_numbers = config['max_number'] - config['min_number'] + 1
            if config['numbers_to_pick'] > total_numbers:
                errors.append("Cannot pick more numbers than available in range")
            
            if config['numbers_to_pick'] <= 0:
                errors.append("Must pick at least 1 number")
            
            if config['min_number'] >= config['max_number']:
                errors.append("Minimum number must be less than maximum number")
        
        # Validate bonus ball settings
        if config.get('has_bonus', False):
            if 'bonus_min' in config and 'bonus_max' in config:
                if config['bonus_min'] >= config['bonus_max']:
                    errors.append("Bonus minimum must be less than bonus maximum")
                
                bonus_count = config.get('bonus_count', 1)
                bonus_range = config['bonus_max'] - config['bonus_min'] + 1
                if bonus_count > bonus_range:
                    errors.append("Cannot pick more bonus numbers than available in range")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        logger.error(f"❌ Game config validation error: {e}")
        return False, [f"Validation error: {e}"]


def calculate_game_odds(config: Dict[str, Any]) -> Dict[str, int]:
    """
    Calculate odds for a game configuration.
    
    Args:
        config: Game configuration
        
    Returns:
        Dictionary with calculated odds
    """
    try:
        import math
        
        def combination(n, r):
            if r > n or r < 0:
                return 0
            return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
        
        numbers_to_pick = config['numbers_to_pick']
        total_numbers = config['max_number'] - config['min_number'] + 1
        
        main_odds = combination(total_numbers, numbers_to_pick)
        
        result = {'main_odds': main_odds}
        
        if config.get('has_bonus', False):
            bonus_count = config.get('bonus_count', 1)
            bonus_range = config.get('bonus_max', 49) - config.get('bonus_min', 1) + 1
            bonus_odds = combination(bonus_range, bonus_count)
            result['bonus_odds'] = bonus_odds
            result['jackpot_odds'] = main_odds * bonus_odds
        else:
            result['jackpot_odds'] = main_odds
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to calculate game odds: {e}")
        return {'main_odds': 0, 'jackpot_odds': 0}