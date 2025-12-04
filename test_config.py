from streamlit_app.core import get_game_config

games = ['Lotto 6/49', 'Lotto Max', 'Daily Grand', 'Powerball']

for game in games:
    try:
        config = get_game_config(game)
        print(f'{game}:')
        print(f'  max_number: {config.get("max_number", "NOT SET")}')
        print(f'  main_numbers: {config.get("main_numbers", "NOT SET")}')
        print(f'  bonus_number: {config.get("bonus_number", "NOT SET")}')
        print()
    except Exception as e:
        print(f'{game}: ERROR - {e}')
        print()
