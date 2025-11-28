# manager.py

import json
import os
from pathlib import Path


def promote_model(game, model_type, version, registry_path: str = None):
    """Promote a model to champion by writing a champion pointer file and updating registry metadata.

    Writes `models/{game}/champion_model.json` containing minimal champion info.
    If registry_path is provided and exists, it will mark the promoted model as champion there too.
    """
    game_key = game if isinstance(game, str) else str(game)
    dest = Path('models') / game_key
    try:
        dest.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        champ = {'game': game_key, 'model_type': model_type, 'version': version, 'promoted_on': datetime.utcnow().isoformat()}
        champ_path = dest / 'champion_model.json'
        # write champion pointer
        with open(champ_path, 'w', encoding='utf-8') as fh:
            json.dump(champ, fh, indent=2)

        # optionally update registry entries
        if registry_path and os.path.exists(registry_path):
            try:
                regs = json.load(open(registry_path, 'r', encoding='utf-8'))
                if isinstance(regs, list):
                    for r in regs:
                        if r.get('version') == version and r.get('type') == model_type and r.get('name'):
                            r['status'] = 'champion'
                        else:
                            if r.get('status') in ('champion', 'Champion', '🏆'):
                                r['status'] = 'challenger'
                    with open(registry_path, 'w', encoding='utf-8') as fh:
                        json.dump(regs, fh, indent=2)
            except Exception:
                pass
        return str(champ_path)
    except Exception:
        return None
