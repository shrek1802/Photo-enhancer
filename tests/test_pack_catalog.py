from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'model_packs' / 'catalog.json'


def test_catalog_has_unique_pack_ids() -> None:
    payload = json.loads(CATALOG.read_text(encoding='utf-8'))
    pack_ids = [pack['pack_id'] for pack in payload['packs']]
    assert len(pack_ids) == len(set(pack_ids))


def test_ready_packs_have_release_tags() -> None:
    payload = json.loads(CATALOG.read_text(encoding='utf-8'))
    for pack in payload['packs']:
        if pack['status'] == 'ready':
            assert pack.get('release_tag')


def test_specialist_templates_do_not_claim_unbuilt_weights() -> None:
    for path in (
        ROOT / 'model_packs' / 'auto-portrait-pro' / 'manifest.template.json',
        ROOT / 'model_packs' / 'auto-heritage' / 'manifest.template.json',
        ROOT / 'model_packs' / 'auto-social-recovery' / 'manifest.template.json',
    ):
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert payload['release_state'] == 'blocked-until-regression-tested'
        for model in payload['files']:
            assert model['sha256'] == ''
            assert model['size'] == 0
