import pytest
import tempfile
import data
import subprocess
import os


@pytest.mark.slow
def test_download_multiwoz_21(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv('DATASETS_PATH', tmpdir)
        scripts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
        subprocess.run([os.path.join(scripts_path, 'download_convlab_multiwoz.py')])

        d = data.load_dataset('multiwoz-2.1-train')
        assert len(d) == 56689
        d = data.load_dataset('multiwoz-2.1-val')
        assert len(d) == 7365
        d = data.load_dataset('multiwoz-2.1-test')
        assert len(d) == 7372
        assert d.database(dict(train={'departure': 'cambridge'}))['train'] > 0
        assert d.lexicalizer('test [name]', dict(hotel=(1, [{'name': '1'}])), belief=dict(hotel={'name': '1'})) == 'test 1'


@pytest.mark.slow
def test_download_multiwoz_20(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv('DATASETS_PATH', tmpdir)
        scripts_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
        subprocess.run([os.path.join(scripts_path, 'download_multiwoz.py')])

        d = data.load_dataset('multiwoz-2.0-train')
        assert len(d) == 56620
        d = data.load_dataset('multiwoz-2.0-val')
        assert len(d) == 7365
        d = data.load_dataset('multiwoz-2.0-test')
        assert len(d) == 7368
        assert d.database(dict(train={'departure': 'cambridge'}))['train'] > 0
        assert d.lexicalizer('test [hotel_name]', dict(hotel=(1, [{'name': '1'}])), belief=dict(hotel={'name': '1'})) == 'test 1'
