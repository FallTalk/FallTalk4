# tests/test_config.py
import os, json, tempfile, pytest
from unittest.mock import patch

def test_config_key_has_key_and_default():
    from src.config.config import ConfigKey
    k = ConfigKey("my_key", "default_val")
    assert k.key == "my_key"
    assert k.default == "default_val"

def test_config_get_returns_default_when_missing():
    from src.config.config import Config, ConfigKey
    with tempfile.TemporaryDirectory() as d:
        with patch("src.config.config.get_app_root", return_value=d):
            os.makedirs(os.path.join(d, "config"), exist_ok=True)
            c = Config()
            k = ConfigKey("missing_key", "hello")
            assert c.get(k) == "hello"

def test_config_set_and_get():
    from src.config.config import Config, ConfigKey
    with tempfile.TemporaryDirectory() as d:
        with patch("src.config.config.get_app_root", return_value=d):
            os.makedirs(os.path.join(d, "config"), exist_ok=True)
            c = Config()
            k = ConfigKey("test_key", 0, int)
            c.set(k, 42)
            assert c.get(k) == 42

def test_config_persists_across_instances():
    from src.config.config import Config, ConfigKey
    with tempfile.TemporaryDirectory() as d:
        with patch("src.config.config.get_app_root", return_value=d):
            os.makedirs(os.path.join(d, "config"), exist_ok=True)
            k = ConfigKey("persist_key", "orig", str)
            c1 = Config()
            c1.set(k, "saved")
            c2 = Config()
            assert c2.get(k) == "saved"

def test_config_reset_restores_defaults():
    from src.config.config import Config, ConfigKey
    with tempfile.TemporaryDirectory() as d:
        with patch("src.config.config.get_app_root", return_value=d):
            os.makedirs(os.path.join(d, "config"), exist_ok=True)
            c = Config()
            k = ConfigKey("reset_key", "orig_default", str)
            c.set(k, "changed")
            c.reset()
            assert c.get(k) == "orig_default"

def test_config_type_coercion():
    from src.config.config import Config, ConfigKey
    with tempfile.TemporaryDirectory() as d:
        with patch("src.config.config.get_app_root", return_value=d):
            os.makedirs(os.path.join(d, "config"), exist_ok=True)
            # Simulate JSON storing a float as string
            settings_path = os.path.join(d, "config", "settings.json")
            with open(settings_path, "w") as f:
                json.dump({"coerce_key": "3.14"}, f)
            c = Config()
            k = ConfigKey("coerce_key", 1.0, float)
            assert c.get(k) == 3.14

def test_config_ini_migration(tmp_path):
    """Old QSettings INI format is migrated on first load."""
    from src.config.config import Config, ConfigKey
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    # Write fake QSettings INI file
    ini = config_dir / "configv2.json"
    ini.write_text("[General]\ndevice=cpu\nspeed=1.5\n")
    with patch("src.config.config.get_app_root", return_value=str(tmp_path)):
        c = Config()
        k_device = ConfigKey("device", "cuda", str)
        assert c.get(k_device) == "cpu"
