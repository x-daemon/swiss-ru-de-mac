import sys
import types

import pytest


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch):
    """Never touch the real system keychain in tests."""
    store = {}

    class PasswordDeleteError(Exception):
        pass

    module = types.ModuleType("keyring")
    module.errors = types.SimpleNamespace(PasswordDeleteError=PasswordDeleteError)
    module.get_password = lambda service, name: store.get((service, name))
    module.set_password = lambda service, name, value: store.__setitem__((service, name), value)

    def delete_password(service, name):
        if (service, name) not in store:
            raise PasswordDeleteError(name)
        del store[(service, name)]

    module.delete_password = delete_password
    monkeypatch.setitem(sys.modules, "keyring", module)
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "DEEPL_AUTH_KEY"):
        monkeypatch.delenv(var, raising=False)
    return store
