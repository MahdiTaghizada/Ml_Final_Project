"""Minimal runner so the tests work without pytest:  python -m tests.run_tests"""
import sys
import time
import traceback

from . import test_models as T
from . import test_bonuses as TB


def main() -> int:
    failed = 0
    for mod in (T, TB):
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t = time.time()
            try:
                getattr(mod, name)()
                print(f"PASS  {name}  ({time.time() - t:.1f}s)")
            except Exception:
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{'ALL TESTS PASSED' if not failed else str(failed) + ' TEST(S) FAILED'}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
