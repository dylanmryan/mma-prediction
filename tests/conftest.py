import os

# torch and xgboost each bundle an OpenMP runtime; loading both in one
# process segfaults on this platform unless OpenMP threading is pinned.
# Must be set before either library is imported, hence conftest.
os.environ.setdefault("OMP_NUM_THREADS", "1")
