# Push this repaired repository to GitHub

## Recommended: WSL / Linux

This repository contains upstream directories that differ only by letter case. Do not unpack / maintain the repository in a normal Windows case-insensitive directory.

If you use the archive that includes `.git`, extract it inside WSL/Linux, then:

```bash
cd ~/vjepa2-ac-codesign
git status
python scripts/validate_repository_layout.py

git add .gitattributes .gitignore \
  AGENT_REVIEW_NOTES.md \
  INTEGRATION_CHECKLIST.md \
  README_BENCHMARK_ADAPTER.md \
  PUSH_TO_GITHUB.md \
  requirements-adapter.txt \
  configs/benchmark_protocol.example.yaml \
  configs/precision_factories.example.yaml \
  scripts vjepa2_bench

git status
git commit -m "Add V-JEPA2-AC benchmark and profiling adapter"
git push origin main
```

Before the commit, `git status` should show only the benchmark/deployment additions and `.gitignore`; it should **not** show hundreds of modified upstream files.

## Server later

When server resources are available:

```bash
git clone https://github.com/<your-user>/vjepa2-ac-codesign.git
cd vjepa2-ac-codesign
pip install -r requirements.txt
pip install -r requirements-adapter.txt
python scripts/validate_repository_layout.py
```

Then download the official V-JEPA2-AC checkpoint separately and run:

```bash
python scripts/check_vjepa2_ac_adapter.py \
  --repo-root . \
  --checkpoint /path/to/vjepa2-ac-vitg.pt
```
