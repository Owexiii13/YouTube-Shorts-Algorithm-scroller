#!/usr/bin/env bash
set -euo pipefail

# Push the rebased/resolved `work` branch to origin.
# Use this from a local/authenticated shell if PR UI still shows stale conflicts.

git checkout work
git push --force-with-lease origin work

echo "✅ Pushed 'work' branch to origin. Refresh PR page."
