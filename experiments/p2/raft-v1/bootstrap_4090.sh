#!/usr/bin/env bash
set -euo pipefail
base=/root/autodl-tmp/RAFT-Benchmark-v1
mkdir -p "$base/source" "$base/checkpoints" "$base/evidence" "$base/deps"
archive="$base/evidence/raft-upstream-2888e15.tar.gz"
if [ ! -f "$archive" ]; then
  curl -fL --retry 3 --output "$archive" https://codeload.github.com/princeton-vl/RAFT/tar.gz/2888e15a51fa41140771d3f498ed8023cff098d1
fi
if [ ! -d "$base/source/RAFT" ]; then
  mkdir "$base/source/RAFT"
  tar -xzf "$archive" -C "$base/source/RAFT" --strip-components=1
fi
checkpoint="$base/checkpoints/raft-things.pth"
if [ ! -f "$checkpoint" ]; then
  curl -fL --retry 3 --output "$checkpoint" https://hf-mirror.com/ddrfan/RAFT/resolve/main/raft-things.pth
fi
printf '%s  %s\n' fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1 "$checkpoint" | sha256sum -c -
printf '%s  %s\n' 5a815fe352c27c8fe94e19e5236bef3856ae22b9b9ee0b38ec406ef213faaaff "$base/source/RAFT/demo-frames/frame_0016.png" | sha256sum -c -
printf '%s  %s\n' 6c6f31f001dadddb9578dcc03f8bd53a045fe752311a53b9f9bcd1e1a1b2a329 "$base/source/RAFT/demo-frames/frame_0017.png" | sha256sum -c -
sha256sum "$archive" "$checkpoint" "$base/source/RAFT/demo-frames/frame_0016.png" "$base/source/RAFT/demo-frames/frame_0017.png" > "$base/evidence/material-sha256.txt"
du -h "$archive" "$checkpoint"
