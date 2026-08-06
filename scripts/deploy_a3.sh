#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Deploy the committed local HEAD to an immutable release directory on a3.

Usage: scripts/deploy_a3.sh [options]

Options:
  --host HOST             SSH host (default: a3)
  --runtime-root PATH     Remote release root (default: ~/autodl-tmp/EdgeCloudRuntime)
  --shared-root PATH      Existing data/model/output root (default: ~/autodl-tmp/EdgeCloud)
  --no-activate           Verify the release but do not update the current symlink
  --dry-run               Build locally and print the intended deployment only
  -h, --help              Show this help

The script never invokes git on the remote host. Local Git is the source of
truth; only committed files from `git archive HEAD` are deployed.
EOF
}

remote_host="a3"
runtime_root='~/autodl-tmp/EdgeCloudRuntime'
shared_root='~/autodl-tmp/EdgeCloud'
activate=1
dry_run=0

while (($#)); do
  case "$1" in
    --host)
      remote_host=${2:?missing value for --host}
      shift 2
      ;;
    --runtime-root)
      runtime_root=${2:?missing value for --runtime-root}
      shift 2
      ;;
    --shared-root)
      shared_root=${2:?missing value for --shared-root}
      shift 2
      ;;
    --no-activate)
      activate=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

if [[ -n $(git status --porcelain) ]]; then
  echo "refusing to deploy: local worktree is not clean" >&2
  git status --short >&2
  exit 1
fi

commit=$(git rev-parse HEAD)
short_commit=$(git rev-parse --short=12 HEAD)
case "$commit" in
  (*[!0-9a-f]*|'')
    echo "unexpected commit id: $commit" >&2
    exit 1
    ;;
esac

deploy_tmp=$(mktemp -d "${TMPDIR:-/tmp}/edgecloud-deploy.XXXXXX")
cleanup() {
  rm -rf "$deploy_tmp"
}
trap cleanup EXIT

stage_dir="$deploy_tmp/release"
archive_path="$deploy_tmp/edgecloud-${commit}.tar.gz"
mkdir -p "$stage_dir"
git archive "$commit" | tar -xf - -C "$stage_dir"
printf '%s\n' "$commit" > "$stage_dir/DEPLOYED_COMMIT"
# macOS can attach a provenance xattr while extracting even a clean Git
# archive; remove it so GNU tar on the Linux host sees a portable archive.
if command -v xattr >/dev/null 2>&1; then
  xattr -rc "$stage_dir"
fi

(
  cd "$stage_dir"
  find . -type f ! -name MANIFEST.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 > MANIFEST.sha256
)
COPYFILE_DISABLE=1 tar -czf "$archive_path" -C "$stage_dir" .

echo "commit:       $commit"
echo "release:      $runtime_root/releases/$commit"
echo "archive:      $archive_path"
echo "archive size: $(du -h "$archive_path" | awk '{print $1}')"

if ((dry_run)); then
  echo "dry-run: no remote changes made"
  exit 0
fi

remote_archive="/tmp/edgecloud-release-${commit}.tar.gz"
scp "$archive_path" "${remote_host}:${remote_archive}"

ssh "$remote_host" bash -s -- \
  "$runtime_root" "$shared_root" "$commit" "$remote_archive" "$activate" <<'REMOTE'
set -euo pipefail

runtime_root=$1
shared_root=$2
commit=$3
remote_archive=$4
activate=$5

# Expand the deliberately supported leading ~/ without eval.
runtime_root=${runtime_root/#\~/$HOME}
shared_root=${shared_root/#\~/$HOME}
release_dir="$runtime_root/releases/$commit"
temporary_release="$runtime_root/releases/.${commit}.tmp.$$"

mkdir -p "$runtime_root/releases" "$runtime_root/shared/checkpoints" \
  "$runtime_root/shared/outputs"

if [[ -e "$release_dir" ]]; then
  if [[ ! -f "$release_dir/DEPLOYED_COMMIT" ]] \
      || [[ $(<"$release_dir/DEPLOYED_COMMIT") != "$commit" ]]; then
    echo "existing release does not match commit marker: $release_dir" >&2
    exit 1
  fi
  echo "release already exists; verifying immutable contents"
else
  mkdir "$temporary_release"
  tar -xzf "$remote_archive" -C "$temporary_release"
  mv "$temporary_release" "$release_dir"
fi
rm -f "$remote_archive"

(
  cd "$release_dir"
  sha256sum -c --quiet MANIFEST.sha256
)

# Link large, untracked runtime assets without replacing tracked README files.
for asset_group in data models; do
  source_dir="$shared_root/$asset_group"
  target_dir="$release_dir/$asset_group"
  mkdir -p "$target_dir"
  if [[ -d "$source_dir" ]]; then
    while IFS= read -r -d '' source_entry; do
      name=${source_entry##*/}
      if [[ ! -e "$target_dir/$name" && ! -L "$target_dir/$name" ]]; then
        ln -s "$source_entry" "$target_dir/$name"
      fi
    done < <(find "$source_dir" -mindepth 1 -maxdepth 1 -print0)
  fi
done

checkpoint_source="$shared_root/module_edge_perception/checkpoints"
checkpoint_target="$release_dir/module_edge_perception/checkpoints"
if [[ -d "$checkpoint_source" && ! -e "$checkpoint_target" ]]; then
  ln -s "$checkpoint_source" "$checkpoint_target"
fi
ln -sfn "$runtime_root/shared/outputs" "$release_dir/outputs"

if ((activate)); then
  next_link="$runtime_root/.current.$$.tmp"
  ln -s "$release_dir" "$next_link"
  mv -Tf "$next_link" "$runtime_root/current"
  echo "activated: $runtime_root/current -> $release_dir"
else
  echo "verified without activation: $release_dir"
fi

echo "deployed commit: $(<"$release_dir/DEPLOYED_COMMIT")"
REMOTE

echo "deployment complete: $short_commit"
