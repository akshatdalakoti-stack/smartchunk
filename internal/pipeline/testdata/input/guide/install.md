---
title: Installing Tidepool
sidebar: guide
---

# Installing Tidepool

Tidepool is a small job scheduler. This page covers every supported way of getting it onto a machine, from a single binary to a container image.

Back to the [documentation index](/index.md).

## Requirements

Tidepool needs very little to run. The scheduler itself is a static binary with no runtime dependencies, so the list below is about the machine and not about libraries.

- A 64-bit Linux, macOS or Windows host
- At least 128 MB of free memory
  - 512 MB if you enable the web console
  - 1 GB for clusters with more than fifty workers
- Network access to the workers on port 7400

Back to the [documentation index](/index.md).

## Install from a release

Download the archive for your platform and unpack it somewhere on your `PATH`:

```sh
curl -LO https://example.com/tidepool/v2.4.0/tidepool-linux-amd64.tar.gz
tar -xzf tidepool-linux-amd64.tar.gz
sudo mv tidepool /usr/local/bin/
tidepool --version
```

The last command should print `tidepool 2.4.0`. If it prints an older number, an earlier copy is shadowing the new one.

Back to the [documentation index](/index.md).

## Install with a package manager

| Platform | Command                         | Notes                      |
|----------|---------------------------------|----------------------------|
| macOS    | `brew install tidepool`         | Updated within a day       |
| Debian   | `apt install tidepool`          | Needs the vendor repo      |
| Windows  | `winget install tidepool`       | Installs to Program Files  |
| Arch     | `pacman -S tidepool`            | Community package          |

Package managers lag behind releases by a few hours. Check the installed version before reporting a bug that may already be fixed.

Back to the [documentation index](/index.md).

## Run in a container

> The container image runs as an unprivileged user. Mount your configuration read-only and give the data directory its own volume.

```sh
docker run -d --name tidepool \
  -v "$PWD/tidepool.yaml:/etc/tidepool.yaml:ro" \
  -v tidepool-data:/var/lib/tidepool \
  -p 7400:7400 example/tidepool:2.4.0
```

Back to the [documentation index](/index.md).

## Self-check

Run the self-check. It starts a scheduler on a random port, submits one job, waits for it to finish and shuts down again. If the check fails, the output names the step that went wrong and the log file to read.

```sh
tidepool selfcheck
```

A passing check ends with the line `selfcheck: ok`.

Back to the [documentation index](/index.md).
