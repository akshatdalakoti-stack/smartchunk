# Command reference

## tidepool submit

Submit a job description to the scheduler.

| Flag         | Default  | Meaning                               |
|--------------|----------|---------------------------------------|
| `--wait`     | `false`  | Block until the job finishes          |
| `--priority` | `normal` | Override the priority in the file     |
| `--dry-run`  | `false`  | Validate the file without submitting  |

Back to the [documentation index](/index.md).

## Self-check

Run the self-check. It starts a scheduler on a random port, submits one job, waits for it to finish and shuts down again. If the check fails, the output names the step that went wrong and the log file to read.

```sh
tidepool selfcheck
```

A passing check ends with the line `selfcheck: ok` and exit code 0.

Back to the [documentation index](/index.md).

## tidepool park

Move a job to the parked list. Parked jobs are skipped by the scheduler until they are unparked or the retry timer fires.

Back to the [documentation index](/index.md).

## Environment variables

| Variable            | Default           | Purpose                              |
|---------------------|-------------------|--------------------------------------|
| `TIDEPOOL_CONFIG`   | `./tidepool.yaml` | Path of the configuration file       |
| `TIDEPOOL_DATA`     | `./data`          | Directory for the queue and the log  |
| `TIDEPOOL_LOG`      | `info`            | One of debug, info, warn or error    |
| `TIDEPOOL_NO_COLOR` | unset             | Disable coloured terminal output     |

## Minimal configuration

```yaml
listen: 0.0.0.0:7400
data_dir: /var/lib/tidepool
stall_timeout: 10m
workers:
  - name: alpha
    slots: 4
    labels: [linux]
  - name: beta
    slots: 8
    labels: [linux, gpu]
```

## Exit codes

- `0` success
- `1` the job failed
- `2` the job description was invalid
- `3` the scheduler could not be reached

[scheduler guide]: guide/scheduling.md
