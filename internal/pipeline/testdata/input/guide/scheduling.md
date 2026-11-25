# Scheduling jobs

How the scheduler works
-----------------------

The scheduler keeps a queue of pending jobs ordered by priority and then by submission time. Every second it looks at the head of the queue and asks each idle worker whether it can take the job. A worker accepts when it has the labels the job requires and enough free slots. If no worker accepts, the job stays at the head and the scheduler tries again on the next tick. Jobs behind it are not considered until it has been placed, which keeps a large job from being starved by a stream of small ones. This strict ordering is simple to reason about but it means one unplaceable job can hold up the queue. To guard against that, a job that has been at the head for longer than the stall timeout is moved to the parked list and an alert is raised. Parked jobs are retried every minute. Operators can also park and unpark jobs by hand, which is useful during maintenance windows. When a job is placed the scheduler records the decision in the audit log together with the worker it chose and the reason the other candidates were rejected.

Retries use exponential backoff. The first retry happens after two seconds, the second after four, and so on up to a ceiling of five minutes.

Database migrations are a different matter entirely. Before upgrading, export the schema with the dump tool and keep the file somewhere safe. The migration runner applies numbered SQL files in order and stores the last applied number in a metadata table.

Each migration runs inside a transaction when the database engine supports transactional schema changes. If a migration fails the transaction is rolled back and the runner stops.

Back to the [documentation index](/index.md).

## Priorities

### Named levels

Jobs carry one of four named priority levels.

1. `critical` is reserved for system jobs.
2. `high` is for user-facing work.
3. `normal` is the default.
4. `low` is for batch work that can wait.

### Numeric priorities

A job may instead carry a number from 0 to 1000. Named levels map onto 900, 700, 500 and 100.

## Submitting a job

```yaml
# job.yaml
name: nightly-report
priority: normal
labels: [linux, gpu]
retries: 3
command: ["python", "report.py", "--date", "yesterday"]
env:
  REPORT_BUCKET: s3://reports
  LOG_LEVEL: info
timeout: 45m
```

Submit it with `tidepool submit job.yaml`. The command prints the job id.

Back to the [documentation index](/index.md).

## Generated client

The block below is generated. It is long on purpose and must never be cut.

```go
package client

type Job struct {
	ID        string
	Name      string
	Priority  int
	Labels    []string
	Retries   int
	Command   []string
	Env       map[string]string
	Timeout   time.Duration
	Submitted time.Time
	Started   time.Time
	Finished  time.Time
	Worker    string
	ExitCode  int
	Attempt   int
	ParkedAt  time.Time
	Reason    string
}

func (c *Client) Submit(ctx context.Context, job Job) (string, error) {
	body, err := json.Marshal(job)
	if err != nil {
		return "", fmt.Errorf("encoding job: %w", err)
	}
	resp, err := c.post(ctx, "/v1/jobs", body)
	if err != nil {
		return "", fmt.Errorf("submitting job: %w", err)
	}
	return resp.ID, nil
}
```

## Empty tail
