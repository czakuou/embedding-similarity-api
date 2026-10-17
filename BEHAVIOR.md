# Document Similarity Service: Behaviour Scenarios

> Business processes of the service, written in Gherkin and implemented as plain `pytest` tests (no `pytest-bdd`).
> These are **integration tests**: each scenario maps 1:1 to a test named `test_given_..._when_..._then_...` in
> `tests/behavior/`, run against the real model, a real Redis, and the real worker loop -- not test doubles.
> They describe observable business behaviour only; the exact status codes, header names, and field names that
> realise each outcome are the API contract in [`SPEC.md`](SPEC.md) §4, not repeated here.

## How these scenarios run

| Aspect | Setup | Why |
|---|---|---|
| HTTP | FastAPI `TestClient` against the real app factory | Exercises routing, validation, Problem Details |
| Redis | **Real Redis via testcontainers** (session-scoped, `FLUSHDB` between tests, `noeviction`) | Correctness relies on real Redis semantics (`MULTI`, consumer groups, `XAUTOCLAIM`, TTL) |
| Worker | One iteration of the `worker.py` loop (`process_batch`) driven in-process | Deterministic: no background threads or sleeps in tests |
| Model | **The real pinned model** (`encoder.py`'s `SentenceTransformerEncoder`), loaded once per test session | These are integration tests: a fake would prove the plumbing works, not that the service works |
| Time & randomness | Injected clock and RNG | Expiry and retry timing are testable without waiting |
| Test overrides | Small rate limit, queue depth and visibility timeout per scenario | Scenarios run in milliseconds, aside from the one real model call per analysis |

## Priorities

| # | Scenario | Business requirement | Priority |
|---|---|---|---|
| 1 | Happy path: submit, poll, get result | 3, 4 | 🔴 Must |
| 2 | Invalid input is rejected before queueing | 1 | 🔴 Must |
| 3 | One client cannot consume another client's capacity | 2 | 🔴 Must |
| 4 | Full queue rejects new work, accepted work still completes | 2 | 🔴 Must |
| 5 | An analysis that can never succeed fails alone, with a reason | 4 | 🔴 Must |
| 6 | Expired and unknown results | 5 | 🟡 Should |
| 7 | Client deletes its result | 5 | 🟡 Should |
| 8 | A crashed worker's work is completed by another | 4 | 🟢 Nice (first to cut) |

---

## Feature: Asynchronous similarity analysis

```gherkin
Feature: Asynchronous similarity analysis
  As an API client
  I want to submit a document with a query and collect the result later
  So that my request is accepted quickly even when processing is busy

  Background:
    Given the service is running
    And client "client-a" is identified to the service

  # 1 ── Must ──────────────────────────────────────────────────────────────
  Scenario: Client submits a document and receives the similarity result
    Given a valid text document containing "The quick brown fox jumps over the lazy dog."
    When "client-a" submits the document with the query "fox over dog"
    Then the submission is accepted for later processing
    And the client is told where to check on it and how long to wait before doing so
    And the client is told the submission will remain valid for 24 hours
    When the worker processes the queue
    And "client-a" checks on the analysis
    Then the analysis is complete
    And a single similarity score is returned, between -1 and 1
    And the client is told whether the document had to be shortened to fit the model

  # 2 ── Must ──────────────────────────────────────────────────────────────
  Scenario Outline: Invalid input is rejected and nothing is queued
    Given <file>
    When "client-a" submits it with the query <query>
    Then the submission is rejected with a reason the client can act on
    And no work is queued because of it

    Examples:
      | file                                       | query           | reason                                   |
      | a file that is not a text document         | "fox over dog"  | the file type is not supported           |
      | a text document larger than the size limit | "fox over dog"  | the file is too large                    |
      | a text document that is not valid UTF-8    | "fox over dog"  | the content could not be validated       |
      | an empty text document                     | "fox over dog"  | the content could not be validated       |
      | a valid text document                      | ""              | the content could not be validated       |
      | a valid text document                      | 1001 characters | the content could not be validated       |

  # 3 ── Must ──────────────────────────────────────────────────────────────
  Scenario: One client cannot consume another client's capacity
    Given "client-a" has already used up its fair share of capacity for the current window
    When "client-a" submits another analysis in that same window
    Then the submission is rejected as too frequent
    And the client is told how long to wait before trying again
    When "client-b" submits a valid analysis in that same window
    Then the submission is accepted for later processing

  # 4 ── Must ──────────────────────────────────────────────────────────────
  Scenario: A full queue rejects new work but accepted work still completes
    Given the service is at its capacity limit for waiting work
    When a client submits another valid analysis
    Then the submission is rejected as the service being busy
    And the client is told how long to wait before trying again
    When the worker processes the queue
    Then all previously accepted analyses have completed
    And a new submission is now accepted

  # 5 ── Must ──────────────────────────────────────────────────────────────
  Scenario: An analysis that can never succeed fails alone and reports why
    Given a batch containing several analyses, one of which cannot ever be processed successfully
    When the worker processes the batch
    Then the healthy analyses complete successfully with a result
    And the unprocessable analysis is still being retried, with one attempt recorded so far
    When the unprocessable analysis has been retried as many times as the service allows
    Then that analysis is marked as permanently failed
    And the client is given a reason for the failure, with the number of attempts made
    And the failure reason contains no internal technical detail
    And the failed analysis is no longer occupying space in the queue

  # 6 ── Should ────────────────────────────────────────────────────────────
  Scenario: An expired result is reported as gone
    Given "client-a" received a completed analysis
    When the result's retention period has passed
    And "client-a" checks on the analysis
    Then the client is told the result is gone, not simply missing

  Scenario: An unknown or foreign analysis is not found
    When "client-a" checks on an analysis that was never submitted
    Then the client is told it does not exist
    Given "client-b" owns a completed analysis
    When "client-a" checks on that analysis
    Then "client-a" is told it does not exist, with no indication that it belongs to someone else

  # 7 ── Should ────────────────────────────────────────────────────────────
  Scenario: Client deletes a finished result
    Given "client-a" owns a completed analysis
    When "client-a" deletes the analysis
    Then the deletion succeeds
    When "client-a" checks on the analysis
    Then the client is told it does not exist

  Scenario: Client cannot delete an analysis that is still in progress
    Given "client-a" owns an analysis that is still waiting to be processed
    When "client-a" deletes the analysis
    Then the deletion is refused because the work is still in progress
    And the analysis still exists, still waiting to be processed

  # 8 ── Nice ──────────────────────────────────────────────────────────────
  Scenario: Work of a crashed worker is completed by another worker
    Given one worker has picked up an analysis but crashed before finishing it
    When enough time has passed that the work is considered abandoned
    And another worker picks up the abandoned work
    Then the analysis completes successfully
    And the record reflects that it took a second attempt to complete
```

---

## Test mapping

| # | Test function |
|---|---|
| 1 | `test_given_valid_document_when_submitted_and_processed_then_result_is_single_float` |
| 2 | `test_given_invalid_input_when_submitted_then_rejected_and_not_queued` (parametrised over the Examples table) |
| 3 | `test_given_client_over_limit_when_submitting_then_429_and_other_client_unaffected` |
| 4 | `test_given_full_queue_when_submitting_then_503_and_accepted_work_completes` |
| 5 | `test_given_unprocessable_analysis_in_batch_when_processed_then_only_it_fails_with_reason` |
| 6a | `test_given_expired_analysis_when_polled_then_410` |
| 6b | `test_given_unknown_or_foreign_analysis_when_polled_then_404` |
| 7a | `test_given_done_analysis_when_deleted_then_204_and_then_404` |
| 7b | `test_given_in_progress_analysis_when_deleted_then_409` |
| 8 | `test_given_crashed_worker_when_message_idle_then_reclaimed_and_completed` |

> Test function names keep the concrete status codes (e.g. `429`, `503`) because they are code, not business
> narrative; the exact mapping from each scenario's outcome to a status code lives in SPEC.md §4.
