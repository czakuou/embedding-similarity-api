# Document Similarity Service: Behaviour Scenarios

> Business processes of the service, written in Gherkin and implemented as plain `pytest` tests (no `pytest-bdd`).
> Each scenario maps 1:1 to a test named `test_given_..._when_..._then_...` in `tests/behavior/`.
> Design context: [`SPEC.md`](SPEC.md).

## How these scenarios run

| Aspect | Setup | Why |
|---|---|---|
| HTTP | FastAPI `TestClient` against the real app factory | Exercises routing, validation, Problem Details |
| Redis | **Real Redis via testcontainers** (session-scoped, `FLUSHDB` between tests, `noeviction`) | Correctness relies on real Redis semantics (`MULTI`, consumer groups, `XAUTOCLAIM`, TTL) |
| Worker | One iteration of the `worker.py` loop (`process_batch`) driven in-process | Deterministic: no background threads or sleeps in tests |
| Model | **Fake encoder** implementing the `Encoder` Protocol from `encoder.py`: deterministic vectors from a text hash; a document containing `__POISON__` raises | Fast, no torch; the real model is covered by the benchmark |
| Time & randomness | Injected clock and RNG | Expiry (`410`) and jitter (`Retry-After`) are testable without waiting |
| Test overrides | Small `RATE_LIMIT`, `MAX_QUEUE_DEPTH`, `VISIBILITY_TIMEOUT_S` per scenario | Scenarios run in milliseconds |

## Priorities

| # | Scenario | Business requirement | Priority |
|---|---|---|---|
| 1 | Happy path: submit, poll, get result | 3, 4 | 🔴 Must |
| 2 | Invalid input is rejected before queueing | 1 | 🔴 Must |
| 3 | One client cannot consume another client's capacity | 2 | 🔴 Must |
| 4 | Full queue rejects new work, accepted work still completes | 2 | 🔴 Must |
| 5 | A poisoned document fails alone, with a reason | 4 | 🔴 Must |
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
    And client "client-a" is identified by the "X-Client-Id" header

  # 1 ── Must ──────────────────────────────────────────────────────────────
  Scenario: Client submits a document and receives the similarity result
    Given a UTF-8 file "fox.txt" containing "The quick brown fox jumps over the lazy dog."
    When "client-a" submits the file with the query "fox over dog"
    Then the response status is 202
    And the response has a "Location" header pointing to the analysis
    And the response has a "Retry-After" header
    And the body contains an "id", status "queued" and an "expires_at" 24 hours after "created_at"
    When the worker processes the queue
    And "client-a" polls the analysis
    Then the response status is 200
    And the status is "done"
    And the result is a list with exactly one float between -1 and 1
    And "document_truncated" is false

  # 2 ── Must ──────────────────────────────────────────────────────────────
  Scenario Outline: Invalid input is rejected and nothing is queued
    Given <file>
    When "client-a" submits it with the query <query>
    Then the response status is <status>
    And the body is a Problem Details document with code "<code>"
    And the queue depth is unchanged

    Examples:
      | file                                     | query           | status | code                   |
      | a file "data.csv" with valid UTF-8 text  | "fox over dog"  | 415    | UNSUPPORTED_MEDIA_TYPE |
      | a "doc.txt" file larger than the limit   | "fox over dog"  | 413    | PAYLOAD_TOO_LARGE      |
      | a "doc.txt" file with invalid UTF-8      | "fox over dog"  | 422    | VALIDATION_ERROR       |
      | an empty "doc.txt" file                  | "fox over dog"  | 422    | VALIDATION_ERROR       |
      | a valid "doc.txt" file                   | ""              | 422    | VALIDATION_ERROR       |
      | a valid "doc.txt" file                   | 1001 characters | 422    | VALIDATION_ERROR       |

  # 3 ── Must ──────────────────────────────────────────────────────────────
  Scenario: One client cannot consume another client's capacity
    Given the per-client rate limit is 20 requests per second
    And "client-a" has submitted 20 valid analyses within the current second
    When "client-a" submits a 21st analysis in the same second
    Then the response status is 429
    And the response has a "Retry-After" header between the base and twice the base
    And the body is a Problem Details document with code "RATE_LIMITED"
    When "client-b" submits a valid analysis in the same second
    Then the response status is 202

  # 4 ── Must ──────────────────────────────────────────────────────────────
  Scenario: A full queue rejects new work but accepted work still completes
    Given the maximum queue depth is 3
    And 3 analyses have been accepted and are waiting in the queue
    When a client submits another valid analysis
    Then the response status is 503
    And the response has a "Retry-After" header between the base and twice the base
    And the body is a Problem Details document with code "OVERLOADED"
    When the worker processes the queue
    Then all 3 previously accepted analyses have status "done"
    And a new submission is accepted with status 202

  # 5 ── Must ──────────────────────────────────────────────────────────────
  Scenario: A poisoned document fails alone and reports why
    Given the maximum number of attempts is 3
    And 4 analyses are queued in one batch, one of which contains "__POISON__"
    When the worker processes the batch
    Then the 3 healthy analyses have status "done" with a result
    And the poisoned analysis has status "processing" with 1 attempt
    When the poisoned message is reclaimed and processed until attempts are exhausted
    Then the poisoned analysis has status "failed"
    And its error has code "PROCESSING_FAILED" and "attempts" equal to 3
    And the error message contains no stack trace
    And the message is no longer in the queue

  # 6 ── Should ────────────────────────────────────────────────────────────
  Scenario: An expired result is reported as gone
    Given "client-a" received a "done" analysis
    When 24 hours and 1 second have passed
    And "client-a" polls the analysis
    Then the response status is 410

  Scenario: An unknown or foreign analysis is not found
    When "client-a" polls an analysis id that was never created
    Then the response status is 404
    Given "client-b" owns a "done" analysis
    When "client-a" polls that analysis
    Then the response status is 404

  # 7 ── Should ────────────────────────────────────────────────────────────
  Scenario: Client deletes a finished result
    Given "client-a" owns a "done" analysis
    When "client-a" deletes the analysis
    Then the response status is 204
    When "client-a" polls the analysis
    Then the response status is 404

  Scenario: Client cannot delete an analysis that is still in progress
    Given "client-a" owns a "queued" analysis
    When "client-a" deletes the analysis
    Then the response status is 409
    And the analysis still exists with status "queued"

  # 8 ── Nice ──────────────────────────────────────────────────────────────
  Scenario: Work of a crashed worker is completed by another worker
    Given the visibility timeout is 1 second
    And worker "w1" has received an analysis but crashed before acknowledging it
    When more than 1 second has passed
    And worker "w2" reclaims idle messages and processes them
    Then the analysis has status "done"
    And its "attempts" is 2
```

---

## Test mapping

| # | Test function |
|---|---|
| 1 | `test_given_valid_document_when_submitted_and_processed_then_result_is_single_float` |
| 2 | `test_given_invalid_input_when_submitted_then_rejected_and_not_queued` (parametrised over the Examples table) |
| 3 | `test_given_client_over_limit_when_submitting_then_429_and_other_client_unaffected` |
| 4 | `test_given_full_queue_when_submitting_then_503_and_accepted_work_completes` |
| 5 | `test_given_poisoned_document_in_batch_when_processed_then_only_it_fails_with_reason` |
| 6a | `test_given_expired_analysis_when_polled_then_410` |
| 6b | `test_given_unknown_or_foreign_analysis_when_polled_then_404` |
| 7a | `test_given_done_analysis_when_deleted_then_204_and_then_404` |
| 7b | `test_given_in_progress_analysis_when_deleted_then_409` |
| 8 | `test_given_crashed_worker_when_message_idle_then_reclaimed_and_completed` |
