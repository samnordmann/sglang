# Reading the serving measurements

The [first pair](pair-1.json) and [second pair](pair-2.json) used different GPU pairs of hosts. Each used one prefill GPU on the first host and two decode GPUs on the second. Both use the same baseline. The candidate revisions are listed separately. The first two exports include transfer diagnostics as well as result callbacks. A [third run on the second host pair](pair-2-minimal.json) used the same final candidate and hardware with only the two buffered scheduler-result hooks. Its 64-token concurrency-four cell also failed the token controls; the other three cells passed.

Each JSON reports the number of distinct prompts with any token mismatch and the number for which the two independent direct P1→D2 references disagreed. Counts cover 26 waves including two warmups and count each input once per phase. The 64-token concurrency-four cell on the second pair failed qualification when the direct controls disagreed on one prompt. Its timing statistics are omitted. The two 2,048-token cells qualified on both pairs.

For qualified cells, `per_request` gives the observed p50 and p95 for 24 measured waves per order (24 requests at depth one, 96 at depth four). Each endpoint interval runs from prefill result to the later first decode callback across ranks. Lower and upper quantiles include host clock uncertainty. HTTP uses the client's single monotonic clock. Quantiles linearly interpolate the sorted observations at `(n−1)p`.

`AB` is baseline then candidate; `BA` is candidate then baseline. Deltas use candidate minus baseline. Each row of `per_wave_deltas_us` contains three relative durations in microseconds, ordered by the preassigned repeat index:

1. Lower bound of the difference in the time from the wave's earliest prefill result to its latest correct first decode across both ranks and all requests.
2. Upper bound of that same difference.
3. Difference in first client departure to last full HTTP response in the wave.

`paired_median_delta_us` bounds the observed median of 24 paired deltas. The separate order-statistic interval uses the seventh and eighteenth ordered delta bounds. Its nominal coverage assumes independent paired waves within an order. `active_http_req_per_s` excludes time between waves. Original clocks, prompt identities, endpoint identifiers and runtime logs are omitted.

To produce the same export from a new minimal run, invoke the offline validator and exporter with the baseline and candidate git revisions that generated the results.

```bash
python validate.py "$OUT"
python export_evidence.py "$OUT/buffered-endpoint-results.json" pair-2-minimal.json \
  --pair 2 --baseline "$BASE_COMMIT" --candidate "$CANDIDATE_COMMIT"
```

The exporter retains eligibility per cell if the validator writes a completed audit with one or more output mismatches. A structural validation failure does not produce a fresh complete audit.
