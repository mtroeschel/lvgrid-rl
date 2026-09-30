# M4.0: validating the Lagrangian reward mode on the M3 setup

**Status: complete, five of five seeds.** The criteria
and the decision rule below were written and committed (`86c32f9`) before the
first run finished, so that the decision on the reward mode for M4 does not
adapt to what the runs happen to show.

**Verdict: V1 and V2 hold, V3 fails. M4 stays on fixed weights.** The mechanism
works -- multipliers converge, reach the workers and hold the constraints -- but
it does not reach a better operating point than the hand-set weight: two of four
policies are dominated by a measured cap -- three of five in the end -- against
at most one of five allowed. The outcome was fixed after four seeds; the fifth
confirmed it.

## Question

`docs/results/m3.md`, consequence 2, left open whether M4 keeps the fixed reward
weights or switches to the constrained formulation of section 6.4. The question
is decided here, on the one setup whose frontier has already been measured: PV
curtailment only, `1-LV-rural1--2-sw`, `moderate_growth`.

## What was changed to make the mode usable

The mode existed in M3 only as half a mechanism: `RewardComposer` accepted
multipliers, but nothing updated them. Two further defects surfaced when the
code was read against the claim that `d = 0.05` is the standard's criterion:

- **The K95 cost was not a rate.** It was normalised by `n_buses ×
  budget_windows`, so its weekly sum was the budget share *averaged over buses*
  and its limit would have been 1, not 0.05 -- and an average lets twelve
  compliant buses dilute one that fails. The voltage costs are now the
  violating windows at the worst bus per window closed. Their mean is a share
  of windows outside the band, which is what EN 50160 bounds; summing the worst
  bus over time is conservative.
- **The dual update did not exist.** `LagrangianCallback` now performs dual
  ascent once per rollout on an EMA-smoothed cost estimate, with bounded
  multipliers, and writes the multiplier history to `lagrangian.json` in the
  run directory.

`fixed_weights` mode is unchanged, bit for bit, so the M3 runs remain a valid
reference.

## Setup

Identical to M3 except for the reward mode:

| | |
|---|---|
| grid, scenario | `1-LV-rural1--2-sw`, `moderate_growth` |
| training | PPO, 600,000 steps, 8 workers, base seeds 1 to 5 (the M3 seeds) |
| PPO settings | Stable-Baselines3 defaults, as in M3: `n_steps` 2,048 per worker (16,384 per rollout), `batch_size` 64, network 64×64. `configs/agent/ppo.yaml` is **not** read by `train.py`; see the correction note in `m3.md` |
| reward | `lagrangian`: objective terms with the M3 weights; constraints priced by multipliers |
| limits | K95 rate 0.05, K100 rate 0, thermal overload 0 |
| dual settings | `DEFAULT_DUALS` in `experiment/callbacks.py` (step size 4 for the voltage rates, 80 for thermal overload; bounds 50 and 200); EMA decay 0.8; one update per rollout, about 37 per run |
| evaluation | test weeks, `EpisodeMode.EVALUATE`, baselines merged from the M3 results |

The dual settings are starting values chosen from the cost scales of the M3 test
weeks, not tuned. They replace the penalty weights as hyperparameters, with one
difference that is the point of the exercise: in theory the weights decide the
solution, whereas the dual settings decide only how fast it is reached.

## What to expect

On this setup the two formulations should land in almost the same place. The
thermal limit is zero, and a hinge penalty with a zero limit is an exact
penalty: above some finite weight, the penalised optimum *is* the constrained
optimum. M3's weight of 10 already drove overload to near zero, so M3 sits at
the zero-overload end of the frontier, and a Lagrangian run with `d = 0` aims
at the same point. K95 does not bind in this scenario (every cap at 0.40 or
tighter passes), so its multiplier should stay near zero.

The validation therefore tests **the mechanism, not an improvement**: whether
dual ascent reaches the constrained optimum at least as well as a hand-set
weight, without oscillating and without degenerating into a fixed weight at its
bound. The payoff of the formulation is expected in M4, where several objectives
compete and the constraint prices can no longer be read off one corner.

## Criteria

All on the test weeks, over five seeds, reported as in M3 (IQM with bootstrap
confidence interval).

| | criterion | reference |
|---|---|---|
| **V1** mechanism | no multiplier ends at its bound; the thermal multiplier changes by less than 10 % over the last 20 % of rollouts | `lagrangian.json` per run |
| **V2** constraints | `pass_rate` IQM = 1.0; overload IQM ≤ 2.0 | M3 policies: 1.0 and 0.00 to 1.98 |
| **V3** objective | curtailment IQM ≤ 45.0 MWh; no more than one of five policies dominated by a measured cap | M3: IQM 39.89, CI upper bound 45.01; one of five dominated |

## Decision rule

- **V1 to V3 hold** → M4 uses the Lagrangian mode. Section 13 D5 is updated
  accordingly, and the fixed-weight mode stays available as the comparison arm
  for M7.
- **V1 fails, multiplier at its bound** → the formulation degenerated into a
  fixed weight. Raise the bound once and rerun one seed; if it recurs, M4 stays
  on fixed weights and the Lagrangian mode returns in M7 as planned.
- **V1 fails, oscillation** → reduce the dual step size once and rerun one seed;
  the same fallback applies.
- **V2 or V3 fails with V1 holding** → the mechanism works but reaches a worse
  point than the hand-set weight. That is a result in its own right and is
  reported; M4 stays on fixed weights.

At most one correction round per failure mode, so that the rule cannot be
satisfied by tuning until it passes.

## Results

Test weeks, `EpisodeMode.EVALUATE`.

| seed | run | final λ thermal | λ change, last 20 % | pass rate | overload | curtailed | dominated by |
|---|---|---|---|---|---|---|---|
| 1 | `3adc98ffd9f1` | 37.76 | +2.4 % | 1.0 | 0.000 | 44.92 MWh | cap 0.11 |
| 2 | `706bacbf9f20` | 33.61 | +3.3 % | 1.0 | 0.079 | 36.98 MWh | -- |
| 3 | `092a9bddbcec` | 43.65 | +4.3 % | 1.0 | 0.000 | 40.56 MWh | -- |
| 4 | `5d1b1b73241d` | 40.00 | +2.7 % | 1.0 | 0.000 | 46.61 MWh | caps 0.10, 0.11 |
| 5 | `c52cce77df58` | 43.17 | +4.3 % | 1.0 | 0.237 | 41.58 MWh | cap 0.12 |

Over five seeds (IQM, 95 % bootstrap CI, `scripts/aggregate.py`):

| | pass rate | k95 windows | overload | curtailed |
|---|---|---|---|---|
| **Lagrangian** | 1.0000 [1.0, 1.0] | 0 [0, 0] | 0.026 [0.00, 0.18] | 42.35 MWh [38.17, 46.04] |
| fixed weights (M3) | 1.0000 [1.0, 1.0] | 0 [0, 0] | 0.101 [0.00, 1.42] | 39.89 MWh [35.02, 45.01] |

The K95 multiplier stayed at zero in every run, and the K100 multiplier below
0.05 -- as expected in a scenario where voltage does not bind.

**V1 holds.** No multiplier reached its bound; the thermal multiplier changed by
2 to 4 % over the last seven of 37 updates. It is still rising, as it must with a
limit of zero: the stochastic training policy produces a residual overload of
about 0.0005 per step, so `J_c − d` stays positive. "Converged" here means the
growth has slowed, not stopped; with a longer budget the multiplier would keep
climbing.

**V2 holds.** Pass rate 1.0 in every seed; overload IQM 0.026, at most 0.24.

**V3 fails.** The curtailment half holds (IQM 42.35 ≤ 45.0), the dominance half
does not: seeds 1 and 4 curtail more than a cap for the same zero overload, and
seed 5 is beaten by `cap 0.12` on both axes. In M3, one of five policies was
dominated.

### Reading

- **The overall picture is a small shift along the frontier, not a better
  frontier.** The Lagrangian policies hold overload slightly lower (IQM 0.026
  against 0.101) and curtail slightly more (42.35 against 39.89 MWh). Both
  intervals overlap widely with M3.
- **The dual prices ended three to four times above the M3 weight** of 10. The
  constraint was already held at that weight, so the Lagrangian runs paid a
  higher price for the same corner, which is consistent with the dominated
  seeds curtailing more. The relation is not clean, though: seed 3 has the
  highest multiplier and the best point, so the data do not show that the price
  alone explains the outcome.
- **The best single point is a Lagrangian one.** Seed 3 reaches zero overload
  with 40.56 MWh, against 41.21 MWh for the best M3 policy and 43.12 MWh for the
  best cap. Seed 2 at 36.98 MWh and 0.079 overload dominates the M3 policy
  `eaca5d1d` (37.00, 0.303). The spread across seeds, 37 to 47 MWh, is as wide
  as in M3 (34 to 47 MWh), so none of this separates the formulations.
- **The zero limit is the structural weak point**, as expected before the runs:
  with `d = 0` the multiplier can only grow, so what it settles at depends on
  the training budget and the exploration noise rather than on the problem.
  A small positive thermal limit, or a cost measured on the deterministic
  policy, would address that. Either one is a different experiment with its own
  pre-registration, not a correction round under this rule.

## Consequence for M4

M4 trains with `fixed_weights`, as section 13 D5 now records. The Lagrangian
mode stays in the code, tested and working, as the comparison arm for M7. The
open question it leaves is the zero limit: before the mode is reconsidered, a
positive thermal limit or a cost measured on the deterministic policy needs its
own pre-registered test.

## Artefacts

| | |
|---|---|
| runs | `results/m4-lagrangian/<run_id>/` with `manifest.json`, `lagrangian.json`, `eval/test.json` |
| aggregation | `results/aggregate-m4-lagrangian/aggregate_test.json` |
| cap points | `results/pareto/pareto_test.json` (from M3, unchanged) |
| baseline rows | merged from `results/m3/4c717e8655ad/eval/test.json` |
