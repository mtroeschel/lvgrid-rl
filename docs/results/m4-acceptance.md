# M4.6: the acceptance runs — plan and criteria

**Status: plan, pre-registered.** The questions, criteria and decision rules
below are committed before the first run starts, so that neither the choice of
PV normalisation, nor the training budget, nor the verdict on M4 adapts to what
the runs happen to show. Results are appended to this file; the plan above them
is not edited afterwards, except to correct an error, which is then marked as
such.

## Questions

1. **Budget.** The action space grows from 8 PV setpoints (M3) to 31 setpoints
   — 8 PV systems, 8 batteries, 8 heat pumps, 7 charge points — and the policy
   is the shared one of action mode 2. Is the M3 budget of 600,000 steps still
   enough?
2. **PV normalisation.** Normalised on the rated limits, about 7 % of the PV
   action range changes anything over the year (architecture §6.3, the dead
   zone). Does normalising on the forecast available power instead
   (`--pv-normalisation available`) learn better?
3. **Acceptance.** Does a policy with the full actuator set beat the best rule
   (B6) and use the flexibility the way M3 said it must — by shifting load
   rather than by curtailing (`docs/results/m3.md`, consequence 1)?

## Setup

Everything not listed here is the code at the commit of this file.

| | |
|---|---|
| grid, scenario | `1-LV-rural1--2-sw`, `moderate_growth` |
| assets | `--storage --heat-pumps --ev`, the M4 defaults (`DEFAULT_STORAGE`, `DEFAULT_HEAT_PUMPS`, `DEFAULT_EV`) |
| policy | `configs/agent/ppo_shared.yaml` unchanged (SharedAssetPolicy, D14; PPO settings as M3) |
| reward | fixed weights as in `RewardConfig`: curtailment −1/MWh, `ev_unserved` −100/MWh, `hp_comfort` −2/Kh, `bess_degradation` −0.2/MWh, smoothness −0.05/MW, losses −0.1/MWh; K95 −10 per window beyond budget, K100 −50, thermal overload −30. Kept as they are, decided before this plan: supply first — comfort and charging energy are priced well above curtailment |
| training | 8 workers (16,384 steps per rollout), `sim_dt` 5 min, `control_dt` 15 min, two-day episodes starting on the control grid, randomised budget |
| throughput | about 34 steps per second (measured, 8 workers, this machine): 600,000 steps ≈ 5 h |
| evaluation | `EpisodeMode.EVALUATE`; validation weeks (8) for phase 1, test weeks (9) for phase 2 |

## Phase 1 — learning-curve diagnostic and PV normalisation

Two arms, `rated` and `available`, two seeds each (base seeds 1 and 2),
1,200,000 steps, a checkpoint every 200,000 steps (`--checkpoint-every`). Every
checkpoint is evaluated on the **validation** weeks, policy only
(`evaluate.py --checkpoint ... --set val --policy-only`). Twenty-four
evaluations; the test weeks are not touched in this phase.

**Score.** The return on the validation weeks — the sum of the reward over the
eight weeks, the quantity the agent optimises. The KPIs of every checkpoint are
reported alongside, but the decisions below use the return only, so that
they cannot be argued after the fact.

**N — normalisation.** `available` is adopted if, at 1,200,000 steps, the mean
validation return of its two seeds exceeds that of `rated` by more than the
larger of the two within-arm spreads (the difference between an arm's two
seeds). Otherwise `rated` stays: it is the status quo and comparable with M3.

**B — budget.** For the adopted arm, the mean validation return over its two
seeds per checkpoint. The acceptance budget is the first checkpoint from which
the remaining gain up to 1,200,000 is less than 10 % of the gain from 200,000
to 1,200,000. If no checkpoint before 1,200,000 qualifies, the budget is
1,200,000 and the record says that the curve was still rising. The budget is
not extended beyond 1,200,000 in M4.

## Phase 2 — acceptance

Five seeds (base seeds 1 to 5), the arm and budget from phase 1, evaluated on
the **test** weeks with all reference methods. Seeds 1 and 2 of phase 1 are not
reused: phase 2 trains afresh, so that every seed has the same budget.

**References.** B0 `do_nothing`, B1 `random`, B2 `fixed_cap(0.4)`, B4
`p_u_droop(1.04/1.10)`, B5 `en14a_dimming`, B6 `greedy_local(theta=0.2,m_h=2.0)`,
all tuned and in the same configuration (`configs/baseline/tuned.json`); and the
M3 policy as recorded (`docs/results/m3.md`: pass rate 1.0, overload IQM 0.101
in schema 1, i.e. 0.034 in schema 2, curtailment 39.89 MWh on the same test
weeks, PV curtailment only). Measured for B6 on the test weeks: pass rate
0.9829, overload 64.3, curtailment 1.57 MWh.

**Criteria**, each on the IQM over the five seeds (95 % bootstrap CI reported,
`scripts/aggregate.py`):

| | criterion | reference |
|---|---|---|
| **A1** grid | pass rate = 1.0 | M3 policy 1.0; B6 0.9829 |
| **A2** grid | overload ≤ 2.0 | M3 policy 0.034; B6 64.3 |
| **A3** value of flexibility | curtailment ≤ 11.6 MWh, half of the least any PV-only controller can do | PV-only optimum 23.2 MWh; M3 policy 39.89 MWh; optimum with batteries 0 MWh |
| **A4** supply | EV energy unserved plus deferred ≤ 0.015 MWh (1 % of the 1.48 MWh charged in the test weeks) and heat pump comfort deviation ≤ 1.0 Kh, summed over the test weeks | every reference: 0 unserved, ≤ 0.003 deferred, 0 Kh |
| **A5** no borrowing | battery energy change ≥ −1.26 MWh and buffer energy change ≥ −0.25 MWh, summed over the nine weeks (−10 % of the total capacity, 1.40 MWh and 0.28 MWh, per week) | B6: +0.40 MWh |

A2 is the same bound as in M4.0 (V2). A3 is the claim of M3's consequence 1
made measurable, against the physics rather than against another policy (next
section). A4 and A5 close the two ways a policy could look good without
being good: by not delivering, or by ending the week with empty storage.

## How low can curtailment go? (the basis of A3)

A3 first read "≤ 20 MWh, half of the M3 policy's". Asked before the first run
whether that is attainable and what the least curtailment is that resolves the
voltage and thermal problems, the question was answered with a bound and A3
re-based on it. No run of this plan had started.

`scripts/curtailment_bound.py --voltage` solves, per test week, a linear
programme with the whole week known in advance: least curtailed PV energy such
that the transformer and every line stay at or below their rating (flows from
the radial topology), every assessed bus stays at or below 1.09 pu at every
step (stricter than EN 50160; linearised, with 0.01 pu for the linearisation
error), and the batteries keep their limits and end each week no emptier than
they began. Heat pumps and charge points stay on their SimBench profiles, so
their flexibility is not used. The plan is then replayed in the full AC power
flow, and the replay is what the table reports.

| test weeks (9) | curtailment | pass rate | K95 | overload |
|---|---|---|---|---|
| optimum with batteries | **0 MWh** | 1.0 | 0 | 0.001 |
| optimum, PV curtailment only | **23.2 MWh** | 1.0 | 0 | 0 |
| M3 policy (PV only) | 39.89 MWh | 1.0 | 0 | 0.034 |
| B6 (best rule) | 1.57 MWh | 0.9829 | 2 | 64.3 |

With batteries and perfect foresight no curtailment at all is needed; the
batteries end the nine weeks 3.8 MWh fuller in total, at 40.6 MWh throughput.
PV curtailment alone needs at least 23.2 MWh; the M3 policy used 1.7 times
that. A first programme without the voltage condition also found 0 MWh but
failed K95 in the replay (pass rate 0.93): it emptied the batteries at night at
full power, which lifts the voltage at the end of the long feeder above
1.10 pu. With the voltage condition at 1.097 pu the margin was too small for
that (predicted 1.095, reached 1.101); at 1.09 the plan holds.

**A3 is therefore set at half the PV-only optimum, 11.6 MWh**: less than any
controller can reach by curtailing PV, even with perfect foresight, so meeting
it shows that the policy uses the flexibility. It is not set nearer the
optimum with batteries, because the policy sees one hour ahead, not the week.

## Decision rule

- **A1 to A5 hold** → M4 is accepted. §12 is marked complete, and this file
  becomes the acceptance record.
- **Any criterion fails** → M4 is not accepted on these runs. The failure is
  reported as measured, with the KPIs of every seed, and what follows is
  decided then — not by retuning until the rule passes. In particular the
  reward weights, the references and the bounds above are not changed in
  response to a failure.
- **Phase 1 produces no usable curve** (a run diverges or ends with a
  non-finite return) → the run is repeated once with the same seed; a second
  failure stops the plan and is reported.
