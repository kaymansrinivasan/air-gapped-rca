# A595 synthetic RCA workflow walkthrough

**DEMONSTRATION ONLY. Original failures are real recorded observations. Every investigation, diagnosis, action and retest below is fictional. No engineer has validated these scenarios.**

Read cases 001 and 002 together: the same observed failure leads to different conclusions only after different simulated evidence is supplied.

## SYN-A595-001 — Continuity branch A: recovery after a simulated contact intervention

**Real starting point:** wafer 02, device 80, test 100 (Open/Short-); [source log](source/a595_tester_log.txt), lines 12704–12706.

```text
   100 Open/Short-         Halt Vector:      0   Halt Cycle:      0   Failing Pins:  1 (F)
          Failed Pins:
               AMSDSM : 40    
```

**Simulated investigation:**

- A fictional contact inspection reports residue at the AMSDSM probe contact.
- A simulated reference contact check fails on the same channel with the suspect interface and passes with the reference interface.
- The simulated investigation therefore localizes the issue to the interface, rather than relying on the DUT failure alone.

**Scenario cause:** Probe-contact contamination in the simulated setup. This is not a diagnosis of the original real device.

**Simulated action:** Apply the fictional approved cleaning procedure to the suspect contact and repeat the configured test flow.

**Closure reference:** The fictional action is followed by 123/123 passing checks for the same synthetic device. This supports recovery within the scripted scenario only. The assigned scenario cause is Probe-contact contamination in the simulated setup. The passing values came from another real passing device as a synthetic template; they are not genuine retest measurements.

**Citations:** [SYN-A595-001-FAILURE](incidents/SYN-A595-001/observed_failure.json), [SYN-A595-001-INVESTIGATION](incidents/SYN-A595-001/simulated_investigation.json), [SYN-A595-001-ACTION](incidents/SYN-A595-001/simulated_action.json), [SYN-A595-001-RETEST-1](incidents/SYN-A595-001/simulated_retest_1.json).

## SYN-A595-002 — Continuity branch B: no recovery from the same initial failure

**Real starting point:** wafer 02, device 80, test 100 (Open/Short-); [source log](source/a595_tester_log.txt), lines 12704–12706.

```text
   100 Open/Short-         Halt Vector:      0   Halt Cycle:      0   Failing Pins:  1 (F)
          Failed Pins:
               AMSDSM : 40    
```

**Simulated investigation:**

- In this alternative fictional history, cleaning does not change the continuity failure.
- The same synthetic device still fails on a reference interface; a control device passes that interface.
- A fictional laboratory finding explicitly reports a die-side discontinuity associated with AMSDSM. This is a scenario assumption, not a real microscopy report or layout-specific diagnosis.

**Scenario cause:** Die-side connection defect in the simulated AMSDSM path. This is not a diagnosis of the original real device.

**Simulated action:** Record that the attempted interface correction failed; isolate the synthetic device for engineering disposition rather than repeatedly changing the tester.

**Closure reference:** The simulated interface correction did not resolve the continuity failure. The correctly linked targeted repeat still fails. The fictional laboratory finding supports the die-side defect only within this scenario. Containment is recorded; repair and release are not established.

**Citations:** [SYN-A595-002-FAILURE](incidents/SYN-A595-002/observed_failure.json), [SYN-A595-002-INVESTIGATION](incidents/SYN-A595-002/simulated_investigation.json), [SYN-A595-002-ACTION](incidents/SYN-A595-002/simulated_action.json), [SYN-A595-002-RETEST-1](incidents/SYN-A595-002/simulated_retest_1.json).

## SYN-A595-003 — Supply-current alarm: simulated fixture isolation and recovery

**Real starting point:** wafer 02, device 41, test 210 (IDD_Static); [source log](source/a595_tester_log.txt), lines 6411–6411.

```text
   210 IDD_Static         curr         35.00 ma <   200.01 ma (A) <  55.00 ma
```

**Simulated investigation:**

- The original 200.01 mA display is preserved as an alarm-marked reading, not treated as a precise clean measurement.
- A fictional isolation check reports excessive supply load even with the synthetic DUT disconnected; that load is absent on a reference interface.
- A fictional fixture inspection identifies an unintended supply load. These added observations are the basis of the scenario diagnosis.

**Scenario cause:** Unintended load in the simulated test fixture supply path. This is not a diagnosis of the original real device.

**Simulated action:** Remove the fictional unintended fixture load under an approved simulated procedure, verify the empty fixture, and retest the same synthetic device.

**Closure reference:** The fictional action is followed by 123/123 passing checks for the same synthetic device. This supports recovery within the scripted scenario only. The assigned scenario cause is Unintended load in the simulated test fixture supply path. The passing values came from another real passing device as a synthetic template; they are not genuine retest measurements.

**Citations:** [SYN-A595-003-FAILURE](incidents/SYN-A595-003/observed_failure.json), [SYN-A595-003-INVESTIGATION](incidents/SYN-A595-003/simulated_investigation.json), [SYN-A595-003-ACTION](incidents/SYN-A595-003/simulated_action.json), [SYN-A595-003-RETEST-1](incidents/SYN-A595-003/simulated_retest_1.json).

## SYN-A595-004 — High delay: simulated test-configuration investigation

**Real starting point:** wafer 02, device 11, test 706 (Delay Line #07); [source log](source/a595_tester_log.txt), lines 1830–1830.

```text
   706 Delay Line #07     delay_line   423.0 ns <    487.1 ns (F) <  483.0 ns
```

**Simulated investigation:**

- A fictional configuration comparison finds that the active timing-measurement setup differs from its approved reference.
- A fictional reference check reproduces the delay shift on the suspect setup and removes it on the reference setup.
- The scenario attributes the apparent delay failure to that configuration difference. The pack contains no real configuration comparison.

**Scenario cause:** Incorrect timing-measurement configuration in the simulated setup. This is not a diagnosis of the original real device.

**Simulated action:** Restore the fictional approved timing configuration, verify the reference, and rerun the same synthetic device without changing its acceptance limits.

**Closure reference:** The fictional action is followed by 123/123 passing checks for the same synthetic device. This supports recovery within the scripted scenario only. The assigned scenario cause is Incorrect timing-measurement configuration in the simulated setup. The passing values came from another real passing device as a synthetic template; they are not genuine retest measurements.

**Citations:** [SYN-A595-004-FAILURE](incidents/SYN-A595-004/observed_failure.json), [SYN-A595-004-INVESTIGATION](incidents/SYN-A595-004/simulated_investigation.json), [SYN-A595-004-ACTION](incidents/SYN-A595-004/simulated_action.json), [SYN-A595-004-RETEST-1](incidents/SYN-A595-004/simulated_retest_1.json).

## SYN-A595-005 — Scan: repeated failure with an unresolved cause

**Real starting point:** wafer 09, device 331, test 606 (SCAN Test); [source log](source/a595_tester_log.txt), lines 50692–50695.

```text
   606 SCAN Test           Halt Vector:  15470   Halt Cycle:  15470   Failing Pins:  1 (F)
       Pattern: scan                                                                            
          Failed Pins:
               TSTOUT : 87    
```

**Simulated investigation:**

- A fictional targeted repeat again reports TSTOUT at cycle 15470.
- No scan-chain or cell mapping, complete pattern definition, approved timing comparison, or physical diagnosis is supplied.
- Repeatability establishes a repeatable observation within this scenario; it does not isolate the cause to silicon or equipment.

**Scenario cause:** Unresolved. This is not a diagnosis of the original real device.

**Simulated action:** Hold the incident open and request scan mapping, pattern/timing context and an independent setup comparison. No correction is claimed.

**Closure reference:** The correctly linked simulated scan repeat still fails at TSTOUT/cycle 15470. No correction or internal root cause is established. Keep the incident open and request discriminating diagnostics.

**Citations:** [SYN-A595-005-FAILURE](incidents/SYN-A595-005/observed_failure.json), [SYN-A595-005-INVESTIGATION](incidents/SYN-A595-005/simulated_investigation.json), [SYN-A595-005-ACTION](incidents/SYN-A595-005/simulated_action.json), [SYN-A595-005-RETEST-1](incidents/SYN-A595-005/simulated_retest_1.json).

## SYN-A595-006 — Low delay: reject a retest linked to the wrong device

**Real starting point:** wafer 03, device 198, test 702 (Delay Line #03); [source log](source/a595_tester_log.txt), lines 30496–30496.

```text
   702 Delay Line #03     delay_line   390.0 ns <    389.8 ns (F) <  430.0 ns
```

**Simulated investigation:**

- A fictional candidate retest is attached to the incident but its synthetic device key does not match the incident subject.
- Its passing result must be excluded from this device's recovery evidence.
- A correctly identified focused repeat is required; its outcome is not yet known. No discriminating cause investigation is available.

**Scenario cause:** Unresolved. This is not a diagnosis of the original real device.

**Simulated action:** Correct the evidence association, preserve the rejected record for audit, and request a properly identified repeat. Keep the device on hold.

**Closure reference:** Reject RETEST-1 as evidence for this incident subject: its synthetic device key differs. The correctly linked RETEST-2 is a targeted check that still fails below the timing minimum. A device pass, successful hardware correction and physical cause are not established.

**Citations:** [SYN-A595-006-FAILURE](incidents/SYN-A595-006/observed_failure.json), [SYN-A595-006-INVESTIGATION](incidents/SYN-A595-006/simulated_investigation.json), [SYN-A595-006-ACTION](incidents/SYN-A595-006/simulated_action.json), [SYN-A595-006-RETEST-1](incidents/SYN-A595-006/simulated_retest_1.json), [SYN-A595-006-RETEST-2](incidents/SYN-A595-006/simulated_retest_2.json).
