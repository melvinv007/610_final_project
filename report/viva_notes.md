# Viva / presentation cheat sheet (all numbers are from report.pdf)

**One-line story:** A wrong answer to the check-in retry is the moment students leave (13.1% vs 0.94% elsewhere); the lesson that follows is linked to better next-try success; leaving is mostly a student trait, and failed-retry quitters come back fast but less reliably.

## Likely questions
1. **Why these three questions?** They follow one process: lesson (what helps), quit (where it breaks), return (what happens next). All are temporal. We replaced topic/year-group comparisons because the student ICC (0.285) dwarfs topic (0.012).
2. **What is an "exit"?** Last step of a session that never reached Q5 (9,426 sessions). Q5 excluded because the quiz ends there. Robust to "pause > 1 h" (ratio 10.6) and "no return in 24 h" (11.0).
3. **Do you measure frustration?** No. We observe exits only; we write "quit after failed retry".
4. **Is the lesson effect causal?** Not proven. Raw +12.7 pp, student fixed effects +5.9 pp, logit +7.0, IPW +8.3. Placebo (accuracy elsewhere in session) ~0 within student. E-value 1.62.
5. **Why does the lesson plateau at ~20 s?** Success 29.6% (<=5 s) -> 47% (10-15 s) -> ~51-52% (20-60 s); reading is what matters (glance ~ skip).
6. **What is a discrete-time hazard model?** Each step is a record; logit of "session ends here | reached here" (Singer & Willett, 1993). Lets us add time-varying covariates.
7. **Why does risk fall across the session but rise with failures?** Survivor selection. Pooled OR for 1 earlier failure 0.72; within student (conditional logit) 1.65.
8. **Why a student-cluster bootstrap?** Steps within a student are dependent; resample students.
9. **What is the 3 May break?** Quit rate after failed retry 9.2% -> 16.7% (LR 511; 95% CI 26 Apr-9 May); 372 students seen in both periods +11.6 pp. Cause unknown; adjusted in all models; ratio unchanged (13.6 -> 14.7).
10. **Is quitting persistent?** After a quit 34.7% quit next time vs 7.9% after completion (x4.4). Within student only +4.3 pp; split-half reliability 0.72; top 10% students = 54% of exits.
11. **Do quitters return?** After failed-retry quit: 37.6% within 1 h, 43.6% in 24 h, 56.5% in 7 d, 65.8% in 30 d vs 85.3% / 90.4% for finishers. Median return among returners 11 min. Almost nobody returns to the same quiz (0.1%).
12. **Biggest limitations?** Observational; no emotion; unexplained break; platform flow constraints (after a lesson retry is forced); no class/teacher IDs; one platform.
13. **What would you do next?** Randomised lesson prompt at the failed-retry step; link to class data; competing-risks return model.
