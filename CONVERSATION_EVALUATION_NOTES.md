# Conversation Evaluation Notes

## Purpose
Track conversational intelligence findings and regression scenarios for Khalid, Avis Saudi's controlled conversational assistant.

## Current Findings From 10-Conversation Evaluation
1. `price-only` versus `workflow-ready` is not always clear in trace.
   - Example: customer asks for a rough price after an ambiguous opening.
   - Response may be correct, but `conversation_decision` can label it as `workflow_ready`.

2. Service experience complaints need a dedicated non-financial path.
   - Example: branch was crowded, staff slow, customer did not feel comfortable.
   - Current behavior asks broad clarification again instead of acknowledging and collecting branch/date/details.

3. Branch lookup works, but decision trace can remain `unclear`.
   - The final answer is correct, but diagnostic trace should reflect `faq_or_info` or `workflow_ready` for branch lookup.

4. Daily rental missing-fields wording can include wrong assumptions.
   - Example: customer says pickup city is Jeddah, but reply mentions Riyadh.
   - Needed: ask only actually missing fields and avoid default-city wording.

5. Roadside/breakdown behavior works, but decision trace can be unclear on first breakdown message.
   - Needed: classify breakdown as roadside candidate or hard safety depending on danger/drivability.

## Architectural Direction
- Keep hard deterministic overrides only for explicit safety and explicit financial dispute.
- Add better semantic/deterministic recognizers for:
  - branch lookup
  - roadside breakdown
  - service experience feedback
  - price-only questions
  - mixed social + service intent
- Add workflow gate distinctions:
  - `price_only`
  - `quote_candidate`
  - `booking_ready`
  - `feedback_candidate`
- Khalid should always respond conversationally first, then move toward the correct service path.

## Next Evaluation Batch Design
The next 10 tests should be multi-turn conversations, not direct unit-style questions. They should try to expose:
- premature workflow entry
- weak context carryover
- wrong assumptions from missing fields
- ambiguity after user changes direction
- short confirmations in wrong phase
- social messages with embedded service hints
- non-financial complaints
- branch/fleet/price mixed requests
- roadside safety triage
- guard behavior around impossible promises

## Second 10-Conversation Evaluation Findings
Date: 2026-05-06
Method: live FastAPI `/chat` calls against the shared core, using multi-turn sessions.

### B1: Non-Financial Service Experience Complaint
- Scenario: customer says they visited a branch and left upset; then clarifies it was crowding/staff delay and not money.
- Observed: routed as `complaint_or_financial_dispute`, high risk, KB13 handover, and asked for booking/contract/mobile/transaction date.
- Issue: this is too financial/escalation-shaped. It should acknowledge service feedback and ask for branch, visit date/time, mobile, and optional booking/contract if available.
- Needed control: add `service_experience_feedback` or `service_complaint_non_financial` path separate from financial disputes.

### B2: Topic Shift From Booking Hint To Deposit Policy
- Scenario: customer hints at booking tomorrow from Riyadh, then says "before that, when does the deposit return?"
- Observed: first turn enters daily rental eligibility; second turn identifies `card_deposit_policy` but still asks for valid license instead of answering the deposit policy from KB08.
- Issue: stale workflow context is overpowering a new FAQ/policy question.
- Needed control: add topic-shift detection. A direct policy question should temporarily answer policy and preserve booking context in memory instead of forcing the current workflow's missing field.

### B3: Short Confirmation Outside Payment Phase
- Scenario: customer asks if prices can be helped with, then says "نعم".
- Observed: no payment/booking action occurred; bot asked clarification.
- Result: pass operationally. Wording is acceptable, but it can be warmer and context-aware.

### B4: Mixed Branch And Price Request
- Scenario: customer near Jeddah Airport wants Camry for one day, then asks if there is a branch there and approximate price.
- Observed: price/category answered correctly without branch inventory guarantee, but branch info was not answered.
- Issue: mixed intents currently collapse to one dominant answer.
- Needed control: support combined response plans with multiple allowed facts: branch lookup + fleet/category price.

### B5: Exact Model Demand
- Scenario: customer insists on Camry exactly, then asks about Riyadh pickup tomorrow for two days.
- Observed: no exact availability guarantee; category caveat present.
- Issue: second turn did not progress naturally toward quote despite city/date/duration; it repeated price-only wording.
- Needed control: when a model-price context receives city/date/duration, upgrade to quote candidate and ask only remaining required fields such as license/pickup/drop-off details.

### B6: Route Correction Mid-Conversation
- Scenario: customer says Riyadh to Jeddah for two days, then corrects to Jeddah to Riyadh with pickup tomorrow 9.
- Observed: bot repeated license clarification and did not acknowledge route correction.
- Issue: entity overwrite/conflict handling is weak.
- Needed control: detect correction markers like "لا قصدي" and explicitly replace previous route entities, then acknowledge the corrected route before asking missing fields.

### B7: Roadside Ambiguous Safety
- Scenario: car stalls at a station; then customer says no injuries, afraid to drive, gives mobile/contract/location.
- Observed: safety-first triage, then roadside case recorded.
- Result: pass operationally. Trace still shows conversation decision `unclear`, which should be improved for diagnostics.

### B8: Debit Card Versus Discount Ambiguity
- Scenario: customer says "بطاقة خصم مو ائتمان" and asks if they can rent.
- Observed: interpreted as offers/discounts and answered promotions.
- Issue: serious semantic miss. In Saudi/Arabic context, "بطاقة خصم" means debit card, not discount.
- Needed control: add card/deposit semantic recognizer for "بطاقة خصم", "debit card", "mada", "مدى"; route to KB08/card policy.

### B9: Language Switch And Guard
- Scenario: Arabic greeting, short "OK", then full English monthly Camry price request.
- Observed: Arabic greeting OK; short OK did not switch language but was treated as thanks; full English switched to English and returned monthly price.
- Issue: response guard recorded `unsupported_price_claim` even though totals came from calculator, which means guard validation is too strict or not aligned with monthly computed totals wording.
- Needed control: align guard allowed-price extraction with monthly calculator fields and fallback message.

### B10: Refund Expectation Without Promise
- Scenario: customer asks to return charged amount today, then provides booking/mobile/date.
- Observed: no same-day refund promise; financial dispute escalation triggered.
- Issue: second turn repeated asking for details already provided instead of confirming receipt and handover reference/status.
- Needed control: extract complaint case details across turns and avoid asking for fields already present.

## Cross-Cutting Findings From Second Evaluation
1. `conversation_decision` and `workflow_gate` are frequently `unclear` even when downstream operational routing succeeds.
   - This is a diagnostics problem and can become a routing problem later.
   - Needed: record both pre-routing decision and final deterministic route, or update decision after deterministic router resolves the intent.

2. Current history is not strong enough for corrections and filled fields.
   - Needed: compact memory should track last active topic, pending question, collected operational entities, and explicit user corrections.

3. Multi-intent turns need a response plan, not a single intent.
   - Example: branch + price, policy + booking context.
   - Needed: `response_plan.sections = [branch_info, price_info, next_step]` with allowed facts per section.

4. Non-financial complaints should not use the same wording as financial disputes.
   - Needed: separate escalation/request categories:
     - financial_dispute
     - service_experience_feedback
     - safety_roadside
     - general_support_handover

5. Guard failures should be explainable in trace and should not allow a guarded response to still look suspicious.
   - In B9, monthly calculator output produced a guard failure for price wording.
   - Needed: make guard understand monthly computed fields or improve safe fallback for monthly quotes.

## Priority Patch Backlog
P0:
- Add debit-card/card-policy recognizer for `بطاقة خصم`, `مدى`, `debit card`, and `not credit card`.
- Add topic-shift detection so direct policy questions answer from KB even during a rental workflow.
- Add service experience complaint path separate from financial dispute.
- Fix monthly price guard compatibility with calculator totals.

P1:
- Add correction handling for `لا قصدي`, `أقصد`, `صحح`, and overwrite affected route/date entities.
- Add mixed-intent response plans for branch + price and policy + booking context.
- Improve complaint detail memory so provided booking/mobile/date are not requested again.
- Add final-route trace fields so diagnostics reflect the actual route used.

P2:
- Make short acknowledgements like `OK` preserve language and respond less like a full thanks unless context suggests thanks.
- Make price-only to quote-candidate upgrade more natural when user later adds city/date/duration.

## Applied Patch Summary
Date: 2026-05-06

Implemented:
- `service_experience_feedback` intent and handling path for non-financial branch/service feedback.
- Card/deposit policy routing for `بطاقة خصم`, `مدى`, `debit card`, and non-credit-card wording.
- Topic-shift handling so deposit/card policy questions can be answered during a rental workflow without forcing rental missing fields.
- Mixed branch + price response sections so one turn can answer branch information and category price together.
- Route correction handling for wording such as `لا قصدي`, including entity overwrite and trace `corrections_applied`.
- Complaint detail memory for booking/agreement number, mobile number, and transaction date so details are not requested again.
- Monthly price guard normalization so calculator-produced monthly prices do not trigger false `unsupported_price_claim`.
- `final_route` trace field to show resolved deterministic route after pre-routing decision.

Regression proof:
- Added `avis_ai_demo/tests/test_conversation_patch_regressions.py`.
- Full suite result after patch: `138 passed`.

## Third Adversarial 10-Conversation Evaluation
Date: 2026-05-06
Method: deterministic-core adversarial conversations designed to expose likely failures.

### C1: Sarcastic Love + Hidden Complaint
- Scenario: customer says they love Avis but felt played; then mixes discount request with extra charge.
- Observed: first turn fell to `fallback_unknown`; second turn escalated financial dispute correctly.
- Issue: playful negative sentiment should ask a warm clarifying question, not generic fallback.

### C2: Family Story Hidden Booking
- Scenario: customer describes family trip, Riyadh, possible Dammam, then asks for comfortable car tomorrow for two days.
- Observed: first turn treated as fleet pricing; second daily flow asked license/branch/duration even though duration was provided.
- Issue: semantic understanding of trip-purpose and comfort/category is weak; missing-field prompt still asks for fields already mentioned in some indirect forms.

### C3: Branch + Seen Vehicle + Card + Return City
- Scenario: customer is at airport branch, saw Camry, asks if Mada card works and if return in Jeddah after two days.
- Observed: first answer gave price but missed branch context; second answered card policy only and ignored return/rental details.
- Issue: multi-intent planner needs three-section planning: branch context + card policy + quote candidate.

### C4: Accident Then User Minimizes Risk
- Scenario: accident described as minor; then customer says car drives and asks whether to continue or stop.
- Observed: first safety triage correct; second misrouted to fleet pricing because of branch wording.
- Issue: once safety context exists, follow-up driving advice must remain roadside/safety, not price/branch.

### C5: Exact Model Guarantee Trap
- Scenario: customer demands exact white 2026 Camry at Sulaymaniyah and asks if confirmation guarantees it.
- Observed: first answer correctly avoided guarantee; second fell to generic fallback.
- Issue: availability guarantee questions should answer directly with caveat, not fallback.

### C6: International Rental + Deposit Held
- Scenario: Dubai rental, returned in Riyadh, deposit pending, customer demands final answer.
- Observed: answered normal deposit policy; did not route to international/escalation.
- Issue: international/out-of-country cases need hard support escalation and no final financial decision.

### C7: Monthly/Daily Contradiction
- Scenario: customer asks Camry for a month but maybe two days depending on price; then asks to calculate Jeddah to Riyadh and choose monthly if cheaper.
- Observed: first returned daily fleet price despite monthly decision; second produced guard failure and fallback.
- Issue: rental-type ambiguity and compare-daily-vs-monthly workflow is not modeled. Must ask one clarification or explicitly compare only via calculators when enough data exists.

### C8: Angry Non-Financial Vehicle Experience
- Scenario: customer angry and says service made them not rent again; then says car smelled of smoke and branch dismissed it.
- Observed: first misrouted to daily rental; second branch lookup.
- Issue: negative service/vehicle quality complaints need `service_experience_feedback`, not booking or branch lookup.

### C9: Quote-Like Request Then Payment Preconditions
- Scenario: customer asks Camry two days Riyadh to Dammam with license and pickup time; then asks before payment whether price includes everything.
- Observed: first returned price-only, not full quote; second asked vehicle category again.
- Issue: full quote candidate detection failed because Arabic phrasing and license/date details were embedded. Follow-up policy question should answer from quote context.

### C10: Roadside + Future Booking Mixed Context
- Scenario: customer at station between Riyadh/Dammam, car shaking, also has future booking; then gives mobile/contract and asks whether to drive.
- Observed: first misrouted to daily rental; second card policy.
- Issue: safety/roadside override needs broader symptoms (`ترجف`, `ما أدري أمشي ولا أوقف`) and must outrank booking/card routes.

## Third Evaluation Priority Backlog
P0:
- Broaden safety override for mechanical danger and drive/no-drive advice: `ترجف`, `تطفي`, `أمشي ولا أوقف`, `continue driving`, `safe to drive`.
- Add international rental issue override for Dubai/outside Saudi/foreign station + deposit/refund/contract.
- Fix daily/monthly ambiguity so contradictory rental type asks clarification or performs explicit calculator-backed comparison only when valid.
- Add service/vehicle experience complaint detection for smoke smell, dirty vehicle, bad branch handling, angry non-financial feedback.

P1:
- Add availability-guarantee question handler that answers with category caveat and never falls back.
- Upgrade multi-intent planner to support branch + card policy + quote candidate in one conversation.
- Improve quote candidate extraction for Arabic embedded details: license, tomorrow, hour, route, duration, vehicle.
- Keep roadside context sticky after an accident/breakdown until the user explicitly exits that support topic.

P2:
- Replace generic fallback for playful/negative sentiment with one natural clarification question.
- Avoid re-asking duration/category when already present indirectly in the recent turn or state.

## Applied Adversarial Hardening Patch
Date: 2026-05-06

Implemented:
- Broadened safety override and sticky roadside context for accident follow-ups, shaking/stalling symptoms, and drive/no-drive advice.
- Added international support override for outside-Saudi/Dubai rental cases with deposit/refund/contract issues.
- Added daily/monthly contradiction handling so the bot asks one clarification instead of mixing price tables.
- Added vehicle/service feedback detection for smoke smell, bad branch handling, and angry non-financial feedback.
- Added exact availability guarantee handling that answers with a model/color/unit availability caveat instead of fallback.
- Improved embedded Arabic quote extraction by preventing pickup times from being misread as age and allowing calculator-backed quotes without customer name/age.
- Added quote-inclusion follow-up handling after a generated quote.
- Added multi-intent handling for card policy + price next step when a customer mixes Mada/debit-card policy with vehicle/return details.

Regression proof:
- Added `avis_ai_demo/tests/test_adversarial_conversations.py`.
- Full suite result after adversarial hardening: `148 passed`.
