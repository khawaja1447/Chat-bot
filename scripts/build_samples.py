"""
Regenerate the sample corpus in samples/.

Three documents about the same fictional company. That overlap is deliberate:
headcount, dates and money appear in more than one document, so retrieval has to
distinguish between genuinely similar passages instead of matching on topic alone.
Run with:  python scripts/build_samples.py
"""

import pathlib

import pymupdf

OUT = pathlib.Path(__file__).resolve().parent.parent / "samples"

ANNUAL = [
    """Meridian Coffee Roasters
2024 Annual Summary

1. Overview

Meridian Coffee Roasters is a specialty coffee company founded in 2016 in
Portland, Oregon. The company roasts, packages and distributes single-origin and
blended coffees to independent cafes and to direct-to-consumer subscribers across
North America.

As of 31 December 2024 Meridian employed 84 people, of whom 51 worked in
production and logistics, 19 in retail and customer support, and 14 in corporate
functions. The company operates two roasting facilities: the original site in
Portland, Oregon, and a second facility in Sacramento, California, which opened
in June 2023.

2. Product Lines

Meridian sells through three product lines.

(a) Origin Series - single-origin beans sourced from individual farms in
Ethiopia, Colombia and Guatemala. Roasted in small 12 kg batches and sold in
250 g bags. Origin Series accounted for 41 percent of 2024 revenue.

(b) Daily Blend - a year-round house blend sold in 340 g and 900 g bags. This is
the volume product and the entry point for most new customers. Daily Blend
accounted for 47 percent of 2024 revenue.

(c) Cold Brew Concentrate - a ready-to-dilute liquid concentrate in 1 litre
bottles, launched in March 2024. It accounted for 12 percent of 2024 revenue
despite being on sale for only ten months of the year.

A fourth line, single-serve compostable pods, was evaluated during 2024 and
shelved. The pilot did not reach the margin threshold set by the board.
""",
    """3. Sourcing and Supply Chain

All green coffee is bought on direct-trade contracts negotiated annually with
producers. The average price paid to producers in 2024 was 3.15 US dollars per
pound, roughly 68 percent above the C-market benchmark over the same period.
Meridian publishes the price paid for every lot on its website.

Green coffee arrives at the Port of Oakland and is trucked to a bonded warehouse
in Sacramento. Average time from landing to roast was 34 days in 2024, down from
41 days in 2023. The company holds roughly eleven weeks of green inventory.

Meridian works with 23 producer partners across seven countries. The three
largest, all in Colombia, together supplied 38 percent of 2024 green volume,
which the board has flagged as a concentration risk.

4. Roasting Operations

The Portland facility runs two 35 kg drum roasters and one 12 kg sample roaster.
Sacramento runs a single 70 kg roaster commissioned in June 2023 and reached
full utilisation in the third quarter of 2024.

Combined 2024 roasted volume was 640 metric tonnes, up from 495 tonnes in 2023.
Roast loss averaged 15.8 percent. Batch-level roast profiles are logged and
retained for three years to support quality investigations.

Quality control cups every production batch. In 2024, 1.9 percent of batches
were rejected and redirected to the discount channel, compared with 2.6 percent
in 2023.
""",
    """5. Financial Results

Total revenue for 2024 was 18.4 million US dollars, up 22 percent from
15.1 million in 2023. Gross margin improved from 38 percent to 41 percent,
driven mainly by the higher-margin Cold Brew Concentrate line and by better
utilisation of the Sacramento facility.

Operating expenses were 6.9 million dollars, of which 2.4 million was payroll
outside production, 1.1 million was marketing, and 0.8 million was rent.

Net income was 612,000 dollars. This was the company's first profitable year.

Cash at year end was 2.3 million dollars against 1.4 million of debt, all of it
the equipment loan taken out for the Sacramento roaster. That loan carries a
6.2 percent fixed rate and matures in 2028.

Revenue by channel:

  Direct-to-consumer subscriptions    7.9 million
  Wholesale to cafes                  6.6 million
  Retail (two Portland storefronts)   2.4 million
  Discount and seconds channel        1.5 million

Wholesale grew slowest at 9 percent year over year. Management attributes this
to cafe closures in the Pacific Northwest during the first half of the year.
""",
    """6. Subscriptions

The direct-to-consumer subscription programme reached 21,300 active subscribers
in December 2024, up from 14,900 a year earlier. Monthly churn averaged
4.1 percent across the year, improving to 3.6 percent in the fourth quarter.

The average subscriber ordered 1.6 bags per month at an average order value of
21.40 dollars. Customer acquisition cost was 38 dollars, giving a payback period
of roughly four months at current margins.

Subscribers on the annual prepay plan, introduced in February 2024, churn at
1.2 percent monthly - roughly a third of the monthly-plan rate. Annual prepay
accounted for 18 percent of subscribers at year end.

7. Retail

The two Portland storefronts generated 2.4 million dollars, up 6 percent. The
Hawthorne location outperformed the Pearl District location by a factor of 1.8
on revenue per square foot. A third storefront was considered and deferred to
2026.

Retail also serves as the primary channel for new subscriber acquisition:
31 percent of 2024 subscription signups originated from an in-store promotion
code.
""",
    """8. Sustainability

Both facilities ran on 100 percent renewable electricity from January 2024,
purchased through the local utility's green tariff rather than through offsets.

Packaging moved to a compostable multilayer film in August 2024, removing an
estimated 31 tonnes of plastic per year. The transition raised packaging cost per
bag by 4.1 cents, absorbed rather than passed to customers.

Roasting itself remains gas-fired. Direct roasting emissions were an estimated
310 tonnes CO2e in 2024. The company has committed to evaluating electric
roasters during 2025 but has not budgeted for replacement.

9. Outlook for 2025

Management expects revenue between 21 and 23 million dollars. The main planned
investments are a third roasting facility in Denver, Colorado, and an expansion
of the Cold Brew Concentrate line into two additional flavours.

10. Principal Risks

Green coffee price volatility is the principal risk, since direct-trade contracts
are renegotiated each year and the company does not hedge on the futures market.

Secondary risks are supplier concentration in Colombia, dependence on a single
bonded warehouse in Sacramento, and the fact that 47 percent of revenue rests on
one product line.
""",
]

HANDBOOK = [
    """Meridian Coffee Roasters
Employee Handbook

Revision 7, effective 1 March 2025.
This handbook supersedes all previous versions. It is not an employment contract.

1. Employment Basics

All new employees serve a probationary period of 90 calendar days. During
probation either party may end the employment relationship with one week of
notice. After probation the notice period is four weeks for both parties.

Employment is confirmed in writing by the People team. Verbal offers are not
binding on the company.

Meridian is an equal opportunity employer. Hiring, promotion and compensation
decisions are made without regard to race, colour, religion, sex, sexual
orientation, gender identity, national origin, age, disability or veteran status.

2. Working Hours

The standard working week is 40 hours. Production staff work one of three fixed
shifts: 05:00-13:30, 13:00-21:30, or 21:00-05:30. Shift assignments rotate on a
four-week cycle and are published at least fourteen days in advance.

Corporate and support staff set their own hours within a core window of 10:00 to
15:00 Pacific time, during which they are expected to be reachable.

Overtime for hourly staff is paid at 1.5 times the base rate beyond 40 hours in a
week, and at 2.0 times on company holidays. Overtime must be approved in advance
by a shift lead. Unapproved overtime is still paid, but repeated instances are
addressed through the performance process.
""",
    """3. Remote and Hybrid Work

Roles that do not require physical presence may work remotely up to three days
per week. Fully remote arrangements require director approval and are reviewed
every six months.

Production, warehouse and retail roles are on-site by nature and are not
eligible for remote work. The company does not offer a stipend for home office
equipment, but will ship a laptop, monitor and chair on request.

Employees working remotely must be within four hours of Pacific time unless an
exception is granted. Employees may work from outside the United States for up
to fifteen working days per calendar year, subject to written approval from both
their manager and the People team at least thirty days ahead.

4. Leave and Time Off

Full-time employees accrue 22 days of paid annual leave per year, accruing at
1.83 days per month from the first day of employment. Leave may be taken during
probation with manager approval.

Up to five unused days may be carried into the following year. They expire on
31 March. Days beyond five are forfeited at year end and are not paid out,
except on termination, where all accrued and unused leave is paid at the base
rate.

Paid sick leave is 10 days per year and does not accrue or carry over. A
doctor's note is required for absences longer than three consecutive days.

Parental leave is 16 weeks at full pay for the primary caregiver and 8 weeks at
full pay for the secondary caregiver, available after 12 months of service.
Bereavement leave is 5 days for an immediate family member and 2 days otherwise.
""",
    """5. Compensation and Benefits

Salaries are reviewed once per year, in March, with any change effective 1 April.
Meridian benchmarks against the 60th percentile of the Portland market for
comparable roles.

Health insurance begins on the first day of the month following the start date.
The company pays 85 percent of the employee premium and 60 percent of dependent
premiums.

The retirement plan matches employee contributions dollar for dollar up to
4 percent of salary. Matching contributions vest immediately; there is no
vesting cliff.

All employees receive two 340 g bags of coffee per month at no cost, and a
40 percent discount on all products for personal use. The discount may not be
used to buy for resale.

Employees who refer a candidate who is hired and completes probation receive a
1,000 dollar referral bonus, paid in the following payroll cycle.

6. Performance

Formal performance reviews are held twice a year, in February and August.
Reviews are written by the direct manager and countersigned by a director.
Ratings do not follow a forced distribution.

Employees may request a written response to any review, which is retained in the
personnel file alongside the review itself.
""",
    """7. Conduct

Employees must disclose any outside employment or consulting that touches the
coffee, food or beverage industry. Other outside work does not require
disclosure but must not be performed during working hours or with company
equipment.

Gifts from suppliers with a value above 75 dollars must be declined or
surrendered to the People team. Meals and samples in the ordinary course of
business are exempt.

Harassment and discrimination are grounds for immediate dismissal. Reports may
be made to any manager, to the People team, or through the anonymous reporting
line, which is operated by a third party and does not log caller identity.

The company prohibits retaliation against anyone who makes a report in good
faith or participates in an investigation.

8. Health and Safety

Closed-toe shoes are mandatory anywhere in the roasting and warehouse areas.
Hearing protection is mandatory within three metres of an operating roaster.

All production staff complete a safety induction before their first shift and a
refresher every twelve months. Records are kept for five years.

Any injury, however minor, must be reported to a shift lead before the end of
the shift in which it occurred. Near misses must also be reported; the company
does not discipline employees for reporting near misses.
""",
    """9. Equipment and Information

Company laptops are issued to corporate and support staff and remain company
property. Personal use is permitted within reason.

Employees must use the company password manager for all work accounts.
Multi-factor authentication is mandatory on email, payroll and the production
systems. Passwords may not be shared, including with a manager.

Customer data may not be copied to personal devices or personal cloud storage.
Requests to export customer data require written approval from a director.

Lost or stolen devices must be reported within 24 hours so the device can be
remotely wiped.

10. Leaving the Company

Resigning employees give four weeks of notice after probation. The company may
waive part of the notice period but will pay it in full where it does so.

On the final day, employees return their laptop, keys, badge and any company
credit card to their manager. Final pay, including all accrued and unused annual
leave, is issued on the next scheduled payroll date and not earlier.

Access to email and internal systems is revoked at 18:00 on the final working
day. Employees who wish to retain personal files should export them beforehand
and may ask the People team for help doing so.

Alumni remain eligible for the 40 percent product discount for twelve months.
""",
]

RUNBOOK = [
    """Meridian Platform Runbook
On-call reference for the subscription and commerce platform.
Last reviewed 14 February 2025.

1. System Overview

The platform is a Python service backed by PostgreSQL 16 and Redis 7, deployed on
three application nodes behind a load balancer. Static assets are served from a
CDN. Background jobs run on a separate worker pool of two nodes.

The subscription biller is the most critical component. It runs nightly at 02:00
Pacific and charges every subscription due that day. A failed biller run is a
Severity 1 incident regardless of time of day, because a missed run cannot simply
be retried the next night without double-charging.

Service level objectives:

  API availability            99.9 percent monthly
  p99 request latency         400 milliseconds
  Checkout success rate       99.5 percent
  Biller completion           by 04:00 Pacific

The error budget is 43 minutes of downtime per month. When more than half the
budget is consumed, feature deploys pause until the following month.

2. Environments

There are three environments: production, staging and development. Staging runs
the same topology at one third the capacity and holds anonymised data refreshed
weekly. Development uses synthetic fixtures only; production data may never be
copied into development.
""",
    """3. Deployment

Deploys go out through the CI pipeline on merge to main. A deploy takes roughly
eleven minutes end to end and is a rolling restart, one node at a time, with a
health check gate between nodes.

Deploys are frozen from 20:00 Pacific on Friday until 08:00 Monday, and during
the last three days of any month, when subscription billing volume peaks.

To roll back, re-run the previous successful pipeline from the CI interface.
Rollback takes about four minutes. Do not roll back by force-pushing to main.

Database migrations are applied separately from application deploys and always
before them. Migrations must be backwards compatible for one release, so that a
rollback of application code does not require a migration rollback.

Any migration expected to take longer than thirty seconds must be run outside
business hours and announced in advance.

4. Monitoring

Dashboards are in Grafana. The four dashboards that matter on call are API
Overview, Biller, Database, and Checkout Funnel.

Alerts route to PagerDuty. Severity 1 pages immediately, day or night. Severity 2
pages during business hours and queues overnight. Severity 3 opens a ticket and
never pages.

Logs are retained for 30 days in the log store and for 400 days in cold storage.
Traces are sampled at 5 percent of requests, raised to 100 percent automatically
for any request that returns a 5xx status.
""",
    """5. Incident Response

The first responder is whoever is paged. Their first duty is to acknowledge
within five minutes, not to fix the problem.

For Severity 1, open an incident channel, post the current impact in plain
language, and name an incident commander. The commander coordinates and does not
debug. If the first responder is the only person available, they are the
commander until relieved.

Declare Severity 1 when checkout is failing for more than 5 percent of attempts,
when the biller has not completed by 04:00, or when the API is returning 5xx for
more than 2 percent of requests for longer than five minutes.

Declare Severity 2 for degraded but working service: elevated latency below the
5xx threshold, a single failed worker node, or a delayed but progressing biller
run.

Customer communication is the responsibility of the support lead, not the
engineer. Do not post status page updates directly.

Every Severity 1 and Severity 2 incident gets a written postmortem within five
working days. Postmortems are blameless and are published to the whole company.
The action items are tracked in the normal backlog and reviewed monthly.
""",
    """6. Database Operations

PostgreSQL runs as a primary with two streaming replicas, one in the same region
and one cross-region. Replication lag is alerted above ten seconds.

Backups are taken as a full base backup nightly at 01:00 with continuous WAL
archiving. The recovery point objective is 5 minutes and the recovery time
objective is 15 minutes.

Restores are tested monthly against the staging environment. A restore that has
not been exercised in the current quarter is treated as untested and may not be
relied on in an incident.

Failover to the same-region replica is automatic after 30 seconds of primary
unavailability. Cross-region failover is manual and requires director approval,
because it cannot be reversed without data loss.

Never run a manual UPDATE or DELETE against production without a transaction and
a colleague watching. Always run the SELECT form of the query first and confirm
the row count.

Long-running queries above 60 seconds are terminated automatically. If a report
needs longer, run it against the same-region replica instead of the primary.

Connection pool size is 40 per application node. Pool exhaustion presents as
timeouts rather than errors, which is the most common cause of a confusing
latency alert.
""",
    """7. The Biller

The nightly biller is the component most likely to page you.

It runs at 02:00 Pacific, processes subscriptions in batches of 500, and writes
an idempotency key for every charge attempt before calling the payment provider.
That key is what makes a retry safe.

If the biller fails partway, do not restart it from the beginning without first
confirming the idempotency keys are present for the completed batches. The
restart command takes a batch offset for this reason.

A biller run that has not completed by 04:00 is a Severity 1. The usual causes,
in order of observed frequency, are payment provider timeouts, connection pool
exhaustion, and a migration that changed a column the biller reads.

Payment provider timeouts are retried three times with exponential backoff.
After the third failure the subscription is marked for manual review rather than
being retried indefinitely, and appears on the Biller dashboard.

Never issue refunds directly against the payment provider. Refunds must go
through the admin interface so the ledger stays consistent.
""",
    """8. Access and Security

Production access requires a hardware security key. Passwords alone are not
sufficient and there is no break-glass password.

Access is granted by role, reviewed quarterly, and revoked automatically after
60 days without use. Requesting access back is a two-minute process, so the
review errs on the side of revoking.

Secrets live in the secret manager and are injected at runtime. Secrets must
never be committed to the repository, written to logs, or pasted into a chat
channel. A secret that touches any of those is considered compromised and must
be rotated the same day.

Customer payment details are never stored by Meridian. The payment provider
returns a token and only the token is persisted.

9. Disaster Recovery

The disaster recovery plan assumes total loss of the primary region. Recovery
runs from the cross-region replica plus object storage backups.

The full recovery drill is performed twice a year, in April and October, and is
timed. The most recent drill, in October 2024, completed in 2 hours 40 minutes
against a target of 4 hours.

The runbook itself is mirrored to a location outside the primary region, because
a regional outage would otherwise take the recovery instructions with it.
""",
]

# The prior-year report exists to make the corpus hard. It mirrors the 2024
# structure heading for heading with different figures, so every numeric question
# has a plausible wrong answer sitting one document away. Retrieval that matches
# on topic alone fails here; that is the point.
ANNUAL_2023 = [
    """Meridian Coffee Roasters
2023 Annual Summary

1. Overview

Meridian Coffee Roasters is a specialty coffee company founded in 2016 in
Portland, Oregon.

As of 31 December 2023 Meridian employed 71 people, of whom 44 worked in
production and logistics, 16 in retail and customer support, and 11 in corporate
functions. The Sacramento facility opened in June 2023 and was still ramping at
year end.

2. Product Lines

Meridian sold through two product lines in 2023.

(a) Origin Series - single-origin beans from Ethiopia, Colombia and Guatemala,
roasted in 12 kg batches. Origin Series accounted for 46 percent of 2023 revenue.

(b) Daily Blend - the year-round house blend. Daily Blend accounted for
54 percent of 2023 revenue.

Cold Brew Concentrate was in development throughout 2023 and did not go on sale
until March 2024. It contributed no 2023 revenue.

3. Sourcing

The average price paid to producers in 2023 was 2.84 US dollars per pound,
roughly 59 percent above the C-market benchmark for the period.

Average time from landing to roast was 41 days. Meridian worked with 19 producer
partners across six countries in 2023.
""",
    """4. Roasting Operations

Combined 2023 roasted volume was 495 metric tonnes. Roast loss averaged
16.4 percent. Quality control rejected 2.6 percent of batches, redirecting them
to the discount channel.

The Sacramento 70 kg roaster was commissioned in June 2023 and ran below half
utilisation for the remainder of the year.

5. Financial Results

Total revenue for 2023 was 15.1 million US dollars, up 17 percent from
12.9 million in 2022. Gross margin was 38 percent.

Operating expenses were 6.1 million dollars. Net loss was 284,000 dollars.
2023 was the company's seventh consecutive unprofitable year.

Cash at year end was 1.1 million dollars against 1.8 million of debt.

Revenue by channel:

  Direct-to-consumer subscriptions    5.8 million
  Wholesale to cafes                  6.1 million
  Retail (two Portland storefronts)   2.3 million
  Discount and seconds channel        0.9 million

Wholesale was the largest channel in 2023. It was overtaken by subscriptions
during 2024.
""",
    """6. Subscriptions

The subscription programme reached 14,900 active subscribers in December 2023,
up from 9,400 a year earlier. Monthly churn averaged 5.7 percent.

The average subscriber ordered 1.5 bags per month at an average order value of
19.80 dollars. Customer acquisition cost was 44 dollars.

There was no annual prepay plan in 2023; it was introduced in February 2024.

7. Retail

The two Portland storefronts generated 2.3 million dollars in 2023. The
Hawthorne location outperformed the Pearl District location on revenue per
square foot by a factor of 1.6.

8. Sustainability

The Portland facility moved to renewable electricity in March 2023; Sacramento
followed in January 2024. Packaging remained conventional laminate throughout
2023; the compostable film transition began in August 2024.

Direct roasting emissions were an estimated 268 tonnes CO2e in 2023.

9. Outlook for 2024

Management expected revenue between 17 and 19 million dollars for 2024, and
identified the launch of Cold Brew Concentrate as the principal growth driver.
""",
]

CATALOG = [
    """Meridian Coffee Roasters
Wholesale Product Catalogue
Effective 1 January 2025. Prices are per unit, excluding tax and freight.

Origin Series

MER-ETH-250   Ethiopia Guji Natural, 250 g
              Tasting notes: blueberry, jasmine, dark chocolate.
              Process: natural. Altitude: 1,950-2,150 m. Roast: light.
              Wholesale 14.20 USD. Case of 12.

MER-COL-250   Colombia Huila Washed, 250 g
              Tasting notes: red apple, caramel, orange peel.
              Process: fully washed. Altitude: 1,600-1,850 m. Roast: light-medium.
              Wholesale 11.80 USD. Case of 12.

MER-GUA-250   Guatemala Antigua Washed, 250 g
              Tasting notes: cocoa, almond, green apple.
              Process: fully washed. Altitude: 1,500-1,700 m. Roast: medium.
              Wholesale 12.40 USD. Case of 12.

MER-ETH-1KG   Ethiopia Guji Natural, 1 kg
              Wholesale 48.00 USD. Case of 6.

Origin Series lots are seasonal. Availability is confirmed at order time and
substitutions are never made without written approval from the buyer.
""",
    """Daily Blend

MER-DB-340    Daily Blend, 340 g
              Tasting notes: milk chocolate, toasted nut, brown sugar.
              Components: Colombia 60 percent, Guatemala 25 percent, Brazil
              15 percent. Roast: medium.
              Wholesale 8.90 USD. Case of 20.

MER-DB-900    Daily Blend, 900 g
              Wholesale 21.50 USD. Case of 8.

MER-DB-5KG    Daily Blend, 5 kg foodservice bag
              Wholesale 104.00 USD. Sold singly.

The Daily Blend recipe is adjusted seasonally to hold the flavour profile
constant as crops change. Component percentages above are indicative.

Cold Brew Concentrate

MER-CB-1L     Cold Brew Concentrate, 1 litre
              Dilute 1:3 with water or milk. Yields roughly 4 litres.
              Shelf life 21 days refrigerated, unopened. 7 days after opening.
              Wholesale 16.75 USD. Case of 6.
              Requires refrigerated transport and storage below 4 degrees C.

MER-CB-5L     Cold Brew Concentrate, 5 litre bag-in-box
              Wholesale 74.00 USD. Sold singly.
""",
    """Ordering

Minimum opening order is 400 USD. Minimum reorder is 250 USD.

Orders received before 14:00 Pacific on a business day ship the following
business day. Orders after that cut-off ship in two business days.

Freight is free on orders above 750 USD within the continental United States.
Below that threshold, freight is charged at cost.

Refrigerated items ship Monday through Wednesday only, to avoid a weekend in
transit.

Payment Terms

New accounts are prepay for the first three orders. After that, net 30 terms are
available on application, subject to a credit check.

Invoices more than 15 days past due incur a 1.5 percent monthly late charge.
Accounts more than 45 days past due are placed on hold.

Volume Discounts

  Annual spend above  10,000 USD    3 percent
  Annual spend above  25,000 USD    5 percent
  Annual spend above  60,000 USD    8 percent

Discounts are applied as a rebate at the end of the calendar year, not at the
point of invoice.
""",
    """Freshness and Returns

All whole bean coffee is roasted to order and shipped within 72 hours of roast.
The roast date is printed on every bag; there is no separate best-before date.

Whole bean coffee is at its best between 7 and 28 days after roast. Ground
coffee is not recommended for wholesale accounts and is supplied only by
exception.

Returns are accepted within 14 days for unopened product in original packaging.
Coffee that has been opened may only be returned where there is a quality
defect, which must be reported with photographs within 7 days of delivery.

Refrigerated Cold Brew Concentrate cannot be returned under any circumstances
once it has left a Meridian facility, because the cold chain cannot be verified
after the fact.

Private Label

Meridian offers private label roasting at a minimum of 200 kg per SKU per year.
Lead time for a new private label SKU is 10 to 14 weeks including packaging
design and food-safety review. Private label pricing is quoted individually and
is not published in this catalogue.

Samples

Prospective accounts may request up to three 250 g samples at no charge. Further
samples are charged at wholesale price.
""",
    """Equipment and Support

Meridian does not sell or lease brewing equipment. Accounts spending above
20,000 USD annually receive one on-site barista training session per year at no
charge, delivered by a Meridian trainer.

Additional training sessions are 450 USD per half day plus travel.

Marketing Support

Accounts receive origin cards, a printed brew guide, and window decals at no
charge. Custom point-of-sale material is quoted separately.

Use of the Meridian name and logo requires written approval and must follow the
brand guidelines supplied on account opening. Approval is typically returned
within five business days.

Contact

Wholesale orders:      wholesale@meridiancoffee.example
Accounts and billing:  ar@meridiancoffee.example
Quality issues:        quality@meridiancoffee.example

The wholesale team is staffed 07:00 to 17:00 Pacific, Monday to Friday. Quality
issues are monitored seven days a week, because a quality problem in a cafe
cannot wait for Monday.
""",
]

SECURITY = [
    """Meridian Coffee Roasters
Information Security Policy

Version 3.1, approved by the board on 20 January 2025.
Applies to all employees, contractors and third parties with access to Meridian
systems or data.

1. Scope and Ownership

The Head of Engineering is the accountable owner of this policy. It is reviewed
annually and after any Severity 1 security incident.

This policy governs information security. It sits alongside, and does not
replace, the acceptable-use provisions in the Employee Handbook. Where the two
appear to conflict, this policy takes precedence for anything touching
production systems or customer data.

2. Data Classification

Meridian classifies data into four levels.

  Public        Published material. No restriction.
  Internal      Default for business data. Employees and contractors only.
  Confidential  Customer records, financials before publication, contracts.
                Access on a need-to-know basis, logged.
  Restricted    Authentication secrets, encryption keys, payment tokens.
                Access requires named approval and is reviewed monthly.

Data must be classified at creation. Unclassified data is treated as
Confidential until classified, not as Internal.
""",
    """3. Access Control

Access follows least privilege and is granted by role, never to an individual
directly. Role membership is reviewed quarterly by the system owner.

Multi-factor authentication is mandatory for all systems holding Internal data
or above. For production systems the second factor must be a hardware security
key; time-based codes and SMS are not accepted.

Shared accounts are prohibited. Where a vendor system cannot support individual
accounts, the exception must be recorded in the risk register with a named owner
and a review date.

Access is revoked within four hours of an employee's departure, and immediately
where the departure is involuntary. This is faster than the general system access
revocation described in the Employee Handbook, which applies to non-production
systems.

Dormant access is removed automatically after 60 days without use.

4. Passwords and Secrets

All work credentials live in the company password manager. Reuse of a work
password on any personal service is prohibited.

Application secrets are held in the secret manager and injected at runtime.
Secrets must never be committed to a repository, written to application logs,
included in an error report, or pasted into a chat or ticket.

Any secret exposed by any of those routes is treated as compromised and rotated
the same working day, whether or not there is evidence of misuse.
""",
    """5. Endpoint Security

Company laptops run full-disk encryption and a managed endpoint agent. Personal
devices may access company email and chat, but may not hold Confidential data
and may not access production systems.

Operating system and browser updates must be applied within 14 days of release,
and within 48 hours for a vulnerability rated critical.

Devices lock automatically after 5 minutes of inactivity.

A lost or stolen device must be reported within 24 hours so it can be remotely
wiped.

6. Vendor and Third-Party Risk

Any vendor processing Confidential or Restricted data requires a security review
before contract signature. The review covers data location, subprocessors,
breach notification terms and deletion on termination.

Vendors handling Restricted data must notify Meridian of a breach within
24 hours. Vendors handling Confidential data must notify within 72 hours.

The vendor register is reviewed twice a year. A vendor that has not been used
for twelve months has its access terminated.
""",
    """7. Secure Development

All code changes require review by someone other than the author before merge.
Changes touching authentication, payments or customer data require review by two
people, at least one of whom is on the platform team.

Dependencies are scanned on every pipeline run. A critical vulnerability blocks
the pipeline; a high vulnerability opens a ticket with a 14-day remediation
target.

Production data may never be copied into development. Staging holds anonymised
data only and is refreshed weekly.

Penetration testing is commissioned annually from an external firm. The most
recent test was completed in November 2024 and produced four findings: one
medium and three low. All four were remediated by 20 December 2024.

8. Logging and Monitoring

Authentication events, access to Confidential data, and all administrative
actions are logged centrally. Logs are immutable and retained for 400 days.

Logs must not contain secrets, full payment details, or customer passwords.
Where a log would otherwise capture such a field, it is redacted at source
rather than filtered later.
""",
    """9. Incident Response

A security incident is any event that compromises, or may compromise, the
confidentiality, integrity or availability of Meridian data.

Anyone who suspects an incident reports it immediately to the platform on-call.
There is no penalty for reporting something that turns out to be benign, and
staff are explicitly encouraged to report early rather than investigate first.

Severity 1 security incidents require notification to the Head of Engineering
and the CEO within one hour of confirmation.

Where personal data is involved, the Head of Engineering determines the
notification obligation. Regulatory notification deadlines are assumed to be
72 hours from confirmation unless legal advice says otherwise.

Every security incident receives a written postmortem within five working days,
following the same blameless format used for availability incidents.

10. Training and Exceptions

All staff complete security awareness training on joining and annually
thereafter. Engineers complete an additional secure-development module.

Any exception to this policy requires written approval from the Head of
Engineering, a documented compensating control, and an expiry date no more than
six months out. Exceptions without an expiry date are not granted.
""",
]

DOCS = {
    "meridian-annual-summary-2024.pdf": ANNUAL,
    "meridian-annual-summary-2023.pdf": ANNUAL_2023,
    "meridian-employee-handbook.pdf": HANDBOOK,
    "meridian-platform-runbook.pdf": RUNBOOK,
    "meridian-product-catalogue.pdf": CATALOG,
    "meridian-security-policy.pdf": SECURITY,
}


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCS.items():
        doc = pymupdf.open()
        for body in pages:
            page = doc.new_page()
            overflow = page.insert_textbox(
                pymupdf.Rect(58, 58, 552, 780), body, fontsize=10, fontname="helv"
            )
            if overflow < 0:
                raise SystemExit(f"{name}: page overflows by {overflow:.0f}pt — shorten it")
        doc.save(OUT / name)
        doc.close()
        print(f"wrote {name}  ({len(pages)} pages)")


if __name__ == "__main__":
    build()
