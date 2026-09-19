---
doc_id: RPT-002
source: "Sample National CERT Advisory"
title: "Operation Ghost: APT29 WellMess Activity Against Healthcare and Government"
publication_date: "2021-04-10"
related_actors: ["APT29"]
related_malware: ["WellMess", "Cobalt Strike"]
related_cves: ["CVE-2021-26855"]
sectors: ["Healthcare", "Government"]
---

# Operation Ghost: APT29 WellMess Activity Against Healthcare and Government

This advisory describes "Operation Ghost", a campaign attributed to APT29
(Cozy Bear / The Dukes / Nobelium) targeting healthcare and government entities.
The group used the WellMess malware family for persistent access and data
exfiltration over application-layer protocols (T1071) and scripting-based
execution (T1059).

Investigators identified that APT29 operators exploited the Microsoft Exchange
Server Server-Side Request Forgery vulnerability, known as ProxyLogon
(CVE-2021-26855), to gain an initial foothold before deploying WellMess and,
in several cases, Cobalt Strike beacons for follow-on access.

Both APT28 and APT29 have been separately observed leveraging spearphishing
attachments (T1566.001) as an initial access technique, underscoring the
continued effectiveness of phishing-based intrusion against both healthcare
and government sector targets.
