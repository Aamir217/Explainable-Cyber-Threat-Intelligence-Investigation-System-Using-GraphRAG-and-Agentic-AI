---
doc_id: RPT-005
source: "Sample Malware Analysis Report"
title: "Technical Profile: X-Agent Modular Implant"
publication_date: "2019-11-20"
related_actors: ["APT28"]
related_malware: ["X-Agent"]
related_cves: []
sectors: ["Defense Contractors"]
---

# Technical Profile: X-Agent Modular Implant

X-Agent is a modular implant attributed to APT28 and used across multiple
intrusion sets, including the Sofacy 2020 Intrusion Set. The implant supports
command execution through scripting interpreters (T1059) and communicates
with command-and-control infrastructure using application layer protocols
(T1071), typically HTTP or HTTPS.

X-Agent has been deployed alongside other APT28 tooling such as Zebrocy and
Cobalt Strike in coordinated campaigns against defense contractors, with
initial access most commonly achieved via spearphishing attachments
(T1566.001).
