---
doc_id: RPT-003
source: "Sample Incident Response Writeup"
title: "Cobalt Strike Deployment via VMware Workspace ONE Command Injection"
publication_date: "2020-12-18"
related_actors: ["APT29"]
related_malware: ["Cobalt Strike"]
related_cves: ["CVE-2020-4006"]
sectors: ["Government"]
---

# Cobalt Strike Deployment via VMware Workspace ONE Command Injection

A joint incident response effort identified exploitation of a command
injection vulnerability in VMware Workspace ONE Access and Identity Manager
(CVE-2020-4006) as the initial access vector in an intrusion later attributed
with moderate confidence to APT29 infrastructure.

Following exploitation, the attackers deployed Cobalt Strike beacons that
communicated over application-layer protocols (T1071) and used
command-and-scripting-interpreter techniques (T1059) for post-exploitation
activity. This aligns with APT29's broader tooling preferences observed in
Operation Ghost and related campaigns.

Defenders are advised to prioritize patching internet-facing identity and
access management infrastructure, as command-injection vulnerabilities in
such systems provide a direct path to full network compromise.
