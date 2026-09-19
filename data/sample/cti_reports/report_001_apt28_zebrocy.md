---
doc_id: RPT-001
source: "Sample CTI Vendor Report"
title: "APT28 Zebrocy Campaign Targeting Government Networks"
publication_date: "2020-04-15"
related_actors: ["APT28"]
related_malware: ["Zebrocy", "X-Agent"]
related_cves: ["CVE-2020-0688"]
sectors: ["Government", "Defense Contractors"]
---

# APT28 Zebrocy Campaign Targeting Government Networks

Researchers observed the threat actor APT28 (also tracked as Fancy Bear or Sofacy)
conducting a spearphishing campaign against government and defense-sector targets
in early 2020. The initial access vector relied on spearphishing attachments
(MITRE ATT&CK T1566.001) delivering the Zebrocy malware family.

Once deployed, Zebrocy was used to move laterally through exploitation of remote
services, and in several confirmed intrusions the operators leveraged the
Microsoft Exchange validation key remote code execution vulnerability
(CVE-2020-0688) to gain code execution on unpatched Exchange servers.

APT28 has historically also relied on the X-Agent implant as part of the same
intrusion set (tracked internally as "Sofacy 2020 Intrusion Set"), using
command-and-scripting-interpreter techniques (T1059) and application layer
protocols (T1071) for command and control communications.

Organizations running Microsoft Exchange Server are strongly advised to apply
the relevant security updates and to provide user training against phishing
(MITRE mitigation M1017) as a primary defense against this activity.
