---
doc_id: RPT-004
source: "Sample Vendor Security Advisory"
title: "Continued Exploitation of Microsoft Exchange Server Vulnerabilities"
publication_date: "2021-05-01"
related_actors: ["APT28", "APT29"]
related_malware: ["Zebrocy", "WellMess"]
related_cves: ["CVE-2020-0688", "CVE-2021-26855"]
sectors: ["Government", "Healthcare", "Defense Contractors"]
---

# Continued Exploitation of Microsoft Exchange Server Vulnerabilities

Multiple state-sponsored threat actors, including APT28 and APT29, have been
observed exploiting vulnerabilities in Microsoft Exchange Server as part of
their intrusion sets. Two vulnerabilities in particular have seen widespread
exploitation:

- CVE-2020-0688 (validation key remote code execution), exploited by APT28's
  Zebrocy malware.
- CVE-2021-26855 (ProxyLogon SSRF), exploited by APT29's WellMess malware.

Both vulnerabilities affect Microsoft Exchange Server and allow attackers to
achieve remote code execution or authentication bypass on unpatched systems.
Organizations are urged to apply update software mitigations (M1051)
immediately and to monitor for indicators associated with Zebrocy and
WellMess activity.
