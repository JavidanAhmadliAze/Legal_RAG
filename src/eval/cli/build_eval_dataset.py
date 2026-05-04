#!/usr/bin/env python3
"""
Build a comprehensive RAG evaluation dataset for both retrieval AND generation.

For each predefined query, this builder:
  1. Optionally restricts the candidate pool to a set of expected acts
     (``expected_acts``) — gives clean ground truth for narrow-scope questions.
     Otherwise falls back to title-keyword filtering (same as the v1 builder).
  2. Runs hybrid BM25 + k-NN against the index, then cross-encoder reranks.
  3. Assigns graded relevance labels (0–3) to the top-N chunks.
  4. Validates each spec: warns if ``must_mention`` strings are missing from
     the labelled chunks, or if an out-of-scope query surfaces a chunk above
     a high rerank threshold (corpus leak).
  5. Emits a build report alongside the dataset.

Each entry has the fields needed by every evaluator in ``src/eval/``:

    Document Retrieval (process eval, ground_truth labels):
        ground_truth        ── chunk_id → relevance 0-3

    Retrieval (LLM judge, no labels needed):
        query, polish_query

    Groundedness, Relevance (LLM judges, no labels needed):
        query

    Response Completeness (process eval, ground-truth answer):
        ground_truth_answer ── reference answer (1–3 sentences)

    Hard-rule checks (cheap programmatic validators):
        must_mention        ── substrings the answer MUST contain
        must_not_mention    ── substrings the answer MUST NOT contain
        expected_acts       ── document_ids the citations should reference

Output:
    data/eval/ground_truth.json   — full dataset
    data/eval/build_report.json   — labelling diagnostics

Usage:
    python -m src.eval.cli.build_eval_dataset
    python -m src.eval.cli.build_eval_dataset --only eu_long_term_5year_clock
    python -m src.eval.cli.build_eval_dataset --top-n 8
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.embedding import embed_query
from src.services.opensearch import INDEX_NAME, get_client
from src.services.reranking import rerank


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class QuerySpec:
    id: str
    query: str
    polish_query: str
    topic: str
    query_type: str            # eligibility | procedure | threshold_numeric |
                               # document_form | authority | penalty | cross_ref |
                               # multi_hop | recent_amendment | adversarial |
                               # out_of_scope
    difficulty: str            # easy | medium | hard
    law_domain: str            # immigration | administrative | ... | mixed | none
    ground_truth_answer: str   # short reference answer (1–3 sentences)

    expected_acts: list[str] = field(default_factory=list)
    title_keywords: list[str] = field(default_factory=list)
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    notes: str = ""
    is_out_of_scope: bool = False


# ---------------------------------------------------------------------------
# Query catalogue
# ---------------------------------------------------------------------------
#
# Coverage rationale:
#   - 28 immigration queries spanning every major sub-topic in the act:
#     residence permits (temp/perm/EU long-term), Schengen+national visas,
#     Karta Polaka, repatriation, Ukrainian temporary protection / PESEL UKR,
#     citizenship, work permits, asylum/international protection, application
#     formalities (forms, photos, fingerprints, MOS portal), border re-entry,
#     family-related grounds, recent 2025/2026 amendments.
#   - 6 administrative queries covering subsidies, e-Doręczenia, ePUAP.
#   - 4 cross-domain hard queries (immigration × work, immigration × delivery).
#   - 6 out-of-scope queries.
#   - Difficulty stratified easy/medium/hard.
#   - Query types stratified to exercise every evaluator dimension.

QUERIES: list[QuerySpec] = [

    # ════════════════════════════════════════════════════════════════════
    # Immigration — residence permits
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="eu_long_term_5year_clock",
        query=(
            "I'm Indian, I've been in Poland on consecutive temporary residence "
            "permits for 4 years and 11 months. My employer wants to move me to "
            "a permanent contract. When can I apply for EU long-term resident status?"
        ),
        polish_query="rezydent długoterminowy UE pięć lat nieprzerwany pobyt cudzoziemiec",
        topic="EU long-term residence — 5-year clock",
        query_type="threshold_numeric",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "EU long-term resident status requires 5 years of uninterrupted legal "
            "stay in Poland immediately preceding the application. At 4 years and "
            "11 months you cannot yet apply — you must wait until the full 5 years "
            "have elapsed."
        ),
        expected_acts=["WDU20250001079", "WDU20250001794"],
        title_keywords=["cudzoziemcach"],
        must_mention=["5 lat", "nieprzerwan"],
        must_not_mention=["3 lata", "10 lat"],
        notes="EU long-term resident requires 5 years of uninterrupted legal stay (Art. 195).",
    ),

    QuerySpec(
        id="permanent_residence_after_marriage",
        query=(
            "My wife is Polish and we have been married for 2 years and 8 months. "
            "I've been on a temporary residence permit the whole time. "
            "Can I already apply for permanent residence based on our marriage?"
        ),
        polish_query="zezwolenie pobyt stały małżeństwo obywatel polski trzy lata",
        topic="Permanent residence by marriage to a Polish citizen",
        query_type="threshold_numeric",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "Permanent residence based on marriage to a Polish citizen requires the "
            "marriage to have lasted at least 3 years AND the applicant to have "
            "resided in Poland uninterruptedly for at least the last 2 years on a "
            "temporary residence permit granted on that basis. At 2 years 8 months "
            "of marriage, the applicant does not yet meet the marriage threshold."
        ),
        expected_acts=["WDU20250001079"],
        title_keywords=["cudzoziemcach"],
        must_mention=["3 lata", "małżeń"],
    ),

    QuerySpec(
        id="temporary_permit_max_duration",
        query="What is the maximum duration of a single temporary residence permit?",
        polish_query="zezwolenie pobyt czasowy okres maksymalny trzy lata",
        topic="Temporary residence permit — maximum duration",
        query_type="threshold_numeric",
        difficulty="easy",
        law_domain="immigration",
        ground_truth_answer=(
            "A temporary residence permit is granted for the period necessary to "
            "achieve the purpose of stay, but no longer than 3 years (Art. 98 ust. 2)."
        ),
        expected_acts=["WDU20250001079"],
        title_keywords=["cudzoziemcach"],
        must_mention=["3 lata", "Art. 98"],
    ),

    QuerySpec(
        id="temporary_permit_application_deadline",
        query=(
            "I'm currently in Poland on a national visa that expires in two weeks. "
            "By when must I file my temporary residence permit application?"
        ),
        polish_query="wniosek zezwolenie pobyt czasowy termin ostatni dzień legalnego pobytu",
        topic="Application deadline — last day of legal stay",
        query_type="procedure",
        difficulty="easy",
        law_domain="immigration",
        ground_truth_answer=(
            "The application must be submitted no later than the last day of the "
            "applicant's legal stay in Poland (Art. 105 ust. 1)."
        ),
        expected_acts=["WDU20250001079", "WDU20250001794"],
        title_keywords=["cudzoziemcach"],
        must_mention=["ostatni", "Art. 105"],
    ),

    QuerySpec(
        id="visa_rejection_appeal",
        query=(
            "My Schengen visa application was rejected by the Polish consulate. "
            "Can I appeal that decision, and what is the procedure?"
        ),
        polish_query="odmowa wizy odwołanie konsul ponowne rozpatrzenie",
        topic="Visa refusal — consul reconsideration",
        query_type="procedure",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "A consul's refusal of a Schengen visa is subject to a request for "
            "reconsideration (wniosek o ponowne rozpatrzenie sprawy) addressed to "
            "the same consul within 14 days of the decision being delivered."
        ),
        expected_acts=["WDU20250001079"],
        title_keywords=["cudzoziemcach"],
        must_mention=["ponowne rozpatrzenie", "14 dni"],
    ),

    QuerySpec(
        id="border_reentry_stamp",
        query=(
            "My residence card expires next month. I need to travel abroad. "
            "How long can I be outside Poland before my legal stay is broken?"
        ),
        polish_query="powrót wjazd cudzoziemiec stempel przerwa pobyt karta",
        topic="Continuity of residence — absence from Poland",
        query_type="threshold_numeric",
        difficulty="hard",
        law_domain="immigration",
        ground_truth_answer=(
            "A stay is treated as uninterrupted if no single absence exceeds 6 months "
            "and the total of all absences during the qualifying period does not "
            "exceed 10 months."
        ),
        expected_acts=["WDU20250001079"],
        title_keywords=["cudzoziemcach"],
        must_mention=["6 miesięcy", "10 miesięcy"],
    ),

    # ════════════════════════════════════════════════════════════════════
    # Immigration — application formalities (forms, photos, fingerprints)
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="residence_photo_requirements",
        query=(
            "I want to file a permanent-residence application this week. I have a "
            "studio photo from 7 months ago — passport-style, correct size. Can I use it?"
        ),
        polish_query="fotografia wniosek zezwolenie pobyt sześć miesięcy wymagania",
        topic="Application photo — 6-month freshness rule",
        query_type="threshold_numeric",
        difficulty="easy",
        law_domain="immigration",
        ground_truth_answer=(
            "The photo attached to a residence application must be no older than "
            "6 months at the time of filing, so a 7-month-old photo cannot be used."
        ),
        expected_acts=["WDU20250001647", "WDU20260000487", "WDU20260000488"],
        title_keywords=["cudzoziem", "wniosk"],
        must_mention=["6 miesięcy"],
    ),

    QuerySpec(
        id="application_form_regulation",
        query=(
            "Which regulation specifies the form of the application for a temporary "
            "residence permit and how many photos must be attached?"
        ),
        polish_query="rozporządzenie wniosek zezwolenie pobyt czasowy formularz fotografia",
        topic="Application form — implementing regulation",
        query_type="document_form",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "The form template, attachments, photo count and technical photo "
            "requirements are set by Rozporządzenie Ministra Spraw Wewnętrznych "
            "i Administracji z dnia 25 listopada 2025 r. (Dz.U. 2025 poz. 1647)."
        ),
        expected_acts=["WDU20250001647"],
        title_keywords=["wniosk", "cudzoziem"],
        must_mention=["1647"],
    ),

    QuerySpec(
        id="certificate_of_application_template",
        query=(
            "What document confirms that I've submitted my temporary residence "
            "application, and which regulation defines its template?"
        ),
        polish_query="zaświadczenie potwierdzające złożenie wniosek zezwolenie pobyt czasowy wzór",
        topic="Certificate of application — regulation template",
        query_type="document_form",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "Filing is confirmed by a certificate (zaświadczenie) whose template is "
            "set by Rozporządzenie MSWiA z dnia 18 marca 2026 r. (Dz.U. 2026 poz. 386)."
        ),
        expected_acts=["WDU20260000386"],
        title_keywords=["zaświadcz", "cudzoziem"],
        must_mention=["386", "zaświadcz"],
    ),

    QuerySpec(
        id="application_form_contents",
        query=(
            "What information must I provide on the temporary residence permit "
            "application form?"
        ),
        polish_query="formularz wniosek zezwolenie pobyt czasowy dane cudzoziemiec rodzina podróże",
        topic="Application form — required fields",
        query_type="document_form",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "The form must contain the foreigner's personal data, information about "
            "family members in Poland, previous and current stays in Poland, travels "
            "and stays abroad in the last 5 years, financial means, health insurance, "
            "the declared purpose of stay, information about detention or arrest, "
            "obligations from court/administrative decisions, justification, and a "
            "truthfulness declaration (Art. 106)."
        ),
        expected_acts=["WDU20250001794", "WDU20250001079"],
        title_keywords=["cudzoziemcach"],
        must_mention=["Art. 106"],
    ),

    QuerySpec(
        id="amendment_of_residence_permit",
        query=(
            "I have a temporary residence permit for work and my employer is changing "
            "me to a new position. What is the procedure for amending the permit?"
        ),
        polish_query="zmiana zezwolenie pobyt czasowy praca cudzoziemiec wniosek formularz",
        topic="Amendment of a temporary residence + work permit",
        query_type="procedure",
        difficulty="hard",
        law_domain="immigration",
        ground_truth_answer=(
            "An amendment is requested on a form containing the foreigner's data, "
            "family members in Poland, health insurance, and a truthfulness "
            "declaration. A completed appendix from the entity entrusting work and "
            "documents confirming the circumstances justifying the amendment must be "
            "attached (Art. 120a)."
        ),
        expected_acts=["WDU20250001079"],
        title_keywords=["cudzoziemcach"],
        must_mention=["Art. 120a"],
    ),

    # ════════════════════════════════════════════════════════════════════
    # Immigration — Karta Polaka, repatriation, citizenship
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="karta_polaka_eligibility",
        query="Who can apply for a Karta Polaka? What are the eligibility criteria?",
        polish_query="karta polaka pochodzenie polskie narodowość organizacja",
        topic="Karta Polaka — eligibility",
        query_type="eligibility",
        difficulty="easy",
        law_domain="immigration",
        ground_truth_answer=(
            "A Karta Polaka may be granted to a person of Polish origin who declares "
            "belonging to the Polish nation, demonstrates basic knowledge of the "
            "Polish language and tradition, and either (a) shows that at least one "
            "parent/grandparent or two great-grandparents were of Polish nationality, "
            "or (b) has been actively engaged in Polish-community organisations for "
            "at least the last 3 years."
        ),
        expected_acts=["WDU20250001868", "WDU20260000203"],
        title_keywords=["polaka", "polaków"],
        must_mention=["pochodzenie", "polski"],
    ),

    QuerySpec(
        id="repatriation_visa_rights",
        query=(
            "My grandmother is of Polish origin from Kazakhstan. What rights does she "
            "have if she comes to Poland on a repatriation visa? Is housing provided?"
        ),
        polish_query="wiza repatriacyjna repatriant prawa lokal mieszkalny obywatelstwo",
        topic="Repatriation visa — rights and housing",
        query_type="eligibility",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "A repatriant acquires Polish citizenship at the moment of crossing the "
            "Polish border on a repatriation visa, and is entitled to a one-off "
            "settlement allowance, partial reimbursement of education costs, and — "
            "where they have no place to live — provision of housing through a "
            "commune that has accepted them."
        ),
        expected_acts=["WDU20250001805"],
        title_keywords=["repatri"],
        must_mention=["obywatelstw", "lokal"],
    ),

    QuerySpec(
        id="citizenship_acquisition_procedure",
        query=(
            "I have lived in Poland for 8 years on permanent residence. I'd like to "
            "apply for Polish citizenship. What is the procedure and what documents "
            "do I need?"
        ),
        polish_query="obywatelstwo polskie nadanie uznanie wniosek dokumenty wojewoda",
        topic="Polish citizenship by recognition (uznanie)",
        query_type="procedure",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "After at least 3 years of permanent residence (or sometimes shorter on "
            "specific grounds), citizenship by recognition is requested at the "
            "voivode's office. The applicant must show stable income, legal title to "
            "housing, and Polish-language competence (B1 certificate)."
        ),
        expected_acts=["WDU20250001079"],
        title_keywords=["obywatelstw"],
        must_mention=["wojewod"],
    ),

    # ════════════════════════════════════════════════════════════════════
    # Immigration — Ukrainian temporary protection
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="temporary_protection_absence",
        query=(
            "My Ukrainian flatmate plans to visit her family back home for about "
            "five or six weeks this summer. She lives in Poland under temporary "
            "protection. Could a trip that long cause her to lose her right to stay?"
        ),
        polish_query="ochrona tymczasowa obywatel ukrainy wyjazd 30 dni utrata",
        topic="Temporary protection — absence beyond 30 days",
        query_type="threshold_numeric",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "An absence from Poland exceeding 30 consecutive days causes a Ukrainian "
            "citizen to lose temporary-protection status under the special act."
        ),
        expected_acts=["WDU20240000086"],
        title_keywords=["ukrai", "ochron"],
        must_mention=["30 dni"],
    ),

    QuerySpec(
        id="pesel_ukr_registration_deadline",
        query=(
            "A Ukrainian woman crossed into Poland three weeks ago and hasn't got a "
            "PESEL yet. Is there a legal deadline by which she must register?"
        ),
        polish_query="PESEL UKR rejestracja 30 dni przekroczenie granicy obywatel ukrainy",
        topic="PESEL UKR — registration deadline",
        query_type="threshold_numeric",
        difficulty="easy",
        law_domain="immigration",
        ground_truth_answer=(
            "There is no fixed statutory deadline by which a Ukrainian citizen must "
            "obtain a PESEL UKR; registration is optional but is required to access "
            "many state benefits and confirms legal residence under temporary "
            "protection."
        ),
        expected_acts=["WDU20240000086"],
        title_keywords=["ukrainy", "pesel"],
        must_mention=["PESEL"],
    ),

    # ════════════════════════════════════════════════════════════════════
    # Immigration — work
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="seasonal_work_permit_activities",
        query=(
            "Which sectors can issue seasonal work permits to foreigners under the "
            "current regulation?"
        ),
        polish_query="zezwolenie praca sezonowa cudzoziemiec rolnictwo ogrodnictwo turystyka",
        topic="Seasonal work permits — eligible sectors",
        query_type="eligibility",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "Seasonal work permits cover activities in agriculture, horticulture and "
            "tourism specified in Rozporządzenie Rady Ministrów z dnia 25 listopada "
            "2025 r. (Dz.U. 2025 poz. 1654)."
        ),
        expected_acts=["WDU20250001654"],
        title_keywords=["praca sezonow"],
        must_mention=["sezon"],
    ),

    QuerySpec(
        id="business_employment_threshold",
        query=(
            "I run a hairdressing salon as a sole-trader foreigner on a temporary "
            "residence permit. I'm hiring my second Polish employee next month. Does "
            "having two employees change my income threshold for renewing my "
            "business-based residence permit?"
        ),
        polish_query="zezwolenie działalność gospodarcza próg dochodu zatrudnienie pracowników cudzoziemiec",
        topic="Business-based permit — eased income test on hiring",
        query_type="threshold_numeric",
        difficulty="hard",
        law_domain="immigration",
        ground_truth_answer=(
            "An entrepreneur who employs at least 2 full-time workers (in particular "
            "Polish or EU citizens, or foreigners holding eligible long-stay titles) "
            "qualifies under an eased income/turnover test for a business-based "
            "permit renewal."
        ),
        expected_acts=["WDU20250001079"],
        title_keywords=["cudzoziem"],
        must_mention=["zatrudnia"],
    ),

    QuerySpec(
        id="work_permit_change_no_new_required",
        query=(
            "I changed my company's registered address and the worker's job title. "
            "Do I need to apply for a new temporary residence + work permit for him?"
        ),
        polish_query="zmiana siedziba pracodawca stanowisko zezwolenie pobyt praca nowe",
        topic="Work permit — changes that do NOT require a new permit",
        query_type="procedure",
        difficulty="hard",
        law_domain="immigration",
        ground_truth_answer=(
            "A new permit is not required when the employer's registered seat, name "
            "or legal form changes, when the workplace is transferred to another "
            "employer, when working time increases proportionally with remuneration, "
            "or when the job title changes without a change in the scope of duties "
            "(Art. 119). The employer must notify the voivode in writing within 15 "
            "working days."
        ),
        expected_acts=["WDU20250001079"],
        title_keywords=["cudzoziemcach"],
        must_mention=["Art. 119", "15"],
    ),

    # ════════════════════════════════════════════════════════════════════
    # Immigration — asylum & international protection
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="asylum_international_protection",
        query=(
            "I fled my country because of religious persecution. What is the difference "
            "between refugee status and subsidiary protection in Poland, and what rights "
            "do I get under each?"
        ),
        polish_query="status uchodźca ochrona uzupełniająca prześladowanie wniosek",
        topic="Refugee status vs subsidiary protection",
        query_type="eligibility",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "Refugee status is granted when the applicant has a well-founded fear of "
            "persecution on account of race, religion, nationality, political opinion "
            "or membership in a social group. Subsidiary protection is granted when "
            "the applicant does not meet the refugee definition but would face a real "
            "risk of serious harm (death penalty, torture, indiscriminate violence) "
            "if returned. Both statuses confer the right to legal stay, work, social "
            "assistance and a residence card."
        ),
        title_keywords=["ochron", "uchodź", "azyl"],
        must_mention=["uchodź", "uzupełniaj"],
    ),

    # ════════════════════════════════════════════════════════════════════
    # Immigration — recent (2025/2026) amendments
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="november_2025_amendment_summary",
        query=(
            "What are the main changes introduced by the November 2025 amendment to "
            "the Foreigners Act?"
        ),
        polish_query="ustawa zmiana cudzoziemcach listopad 2025 nowelizacja",
        topic="Foreigners Act — November 2025 amendment overview",
        query_type="recent_amendment",
        difficulty="hard",
        law_domain="immigration",
        ground_truth_answer=(
            "The November 2025 amending act (Dz.U. 2025 poz. 1794) revises rules on "
            "the application form for temporary residence permits, introduces new "
            "summons-and-deadlines provisions, and adjusts conditions for residence "
            "and work permits."
        ),
        expected_acts=["WDU20250001794"],
        title_keywords=["cudzoziemcach"],
        must_mention=["1794"],
    ),

    QuerySpec(
        id="april_2025_amendment_summary",
        query=(
            "What changed in the Foreigners Act with the April 2025 amendment?"
        ),
        polish_query="ustawa zmiana cudzoziemcach kwiecień 2025 nowelizacja",
        topic="Foreigners Act — April 2025 amendment overview",
        query_type="recent_amendment",
        difficulty="hard",
        law_domain="immigration",
        ground_truth_answer=(
            "The April 2025 amending act (Dz.U. 2025 poz. 619) introduced "
            "transitional and substantive changes including provisions on residence "
            "and certain administrative procedures."
        ),
        expected_acts=["WDU20250000619"],
        title_keywords=["cudzoziemcach"],
        must_mention=["619"],
    ),

    QuerySpec(
        id="mos_digital_portal_application",
        query=(
            "I heard that the MOS portal launched recently lets foreigners file "
            "residence applications digitally. Is paper still accepted, or must I file "
            "everything electronically now?"
        ),
        polish_query="MOS wniosek elektroniczny system teleinformatyczny portal cudzoziemiec",
        topic="MOS digital portal — electronic vs paper filing",
        query_type="procedure",
        difficulty="hard",
        law_domain="immigration",
        ground_truth_answer=(
            "Applications may be submitted through the dedicated information system "
            "(MOS) administered by the minister of internal affairs; paper filing "
            "remains available where the act permits. The act specifies which "
            "categories of applications must be filed in the MOS portal."
        ),
        title_keywords=["cudzoziem", "system teleinformat"],
        must_mention=["system teleinformat"],
    ),

    # ════════════════════════════════════════════════════════════════════
    # Immigration — adversarial / paraphrased
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="adv_short_query_eu_lt",
        query="EU long-term residence — how many years?",
        polish_query="rezydent długoterminowy UE liczba lat warunek",
        topic="Adversarial — minimal-token query",
        query_type="adversarial",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer="5 years of uninterrupted legal stay in Poland.",
        expected_acts=["WDU20250001079"],
        must_mention=["5 lat"],
        notes="Tests retrieval robustness on very short queries.",
    ),

    QuerySpec(
        id="adv_polish_only_query",
        query="Ile lat trzeba mieszkać w Polsce, żeby dostać kartę stałego pobytu UE?",
        polish_query="rezydent długoterminowy UE pięć lat nieprzerwany pobyt",
        topic="Adversarial — Polish-only query phrasing",
        query_type="adversarial",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer="5 lat nieprzerwanego legalnego pobytu w Polsce.",
        expected_acts=["WDU20250001079"],
        must_mention=["5 lat"],
    ),

    QuerySpec(
        id="adv_typo_karta_polaka",
        query="Wat are eligbility criteria for Karte Polaka?",
        polish_query="karta polaka pochodzenie polskie kryteria",
        topic="Adversarial — typos and misspellings",
        query_type="adversarial",
        difficulty="medium",
        law_domain="immigration",
        ground_truth_answer=(
            "See karta_polaka_eligibility — answer should match despite typos."
        ),
        expected_acts=["WDU20250001868", "WDU20260000203"],
        must_mention=["pochodzenie"],
    ),

    # ════════════════════════════════════════════════════════════════════
    # Administrative — e-Doręczenia, ePUAP, subsidies
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="e_doreczenia_fiction",
        query=(
            "An e-Doręczenia notice was delivered to my electronic mailbox 16 days ago "
            "and I haven't opened it. Is it considered legally delivered now?"
        ),
        polish_query="e-doręczenia fikcja doręczenia korespondencja elektroniczna 14 dni",
        topic="e-Doręczenia — delivery fiction after 14 days",
        query_type="threshold_numeric",
        difficulty="medium",
        law_domain="administrative",
        ground_truth_answer=(
            "Yes — under the e-Doręczenia rules a message left unread in the "
            "electronic mailbox is treated as delivered after 14 days from the date "
            "the notice was placed in the mailbox."
        ),
        title_keywords=["doręczeni", "cyfryz", "elektronic"],
        must_mention=["14 dni"],
    ),

    QuerySpec(
        id="digital_signature_public_services",
        query=(
            "Which digital signature methods are accepted for filing documents with "
            "Polish public administration today?"
        ),
        polish_query="podpis elektroniczny kwalifikowany profil zaufany ePUAP administracja",
        topic="Accepted digital signatures for public administration",
        query_type="eligibility",
        difficulty="easy",
        law_domain="administrative",
        ground_truth_answer=(
            "Documents filed with Polish public administration may be signed using a "
            "qualified electronic signature, a trusted profile (profil zaufany) on "
            "ePUAP, or a personal signature on the e-ID."
        ),
        title_keywords=["cyfryz", "informatyz", "elektronic"],
        must_mention=["profil zaufany"],
    ),

    QuerySpec(
        id="regional_funds_eligibility",
        query=(
            "What are the eligibility criteria for receiving regional development "
            "funds for an SME under the current operational programme?"
        ),
        polish_query="fundusze regionalne mali średni przedsiębiorcy program operacyjny pomoc",
        topic="Regional funds — SME eligibility",
        query_type="eligibility",
        difficulty="medium",
        law_domain="administrative",
        ground_truth_answer=(
            "Under the current operational programme, regional development funds are "
            "available to SMEs meeting EU SME definition, registered in the eligible "
            "region, and pursuing investment in the priority areas defined by the "
            "implementing regulation."
        ),
        title_keywords=["funduszy", "operacyjn", "regionaln"],
    ),

    # ════════════════════════════════════════════════════════════════════
    # Cross-domain hard queries
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="hard_immigration_plus_edoreczenia",
        query=(
            "If a voivode sends me a summons via e-Doręczenia and I don't open it, "
            "can my temporary residence permit application be discontinued?"
        ),
        polish_query="wezwanie wojewoda e-doręczenia fikcja doręczenia zezwolenie pobyt czasowy",
        topic="Cross-ref — e-Doręczenia delivery fiction in residence proceedings",
        query_type="cross_ref",
        difficulty="hard",
        law_domain="mixed",
        ground_truth_answer=(
            "Yes. A summons left unread in the e-Doręczenia mailbox is treated as "
            "delivered after 14 days, and the deadline set by the voivode (no less "
            "than 14 days for additional documents) starts running. Failing to "
            "respond may result in the application being discontinued."
        ),
        expected_acts=["WDU20250001079", "WDU20250001794"],
        title_keywords=["cudzoziemcach"],
        must_mention=["14 dni"],
    ),

    QuerySpec(
        id="hard_minor_application_curator",
        query=(
            "An unaccompanied 14-year-old foreign minor is in Poland. Who can submit "
            "the application for a temporary residence permit on his behalf?"
        ),
        polish_query="małoletni cudzoziemiec bez opieki kurator wniosek zezwolenie pobyt czasowy",
        topic="Application by an unaccompanied minor — curator",
        query_type="procedure",
        difficulty="hard",
        law_domain="immigration",
        ground_truth_answer=(
            "For an unaccompanied minor the application is submitted by an appointed "
            "curator (Art. 105 ust. 2 pkt 3)."
        ),
        expected_acts=["WDU20250001794", "WDU20250001079"],
        title_keywords=["cudzoziemcach"],
        must_mention=["kurator"],
    ),

    QuerySpec(
        id="hard_voivode_summons_deadline",
        query=(
            "How much time can the voivode give me to provide additional documents "
            "after issuing a summons during my residence application?"
        ),
        polish_query="wojewoda wezwanie dokumenty termin 14 dni zezwolenie pobyt cudzoziemiec",
        topic="Voivode summons — minimum response deadline",
        query_type="threshold_numeric",
        difficulty="hard",
        law_domain="immigration",
        ground_truth_answer=(
            "The voivode may set a deadline of no less than 14 days from the delivery "
            "of the summons (Art. 106f)."
        ),
        expected_acts=["WDU20250001794"],
        title_keywords=["cudzoziemcach"],
        must_mention=["14 dni", "Art. 106f"],
    ),

    QuerySpec(
        id="hard_foreigner_employer_threshold",
        query=(
            "I run a one-person consulting firm as a foreigner and I want to hire "
            "two employees next month. Will hiring them help me satisfy the income "
            "test for renewing my business-based residence permit?"
        ),
        polish_query="zezwolenie działalność gospodarcza próg dochodu zatrudnienie pracowników cudzoziemiec",
        topic="Income test on hiring — combined immigration + business question",
        query_type="multi_hop",
        difficulty="hard",
        law_domain="mixed",
        ground_truth_answer=(
            "Yes. Employing at least 2 eligible workers triggers an eased income test "
            "for the renewal of a business-based temporary residence permit."
        ),
        expected_acts=["WDU20250001079"],
        title_keywords=["cudzoziem"],
        must_mention=["zatrudnia"],
    ),

    # ════════════════════════════════════════════════════════════════════
    # Out of scope (corpus has no answer — system must refuse)
    # ════════════════════════════════════════════════════════════════════

    QuerySpec(
        id="oos_weather",
        query="What is the weather forecast for Warsaw next weekend?",
        polish_query="prognoza pogody warszawa weekend",
        topic="Out of scope — weather",
        query_type="out_of_scope",
        difficulty="easy",
        law_domain="none",
        ground_truth_answer=(
            "The system should refuse — this is outside Polish foreigners law."
        ),
        is_out_of_scope=True,
    ),

    QuerySpec(
        id="oos_recipe",
        query="Give me a recipe for traditional Polish bigos.",
        polish_query="przepis bigos kapusta kulinarny",
        topic="Out of scope — recipe",
        query_type="out_of_scope",
        difficulty="easy",
        law_domain="none",
        ground_truth_answer=(
            "The system should refuse — this is outside Polish foreigners law."
        ),
        is_out_of_scope=True,
    ),

    QuerySpec(
        id="oos_math",
        query="What is the integral of sin(x) divided by x from 0 to infinity?",
        polish_query="całka sinus matematyka analiza",
        topic="Out of scope — math problem",
        query_type="out_of_scope",
        difficulty="easy",
        law_domain="none",
        ground_truth_answer=(
            "The system should refuse — this is outside Polish foreigners law."
        ),
        is_out_of_scope=True,
    ),

    QuerySpec(
        id="oos_python_code",
        query="Write a Python function that reverses a linked list.",
        polish_query="programowanie python lista",
        topic="Out of scope — programming question",
        query_type="out_of_scope",
        difficulty="easy",
        law_domain="none",
        ground_truth_answer=(
            "The system should refuse — this is outside Polish foreigners law."
        ),
        is_out_of_scope=True,
    ),

    QuerySpec(
        id="oos_us_immigration",
        query="How do I apply for a US H-1B visa from Poland?",
        polish_query="wiza USA H-1B amerykańska procedura",
        topic="Out of scope — non-Polish immigration law",
        query_type="out_of_scope",
        difficulty="medium",
        law_domain="none",
        ground_truth_answer=(
            "The system should refuse — US immigration is not in the Polish law corpus."
        ),
        is_out_of_scope=True,
        notes="Adversarial OOS — topic is immigration but jurisdiction is wrong.",
    ),

    QuerySpec(
        id="oos_eu_directive_text",
        query="Give me the full text of EU Directive 2003/86/EC on family reunification.",
        polish_query="dyrektywa unijna 2003/86 łączenie rodzin tekst",
        topic="Out of scope — verbatim EU directive text",
        query_type="out_of_scope",
        difficulty="hard",
        law_domain="none",
        ground_truth_answer=(
            "The corpus indexes only Polish Dz.U./M.P. acts; raw EU directive text "
            "is not available."
        ),
        is_out_of_scope=True,
        notes="Adversarial OOS — closely related topic, but not in corpus.",
    ),
]


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

def _candidate_chunks(
    client, spec: QuerySpec, fetch_k: int = 30,
) -> tuple[list[dict], str]:
    """Run hybrid BM25 + k-NN; return candidates plus the strategy used.

    Strategy preference (most → least specific):
      1. ``expected_acts``        → ``terms`` filter on document_id
      2. ``title_keywords``       → ``should`` clause on title
      3. unconstrained
    """
    must = {"match": {"text": spec.polish_query}}

    if spec.expected_acts and not spec.is_out_of_scope:
        strategy = "expected_acts"
        bm25_query = {
            "bool": {
                "must": [must],
                "filter": [{"terms": {"document_id": spec.expected_acts}}],
            }
        }
    elif spec.title_keywords and not spec.is_out_of_scope:
        strategy = "title_keywords"
        title_clause = {
            "bool": {
                "should": (
                    [{"match_phrase": {"title": kw}} for kw in spec.title_keywords]
                    + [{"match": {"title": kw}} for kw in spec.title_keywords]
                ),
                "minimum_should_match": 1,
            }
        }
        bm25_query = {
            "bool": {
                "must": [must],
                "should": [title_clause],
                "minimum_should_match": 1,
            }
        }
    else:
        strategy = "unconstrained"
        bm25_query = must

    body = {"size": fetch_k, "_source": {"excludes": ["embedding"]}, "query": bm25_query}
    bm25_hits = client.search(index=INDEX_NAME, body=body)["hits"]["hits"]

    vec = embed_query(spec.polish_query)
    knn_body = {
        "size": fetch_k,
        "_source": {"excludes": ["embedding"]},
        "query": {"knn": {"embedding": {"vector": vec, "k": fetch_k}}},
    }
    if spec.expected_acts and not spec.is_out_of_scope:
        knn_body["query"] = {
            "bool": {
                "must": [knn_body["query"]],
                "filter": [{"terms": {"document_id": spec.expected_acts}}],
            }
        }
    knn_hits = client.search(index=INDEX_NAME, body=knn_body)["hits"]["hits"]

    # Fallback: if a constrained search returned nothing, retry unconstrained
    # so a typo'd act ID doesn't silently produce empty labels.
    if strategy != "unconstrained" and not bm25_hits and not knn_hits:
        body["query"] = must
        bm25_hits = client.search(index=INDEX_NAME, body=body)["hits"]["hits"]
        knn_body["query"] = {"knn": {"embedding": {"vector": vec, "k": fetch_k}}}
        knn_hits = client.search(index=INDEX_NAME, body=knn_body)["hits"]["hits"]
        strategy += "→fallback_unconstrained"

    seen: set[str] = set()
    merged: list[dict] = []
    for hit in bm25_hits + knn_hits:
        if hit["_id"] in seen:
            continue
        seen.add(hit["_id"])
        merged.append(hit["_source"])
    return merged, strategy


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

# Graded label assignment.  Top-1 is the canonical answer (3); ranks 2-3 are
# strongly relevant (2); ranks 4-6 are marginally relevant (1).
_LABEL_BY_RANK = {0: 3, 1: 2, 2: 2, 3: 1, 4: 1, 5: 1}


def _label_topk(
    spec: QuerySpec, candidates: list[dict], top_n: int,
) -> tuple[dict[str, int], list[dict]]:
    """Cross-encoder rerank, then assign 0-3 graded labels."""
    if spec.is_out_of_scope or not candidates:
        return {}, []

    reranked = rerank(spec.polish_query, candidates, top_k=top_n)
    labels: dict[str, int] = {}
    for rank, c in enumerate(reranked):
        chunk_id = f"{c.get('document_id', '')}:{c.get('page_num', '')}"
        if chunk_id in labels:
            continue
        labels[chunk_id] = _LABEL_BY_RANK.get(rank, 1)
    return labels, reranked


# ---------------------------------------------------------------------------
# Per-spec validation
# ---------------------------------------------------------------------------

# A chunk that scores above this threshold for an out-of-scope query is a
# corpus leak — flag it so the spec can be tightened or the chunk reviewed.
_OOS_LEAK_THRESHOLD = 0.5


def _validate(
    spec: QuerySpec, reranked: list[dict], labels: dict[str, int],
) -> dict:
    """Return per-spec diagnostics: mention coverage, expected-act match,
    OOS leakage."""
    diag: dict = {
        "id": spec.id,
        "labels_count": len(labels),
        "warnings": [],
    }
    if not reranked and not spec.is_out_of_scope:
        diag["warnings"].append("no_candidates_found")
        return diag

    top_score = float(reranked[0].get("_rerank_score", 0.0)) if reranked else 0.0
    diag["top_rerank_score"] = round(top_score, 4)

    # Out-of-scope: any chunk above the leak threshold is suspicious.
    if spec.is_out_of_scope:
        leaks = [
            f"{c.get('document_id', '')}:{c.get('page_num', '')}"
            for c in reranked
            if c.get("_rerank_score", 0.0) >= _OOS_LEAK_THRESHOLD
        ]
        if leaks:
            diag["warnings"].append(f"oos_leak: top chunks {leaks[:3]} pass threshold")
        return diag

    # Expected-act match — top chunks should originate from the listed acts.
    if spec.expected_acts:
        top3_acts = {
            (reranked[i].get("document_id") or "") for i in range(min(3, len(reranked)))
        }
        diag["expected_acts_in_top3"] = sorted(top3_acts & set(spec.expected_acts))
        if not diag["expected_acts_in_top3"]:
            diag["warnings"].append("expected_acts_missing_from_top3")

    # Mention validation — every must_mention substring should appear in at
    # least one labelled chunk's text (case-insensitive).
    if spec.must_mention:
        joined_text = "\n".join(c.get("text", "") for c in reranked).lower()
        missing = [m for m in spec.must_mention if m.lower() not in joined_text]
        if missing:
            diag["warnings"].append(f"must_mention_not_in_chunks: {missing}")

    return diag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _serialise_spec(spec: QuerySpec, labels: dict[str, int]) -> dict:
    """Produce the dataset-entry shape consumed by every evaluator."""
    out = asdict(spec)
    out["ground_truth"] = labels
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the legal-RAG ground-truth evaluation dataset.",
    )
    parser.add_argument(
        "--only", action="append", default=None,
        help="Limit build to these query ids (repeatable).",
    )
    parser.add_argument(
        "--top-n", type=int, default=6,
        help="Number of chunks to label per query (default: 6).",
    )
    parser.add_argument(
        "--fetch-k", type=int, default=30,
        help="Candidate pool size before reranking (default: 30).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[3] / "data" / "eval",
        help="Output directory (default: data/eval).",
    )
    args = parser.parse_args()

    selected = (
        [s for s in QUERIES if s.id in set(args.only)] if args.only else QUERIES
    )
    if args.only and not selected:
        print(f"No queries match --only {args.only!r}.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "ground_truth.json"
    report_path = args.output_dir / "build_report.json"

    client = get_client()
    dataset: list[dict] = []
    report: list[dict] = []
    warnings_total = 0

    for spec in selected:
        print(f"  [{spec.id}] {spec.query[:80]}…")
        candidates, strategy = _candidate_chunks(client, spec, fetch_k=args.fetch_k)
        labels, reranked = _label_topk(spec, candidates, top_n=args.top_n)
        diag = _validate(spec, reranked, labels)
        diag["strategy"] = strategy
        report.append(diag)
        warnings_total += len(diag["warnings"])

        msg = f"    strategy={strategy:38s} labelled={len(labels):2d}"
        if diag.get("top_rerank_score") is not None:
            msg += f" top_score={diag['top_rerank_score']:.3f}"
        if diag["warnings"]:
            msg += f"  WARN={diag['warnings']}"
        print(msg)

        dataset.append(_serialise_spec(spec, labels))

    dataset_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "queries": len(report),
                "warnings_total": warnings_total,
                "by_query": report,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Wrote dataset : {dataset_path}  ({len(dataset)} queries)")
    print(f"Wrote report  : {report_path}  ({warnings_total} warnings)")


if __name__ == "__main__":
    main()
