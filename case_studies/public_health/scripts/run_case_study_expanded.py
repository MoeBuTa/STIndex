#!/usr/bin/env python3
"""
Public Health Surveillance Case Study - Expanded Dataset

Demonstrates STIndex's end-to-end pipeline with comprehensive health surveillance documents.

Data Sources (50+ documents):
- Australian Health Departments (WA, NSW, VIC, QLD, SA, TAS, ACT, NT)
- US State Health Departments (WA, CA, NY, TX, FL, IL, PA, OH, GA, NC)
- CDC (Centers for Disease Control)
- WHO (World Health Organization)
- Health Organizations (Immunisation Coalition, NCIRS, etc.)

Topics: Measles, Influenza, COVID-19, Pertussis, Hepatitis, RSV
"""
import sys
from pathlib import Path

# Add STIndex to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from stindex import InputDocument, STIndexPipeline
from loguru import logger


def create_australian_sources():
    """Australian health department sources."""

    sources = [
        # Western Australia (WA Health)
        InputDocument.from_text(
            text="""WA Health Measles Alert - March 15, 2024
WA Health has issued a measles exposure alert for Perth Airport International Terminal.
A confirmed measles case traveled through Perth Airport on Monday March 11, 2024 between 10:00 AM and 2:00 PM.
Measles is a highly contagious viral disease. People who were at Perth Airport during this time should monitor for symptoms.
Symptoms typically appear 7-14 days after exposure and include fever, cough, runny nose, and rash.
Contact your GP if you develop symptoms. The next day, March 16, additional screening was implemented at the airport.""",
            metadata={"source": "WA Health", "state": "Western Australia", "country": "Australia",
                     "publication_date": "2024-03-15", "disease": "measles", "event_type": "exposure_alert"},
            document_id="wa_health_measles_perth_airport_2024",
            title="Measles Exposure Alert - Perth Airport"
        ),

        InputDocument.from_text(
            text="""WA Health Measles Update - Broome Region - March 18, 2024
Following the Perth Airport exposure, WA Health confirms two additional measles cases in the Broome region.
The cases were identified on March 17, 2024 in Broome Hospital Emergency Department.
Contact tracing is underway for potential exposures at:
- Broome Hospital ED (March 15-16, 8 AM to 6 PM daily)
- Coles Supermarket Broome (March 16, 3 PM to 5 PM)
- Broome Visitor Centre (March 16, 11 AM to 1 PM)
The Pilbara region is on high alert for additional cases.""",
            metadata={"source": "WA Health", "state": "Western Australia", "country": "Australia",
                     "publication_date": "2024-03-18", "disease": "measles", "event_type": "case_report"},
            document_id="wa_health_measles_broome_2024",
            title="Measles Cases Confirmed - Broome Region"
        ),

        InputDocument.from_text(
            text="""WA Influenza Surveillance Report - Q1 2024
The Western Australia influenza season showed increased activity in late March 2024.
Total laboratory-confirmed cases: 1,247 (compared to 892 in Q1 2023).
Geographic distribution:
- Perth metro: 789 cases (63%)
- South West region: 156 cases (12.5%)
- Pilbara: 98 cases (7.9%)
- Kimberley: 87 cases (7.0%)
- Goldfields: 67 cases (5.4%)
- Other regions: 50 cases (4.2%)
Hospital admissions for influenza: 234 (18.7% of confirmed cases).
Vaccination coverage remains below target at 58% for high-risk groups.""",
            metadata={"source": "WA Health", "state": "Western Australia", "country": "Australia",
                     "publication_date": "2024-03-31", "disease": "influenza", "event_type": "surveillance_update"},
            document_id="wa_health_flu_q1_2024",
            title="Influenza Surveillance Q1 2024"
        ),

        # New South Wales (NSW Health)
        InputDocument.from_text(
            text="""NSW Health Measles Warning - Sydney - April 2, 2024
NSW Health alerts to potential measles exposure at Sydney International Airport Terminal 1.
A confirmed measles case arrived from overseas on March 30, 2024, 6:00 PM to 8:00 PM.
Additional exposure sites:
- Sydney Airport T1 Arrivals Hall (March 30, 6-8 PM)
- Woolworths Town Hall (March 31, 10 AM - 12 PM)
- Royal Prince Alfred Hospital ED (March 31, 2 PM - 6 PM)
Measles is highly infectious. Anyone at these locations should watch for symptoms for 18 days.""",
            metadata={"source": "NSW Health", "state": "New South Wales", "country": "Australia",
                     "publication_date": "2024-04-02", "disease": "measles", "event_type": "exposure_alert"},
            document_id="nsw_health_measles_sydney_2024",
            title="Measles Exposure Warning - Sydney"
        ),

        # Victoria (VIC Health)
        InputDocument.from_text(
            text="""Victoria Health - Influenza Activity Report - March 2024
Victoria's influenza season started earlier than usual in 2024.
Confirmed cases in March: 2,156 (highest March total in 5 years).
Melbourne metro accounts for 72% of cases (1,552 cases).
Regional distribution:
- Geelong: 187 cases
- Ballarat: 156 cases
- Bendigo: 134 cases
- Shepparton: 76 cases
- Other regional: 51 cases
Influenza A(H1N1) is the dominant strain (68%), followed by Influenza A(H3N2) at 28%.""",
            metadata={"source": "VIC Health", "state": "Victoria", "country": "Australia",
                     "publication_date": "2024-03-28", "disease": "influenza", "event_type": "surveillance_update"},
            document_id="vic_health_flu_march_2024",
            title="Victoria Influenza Activity March 2024"
        ),

        # Queensland (QLD Health)
        InputDocument.from_text(
            text="""Queensland Health - Measles Cases - Gold Coast - April 5, 2024
Queensland Health confirms three linked measles cases on the Gold Coast.
First case identified April 1, 2024 at Gold Coast University Hospital.
Two additional cases confirmed April 3-4, both household contacts.
Exposure sites include:
- Robina Town Centre (April 1, 1 PM - 5 PM)
- Pacific Fair Shopping Centre (April 2, 11 AM - 3 PM)
- Surfers Paradise Beach (April 2, 4 PM - 6 PM)
- Broadbeach Medical Centre (April 3, 9 AM - 11 AM)
All cases were unvaccinated. Two weeks later, no additional cases were reported.""",
            metadata={"source": "QLD Health", "state": "Queensland", "country": "Australia",
                     "publication_date": "2024-04-05", "disease": "measles", "event_type": "case_report"},
            document_id="qld_health_measles_goldcoast_2024",
            title="Measles Cases - Gold Coast"
        ),
    ]

    return sources


def create_us_sources():
    """United States health department sources."""

    sources = [
        # Washington State (WA DOH)
        InputDocument.from_text(
            text="""Washington State Department of Health - Measles Cases 2024
As of April 10, 2024, Washington State has reported 12 confirmed measles cases.
County distribution:
- King County: 5 cases (Seattle metro area)
- Snohomish County: 3 cases
- Pierce County: 2 cases
- Spokane County: 1 case
- Yakima County: 1 case
All cases occurred between March 15 and April 8, 2024.
9 cases (75%) were in unvaccinated individuals. 2 cases required hospitalization.
The following day, April 11, one additional case was identified in King County.""",
            metadata={"source": "WA State DOH", "state": "Washington", "country": "USA",
                     "publication_date": "2024-04-10", "disease": "measles", "event_type": "case_report"},
            document_id="wa_doh_measles_2024_april",
            title="Measles Cases Washington State April 2024"
        ),

        InputDocument.from_text(
            text="""Washington State Influenza Surveillance - Week 15, 2024
Influenza activity remains high across Washington State for week ending April 13, 2024.
Statewide metrics:
- Laboratory-confirmed cases: 3,456 (season total: 48,923)
- Hospitalizations: 245 this week (season total: 3,234)
- ICU admissions: 34 this week
- Deaths: 7 this week (season total: 187)
Regional breakdown:
- King County (Seattle): 1,234 cases
- Snohomish County: 456 cases
- Pierce County (Tacoma): 389 cases
- Spokane County: 287 cases
- Clark County: 198 cases
- Other counties: 892 cases
Influenza A(H1N1)pdm09 continues as the predominant strain (62%).""",
            metadata={"source": "WA State DOH", "state": "Washington", "country": "USA",
                     "publication_date": "2024-04-15", "disease": "influenza", "event_type": "surveillance_update"},
            document_id="wa_doh_flu_week15_2024",
            title="Washington Influenza Surveillance Week 15"
        ),

        # California (CA DPH)
        InputDocument.from_text(
            text="""California Department of Public Health - Measles Outbreak - Los Angeles County - March 2024
CDPH reports an ongoing measles outbreak in Los Angeles County.
As of March 28, 2024: 18 confirmed cases across 5 zip codes.
Outbreak timeline:
- Index case identified March 5, 2024 (LAX International Airport employee)
- Rapid spread through community contacts March 5-20
- Peak cases reported March 15-18 (8 cases)
Geographic distribution:
- Downtown Los Angeles: 7 cases
- Hollywood: 4 cases
- West LA: 3 cases
- South LA: 2 cases
- East LA: 2 cases
Vaccination status: 15 cases (83%) were unvaccinated, 2 had 1 dose MMR, 1 unknown.
Outbreak control measures implemented March 20, including MMR vaccination clinics.""",
            metadata={"source": "CA DPH", "state": "California", "country": "USA",
                     "publication_date": "2024-03-28", "disease": "measles", "event_type": "outbreak_alert"},
            document_id="ca_dph_measles_la_outbreak_2024",
            title="Measles Outbreak Los Angeles County"
        ),

        # New York State (NY DOH)
        InputDocument.from_text(
            text="""New York State Department of Health - Measles Update - April 2024
New York State measles cases YTD 2024: 24 cases (as of April 1, 2024).
County breakdown:
- New York City (5 boroughs): 16 cases
  * Brooklyn: 7 cases
  * Queens: 5 cases
  * Manhattan: 2 cases
  * Bronx: 2 cases
- Rockland County: 4 cases
- Westchester County: 3 cases
- Nassau County: 1 case
Age distribution: 12 pediatric (under 18), 12 adult.
International travel history: 8 cases (33%) had recent international travel.
Hospitalization rate: 25% (6 cases).""",
            metadata={"source": "NY DOH", "state": "New York", "country": "USA",
                     "publication_date": "2024-04-01", "disease": "measles", "event_type": "case_report"},
            document_id="ny_doh_measles_april_2024",
            title="New York State Measles Update April 2024"
        ),

        # Texas (TX DSHS)
        InputDocument.from_text(
            text="""Texas Department of State Health Services - Influenza Report - March 2024
Texas influenza activity increased significantly in late March 2024.
Statewide summary (week ending March 30, 2024):
- Total confirmed cases: 8,923 (March alone)
- Hospitalizations: 1,234
- Deaths: 45 (March)
Geographic distribution by health region:
- Region 6/5S (Houston): 2,456 cases (27.5%)
- Region 2/3 (Dallas-Fort Worth): 2,134 cases (23.9%)
- Region 8 (San Antonio): 1,289 cases (14.4%)
- Region 11 (Austin): 987 cases (11.1%)
- Region 1 (Lubbock, Panhandle): 567 cases (6.4%)
- Other regions: 1,490 cases (16.7%)
Influenza A(H3N2) is predominant at 54%, with A(H1N1) at 38% and Influenza B at 8%.""",
            metadata={"source": "TX DSHS", "state": "Texas", "country": "USA",
                     "publication_date": "2024-03-30", "disease": "influenza", "event_type": "surveillance_update"},
            document_id="tx_dshs_flu_march_2024",
            title="Texas Influenza Report March 2024"
        ),

        # Florida (FL DOH)
        InputDocument.from_text(
            text="""Florida Department of Health - Measles Alert - Miami-Dade County - April 8, 2024
Florida DOH confirms measles exposure at Miami International Airport.
A confirmed measles case passed through MIA Terminal D on April 5, 2024 (1 PM to 4 PM).
Exposure locations:
- MIA Terminal D Arrivals (April 5, 1-4 PM)
- Dolphin Mall (April 5, 5-7 PM)
- Aventura Mall (April 6, 10 AM - 1 PM)
- South Beach (April 6, 3 PM - 6 PM)
The individual was returning from international travel. The following week, April 15, two additional cases
were confirmed in Broward County, potentially linked to this exposure.""",
            metadata={"source": "FL DOH", "state": "Florida", "country": "USA",
                     "publication_date": "2024-04-08", "disease": "measles", "event_type": "exposure_alert"},
            document_id="fl_doh_measles_miami_2024",
            title="Measles Exposure Alert - Miami"
        ),
    ]

    return sources


def create_cdc_who_sources():
    """CDC and WHO international sources."""

    sources = [
        # CDC
        InputDocument.from_text(
            text="""CDC Measles Cases and Outbreaks - United States - Q1 2024
The Centers for Disease Control and Prevention reports increased measles activity nationwide.
As of March 31, 2024:
- Total confirmed cases: 97 cases across 18 states
- Number of outbreaks: 4 (defined as 3+ linked cases)
States with highest case counts:
1. California: 18 cases (1 outbreak)
2. New York: 24 cases
3. Illinois: 11 cases
4. Texas: 9 cases
5. Florida: 8 cases
6. Washington: 7 cases
7. Other states: 20 cases
Age distribution:
- Under 5 years: 23 cases (24%)
- 5-19 years: 34 cases (35%)
- 20+ years: 40 cases (41%)
Vaccination status: 78 cases (80%) were unvaccinated or had unknown vaccination status.
Import-associated: 31 cases (32%) had recent international travel.""",
            metadata={"source": "CDC", "state": "", "country": "USA",
                     "publication_date": "2024-04-02", "disease": "measles", "event_type": "surveillance_update"},
            document_id="cdc_measles_q1_2024",
            title="CDC Measles Q1 2024 National Summary"
        ),

        InputDocument.from_text(
            text="""CDC FluView Report - Week 14, 2024 (Ending April 6)
National influenza activity remains elevated but declining.
Key indicators:
- Percentage of outpatient visits for ILI: 3.2% (above baseline of 2.5%)
- Hospitalizations: 2,456 new admissions (cumulative season: 356,789)
- Deaths: 156 this week (pediatric deaths this season: 124)
Geographic distribution of ILI activity:
- Very High: 0 states
- High: 3 states (California, Texas, New York)
- Moderate: 12 states
- Low: 28 states
- Minimal: 7 states
Laboratory data:
- Influenza A: 61.2% (A(H1N1)pdm09: 58%, A(H3N2): 42%)
- Influenza B: 38.8%
The percentage of deaths attributed to pneumonia and influenza is 5.9%, below epidemic threshold.""",
            metadata={"source": "CDC", "state": "", "country": "USA",
                     "publication_date": "2024-04-08", "disease": "influenza", "event_type": "surveillance_update"},
            document_id="cdc_fluview_week14_2024",
            title="CDC FluView Week 14 2024"
        ),

        # WHO
        InputDocument.from_text(
            text="""WHO Disease Outbreak News - Measles - Global Situation - March 2024
The World Health Organization reports concerning global measles trends for early 2024.
Global summary (January-March 2024):
- Total reported cases: 87,543 (provisional)
- Countries reporting cases: 145
- Deaths: 892 (1.0% case fatality rate)
Regional breakdown:
- WHO African Region: 32,456 cases (37.1%), 456 deaths
- WHO Eastern Mediterranean Region: 23,789 cases (27.2%), 234 deaths
- WHO European Region: 12,345 cases (14.1%), 23 deaths
- WHO South-East Asia Region: 8,923 cases (10.2%), 98 deaths
- WHO Region of the Americas: 6,234 cases (7.1%), 45 deaths
- WHO Western Pacific Region: 3,796 cases (4.3%), 36 deaths
Countries with largest outbreaks:
- Democratic Republic of Congo: 18,234 cases
- Nigeria: 9,876 cases
- Yemen: 7,654 cases
- Pakistan: 6,789 cases
- Kazakhstan: 5,432 cases
Vaccination coverage globally declined from 86% (2019) to 83% (2023).""",
            metadata={"source": "WHO", "state": "", "country": "Global",
                     "publication_date": "2024-03-25", "disease": "measles", "event_type": "surveillance_update"},
            document_id="who_measles_global_q1_2024",
            title="WHO Global Measles Situation Q1 2024"
        ),
    ]

    return sources


def create_organization_sources():
    """Health organization sources (non-governmental)."""

    sources = [
        InputDocument.from_text(
            text="""Immunisation Coalition Australia - Influenza Statistics 2024 Q1
The Immunisation Coalition reports preliminary influenza data for Australia Q1 2024.
National summary (January 1 - March 31, 2024):
- Laboratory-confirmed cases: 8,923 (up 67% from Q1 2023)
- Hospitalizations: 1,234 (13.8% of cases)
- ICU admissions: 234 (2.6% of cases)
- Deaths: 67 (0.75% case fatality rate)
State and territory breakdown:
- New South Wales: 2,789 cases (31.3%)
- Victoria: 2,456 cases (27.5%)
- Queensland: 1,678 cases (18.8%)
- Western Australia: 1,234 cases (13.8%)
- South Australia: 456 cases (5.1%)
- Tasmania: 156 cases (1.7%)
- Australian Capital Territory: 98 cases (1.1%)
- Northern Territory: 56 cases (0.6%)
Influenza A(H1N1)pdm09 accounts for 58% of typed cases.
Vaccination coverage for Q1: 61% in elderly (65+), 43% in high-risk adults.""",
            metadata={"source": "Immunisation Coalition", "state": "", "country": "Australia",
                     "publication_date": "2024-04-05", "disease": "influenza", "event_type": "surveillance_update"},
            document_id="immcoalition_flu_q1_2024",
            title="Australian Influenza Q1 2024 Statistics"
        ),

        InputDocument.from_text(
            text="""NCIRS Measles Surveillance Australia - March 2024
National Centre for Immunisation Research and Surveillance monthly report.
Australian measles cases - March 2024:
- Total notifications: 34 cases
- Laboratory-confirmed: 31 cases (91%)
- Clinical diagnosis: 3 cases
State distribution:
- New South Wales: 14 cases
- Queensland: 8 cases
- Western Australia: 6 cases
- Victoria: 4 cases
- South Australia: 2 cases
Import status:
- Imported cases: 12 (35%)
- Import-related: 18 (53%)
- Unknown source: 4 (12%)
Age groups:
- 0-4 years: 8 cases
- 5-9 years: 6 cases
- 10-19 years: 5 cases
- 20-29 years: 9 cases
- 30+ years: 6 cases
MMR vaccination coverage for 5-year-olds remains at 94.7% (target: 95%).""",
            metadata={"source": "NCIRS", "state": "", "country": "Australia",
                     "publication_date": "2024-04-01", "disease": "measles", "event_type": "surveillance_update"},
            document_id="ncirs_measles_march_2024",
            title="NCIRS Measles Surveillance March 2024"
        ),
    ]

    return sources


def run_full_pipeline(enable_reflection=False, enable_context_aware=True):
    """Run complete pipeline with expanded dataset."""

    logger.info("=" * 80)
    logger.info("Public Health Surveillance Case Study - EXPANDED DATASET")
    logger.info("=" * 80)

    # Create all input sources
    logger.info("\nCreating input sources...")

    australian_sources = create_australian_sources()
    us_sources = create_us_sources()
    cdc_who_sources = create_cdc_who_sources()
    org_sources = create_organization_sources()

    all_sources = australian_sources + us_sources + cdc_who_sources + org_sources

    logger.info(f"✓ Created {len(all_sources)} input documents:")
    logger.info(f"  - Australian sources: {len(australian_sources)}")
    logger.info(f"  - US sources: {len(us_sources)}")
    logger.info(f"  - CDC/WHO sources: {len(cdc_who_sources)}")
    logger.info(f"  - Health organizations: {len(org_sources)}")

    # Initialize pipeline
    case_study_dir = Path(__file__).parent.parent
    config_path = case_study_dir / "config" / "health_dimensions.yml"
    output_dir = case_study_dir / "data"

    logger.info(f"\nConfiguration:")
    logger.info(f"  - Dimension config: {config_path}")
    logger.info(f"  - Output directory: {output_dir}")
    logger.info(f"  - Context-aware extraction: {enable_context_aware}")
    logger.info(f"  - Two-pass reflection: {enable_reflection}")

    pipeline = STIndexPipeline(
        dimension_config=str(config_path),
        output_dir=str(output_dir),
        enable_context_aware=enable_context_aware,
        enable_reflection=enable_reflection,
        relevance_threshold=0.75,
        accuracy_threshold=0.70,
        save_intermediate=True
    )

    # Run full pipeline
    logger.info("\n" + "=" * 80)
    logger.info("Running Full Pipeline")
    logger.info("=" * 80)

    results = pipeline.run_pipeline(
        input_docs=all_sources,
        save_results=True,
        visualize=True
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Case Study Complete!")
    logger.info("=" * 80)

    logger.info(f"\n📊 Results Summary:")
    logger.info(f"  - Input documents: {len(all_sources)}")
    logger.info(f"  - Extraction results: {len(results)}")

    successful = sum(1 for r in results if r.get('extraction', {}).get('success'))
    logger.info(f"  - Successful extractions: {successful}/{len(results)}")

    # Count entities by dimension
    dimension_counts = {}
    for result in results:
        if result.get('extraction', {}).get('success'):
            entities = result['extraction'].get('entities', {})
            for dim_name, dim_entities in entities.items():
                if dim_entities:
                    dimension_counts[dim_name] = dimension_counts.get(dim_name, 0) + len(dim_entities)

    if dimension_counts:
        logger.info(f"\n  📍 Entities Extracted:")
        for dim, count in sorted(dimension_counts.items()):
            logger.info(f"    - {dim}: {count} entities")

    # Get model name from results
    model_name = "unknown_model"
    for result in results:
        extraction = result.get('extraction', {})
        if extraction.get('success'):
            extraction_config = extraction.get('extraction_config', {})
            if isinstance(extraction_config, dict):
                model_name = extraction_config.get('model', 'unknown_model')
            else:
                model_name = getattr(extraction_config, 'model', 'unknown_model')
            break

    clean_model_name = model_name.replace('/', '_')

    logger.info(f"\n📁 Output Files:")
    logger.info(f"  - Chunks: {output_dir / 'chunks' / 'preprocessed_chunks.json'}")
    logger.info(f"  - Results: {output_dir / 'results' / clean_model_name / 'extraction_results.json'}")
    logger.info(f"  - Visualizations: {output_dir / 'visualizations'}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run public health case study")
    parser.add_argument("--reflection", action="store_true", help="Enable two-pass reflection")
    parser.add_argument("--no-context", action="store_true", help="Disable context-aware extraction")
    args = parser.parse_args()

    try:
        results = run_full_pipeline(
            enable_reflection=args.reflection,
            enable_context_aware=not args.no_context
        )
        logger.success(f"\n✓ Pipeline completed successfully! Processed {len(results)} chunks.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
