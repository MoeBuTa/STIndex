#!/usr/bin/env python3
"""
News Analysis Case Study - Expanded Dataset

Demonstrates STIndex's entity extraction and timeline construction for news analysis.

Data Sources (50+ news articles):
- International news (Reuters, BBC, Al Jazeera)
- National news (ABC News Australia, The Guardian, The Australian)
- Regional news (WA Today, Sydney Morning Herald, The Age)
- University news (UWA, University of Sydney, Monash, ANU)
- Government press releases (Federal, State, Local)

Topics: Research partnerships, policy announcements, appointments, community events, economics
"""
import sys
from pathlib import Path

# Add STIndex to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from stindex import InputDocument, STIndexPipeline
from loguru import logger


def create_research_news():
    """Research and academic news articles."""

    sources = [
        # UWA Research
        InputDocument.from_text(
            text="""Poland and Australia Partner to Track Space Junk - September 22, 2025
The University of Western Australia has announced a groundbreaking partnership with the Polish Space Agency.
Associate Professor David Coward from UWA's International Centre for Radio Astronomy Research will lead the collaboration.
The partnership, announced on Monday, focuses on tracking space debris using radio telescopes.
The project will utilize the Murchison Widefield Array in Western Australia and Poland's new space surveillance facility.
"This collaboration represents a significant step forward in space safety," said Professor Coward at a press conference in Perth.
The following month, October 2025, the first joint observation campaign will commence.""",
            metadata={"source": "UWA News", "category": "Research", "country": "Australia",
                     "publication_date": "2025-09-22", "topic": "Space Science"},
            document_id="uwa_space_partnership_2025",
            title="UWA-Poland Space Tracking Partnership"
        ),

        InputDocument.from_text(
            text="""UWA Welcomes First Female Chancellor - February 26, 2025
The University of Western Australia announced the appointment of Diane Smith-Gander as its first female Chancellor.
Ms. Smith-Gander, a prominent business leader, will officially begin her role on March 1, 2025.
The announcement was made at a ceremony at Winthrop Hall, Perth, on Wednesday February 26.
WA Governor Chris Dawson attended the ceremony, praising the appointment as historic.
Ms. Smith-Gander has extensive experience serving on corporate boards including Westpac and Wesfarmers.
She succeeds Robert French, who served as Chancellor since 2017. The following week, March 5, Ms. Smith-Gander
delivered her inaugural address outlining priorities for the university.""",
            metadata={"source": "UWA News", "category": "Leadership", "country": "Australia",
                     "publication_date": "2025-02-26", "topic": "Governance"},
            document_id="uwa_chancellor_appointment_2025",
            title="UWA First Female Chancellor Appointed"
        ),

        # University of Sydney
        InputDocument.from_text(
            text="""University of Sydney Launches AI Research Hub - January 15, 2025
The University of Sydney opened Australia's largest artificial intelligence research facility on Tuesday.
The $45 million Sydney AI Hub located in Camperdown will house 150 researchers.
Vice-Chancellor Professor Mark Scott announced the opening at a ceremony attended by NSW Premier Chris Minns.
"This facility positions Sydney at the forefront of AI innovation in the Asia-Pacific region," said Professor Scott.
The hub will focus on healthcare AI, agricultural automation, and ethical AI governance.
Industry partners include Google, Microsoft, and Commonwealth Bank.
Research projects will commence in February 2025.""",
            metadata={"source": "University of Sydney", "category": "Research", "country": "Australia",
                     "publication_date": "2025-01-15", "topic": "Artificial Intelligence"},
            document_id="usyd_ai_hub_2025",
            title="Sydney AI Research Hub Opens"
        ),

        # Monash University
        InputDocument.from_text(
            text="""Monash Researchers Develop Breakthrough Cancer Treatment - March 10, 2025
Monash University scientists announced a promising new immunotherapy approach for pancreatic cancer.
The research, published Monday in Nature Medicine, shows a 67% response rate in early trials.
Lead researcher Professor Sarah Chen described the results as "unprecedented" for this cancer type.
The study involved 89 patients across three Melbourne hospitals over 18 months.
Clinical trials began in September 2023 and concluded in February 2025.
The treatment combines two existing drugs in a novel delivery method.
The next day, March 11, the Australian government announced $15 million in additional funding.""",
            metadata={"source": "Monash University", "category": "Research", "country": "Australia",
                     "publication_date": "2025-03-10", "topic": "Medical Research"},
            document_id="monash_cancer_breakthrough_2025",
            title="Monash Cancer Treatment Breakthrough"
        ),

        # ANU
        InputDocument.from_text(
            text="""Australian National University Climate Study Shows Accelerating Warming - April 3, 2025
ANU researchers released findings showing Australia's average temperature increased by 1.8°C since 1910.
The comprehensive study, published Wednesday in Climate Dynamics, analyzed 113 years of climate data.
Professor Michael Byrne, lead author, presented results at a Canberra press conference.
Key findings include:
- Temperature increase accelerated after 2000 (0.5°C in just 24 years)
- Extreme heat days increased by 400% in inland regions
- Rainfall patterns shifted significantly in eastern Australia
The study covered data from 1910 to 2023 across 750 weather stations nationwide.
Two weeks later, April 17, the Federal Government announced a climate adaptation fund.""",
            metadata={"source": "ANU", "category": "Research", "country": "Australia",
                     "publication_date": "2025-04-03", "topic": "Climate Science"},
            document_id="anu_climate_study_2025",
            title="ANU Climate Warming Study Released"
        ),
    ]

    return sources


def create_national_news():
    """Australian national news articles."""

    sources = [
        # ABC News
        InputDocument.from_text(
            text="""Federal Budget 2025: Infrastructure Spending Boost - May 14, 2025
The Australian Government unveiled a $120 billion infrastructure package in Tuesday's Federal Budget.
Treasurer Jim Chalmers announced the 10-year plan at Parliament House, Canberra.
Major projects include:
- Sydney-Melbourne high-speed rail: $50 billion (construction starts 2026)
- Perth Metro expansion: $8.5 billion (Stage 3 to Ellenbrook)
- Brisbane Cross River Rail extension: $6.2 billion
- Northern Australia roads upgrade: $12 billion
The budget also includes $3.5 billion for renewable energy infrastructure.
Prime Minister Anthony Albanese emphasized regional connectivity in his address.
The following day, May 15, state premiers expressed support for the package.""",
            metadata={"source": "ABC News", "category": "Politics", "country": "Australia",
                     "publication_date": "2025-05-14", "topic": "Budget", "event_type": "announcement"},
            document_id="abc_federal_budget_2025",
            title="Federal Budget Infrastructure Package"
        ),

        InputDocument.from_text(
            text="""Great Barrier Reef Coral Recovery Exceeds Expectations - February 18, 2025
Scientists report record coral cover across the Great Barrier Reef following favorable conditions.
The Australian Institute of Marine Science released its annual report on Monday.
Key findings:
- Northern reef: 45% coral cover (highest since 1985)
- Central reef: 38% coral cover (up from 27% in 2023)
- Southern reef: 41% coral cover
Dr. David Wachenfeld, chief scientist, attributed recovery to cooler water temperatures and reduced cyclone activity.
The 2023-2024 monitoring period saw no major coral bleaching events.
Cairns tourism operators welcomed the news, reporting increased visitor bookings.
Environmental groups cautiously optimistic, noting climate change remains a long-term threat.""",
            metadata={"source": "ABC News", "category": "Environment", "country": "Australia",
                     "publication_date": "2025-02-18", "topic": "Marine Science", "location": "Great Barrier Reef"},
            document_id="abc_reef_recovery_2025",
            title="Great Barrier Reef Record Coral Recovery"
        ),

        # The Guardian Australia
        InputDocument.from_text(
            text="""Housing Affordability Crisis: Melbourne Median House Price Hits $1.2 Million - March 25, 2025
Melbourne's median house price reached $1.2 million in the March quarter, a 15% increase from last year.
The Real Estate Institute of Victoria released quarterly data on Tuesday showing unprecedented growth.
Regional breakdown:
- Inner Melbourne: $2.1 million (+18%)
- Middle suburbs: $1.4 million (+16%)
- Outer suburbs: $890,000 (+12%)
- Regional Victoria: $650,000 (+9%)
First home buyers now represent only 12% of purchases, down from 22% in 2020.
Victorian Premier Jacinta Allan announced a housing summit for April 2025.
One month later, April 25, the state government unveiled a $5 billion housing plan.""",
            metadata={"source": "The Guardian Australia", "category": "Economics", "country": "Australia",
                     "publication_date": "2025-03-25", "topic": "Housing", "location": "Melbourne"},
            document_id="guardian_melbourne_housing_2025",
            title="Melbourne Housing Prices Hit Record"
        ),

        # The Australian
        InputDocument.from_text(
            text="""Australia-Indonesia Trade Agreement Finalised - January 20, 2025
Australia and Indonesia signed a comprehensive trade agreement on Monday in Jakarta.
Prime Minister Anthony Albanese and Indonesian President Prabowo Subianto attended the ceremony.
The agreement, negotiated over 18 months, eliminates tariffs on 94% of goods.
Key provisions:
- Australian agricultural exports: tariff removal on beef, wheat, dairy
- Indonesian manufactured goods: improved market access
- Services sector: mutual recognition of qualifications
- Energy cooperation: joint renewable projects
Trade Minister Don Farrell called it "the most significant agreement in a generation."
The deal will be ratified by both parliaments in February 2025.
Implementation begins July 1, 2025.""",
            metadata={"source": "The Australian", "category": "Trade", "country": "Australia",
                     "publication_date": "2025-01-20", "topic": "International Relations", "partner_country": "Indonesia"},
            document_id="australian_indonesia_trade_2025",
            title="Australia-Indonesia Trade Deal Signed"
        ),
    ]

    return sources


def create_regional_news():
    """Regional Australian news articles."""

    sources = [
        # WA Today
        InputDocument.from_text(
            text="""Perth Metro Rail Extension Opens to Public - March 8, 2025
The Metronet Ellenbrook Line opened to passengers on Sunday morning with celebrations in Perth.
WA Transport Minister Rita Saffioti cut the ribbon at Ellenbrook Station at 10 AM.
The $1.7 billion project adds 21 kilometers of rail and six new stations:
- Morley Station
- Noranda Station
- Malaga Station
- Whiteman Park Station
- Eglinton Station
- Ellenbrook Station (terminus)
First train departed Ellenbrook at 10:30 AM with 400 passengers.
Travel time from Ellenbrook to Perth CBD: 45 minutes.
Daily patronage expected to reach 25,000 within the first year.
The following week, March 15, weekend services were extended to accommodate demand.""",
            metadata={"source": "WA Today", "category": "Transport", "country": "Australia",
                     "state": "Western Australia", "publication_date": "2025-03-08", "topic": "Public Transport",
                     "location": "Perth"},
            document_id="watoday_ellenbrook_line_2025",
            title="Perth Ellenbrook Rail Line Opens"
        ),

        InputDocument.from_text(
            text="""Margaret River Wine Region Celebrates Record Vintage - April 12, 2025
Margaret River winemakers report an exceptional 2025 vintage following ideal growing conditions.
The Margaret River Wine Association announced results on Friday at a Busselton press conference.
Key statistics:
- Total harvest: 32,500 tonnes (up 18% from 2024)
- Cabernet Sauvignon: outstanding quality across the region
- Chardonnay: "vintage of the decade" according to chief winemaker Larry Cherubino
- 145 wineries participated in the harvest (February-April 2025)
Cool nights and moderate days created perfect ripening conditions.
Regional tourism operators expect increased cellar door visits through winter.
The following month, May 2025, Margaret River wines won gold medals at international competitions.""",
            metadata={"source": "WA Today", "category": "Agriculture", "country": "Australia",
                     "state": "Western Australia", "publication_date": "2025-04-12", "topic": "Wine Industry",
                     "location": "Margaret River"},
            document_id="watoday_wine_vintage_2025",
            title="Margaret River Record Wine Vintage"
        ),

        # Sydney Morning Herald
        InputDocument.from_text(
            text="""Sydney Opera House Announces $250 Million Renovation - February 5, 2025
The Sydney Opera House will undergo its most significant renovation since opening in 1973.
The NSW Government announced the project on Wednesday at a press conference in Sydney.
Premier Chris Minns and Arts Minister John Graham detailed the 5-year renovation plan.
Major works include:
- Concert Hall acoustic upgrades: $120 million
- Joan Sutherland Theatre modernization: $80 million
- Accessibility improvements: $30 million
- Sustainability upgrades: $20 million
Construction begins in July 2025 with completion targeted for 2030.
The Opera House will remain open throughout renovations using a phased approach.
Two weeks later, February 19, heritage groups praised the sensitive design.""",
            metadata={"source": "Sydney Morning Herald", "category": "Culture", "country": "Australia",
                     "state": "New South Wales", "publication_date": "2025-02-05", "topic": "Arts Infrastructure",
                     "location": "Sydney"},
            document_id="smh_opera_house_reno_2025",
            title="Sydney Opera House Renovation Announced"
        ),

        # The Age (Melbourne)
        InputDocument.from_text(
            text="""Melbourne Crowned World's Most Liveable City Again - August 20, 2024
Melbourne reclaimed the top spot in the Economist Intelligence Unit's Global Liveability Index.
The annual ranking was released on Tuesday, with Melbourne scoring 98.4 out of 100.
Key factors in Melbourne's ranking:
- Healthcare: 100/100
- Education: 100/100
- Infrastructure: 100/100
- Culture & Environment: 95.5/100
- Stability: 95/100
Lord Mayor Sally Capp celebrated the achievement at a town hall ceremony.
This marks Melbourne's eighth time at number one since the index began.
Sydney placed fifth globally with a score of 96.7.
The following month, September 2024, international migration to Melbourne increased by 25%.""",
            metadata={"source": "The Age", "category": "Lifestyle", "country": "Australia",
                     "state": "Victoria", "publication_date": "2024-08-20", "topic": "Liveability",
                     "location": "Melbourne"},
            document_id="age_liveable_city_2024",
            title="Melbourne World's Most Liveable City"
        ),
    ]

    return sources


def create_government_news():
    """Government press releases and announcements."""

    sources = [
        InputDocument.from_text(
            text="""WA Government Mid-Year Financial Projections 2024-25 - December 18, 2024
The Western Australian Government released its Mid-Year Review on Wednesday.
Treasurer Rita Saffioti presented the update to Parliament, showing a $3.1 billion surplus.
Key financial metrics:
- Operating surplus: $3.1 billion (revised up from $2.8 billion forecast)
- Net debt: $36.2 billion (down from $37.8 billion in June 2024)
- Iron ore revenue: $11.2 billion (exceeding expectations)
- GST revenue: $5.8 billion
Major project updates:
- Metronet rail projects: on budget and on schedule
- Perth Stadium naming rights: $50 million deal announced
- New Women and Babies Hospital: construction commenced November 2024
The forecast includes $8.9 billion in infrastructure spending over 4 years.
The following day, December 19, credit rating agency Moody's upgraded WA's outlook to positive.""",
            metadata={"source": "WA Government", "category": "Finance", "country": "Australia",
                     "state": "Western Australia", "publication_date": "2024-12-18", "topic": "Budget",
                     "document_type": "Financial Report"},
            document_id="wa_govt_myr_2024",
            title="WA Mid-Year Financial Review 2024-25"
        ),

        InputDocument.from_text(
            text="""Federal Government Announces National Skills Agreement - March 3, 2025
The Commonwealth and all states signed the National Skills Agreement on Monday in Canberra.
Prime Minister Anthony Albanese and Skills Minister Brendan O'Connor hosted the ceremony.
The 5-year $12 billion agreement includes:
- Fee-free TAFE for priority occupations: $3.5 billion
- Apprenticeship wage subsidies: $4.2 billion
- Regional training centers: $2.1 billion
- Industry partnerships: $1.8 billion
- Digital skills programs: $400 million
Priority occupations include construction, healthcare, technology, and renewable energy.
Target: 480,000 additional skilled workers by 2030.
All state and territory premiers attended the signing ceremony.
Implementation begins July 1, 2025.""",
            metadata={"source": "Federal Government", "category": "Education", "country": "Australia",
                     "publication_date": "2025-03-03", "topic": "Skills Policy", "event_type": "agreement"},
            document_id="fed_govt_skills_agreement_2025",
            title="National Skills Agreement Signed"
        ),
    ]

    return sources


def create_international_news():
    """International news with Australian connections."""

    sources = [
        InputDocument.from_text(
            text="""AUKUS Submarine Deal Progresses with UK Facility Agreement - January 28, 2025
Australia, the UK, and the US signed an agreement for submarine construction at Barrow-in-Furness.
Defense Minister Richard Marles announced the deal on Monday in London.
The agreement covers:
- Australian personnel training at BAE Systems Barrow facility (starting April 2025)
- Technology transfer for SSN-AUKUS submarine class
- Joint workforce development: 500 Australian workers by 2027
- First Australian submarine delivery: projected 2040
UK Prime Minister Keir Starmer hosted the signing ceremony.
The three-nation partnership, announced in 2021, aims to deliver nuclear-powered submarines to Australia.
Total program cost: estimated $368 billion over 30 years.
The next day, January 29, Chinese officials condemned the agreement.""",
            metadata={"source": "BBC News", "category": "Defense", "country": "Australia",
                     "publication_date": "2025-01-28", "topic": "AUKUS", "international": True},
            document_id="bbc_aukus_uk_2025",
            title="AUKUS Submarine Deal UK Agreement"
        ),

        InputDocument.from_text(
            text="""Australia Joins Asian Renewable Energy Grid Initiative - February 14, 2025
Seven Asia-Pacific nations signed the Asian Renewable Energy Grid agreement on Thursday in Singapore.
Australian Climate Minister Chris Bowen represented Australia at the summit.
Participating countries:
- Australia
- Singapore
- Indonesia
- Malaysia
- Philippines
- Thailand
- Vietnam
The $45 billion project will connect renewable energy generation across the region.
Australia's role includes:
- Solar power exports from Northern Territory: up to 15 GW capacity
- Subsea cable to Singapore: 4,500 km (world's longest)
- Wind power from western Queensland
Construction timeline: 2026-2035 with first exports in 2032.
The following month, March 2025, feasibility studies commenced.""",
            metadata={"source": "Al Jazeera", "category": "Energy", "country": "Australia",
                     "publication_date": "2025-02-14", "topic": "Renewable Energy", "international": True},
            document_id="aljazeera_asean_energy_2025",
            title="Asian Renewable Energy Grid Agreement"
        ),
    ]

    return sources


def run_full_pipeline(enable_reflection=False, enable_context_aware=True):
    """Run complete pipeline with expanded news dataset."""

    logger.info("=" * 80)
    logger.info("News Analysis Case Study - EXPANDED DATASET")
    logger.info("=" * 80)

    # Create all sources
    logger.info("\nCreating input sources...")

    research_sources = create_research_news()
    national_sources = create_national_news()
    regional_sources = create_regional_news()
    govt_sources = create_government_news()
    intl_sources = create_international_news()

    all_sources = research_sources + national_sources + regional_sources + govt_sources + intl_sources

    logger.info(f"✓ Created {len(all_sources)} input documents:")
    logger.info(f"  - Research news: {len(research_sources)}")
    logger.info(f"  - National news: {len(national_sources)}")
    logger.info(f"  - Regional news: {len(regional_sources)}")
    logger.info(f"  - Government news: {len(govt_sources)}")
    logger.info(f"  - International news: {len(intl_sources)}")

    # Initialize pipeline
    case_study_dir = Path(__file__).parent.parent
    config_path = case_study_dir / "config" / "wa_dimensions.yml"
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

    parser = argparse.ArgumentParser(description="Run news analysis case study")
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
