# Public Health Surveillance Case Study

Demonstrates STIndex's multi-dimensional extraction for public health surveillance, tracking measles and influenza outbreaks with context-aware location disambiguation and relative temporal resolution.

## Overview

This case study showcases STIndex's capabilities for **public health surveillance** by extracting disease outbreak information from official health sources across Australia, United States, and international organizations. The pipeline demonstrates:
- **Disease outbreak tracking**: Measles and influenza case extraction
- **Context-aware location disambiguation**: Distinguishes between Western Australia (WA) and Washington State (WA, USA)
- **Relative temporal resolution**: Converts expressions like "this week" to absolute dates
- **Multi-source integration**: Combines data from Australian, US, and international health authorities

## Data Sources (10 Documents)

This case study processes **10 documents** from health surveillance sources across multiple jurisdictions, demonstrating STIndex's capability for multi-jurisdictional public health monitoring.

### 1. Washington State DOH Measles Cases
**Source**: Washington State Department of Health (USA)
**URL**: https://doh.wa.gov/you-and-your-family/illness-and-disease-z/measles/measles-cases-washington-state-2025
**Type**: Case data
**Category**: Measles surveillance

Official measles case surveillance for Washington State 2025.

**Contains**:
- Confirmed case counts by county
- Case investigation status
- Outbreak timeline and progression
- Vaccination status of cases
- Public exposure locations

**Key Extraction Challenges**: County-level spatial granularity, relative temporal expressions, vaccination status tracking

---

### 2. Australian Influenza Statistics
**Source**: Immunisation Coalition Australia
**URL**: https://immunisationcoalition.org.au/influenza-statistics/
**Type**: Statistical dashboard
**Category**: Influenza surveillance

Comprehensive national influenza surveillance statistics aggregated across all Australian states and territories.

**Contains**:
- Weekly influenza-like illness (ILI) rates
- Laboratory-confirmed case counts by state/territory
- Seasonal trends and historical comparisons
- Hospitalization and mortality data
- Age-stratified statistics

**Key Extraction Challenges**: Temporal ranges (flu seasons, weeks), state-level granularity, normalizing epidemiological weeks to calendar dates

---

### 3. CDC Measles Cases and Outbreaks
**Source**: US Centers for Disease Control and Prevention
**URL**: https://www.cdc.gov/measles/data-research/index.html
**Type**: National surveillance
**Category**: Measles surveillance

National-level measles surveillance for the United States, aggregating data from all state health departments.

**Contains**:
- National case counts and trends
- Multi-state outbreak investigations
- Geographic distribution by state
- Temporal trends and seasonality
- International importations

---

### 4. WHO Measles Fact Sheet
**Source**: World Health Organization
**URL**: https://www.who.int/news-room/fact-sheets/detail/measles
**Type**: Global fact sheet
**Category**: Measles information

Global measles epidemiology, burden of disease, and elimination efforts across WHO regions.

**Contains**:
- Global case counts and mortality
- Regional variations (WHO regions)
- Vaccination coverage worldwide
- Elimination progress by country

**Key Extraction Challenges**: Global spatial granularity (countries, WHO regions), international date formats, multi-year temporal trends

---

### 5. WHO Influenza (Seasonal) Fact Sheet
**Source**: World Health Organization
**URL**: https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)
**Type**: Global fact sheet
**Category**: Influenza information

Global influenza epidemiology, seasonal patterns, and prevention strategies.

**Contains**:
- Seasonal influenza burden of disease
- Global circulation patterns
- High-risk populations
- Vaccination recommendations
- Prevention and control strategies

---

### 6. WHO Immunization Coverage
**Source**: World Health Organization
**URL**: https://www.who.int/news-room/fact-sheets/detail/immunization-coverage
**Type**: Global fact sheet
**Category**: Immunization statistics

Global immunization coverage statistics for vaccine-preventable diseases.

**Contains**:
- Global vaccination coverage rates
- Regional disparities in immunization
- Vaccine-preventable disease trends
- Immunization program challenges
- Coverage targets and progress

---

### 7. WHO Vaccines and Immunization
**Source**: World Health Organization
**URL**: https://www.who.int/health-topics/vaccines-and-immunization
**Type**: Health topics overview
**Category**: Vaccines and immunization

Comprehensive overview of vaccine development, deployment, and immunization programs worldwide.

**Contains**:
- Vaccine types and technologies
- Immunization schedules
- Global vaccine access
- Vaccine safety and efficacy
- Research and development

---

### 8. Victoria Health Measles
**Source**: Victoria Health (Australia)
**URL**: https://www.health.vic.gov.au/infectious-diseases/measles
**Type**: Infectious disease information
**Category**: Measles information

State-level measles information and guidance for Victoria, Australia.

**Contains**:
- Measles symptoms and transmission
- Vaccination recommendations
- Public health responses
- Notification requirements
- Resources for healthcare providers

---

### 9. WHO Poliomyelitis
**Source**: World Health Organization
**URL**: https://www.who.int/news-room/fact-sheets/detail/poliomyelitis
**Type**: Global fact sheet
**Category**: Polio information

Global polio eradication efforts and remaining endemic areas.

**Contains**:
- Polio epidemiology
- Eradication progress by country
- Vaccination strategies
- Surveillance systems
- Remaining challenges

---

### 10. WHO Dengue and Severe Dengue
**Source**: World Health Organization
**URL**: https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue
**Type**: Global fact sheet
**Category**: Dengue information

Global dengue burden, geographic distribution, and control strategies.

**Contains**:
- Global dengue incidence
- Geographic distribution and risk factors
- Clinical presentation and severity
- Vector control strategies
- Vaccine development

---

## Run

```bash
python case_studies/public_health/scripts/run_case_study.py
```

## Expected Output

The pipeline generates:
- **Preprocessed chunks**: `data/chunks/preprocessed_chunks.json`
- **Extraction results**: `data/results/extraction_results.json`
- **Interactive visualizations**: `data/visualizations/*.html`
  - Disease outbreak maps showing case locations
  - Temporal timelines of outbreak progression
  - Comparative visualizations (Australia vs USA vs Global)
  - Exposure site maps with time ranges

## Key Features Demonstrated

### 1. Multi-Jurisdictional Health Surveillance
The system processes health data from multiple jurisdictions and organizational levels:
- **US State-level**: Washington State DOH measles surveillance
- **US National**: CDC measles and outbreak data
- **Australian State**: Victoria Health infectious disease information
- **Australian National**: Immunisation Coalition influenza statistics
- **Global/International**: WHO fact sheets covering multiple diseases

### 2. Disease-Specific Dimensional Extraction
Custom dimension configuration for multiple health surveillance dimensions:
- **Temporal**: Outbreak dates, epidemiological weeks, vaccination schedules
- **Spatial**: Case locations, affected regions, endemic areas, WHO regions
- **Disease entities**: Measles, influenza, polio, dengue, and other vaccine-preventable diseases
- **Health metrics**: Case counts, vaccination coverage, mortality rates, disease burden

### 3. Relative Temporal Resolution
Automatically resolves relative temporal expressions using document publication dates:
- "This flu season" → Specific date range
- "Recent weeks" → Date range based on publication date
- Epidemiological weeks → ISO 8601 calendar dates

### 4. Geographic Normalization and Hierarchies
Maintains consistent spatial hierarchies across multiple jurisdictions:
- **US hierarchy**: County → State → Country
- **Australian hierarchy**: Suburb/Region → State → Country
- **WHO regions**: Country → WHO Region → Global
- Geocoding with country-specific disambiguation

## Use Case: Disease Outbreak Response

This case study mirrors real-world public health surveillance workflows:
1. **Alert monitoring**: Track official health alerts from multiple jurisdictions
2. **Case extraction**: Automatically extract case dates, locations, and counts
3. **Exposure mapping**: Geocode exposure sites for contact tracing
4. **Timeline reconstruction**: Build temporal sequences of outbreak events
5. **Visualization**: Generate maps and timelines for situation awareness

Potential applications include:
- Automated surveillance dashboards
- Early outbreak detection
- Contact tracing support
- Public health reporting
- Research and epidemiological analysis
