# STIndex Test Suite - Pure English Test Cases

## Test Categories

### 1. Temporal Extraction Tests (10 cases)
### 2. Spatial Extraction Tests (10 cases)
### 3. Combined Spatiotemporal Tests (10 cases)
### 4. Edge Cases and Complex Scenarios (5 cases)

---

## Section 1: Temporal Extraction Tests

### T1.1: Absolute Dates with Explicit Years
```
The project started on January 15, 2020, was paused on March 20, 2021, and resumed on September 5, 2022.
```

### T1.2: Dates Without Years - Year Inference
```
In 2023, the conference began on March 10. The workshop was on March 11, and the closing ceremony happened on March 12.
```

### T1.3: Date Intervals
```
The exhibition will run from May 1, 2024 to May 31, 2024.
```

### T1.4: Mixed Date Formats
```
The event on 2024-06-15 follows the announcement from June 1, 2024, and precedes the deadline of July 15, 2024.
```

### T1.5: Relative Time Expressions
```
The meeting was yesterday, the report is due tomorrow, and the review happens next week.
```

### T1.6: Dates with Specific Times
```
The webinar starts at 2:00 PM on March 15, 2024.
```

### T1.7: Duration Expressions
```
The training program lasts 3 weeks.
```

### T1.8: Complex Temporal Context
```
The study began in January 2020, was interrupted in March 2020 due to COVID-19, and resumed in September 2021.
```

### T1.9: Cross-Year Intervals
```
The study ran from December 2022 to February 2023.
```

### T1.10: Historical Dates
```
World War II ended on September 2, 1945. The Berlin Wall fell on November 9, 1989.
```

---

## Section 2: Spatial Extraction Tests

### S2.1: Major World Cities
```
The tour includes stops in Paris, Tokyo, New York, and Sydney.
```

### S2.2: Cities with Country Context
```
The conference has venues in Berlin, Germany; Toronto, Canada; and Melbourne, Australia.
```

### S2.3: Ambiguous Place Names with Context
```
Springfield, Illinois is the state capital. Springfield, Massachusetts has a different history.
```

### S2.4: States and Regions
```
California, Texas, and Florida are the most populous US states.
```

### S2.5: Landmarks and Specific Locations
```
The Eiffel Tower in Paris and the Statue of Liberty in New York are iconic landmarks.
```

### S2.6: Multiple Locations in Same Country
```
The Australian tour covers Sydney, Melbourne, Brisbane, Perth, and Adelaide.
```

### S2.7: Small Towns with State Context
```
The study was conducted in Boulder, Colorado and Ann Arbor, Michigan.
```

### S2.8: African Cities
```
The research team visited Lagos, Nigeria; Nairobi, Kenya; and Cairo, Egypt.
```

### S2.9: Asian Cities
```
The company has offices in Singapore, Seoul, Bangkok, and Mumbai.
```

### S2.10: European Capitals
```
The summit rotates between Brussels, Geneva, Vienna, and Copenhagen.
```

---

## Section 3: Combined Spatiotemporal Tests

### C3.1: News Report - Hurricane Event
```
On August 29, 2005, Hurricane Katrina made landfall near New Orleans, Louisiana. By August 31, the storm had moved through Mississippi.
```

### C3.2: Travel Itinerary
```
We'll arrive in Rome on June 5, 2024, stay three days, then travel to Florence on June 8.
```

### C3.3: Conference Announcement
```
The International AI Conference will be held in Singapore from September 15-20, 2024.
```

### C3.4: Historical Event - Moon Landing
```
On July 20, 1969, Apollo 11 landed on the Moon.
```

### C3.5: Business Expansion Timeline
```
The company opened its Tokyo office in March 2020, followed by Shanghai in July 2020.
```

### C3.6: Research Field Study
```
The expedition began in Nairobi, Kenya on February 1, 2023. Researchers spent two weeks in the Serengeti.
```

### C3.7: Sports Event
```
The 2026 FIFA World Cup will be jointly hosted by the United States, Canada, and Mexico from June 11 to July 19, 2026.
```

### C3.8: Climate Event (PDF Example)
```
On March 15, 2022, a strong cyclone hit the coastal areas near Broome, Western Australia and later moved inland by March 17.
```

### C3.9: Political Event - Summit
```
The G20 Summit took place in Bali, Indonesia on November 15-16, 2022.
```

### C3.10: Natural Disaster Timeline
```
The earthquake struck off the coast of Sumatra on December 26, 2004. The tsunami affected Thailand and Sri Lanka.
```

---

## Section 4: Edge Cases and Complex Scenarios

### E4.1: No Spatiotemporal Information
```
The algorithm uses machine learning to optimize performance.
```

### E4.2: Dense Information
```
Between January 5 and January 10, 2024, the team visited Paris, London, Berlin, and Amsterdam.
```

### E4.3: Nested Locations
```
The office is located in Austin, Texas, United States, near the University of Texas campus.
```

### E4.4: Non-English Place Names
```
The meeting will be held in Beijing, China on December 1, 2024.
```

### E4.5: Multiple Ambiguous References
```
Cambridge researchers met with Cambridge colleagues to discuss the Cambridge study.
```

---

## Additional Challenge Cases

### A1: Scientific Abstract
```
The study, conducted between March 2019 and August 2021 at Stanford University, California, analyzed climate data from Alaska, Greenland, and Antarctica.
```

### A2: Historical Narrative
```
The expedition departed from London on May 19, 1845. The ships were last seen in Baffin Bay in July 1845, and the fate of the crew remained unknown until artifacts were discovered in the Canadian Arctic in 2014.
```

### A3: Flight Schedule
```
Flight AA123 departs Los Angeles at 8:15 AM on Monday, arrives in Chicago at 2:30 PM, and continues to New York, landing at 6:45 PM.
```

### A4: Medical Record
```
Patient admitted to Massachusetts General Hospital on January 5, 2024. Symptoms began three days earlier. Follow-up scheduled for February 12.
```

### A5: Corporate Timeline
```
Founded in Seattle in 1994, the company expanded to San Francisco in 2000, opened European headquarters in Dublin in 2008, and established Asian operations in Singapore by 2015.
```

---

## Test Statistics

- **Total Test Cases**: 40
- **Temporal Focus**: 15 cases
- **Spatial Focus**: 15 cases
- **Combined**: 10 cases

## Key Testing Objectives

1. **Temporal Extraction**:
   - Absolute dates with explicit years
   - Year inference from context
   - Date intervals and ranges
   - Relative time expressions
   - Duration expressions
   - Historical dates (1940s-1980s)

2. **Spatial Extraction**:
   - Major world cities
   - Ambiguous place name disambiguation
   - Countries, states, and regions
   - Landmarks and specific locations
   - Multi-location contexts

3. **Combined Extraction**:
   - News reports
   - Travel itineraries
   - Event announcements
   - Historical narratives
   - Business timelines

4. **Edge Cases**:
   - No spatiotemporal information
   - Dense information
   - Nested locations
   - Multiple ambiguous references

## Expected Capabilities

- **Year Inference**: "March 17" should resolve to "March 17, 2022" when context year is 2022
- **Geographic Disambiguation**: "Broome" should resolve to Broome, Western Australia (not Broome, NY)
- **Interval Normalization**: "from X to Y" should produce ISO 8601 interval format
- **Context Propagation**: Year information should propagate across mentions in the same document
- **ISO 8601 Compliance**: All temporal outputs should follow ISO 8601 standard

## Evaluation Metrics

1. **Precision**: What percentage of extracted entities are correct?
2. **Recall**: What percentage of actual entities were extracted?
3. **F1 Score**: Harmonic mean of precision and recall
4. **Context Accuracy**: Year inference correctness
5. **Disambiguation Accuracy**: Correct location identification for ambiguous place names
