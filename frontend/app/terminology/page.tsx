'use client'

import {
  Box,
  Container,
  Heading,
  Text,
  VStack,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Badge,
  Divider,
  Button,
  HStack,
} from '@chakra-ui/react'
import { useRouter } from 'next/navigation'
import { MdArrowBack } from 'react-icons/md'

export default function TerminologyPage() {
  const router = useRouter()

  const sections = [
    {
      title: 'Core Concepts',
      icon: '📚',
      terms: [
        {
          term: 'Multi-Dimensional Extraction',
          definition:
            'The process of extracting multiple types of information (dimensions) from text simultaneously, including temporal (time), spatial (location), and custom categorical dimensions (events, diseases, etc.).',
        },
        {
          term: 'Dimension',
          definition:
            'A category of information extracted from text. STIndex supports temporal dimensions (dates/times), spatial dimensions (locations), and custom dimensions (event types, diseases, organizations, etc.).',
        },
        {
          term: 'Entity',
          definition:
            'A piece of information extracted from text, such as a date, location, or event mention. Each entity belongs to a dimension and includes metadata like confidence scores and reflection scores.',
        },
        {
          term: 'Event',
          definition:
            'A spatiotemporal occurrence extracted from text that combines multiple dimensions. An event typically includes temporal information (when it happened), spatial information (where it happened), and categorical information (what type of event). Events are the primary unit of analysis in the dashboard visualizations.',
        },
        {
          term: 'Cluster',
          definition:
            'A group of events that are close together in both space and time. Clusters help identify related events that may be part of the same incident or phenomenon. The system uses DBSCAN (Density-Based Spatial Clustering) to automatically detect spatiotemporal clusters based on geographic and temporal proximity.',
        },
        {
          term: 'Chunk',
          definition:
            'A segment of a larger document split for processing. Long documents are divided into overlapping chunks to manage context size limits while maintaining continuity.',
        },
      ],
    },
    {
      title: 'Extraction Quality Metrics',
      icon: '📊',
      terms: [
        {
          term: 'Relevance',
          definition:
            'Measures how relevant the extracted entity is to the document content. Higher relevance (closer to 100%) means the entity is explicitly mentioned and contextually appropriate.',
        },
        {
          term: 'Accuracy',
          definition:
            'Measures how accurately the entity was extracted and classified. High accuracy means the entity type, attributes, and normalized values are correct.',
        },
        {
          term: 'Completeness',
          definition:
            'Measures whether all important attributes of an entity were captured. For example, a location with coordinates and parent region is more complete than one with just a name.',
        },
        {
          term: 'Consistency',
          definition:
            'Measures whether the extracted entity is consistent with other entities in the document and follows expected patterns. Helps identify conflicting or anomalous extractions.',
        },
        {
          term: 'Confidence Score',
          definition:
            'A numerical value (0-1) indicating how confident the extraction model is about an entity. Scores ≥0.8 are considered high confidence, 0.5-0.8 medium, and <0.5 low.',
        },
        {
          term: 'Reflection Scores',
          definition:
            'Quality scores assigned through a two-pass extraction process where the LLM reviews and scores its own extractions for relevance, accuracy, completeness, and consistency.',
        },
      ],
    },
    {
      title: 'Temporal Concepts',
      icon: '⏰',
      terms: [
        {
          term: 'Temporal Entity',
          definition:
            'Any time-related information extracted from text, including specific dates (2023-03-15), date ranges (2021-2030), durations (P9Y for 9 years), or relative references (yesterday, Monday).',
        },
        {
          term: 'Normalization',
          definition:
            'The process of converting extracted temporal expressions into standardized ISO 8601 format (e.g., "March 15, 2023" → "2023-03-15"). Enables consistent comparison and analysis.',
        },
        {
          term: 'Temporal Coverage',
          definition:
            'The percentage of extracted events that have associated temporal information. Higher coverage indicates better time-based documentation.',
        },
        {
          term: 'Time Span',
          definition:
            'The duration between the earliest and latest temporal entities in the dataset, measured in days or years. Shows the temporal scope of the extracted information.',
        },
        {
          term: 'Relative Temporal Expression',
          definition:
            'Time references relative to a context date, such as "yesterday," "next Monday," or "last year." These are resolved to absolute dates using document publication dates.',
        },
      ],
    },
    {
      title: 'Spatial Concepts',
      icon: '🌍',
      terms: [
        {
          term: 'Spatial Entity',
          definition:
            'Any location-related information extracted from text, including countries, cities, regions, addresses, or geographic features.',
        },
        {
          term: 'Geocoding',
          definition:
            'The process of converting location names into geographic coordinates (latitude/longitude). Enables mapping and spatial analysis of extracted locations.',
        },
        {
          term: 'Spatial Coverage',
          definition:
            'The percentage of spatial entities that were successfully geocoded with coordinates. Higher coverage enables better map visualization.',
        },
        {
          term: 'Location Type',
          definition:
            'The category of a spatial entity: region (South-East Asia), country (Australia), city (Perth), or venue (hospital, airport). Helps understand geographic granularity.',
        },
        {
          term: 'Unique Locations',
          definition:
            'The count of distinct place names extracted from the dataset. Multiple mentions of the same location count as one unique location.',
        },
      ],
    },
    {
      title: 'Event Analysis',
      icon: '🔥',
      terms: [
        {
          term: 'Event Burst',
          definition:
            'A time period with significantly higher event activity than normal. Bursts indicate periods of intense activity, outbreaks, or concentrated incidents.',
        },
        {
          term: 'Burst Intensity',
          definition:
            'A measure of how concentrated events are during a burst period. Higher intensity means more events occurred in a shorter timeframe.',
        },
        {
          term: 'Burst Coverage',
          definition:
            'The percentage of total events that fall within detected burst periods. Shows how much of the activity is concentrated vs. evenly distributed.',
        },
        {
          term: 'Dominant Category',
          definition:
            'The most frequent event type or category within a burst period. Helps identify what type of events drove the burst (e.g., disease outbreaks, natural disasters).',
        },
      ],
    },
    {
      title: 'Categories & Dimensions',
      icon: '🏷️',
      terms: [
        {
          term: 'Category Distribution',
          definition:
            'The breakdown of events by category (e.g., disease type, event type). Shows what types of information dominate the dataset.',
        },
        {
          term: 'Custom Dimension',
          definition:
            'A user-defined dimension beyond temporal and spatial, such as disease types, event categories, organizations, or risk levels. Configured in the dimension schema.',
        },
        {
          term: 'Category Confidence',
          definition:
            'The confidence score specifically for categorical classifications. Indicates how certain the model is about assigning an entity to a particular category.',
        },
        {
          term: 'Dimension Statistics',
          definition:
            'Aggregate metrics about extracted dimensions, including total counts, coverage percentages, and top categories/sources.',
        },
      ],
    },
    {
      title: 'Visualization Types',
      icon: '📈',
      terms: [
        {
          term: 'Basic Timeline',
          definition:
            'A multi-track D3 visualization showing temporal entities over time with quality scores. Each track represents a different category or dimension.',
        },
        {
          term: 'Dimension Breakdown',
          definition:
            'A dimension-agnostic analysis showing how entities are distributed across all extracted dimensions, regardless of type.',
        },
        {
          term: 'Interactive Map',
          definition:
            'A Mapbox-powered visualization showing spatial entities with clustering, story arcs, and optional animation over time.',
        },
        {
          term: 'Story Timeline',
          definition:
            'A visualization combining temporal progression with burst detection, showing how event intensity changes over time.',
        },
        {
          term: 'Entity Network',
          definition:
            'A graph visualization showing co-occurrence relationships between entities. Connected entities appear together frequently in the source documents.',
        },
      ],
    },
  ]

  return (
    <Box minH="100vh" bg="gray.50" py={8}>
      <Container maxW="5xl" px={4}>
        <VStack spacing={6} align="stretch">
          <HStack>
            <Button
              leftIcon={<MdArrowBack />}
              variant="ghost"
              onClick={() => router.push('/')}
              size="sm"
            >
              Back to Dashboard
            </Button>
          </HStack>

          <Box>
            <Heading as="h1" size="2xl" mb={2}>
              Terminology Guide
            </Heading>
            <Text color="gray.600" fontSize="lg">
              Understanding the concepts and metrics used in STIndex
            </Text>
          </Box>

          <Divider />

          <Accordion allowMultiple defaultIndex={[0, 1, 2, 3, 4, 5, 6]} allowToggle>
            {sections.map((section, idx) => (
              <AccordionItem key={idx} bg="white" mb={4} borderRadius="lg" border="none" shadow="md">
                <AccordionButton py={4} _hover={{ bg: 'gray.50' }} borderRadius="lg">
                  <HStack flex="1" textAlign="left" spacing={3}>
                    <Text fontSize="2xl">{section.icon}</Text>
                    <Heading as="h2" size="md">
                      {section.title}
                    </Heading>
                    <Badge colorScheme="blue">{section.terms.length} terms</Badge>
                  </HStack>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel pb={6} pt={2}>
                  <VStack align="stretch" spacing={4} divider={<Divider />}>
                    {section.terms.map((item, termIdx) => (
                      <Box key={termIdx}>
                        <Text fontWeight="bold" fontSize="md" color="blue.600" mb={2}>
                          {item.term}
                        </Text>
                        <Text fontSize="sm" color="gray.700" lineHeight="tall">
                          {item.definition}
                        </Text>
                      </Box>
                    ))}
                  </VStack>
                </AccordionPanel>
              </AccordionItem>
            ))}
          </Accordion>

          <Box bg="blue.50" p={6} borderRadius="lg" borderLeft="4px solid" borderColor="blue.500">
            <Heading as="h3" size="sm" mb={2} color="blue.900">
              About STIndex
            </Heading>
            <Text fontSize="sm" color="blue.900">
              STIndex is a multi-dimensional information extraction system that uses Large Language
              Models (LLMs) to extract temporal, spatial, and custom dimensional data from
              unstructured text. The system features context-aware extraction, two-pass reflection
              for quality filtering, and comprehensive post-processing tools for geocoding and
              temporal normalization.
            </Text>
          </Box>
        </VStack>
      </Container>
    </Box>
  )
}
