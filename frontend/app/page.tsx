'use client'

import { useEffect, useState, useMemo } from 'react'
import dynamic from 'next/dynamic'
import { Box, Container, Heading, Text, Spinner, Center, VStack, Tabs, TabList, TabPanels, Tab, TabPanel } from '@chakra-ui/react'
import { DashboardStats } from './components/DashboardStats'
import { TemporalTimeline } from './components/TemporalTimeline'
import { DimensionBreakdown } from './components/DimensionBreakdown'
import { AnalyticsPanels } from './components/AnalyticsPanels'
import { SpatioTemporalEvent } from './lib/analytics'

// Dynamically import heavy components with SSR disabled
const InteractiveMap = dynamic(() => import('./components/InteractiveMap').then(mod => mod.InteractiveMap), {
  ssr: false,
  loading: () => (
    <Center h="600px">
      <Spinner size="xl" color="blue.500" />
    </Center>
  ),
})

const StoryTimeline = dynamic(() => import('./components/StoryTimeline').then(mod => mod.StoryTimeline), {
  ssr: false,
  loading: () => (
    <Center h="400px">
      <Spinner size="xl" color="blue.500" />
    </Center>
  ),
})

const EntityNetwork = dynamic(() => import('./components/EntityNetwork').then(mod => mod.EntityNetwork), {
  ssr: false,
  loading: () => (
    <Center h="600px">
      <Spinner size="xl" color="blue.500" />
    </Center>
  ),
})

interface ExtractionResult {
  chunk_id: string
  document_id: string
  document_title: string
  source: string | null
  text: string
  extraction: {
    success: boolean
    entities?: {
      temporal?: any[]
      spatial?: any[]
      [key: string]: any[] | undefined
    }
    error?: string
  }
}

interface StoryArc {
  story_id: string
  length: number
  progression_type: string
  confidence: number
  temporal_span: {
    start: string
    end: string
    duration_days: number
  }
  spatial_span: any
  narrative_summary: any
  key_dimensions: any
  event_ids: string[]
}

interface BackendClusters {
  clusters: any[]
  burst_periods: any[]
  statistics: any
}

export default function Home() {
  const [data, setData] = useState<ExtractionResult[]>([])
  const [storyArcs, setStoryArcs] = useState<StoryArc[]>([])
  const [backendClusters, setBackendClusters] = useState<BackendClusters | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTabIndex, setActiveTabIndex] = useState(0)
  const [isMounted, setIsMounted] = useState(false)

  // Ensure component only fully renders after client-side hydration
  useEffect(() => {
    setIsMounted(true)
  }, [])

  useEffect(() => {
    Promise.all([
      fetch('/data/extraction_results.json').then((res) => res.json()),
      fetch('/data/story_arcs.json').then((res) => res.json()),
      fetch('/data/clusters.json').then((res) => res.json())
    ])
      .then(([results, arcs, clusters]) => {
        setData(results)
        setStoryArcs(arcs)
        setBackendClusters(clusters)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  // Filter successful extractions - memoize to prevent infinite re-renders
  // IMPORTANT: Must be called before any conditional returns to satisfy Rules of Hooks
  const successfulExtractions = useMemo(
    () => data.filter((item) => item.extraction.success),
    [data]
  )

  // Transform extraction data into SpatioTemporalEvent format for analytics
  const spatioTemporalEvents = useMemo<SpatioTemporalEvent[]>(() => {
    if (!data || data.length === 0) return []

    const successful = data.filter((item) => item.extraction.success)
    const events: SpatioTemporalEvent[] = []

    successful.forEach((item) => {
      const { extraction, document_id, document_title, source } = item
      const entities = extraction.entities || {}

      const temporal_entities = entities.temporal || []
      const spatial_entities = entities.spatial || []

      // Use document_title as source if source is null
      const eventSource = source || document_title || 'Unknown'

      // Process spatial entities
      spatial_entities.forEach((entity: any, idx: number) => {
        if (entity.latitude && entity.longitude) {
          events.push({
            id: `${document_id}-spatial-${idx}`,
            text: entity.text,
            latitude: entity.latitude,
            longitude: entity.longitude,
            timestamp: temporal_entities[0]?.normalized || undefined,
            normalized_date: temporal_entities[0]?.normalized || undefined,
            // Use location_type from original data
            category: entity.location_type || entity.dimension_name || 'spatial',
            confidence: entity.confidence,
            document_id,
            document_title,
            source: eventSource,
            custom_dimensions: {
              ...entity,
              dimension_name: entity.dimension_name,
              reflection_scores: entity.reflection_scores,
            },
          })
        }
      })

      // Process temporal entities
      temporal_entities.forEach((entity: any, idx: number) => {
        events.push({
          id: `${document_id}-temporal-${idx}`,
          text: entity.text,
          timestamp: entity.normalized,
          normalized_date: entity.normalized,
          // Use normalization_type from original data (date, duration, etc.)
          category: entity.normalization_type || entity.dimension_name || 'temporal',
          confidence: entity.confidence,
          document_id,
          document_title,
          source: eventSource,
          custom_dimensions: {
            ...entity,
            dimension_name: entity.dimension_name,
            normalized: entity.normalized,
            normalization_type: entity.normalization_type,
            reflection_scores: entity.reflection_scores,
          },
        })
      })

      // Process custom dimension entities (event_type, disease, venue_type, etc.)
      Object.entries(entities).forEach(([dimName, dimEntities]) => {
        if (dimName === 'temporal' || dimName === 'spatial') return
        if (!Array.isArray(dimEntities)) return

        dimEntities.forEach((entity: any, idx: number) => {
          events.push({
            id: `${document_id}-${dimName}-${idx}`,
            text: entity.text,
            // Use actual category from entity data, fallback to dimension name
            category: entity.category || dimName,
            confidence: entity.confidence,
            // Include timestamp if available from temporal entities
            timestamp: temporal_entities[0]?.normalized || undefined,
            normalized_date: temporal_entities[0]?.normalized || undefined,
            document_id,
            document_title,
            source: eventSource,
            custom_dimensions: {
              ...entity,
              dimension_name: entity.dimension_name,
              category_confidence: entity.category_confidence,
              reflection_scores: entity.reflection_scores,
            },
          })
        })
      })
    })

    return events
  }, [data])

  // Prevent hydration mismatch by not rendering until mounted on client
  if (!isMounted) {
    return null
  }

  if (loading) {
    return (
      <Center minH="100vh" bg="gray.50">
        <VStack spacing={4}>
          <Spinner size="xl" color="blue.500" thickness="4px" />
          <Text color="gray.600">Loading extraction data...</Text>
        </VStack>
      </Center>
    )
  }

  if (error) {
    return (
      <Center minH="100vh" bg="gray.50">
        <Text color="red.500" fontSize="xl">Error loading data: {error}</Text>
      </Center>
    )
  }

  return (
    <Box minH="100vh" bg="gray.50" py={8}>
      <Container maxW="7xl" px={4}>
        <VStack spacing={8} align="stretch">
          <Box>
            <Heading as="h1" size="2xl" mb={2}>
              STIndex Dashboard
            </Heading>
            <Text color="gray.600" fontSize="lg">
              Multi-Dimensional Spatiotemporal Information Extraction & Analysis
            </Text>
          </Box>

          {/* Overview Stats */}
          <DashboardStats data={successfulExtractions} />

          {/* Analytics Panels */}
          <Box>
            <Heading as="h2" size="lg" mb={4}>
              Advanced Analytics
            </Heading>
            <AnalyticsPanels
              events={spatioTemporalEvents}
              storyArcs={storyArcs}
              backendClusters={backendClusters}
            />
          </Box>

          {/* Tabbed Visualizations */}
          <Box bg="white" p={6} borderRadius="lg" shadow="md">
            <Tabs
              colorScheme="blue"
              variant="enclosed"
              index={activeTabIndex}
              onChange={(index) => setActiveTabIndex(index)}
            >
              <TabList>
                <Tab>Basic Timeline</Tab>
                <Tab>Dimension Breakdown</Tab>
                <Tab>Interactive Map</Tab>
                <Tab>Story Timeline</Tab>
                <Tab>Entity Network</Tab>
              </TabList>

              <TabPanels>
                {/* Basic Timeline */}
                <TabPanel>
                  <VStack align="stretch" spacing={4}>
                    <Text fontSize="sm" color="gray.600">
                      Temporal entity timeline with quality scores
                    </Text>
                    {activeTabIndex === 0 && (
                      <TemporalTimeline data={successfulExtractions} />
                    )}
                  </VStack>
                </TabPanel>

                {/* Dimension Breakdown */}
                <TabPanel>
                  <VStack align="stretch" spacing={4}>
                    <Text fontSize="sm" color="gray.600">
                      Dimension-agnostic entity analysis
                    </Text>
                    {activeTabIndex === 1 && (
                      <DimensionBreakdown data={successfulExtractions} />
                    )}
                  </VStack>
                </TabPanel>

                {/* Interactive Map */}
                <TabPanel p={0} display="flex" flexDirection="column" height="800px">
                  <Box px={6} pt={4} pb={2} flexShrink={0}>
                    <Text fontSize="sm" color="gray.600">
                      Spatiotemporal event clustering with story arc visualization
                    </Text>
                  </Box>
                  <Box flex="1">
                    {activeTabIndex === 2 && (
                      <InteractiveMap
                        events={spatioTemporalEvents}
                        storyArcs={storyArcs}
                        backendClusters={backendClusters}
                        height="100%"
                        showClusters={true}
                        showStoryArcs={true}
                        enableAnimation={true}
                      />
                    )}
                  </Box>
                </TabPanel>

                {/* Story Timeline */}
                <TabPanel>
                  <VStack align="stretch" spacing={4}>
                    <Text fontSize="sm" color="gray.600">
                      Multi-track timeline with burst detection and story arcs
                    </Text>
                    {activeTabIndex === 3 && (
                      <StoryTimeline
                        events={spatioTemporalEvents}
                        storyArcs={storyArcs}
                        burstPeriods={backendClusters?.burst_periods}
                        height={500}
                        showBursts={true}
                        showStoryArcs={true}
                      />
                    )}
                  </VStack>
                </TabPanel>

                {/* Entity Network */}
                <TabPanel>
                  <VStack align="stretch" spacing={4}>
                    <Text fontSize="sm" color="gray.600">
                      Entity co-occurrence network graph
                    </Text>
                    {activeTabIndex === 4 && (
                      <EntityNetwork
                        events={spatioTemporalEvents}
                        height="600px"
                        minCoOccurrence={2}
                      />
                    )}
                  </VStack>
                </TabPanel>
              </TabPanels>
            </Tabs>
          </Box>
        </VStack>
      </Container>
    </Box>
  )
}
