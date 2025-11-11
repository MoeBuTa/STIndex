'use client'

import { useEffect, useState, useMemo } from 'react'
import { Box, Container, Heading, Text, Spinner, Center, VStack, Tabs, TabList, TabPanels, Tab, TabPanel } from '@chakra-ui/react'
import { DashboardStats } from './components/DashboardStats'
import { TemporalTimeline } from './components/TemporalTimeline'
import { DimensionBreakdown } from './components/DimensionBreakdown'
import { InteractiveMap } from './components/InteractiveMap'
import { StoryTimeline } from './components/StoryTimeline'
import { AnalyticsPanels } from './components/AnalyticsPanels'
import { EntityNetwork } from './components/EntityNetwork'
import { SpatioTemporalEvent } from './lib/analytics'

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
            category: entity.location_type || 'unknown',
            document_id,
            document_title,
            source: eventSource,
            custom_dimensions: entity,
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
          category: 'unknown',
          document_id,
          document_title,
          source: eventSource,
          custom_dimensions: entity,
        })
      })

      // Process custom dimension entities (event_type, disease, etc.)
      Object.entries(entities).forEach(([dimName, dimEntities]) => {
        if (dimName === 'temporal' || dimName === 'spatial') return
        if (!Array.isArray(dimEntities)) return

        dimEntities.forEach((entity: any, idx: number) => {
          events.push({
            id: `${document_id}-${dimName}-${idx}`,
            text: entity.text,
            category: entity.category || dimName,
            document_id,
            document_title,
            source: eventSource,
            custom_dimensions: entity,
          })
        })
      })
    })

    return events
  }, [data])

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
