'use client'

import { useEffect, useState, useMemo } from 'react'
import dynamic from 'next/dynamic'
import { Box, Container, Heading, Text, Spinner, Center, VStack, Tabs, TabList, TabPanels, Tab, TabPanel, Button, HStack } from '@chakra-ui/react'
import { useRouter } from 'next/navigation'
import { MdHelp } from 'react-icons/md'
import { DashboardStats } from './DashboardStats'
import { TemporalTimeline } from './TemporalTimeline'
import { DimensionBreakdown } from './DimensionBreakdown'
import { AnalyticsPanels } from './AnalyticsPanels'
import { SpatioTemporalEvent } from '../lib/analytics'

// Dynamically import heavy components with SSR disabled
const InteractiveMap = dynamic(() => import('./InteractiveMap').then(mod => mod.InteractiveMap), {
  ssr: false,
  loading: () => (
    <Center h="600px">
      <Spinner size="xl" color="blue.500" />
    </Center>
  ),
})

const StoryTimeline = dynamic(() => import('./StoryTimeline').then(mod => mod.StoryTimeline), {
  ssr: false,
  loading: () => (
    <Center h="400px">
      <Spinner size="xl" color="blue.500" />
    </Center>
  ),
})

const EntityNetwork = dynamic(() => import('./EntityNetwork').then(mod => mod.EntityNetwork), {
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

interface BackendClusters {
  clusters: any[]
  burst_periods: any[]
  statistics: any
}

export default function DashboardContent() {
  const router = useRouter()
  const [data, setData] = useState<ExtractionResult[]>([])
  const [backendClusters, setBackendClusters] = useState<BackendClusters | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTabIndex, setActiveTabIndex] = useState(0)

  useEffect(() => {
    Promise.all([
      fetch('/data/extraction_results.json').then((res) => res.json()),
      fetch('/data/clusters.json').then((res) => res.json())
    ])
      .then(([results, clusters]) => {
        setData(results)
        setBackendClusters(clusters)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  const successfulExtractions = useMemo(
    () => data.filter((item) => item.extraction.success),
    [data]
  )

  const spatioTemporalEvents = useMemo<SpatioTemporalEvent[]>(() => {
    if (!data || data.length === 0) return []

    const successful = data.filter((item) => item.extraction.success)
    const events: SpatioTemporalEvent[] = []

    successful.forEach((item) => {
      const { extraction, document_id, document_title, source } = item
      const entities = extraction.entities || {}

      const temporal_entities = entities.temporal || []
      const spatial_entities = entities.spatial || []

      const eventSource = source || document_title || 'Unknown'

      spatial_entities.forEach((entity: any, idx: number) => {
        if (entity.latitude && entity.longitude) {
          events.push({
            id: `${document_id}-spatial-${idx}`,
            text: entity.text,
            latitude: entity.latitude,
            longitude: entity.longitude,
            timestamp: temporal_entities[0]?.normalized || undefined,
            normalized_date: temporal_entities[0]?.normalized || undefined,
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

      temporal_entities.forEach((entity: any, idx: number) => {
        events.push({
          id: `${document_id}-temporal-${idx}`,
          text: entity.text,
          timestamp: entity.normalized,
          normalized_date: entity.normalized,
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

      Object.entries(entities).forEach(([dimName, dimEntities]) => {
        if (dimName === 'temporal' || dimName === 'spatial') return
        if (!Array.isArray(dimEntities)) return

        dimEntities.forEach((entity: any, idx: number) => {
          events.push({
            id: `${document_id}-${dimName}-${idx}`,
            text: entity.text,
            category: entity.category || dimName,
            confidence: entity.confidence,
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
          <HStack justify="space-between" align="flex-start">
            <Box>
              <Heading as="h1" size="2xl" mb={2}>
                STIndex Dashboard
              </Heading>
              <Text color="gray.600" fontSize="lg">
                Multi-Dimensional Spatiotemporal Information Extraction & Analysis
              </Text>
            </Box>
            <Button
              leftIcon={<MdHelp />}
              colorScheme="blue"
              variant="outline"
              onClick={() => router.push('/terminology')}
              size="md"
            >
              Terminology Guide
            </Button>
          </HStack>

          <DashboardStats data={successfulExtractions} />

          <Box>
            <Heading as="h2" size="lg" mb={4}>
              Advanced Analytics
            </Heading>
            <AnalyticsPanels
              events={spatioTemporalEvents}
              backendClusters={backendClusters}
            />
          </Box>

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
                <Tab>Event Timeline</Tab>
                <Tab>Entity Network</Tab>
              </TabList>

              <TabPanels>
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
                        backendClusters={backendClusters}
                        height="100%"
                        showClusters={true}
                        enableAnimation={true}
                      />
                    )}
                  </Box>
                </TabPanel>

                <TabPanel>
                  <VStack align="stretch" spacing={4}>
                    <Text fontSize="sm" color="gray.600">
                      Multi-track event timeline with category analysis
                    </Text>
                    {activeTabIndex === 3 && (
                      <StoryTimeline
                        events={spatioTemporalEvents}
                        height={500}
                        showBursts={false}
                      />
                    )}
                  </VStack>
                </TabPanel>

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
