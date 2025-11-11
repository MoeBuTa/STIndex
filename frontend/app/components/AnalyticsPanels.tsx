'use client'

import { useMemo } from 'react'
import {
  Box,
  SimpleGrid,
  VStack,
  HStack,
  Text,
  Badge,
  Progress,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Divider,
  List,
  ListItem,
  ListIcon,
} from '@chakra-ui/react'
import { MdTrendingUp, MdLocationOn, MdEvent, MdTimeline } from 'react-icons/md'
import {
  detectBursts,
  extractStoryArcs,
  clusterEvents,
  SpatioTemporalEvent,
} from '../lib/analytics'

interface AnalyticsPanelsProps {
  events: SpatioTemporalEvent[]
}

export function AnalyticsPanels({ events }: AnalyticsPanelsProps) {
  // Compute analytics
  const { bursts, stories, clusters, qualityMetrics, dimensionStats } = useMemo(() => {
    const temporalEvents = events.filter((e) => e.timestamp || e.normalized_date)
    const spatialEvents = events.filter((e) => e.latitude && e.longitude)

    const burstsData = detectBursts(temporalEvents, 1, 3)
    const clustersData = clusterEvents(events, 50, 7, 3)
    const storiesData = extractStoryArcs(clustersData, 0.3)

    // Calculate quality metrics from reflection scores
    const eventsWithScores = events.filter(
      (e) => e.custom_dimensions && 'reflection_scores' in e.custom_dimensions
    )

    let avgRelevance = 0
    let avgAccuracy = 0
    let avgCompleteness = 0
    let avgConsistency = 0

    if (eventsWithScores.length > 0) {
      eventsWithScores.forEach((e: any) => {
        const scores = e.custom_dimensions?.reflection_scores
        if (scores) {
          avgRelevance += scores.relevance || 0
          avgAccuracy += scores.accuracy || 0
          avgCompleteness += scores.completeness || 0
          avgConsistency += scores.consistency || 0
        }
      })

      const count = eventsWithScores.length
      avgRelevance /= count
      avgAccuracy /= count
      avgCompleteness /= count
      avgConsistency /= count
    }

    const quality = {
      relevance: avgRelevance,
      accuracy: avgAccuracy,
      completeness: avgCompleteness,
      consistency: avgConsistency,
      totalEvents: events.length,
      withScores: eventsWithScores.length,
      avgConfidence:
        events.reduce((sum, e: any) => sum + (e.confidence || 0), 0) / events.length || 0,
    }

    // Dimension statistics
    const categories = new Map<string, number>()
    events.forEach((e) => {
      const cat = e.category || 'unknown'
      categories.set(cat, (categories.get(cat) || 0) + 1)
    })

    const sources = new Map<string, number>()
    events.forEach((e) => {
      sources.set(e.source, (sources.get(e.source) || 0) + 1)
    })

    const dimStats = {
      totalEvents: events.length,
      temporalEvents: temporalEvents.length,
      spatialEvents: spatialEvents.length,
      categories: Array.from(categories.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5),
      topSources: Array.from(sources.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3),
      temporalCoverage:
        temporalEvents.length > 0
          ? ((temporalEvents.length / events.length) * 100).toFixed(1)
          : '0',
      spatialCoverage:
        spatialEvents.length > 0
          ? ((spatialEvents.length / events.length) * 100).toFixed(1)
          : '0',
    }

    return {
      bursts: burstsData,
      stories: storiesData,
      clusters: clustersData,
      qualityMetrics: quality,
      dimensionStats: dimStats,
    }
  }, [events])

  return (
    <SimpleGrid columns={{ base: 1, lg: 2 }} spacing={6}>
      {/* Panel 1: Extraction Quality Metrics */}
      <Box bg="white" p={6} borderRadius="lg" shadow="md">
        <VStack align="stretch" spacing={4}>
          <HStack>
            <MdTrendingUp size={24} color="#3182ce" />
            <Text fontSize="lg" fontWeight="bold">
              Extraction Quality
            </Text>
          </HStack>
          <Divider />

          <SimpleGrid columns={2} spacing={4}>
            <Stat>
              <StatLabel fontSize="xs">Relevance</StatLabel>
              <StatNumber fontSize="2xl">
                {(qualityMetrics.relevance * 100).toFixed(0)}%
              </StatNumber>
              <Progress
                value={qualityMetrics.relevance * 100}
                colorScheme="green"
                size="sm"
                mt={2}
              />
            </Stat>

            <Stat>
              <StatLabel fontSize="xs">Accuracy</StatLabel>
              <StatNumber fontSize="2xl">
                {(qualityMetrics.accuracy * 100).toFixed(0)}%
              </StatNumber>
              <Progress
                value={qualityMetrics.accuracy * 100}
                colorScheme="blue"
                size="sm"
                mt={2}
              />
            </Stat>

            <Stat>
              <StatLabel fontSize="xs">Completeness</StatLabel>
              <StatNumber fontSize="2xl">
                {(qualityMetrics.completeness * 100).toFixed(0)}%
              </StatNumber>
              <Progress
                value={qualityMetrics.completeness * 100}
                colorScheme="purple"
                size="sm"
                mt={2}
              />
            </Stat>

            <Stat>
              <StatLabel fontSize="xs">Consistency</StatLabel>
              <StatNumber fontSize="2xl">
                {(qualityMetrics.consistency * 100).toFixed(0)}%
              </StatNumber>
              <Progress
                value={qualityMetrics.consistency * 100}
                colorScheme="orange"
                size="sm"
                mt={2}
              />
            </Stat>
          </SimpleGrid>

          <Box mt={2}>
            <Text fontSize="xs" color="gray.600">
              Avg Confidence: {(qualityMetrics.avgConfidence * 100).toFixed(1)}%
            </Text>
            <Text fontSize="xs" color="gray.600">
              Events with Scores: {qualityMetrics.withScores} / {qualityMetrics.totalEvents}
            </Text>
          </Box>
        </VStack>
      </Box>

      {/* Panel 2: Event Burst Detection */}
      <Box bg="white" p={6} borderRadius="lg" shadow="md">
        <VStack align="stretch" spacing={4}>
          <HStack>
            <MdEvent size={24} color="#e53e3e" />
            <Text fontSize="lg" fontWeight="bold">
              Event Bursts
            </Text>
            <Badge colorScheme="red">{bursts.length}</Badge>
          </HStack>
          <Divider />

          {bursts.length === 0 ? (
            <Text fontSize="sm" color="gray.500">
              No significant burst periods detected
            </Text>
          ) : (
            <VStack align="stretch" spacing={3} maxH="300px" overflowY="auto">
              {bursts.slice(0, 5).map((burst, idx) => (
                <Box key={idx} p={3} bg="red.50" borderRadius="md" borderLeft="3px solid" borderColor="red.500">
                  <HStack justify="space-between" mb={1}>
                    <Text fontSize="sm" fontWeight="bold">
                      Burst {idx + 1}
                    </Text>
                    <Badge colorScheme="red" fontSize="xs">
                      {burst.eventCount} events
                    </Badge>
                  </HStack>
                  <Text fontSize="xs" color="gray.600">
                    {burst.start} → {burst.end}
                  </Text>
                  <Text fontSize="xs" color="gray.600">
                    Peak: {burst.peakTime} (Intensity: {burst.intensity.toFixed(1)})
                  </Text>
                  {burst.dominantLocation && (
                    <Text fontSize="xs" color="gray.600">
                      Location: {burst.dominantLocation}
                    </Text>
                  )}
                  {burst.dominantCategory && (
                    <Badge colorScheme="purple" fontSize="xs" mt={1}>
                      {burst.dominantCategory}
                    </Badge>
                  )}
                </Box>
              ))}
            </VStack>
          )}
        </VStack>
      </Box>

      {/* Panel 3: Story Arc Summary */}
      <Box bg="white" p={6} borderRadius="lg" shadow="md">
        <VStack align="stretch" spacing={4}>
          <HStack>
            <MdTimeline size={24} color="#805ad5" />
            <Text fontSize="lg" fontWeight="bold">
              Story Arcs
            </Text>
            <Badge colorScheme="purple">{stories.length}</Badge>
          </HStack>
          <Divider />

          {stories.length === 0 ? (
            <Text fontSize="sm" color="gray.500">
              No story arcs detected
            </Text>
          ) : (
            <VStack align="stretch" spacing={3} maxH="300px" overflowY="auto">
              {stories.slice(0, 5).map((story, idx) => (
                <Box
                  key={idx}
                  p={3}
                  bg="purple.50"
                  borderRadius="md"
                  borderLeft="3px solid"
                  borderColor="purple.500"
                >
                  <HStack justify="space-between" mb={1}>
                    <Text fontSize="sm" fontWeight="bold">
                      Story {idx + 1}
                    </Text>
                    <Badge colorScheme="purple" fontSize="xs">
                      {story.clusters.length} clusters
                    </Badge>
                  </HStack>
                  <Text fontSize="xs" color="gray.600" mb={1}>
                    {story.narrative}
                  </Text>
                  <Text fontSize="xs" color="gray.500">
                    Strength: {story.strength.toFixed(1)} | Entities: {story.entities.length}
                  </Text>
                  <HStack spacing={1} mt={1} flexWrap="wrap">
                    {story.locations.slice(0, 3).map((loc, locIdx) => (
                      <Badge key={locIdx} colorScheme="blue" fontSize="xs">
                        {loc.name}
                      </Badge>
                    ))}
                  </HStack>
                </Box>
              ))}
            </VStack>
          )}
        </VStack>
      </Box>

      {/* Panel 4: Dimension Statistics */}
      <Box bg="white" p={6} borderRadius="lg" shadow="md">
        <VStack align="stretch" spacing={4}>
          <HStack>
            <MdLocationOn size={24} color="#38a169" />
            <Text fontSize="lg" fontWeight="bold">
              Dimension Statistics
            </Text>
          </HStack>
          <Divider />

          <SimpleGrid columns={2} spacing={4}>
            <Stat>
              <StatLabel fontSize="xs">Total Events</StatLabel>
              <StatNumber fontSize="2xl">{dimensionStats.totalEvents}</StatNumber>
            </Stat>

            <Stat>
              <StatLabel fontSize="xs">Clusters</StatLabel>
              <StatNumber fontSize="2xl">{clusters.length}</StatNumber>
            </Stat>

            <Stat>
              <StatLabel fontSize="xs">Temporal Coverage</StatLabel>
              <StatNumber fontSize="2xl">{dimensionStats.temporalCoverage}%</StatNumber>
              <StatHelpText fontSize="xs">
                {dimensionStats.temporalEvents} events
              </StatHelpText>
            </Stat>

            <Stat>
              <StatLabel fontSize="xs">Spatial Coverage</StatLabel>
              <StatNumber fontSize="2xl">{dimensionStats.spatialCoverage}%</StatNumber>
              <StatHelpText fontSize="xs">
                {dimensionStats.spatialEvents} events
              </StatHelpText>
            </Stat>
          </SimpleGrid>

          <Box>
            <Text fontSize="sm" fontWeight="medium" mb={2}>
              Top Categories
            </Text>
            <List spacing={1}>
              {dimensionStats.categories.map(([cat, count]) => (
                <ListItem key={cat} fontSize="xs">
                  <HStack justify="space-between">
                    <Text>{cat}</Text>
                    <Badge colorScheme="blue" fontSize="xs">
                      {count}
                    </Badge>
                  </HStack>
                </ListItem>
              ))}
            </List>
          </Box>

          <Box>
            <Text fontSize="sm" fontWeight="medium" mb={2}>
              Top Sources
            </Text>
            <List spacing={1}>
              {dimensionStats.topSources.map(([source, count]) => (
                <ListItem key={source} fontSize="xs">
                  <HStack justify="space-between">
                    <Text isTruncated maxW="150px">
                      {source}
                    </Text>
                    <Badge colorScheme="green" fontSize="xs">
                      {count}
                    </Badge>
                  </HStack>
                </ListItem>
              ))}
            </List>
          </Box>
        </VStack>
      </Box>
    </SimpleGrid>
  )
}
